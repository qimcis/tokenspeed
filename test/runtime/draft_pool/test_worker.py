# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tokenspeed.runtime.draft_pool.worker import (
    DFlash2WorkerEngine,
    WorkerDraftJob,
    WorkerSessionTable,
)
from tokenspeed.runtime.draft_pool.worker_cache import (
    WorkerAttentionBackend,
    WorkerCacheGeometry,
    WorkerKVPool,
    validate_worker_attention,
)
from tokenspeed.runtime.draft_pool.worker_weights import (
    TARGET_VOCABULARY_NAMES,
    CheckpointFiles,
    iter_checkpoint_weights,
    load_checkpoint_config,
    load_complete_draft_weights,
    resolve_checkpoint,
    selected_shards,
)


def test_worker_import_does_not_initialize_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tokenspeed.runtime.draft_pool.worker; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_checkpoint_config_uses_shared_target_normalization(monkeypatch, tmp_path):
    calls = []

    def get_config(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(normalized=True)

    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.utils.hf_transformers_utils",
        SimpleNamespace(get_config=get_config),
    )
    for is_draft, name in ((False, "target"), (True, "draft")):
        checkpoint = CheckpointFiles(tmp_path / name / "config.json", (), None)
        config = load_checkpoint_config(checkpoint, "a" * 40, is_draft)
        assert config.normalized
        assert calls[-1] == {
            "model": str(tmp_path / name),
            "trust_remote_code": False,
            "revision": "a" * 40,
            "model_override_args": None,
            "is_draft_worker": is_draft,
            "speculative_algorithm": "DFLASH" if is_draft else None,
        }


def test_session_capacity_and_unique_reset_identity():
    sessions = WorkerSessionTable(1, 7)
    original = sessions.open("a")
    with pytest.raises(ValueError, match="unique"):
        sessions.open("a")
    with pytest.raises(RuntimeError, match="capacity"):
        sessions.open("b")
    assert sessions.close("a") == original.slot
    assert sessions.close("a") is None
    assert sessions.open("b").slot == original.slot
    with pytest.raises(ValueError, match="missing"):
        sessions.get("a")


def test_snapshot_window_delta_and_transactional_endpoint():
    sessions = WorkerSessionTable(1, 7)
    session = sessions.open("a")
    with pytest.raises(ValueError, match="entire retained"):
        sessions.validate_update("a", 95, 100, True)
    assert sessions.validate_update("a", 93, 100, True) is session
    assert session.confirmed_endpoint is None
    session.confirmed_endpoint = 100
    assert sessions.validate_update("a", 100, 106, False) is session
    assert session.confirmed_endpoint == 100
    with pytest.raises(ValueError, match="installed endpoint"):
        sessions.validate_update("a", 99, 105, False)
    with pytest.raises(ValueError, match="new session"):
        sessions.validate_update("a", 93, 100, True)
    with pytest.raises(ValueError, match="history bound"):
        sessions.validate_update("a", 100, 108, False)


@pytest.mark.parametrize("endpoint", [0, 1, 7, 8, 15, 1000003])
def test_context_ring_order_and_scratch_isolation(endpoint):
    geometry = WorkerCacheGeometry(2, 8, 8)
    start = max(0, endpoint - 7)
    context = geometry.context_locations(0, start, endpoint)
    scratch = geometry.draft_locations(0)
    assert len(set(context)) == endpoint - start
    assert not set(context).intersection(scratch)
    assert geometry.page_row(0, endpoint) == context + scratch
    assert not set(geometry.page_row(0, endpoint)).intersection(
        geometry.page_row(1, endpoint)
    )
    assert 0 not in context + scratch


def test_cold_snapshot_matches_incremental_ring_after_wrap():
    geometry = WorkerCacheGeometry(2, 8, 8)
    pool = WorkerKVPool(geometry, 1, 1, 2, "cpu")
    layer = SimpleNamespace(layer_id=0)

    def append(slot, start, stop):
        locations = torch.tensor(
            geometry.context_locations(slot, start, stop), dtype=torch.int64
        )
        values = (
            torch.arange(start, stop, dtype=torch.float32)
            .view(-1, 1, 1)
            .expand(-1, 1, 2)
            .to(torch.bfloat16)
        )
        pool.set_kv_buffer(layer, locations, values, -values, None, None)

    append(0, 0, 7)
    append(0, 7, 12)
    append(0, 12, 18)
    append(1, 11, 18)
    lhs = geometry.context_locations(0, 11, 18)
    rhs = geometry.context_locations(1, 11, 18)
    keys, values = pool.get_kv_buffer(0)
    assert torch.equal(keys[lhs], keys[rhs])
    assert torch.equal(values[lhs], values[rhs])
    before = keys[lhs].clone()
    scratch = torch.tensor(geometry.draft_locations(0), dtype=torch.int64)
    poison = torch.full((8, 1, 2), 500.0, dtype=torch.bfloat16)
    pool.set_kv_buffer(layer, scratch, poison, poison, None, None)
    assert torch.equal(keys[lhs], before)
    pool.clear_slot(0)
    assert (
        torch.count_nonzero(
            keys[geometry.base(0) : geometry.base(0) + geometry.slot_tokens]
        )
        == 0
    )
    assert torch.equal(keys[rhs], before)
    assert torch.count_nonzero(keys[0]) == 0


def test_native_queries_share_noncausal_block_end_metadata():
    geometry = WorkerCacheGeometry(3, 8, 8)
    backend = WorkerAttentionBackend(geometry, "cpu")
    backend.prepare([2, 0], [103, 3])
    assert backend._page_table.shape == (16, 15)
    assert backend._seq_lens.tolist() == [15] * 8 + [11] * 8
    assert all(
        torch.equal(backend._page_table[0], backend._page_table[i]) for i in range(8)
    )
    assert backend._page_table[8, 11:].tolist() == [0] * 4
    assert backend._write_locations.tolist() == geometry.draft_locations(
        2
    ) + geometry.draft_locations(0)
    with pytest.raises(ValueError, match="distinct"):
        backend.prepare([0, 0], [3, 3])


def test_bounded_cache_rejects_incompatible_built_attention():
    geometry = WorkerCacheGeometry(2, 8, 8)
    attention = SimpleNamespace(
        attn=SimpleNamespace(sliding_window_size=7), num_kv_heads=8, head_dim=128
    )
    model = SimpleNamespace(
        _uses_mla=False, layers=[SimpleNamespace(self_attn=attention)]
    )
    validate_worker_attention(model, geometry, 8, 128)
    attention.attn.sliding_window_size = -1
    with pytest.raises(ValueError, match="sliding window"):
        validate_worker_attention(model, geometry, 8, 128)
    attention.attn.sliding_window_size = 7
    model._uses_mla = True
    with pytest.raises(ValueError, match="GQA"):
        validate_worker_attention(model, geometry, 8, 128)
    model._uses_mla = False
    attention.num_kv_heads = 4
    with pytest.raises(ValueError, match="KV geometry"):
        validate_worker_attention(model, geometry, 8, 128)


def test_data_plane_thread_affinity_is_enforced():
    engine = object.__new__(DFlash2WorkerEngine)
    engine._thread_id = threading.get_ident()
    engine._closed = False
    engine._check_thread()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="execution thread"):
            executor.submit(engine._check_thread).result()
    engine._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        engine._check_thread()


def test_empty_snapshot_installs_without_launching_zero_row_kernels(monkeypatch):
    engine = object.__new__(DFlash2WorkerEngine)
    engine._thread_id = threading.get_ident()
    engine._closed = False
    engine.config = SimpleNamespace(device_id=0)
    engine.contract = SimpleNamespace(feature_width=2)
    engine.geometry = WorkerCacheGeometry(1, 8, 8)
    engine.sessions = WorkerSessionTable(1, 7)
    engine.sessions.open("empty")
    engine.device = "cpu"
    engine.max_position = 100
    engine.pool = object()

    def unexpected_write(*args):
        raise AssertionError("Empty context must not launch KV projection kernels")

    engine.model = SimpleNamespace(write_context_kv=unexpected_write)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    engine.install_features(
        "empty", 0, 0, torch.empty(0, 2, dtype=torch.bfloat16), True
    )
    assert engine.sessions.get("empty").confirmed_endpoint == 0


def test_worker_forwards_native_geometry_and_unscaled_embeddings(monkeypatch):
    """Execute the worker orchestration on CPU with only model/kernel seams stubbed."""
    modes = SimpleNamespace(DECODE="decode")
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.execution.context",
        SimpleNamespace(ForwardContext=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.execution.forward_batch_info",
        SimpleNamespace(
            ForwardMode=modes, CaptureHiddenMode=SimpleNamespace(FULL="full")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.layers.logits_processor",
        SimpleNamespace(LogitsMetadata=SimpleNamespace),
    )
    observations = {}

    def select(selector, candidates, unary, hidden, anchors, out, vocab_size):
        assert hidden.shape == (2, 8, 2)
        assert candidates.shape == unary.shape == (2, 7, 2)
        assert unary.dtype == torch.float32
        out[:, 0].copy_(anchors)
        out[:, 1:].copy_(torch.arange(7, dtype=torch.int32).view(1, 7))
        return out

    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.execution.drafter.dflash2",
        SimpleNamespace(select_dflash2_block=select),
    )

    class Model:
        input_embedding_scale = 3.0
        candidate_selector = SimpleNamespace(top_k=2)

        def __call__(self, **kwargs):
            observations.update(kwargs)
            # The native model, not the worker, owns this scale.
            return SimpleNamespace(
                hidden_states=kwargs["input_embeds"] * self.input_embedding_scale
            )

    engine = object.__new__(DFlash2WorkerEngine)
    engine._thread_id = threading.get_ident()
    engine._closed = False
    engine.config = SimpleNamespace(max_batch_size=2)
    engine.contract = SimpleNamespace(vocab_size=32, feature_width=2)
    engine.geometry = WorkerCacheGeometry(2, 8, 8)
    engine.sessions = WorkerSessionTable(2, 7)
    engine.sessions.open("a").confirmed_endpoint = 4
    engine.sessions.open("b").confirmed_endpoint = 99
    engine.device = "cpu"
    engine.backend = WorkerAttentionBackend(engine.geometry, "cpu")
    engine.pool = object()
    engine.model = Model()
    engine.model_config = SimpleNamespace(dflash_config={"mask_token_id": 31})
    engine.embed_tokens = lambda ids, reduce_results: torch.ones(ids.numel(), 2)
    engine.lm_head = object()
    engine.logits_processor = SimpleNamespace(
        _get_logits=lambda hidden, head, metadata: torch.arange(32)
        .float()
        .expand(hidden.shape[0], 32)
    )
    results = engine.draft_batch(
        [WorkerDraftJob("a", 4, 5), WorkerDraftJob("b", 99, 6)]
    )
    assert torch.equal(observations["input_embeds"], torch.ones(16, 2))
    assert observations["positions"].tolist() == list(range(4, 12)) + list(
        range(99, 107)
    )
    assert observations["input_ids"].view(2, 8).tolist() == [
        [5] + [31] * 7,
        [6] + [31] * 7,
    ]
    assert [result.candidate_ids for result in results] == [tuple(range(7))] * 2


def test_vocabulary_selection_never_opens_unrelated_target_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weight_map = {
        TARGET_VOCABULARY_NAMES[0]: "embedding.safetensors",
        TARGET_VOCABULARY_NAMES[1]: "head.safetensors",
        "model.layers.0.experts.weight": "absent-expert.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    save_file(
        {TARGET_VOCABULARY_NAMES[0]: torch.ones(2, 2)},
        tmp_path / "embedding.safetensors",
    )
    save_file(
        {TARGET_VOCABULARY_NAMES[1]: torch.ones(2, 2), "unused": torch.zeros(1)},
        tmp_path / "head.safetensors",
    )
    checkpoint = resolve_checkpoint(str(tmp_path), "a" * 40, TARGET_VOCABULARY_NAMES)
    assert {path.name for path in checkpoint.weight_paths} == {
        "embedding.safetensors",
        "head.safetensors",
    }
    assert {name for name, _ in iter_checkpoint_weights(checkpoint)} == set(
        TARGET_VOCABULARY_NAMES
    )


def test_checkpoint_missing_or_unsafe_shards_fail():
    with pytest.raises(ValueError, match="missing weights"):
        selected_shards({"a": "one.safetensors"}, ("b",))
    with pytest.raises(ValueError, match="Invalid"):
        selected_shards({"a": "../one.safetensors"}, ("a",))


def test_partial_qkv_checkpoint_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    save_file(
        {"layers.0.self_attn.q_proj.weight": torch.ones(2, 2)},
        tmp_path / "model.safetensors",
    )
    model = SimpleNamespace(
        named_parameters=lambda: [("layers.0.self_attn.qkv_proj.weight", None)],
        load_weights=lambda weights: list(weights),
    )
    with pytest.raises(ValueError, match="k_proj.weight"):
        load_complete_draft_weights(
            model, resolve_checkpoint(str(tmp_path), "a" * 40, None)
        )
