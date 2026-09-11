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

"""Opt-in actual checkpoint gate; CPU/storage tests do not qualify A10 inference.

Set TOKENSPEED_TEST_DRAFT_WORKER_CUDA=1 and the four checkpoint environment
variables below on an explicitly allocated CUDA worker before running.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.draft_pool.worker import (
    DFlash2WorkerEngine,
    WorkerDraftJob,
    WorkerModelConfig,
)
from tokenspeed.runtime.draft_pool.worker_cache import (
    WorkerAttentionBackend,
    WorkerCacheGeometry,
    WorkerKVPool,
)


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_DRAFT_WORKER_CUDA") != "1",
    reason="Requires explicit CUDA allocation and the portable kernel build",
)
def test_portable_native_block_attention_matches_dense_reference():
    """Exercise real Triton GQA against independent FP32 attention on one GPU."""
    torch.cuda.set_device(0)
    geometry = WorkerCacheGeometry(2, 16, 8)
    pool = WorkerKVPool(geometry, 1, 8, 128, "cuda:0")
    generator = torch.Generator(device="cuda:0").manual_seed(17)
    for plane in pool.get_kv_buffer(0):
        plane.copy_(
            torch.randn(
                plane.shape, generator=generator, dtype=torch.bfloat16, device="cuda:0"
            )
        )
        plane[0].zero_()
    backend = WorkerAttentionBackend(geometry, "cuda:0")
    endpoints = [3, 101]
    backend.prepare([0, 1], endpoints)
    q = torch.randn(
        16, 64, 128, generator=generator, dtype=torch.bfloat16, device="cuda:0"
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=64,
        qk_head_dim=128,
        sliding_window_size=15,
        scaling=128**-0.5,
    )
    actual = backend.forward(q, None, None, layer, pool, "decode", 2, False)
    keys, values = pool.get_kv_buffer(0)
    expected = []
    for slot, endpoint in enumerate(endpoints):
        visible = geometry.page_row(slot, endpoint)[-16:]
        k = keys[visible, 0].float().repeat_interleave(8, dim=1).transpose(0, 1)
        v = values[visible, 0].float().repeat_interleave(8, dim=1).transpose(0, 1)
        queries = q[slot * 8 : (slot + 1) * 8].float().transpose(0, 1)
        probabilities = (queries @ k.transpose(-1, -2) * layer.scaling).softmax(dim=-1)
        expected.append((probabilities @ v).transpose(0, 1))
    torch.testing.assert_close(
        actual.float(), torch.cat(expected), rtol=0.03, atol=0.03
    )


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_DRAFT_WORKER_CUDA") != "1",
    reason="Requires explicit CUDA allocation and the actual pinned checkpoint pair",
)
def test_native_eight_batched_forward_and_cold_context_equivalence():
    config = WorkerModelConfig(
        target_model_path=os.environ["TOKENSPEED_TEST_TARGET_CHECKPOINT"],
        target_revision=os.environ["TOKENSPEED_TEST_TARGET_REVISION"],
        draft_model_path=os.environ["TOKENSPEED_TEST_DRAFT_CHECKPOINT"],
        draft_revision=os.environ["TOKENSPEED_TEST_DRAFT_REVISION"],
        device_id=0,
        max_resident_sessions=2,
        max_batch_size=2,
    )
    engine = DFlash2WorkerEngine(config, pipeline_host_budget_bytes=1024**3)
    try:
        engine.open_session("incremental")
        engine.open_session("snapshot")
        history = engine.geometry.history_tokens
        before = 4090
        endpoint = before + 7
        start = before - history
        generator = torch.Generator(device="cpu").manual_seed(13)
        features = torch.randn(
            history + 7,
            engine.contract.feature_width,
            generator=generator,
            dtype=torch.bfloat16,
        )
        engine.install_features("incremental", start, before, features[:history], True)
        first = engine.draft_batch([WorkerDraftJob("incremental", before, 42)])
        assert len(first[0].candidate_ids) == 7
        engine.install_features(
            "incremental", before, endpoint, features[history:], False
        )
        engine.install_features(
            "snapshot", endpoint - history, endpoint, features[7:], True
        )
        a = engine.geometry.context_locations(
            engine.sessions.get("incremental").slot, endpoint - history, endpoint
        )
        b = engine.geometry.context_locations(
            engine.sessions.get("snapshot").slot, endpoint - history, endpoint
        )
        for layer_id in range(len(engine.model.layers)):
            for plane in engine.pool.get_kv_buffer(layer_id):
                # Different GEMM row counts can round BF16 at different points.
                torch.testing.assert_close(plane[a], plane[b], rtol=0.02, atol=0.02)
        context_before = engine.pool.keys[0][a].clone()
        results = engine.draft_batch(
            [
                WorkerDraftJob("incremental", endpoint, 43),
                WorkerDraftJob("snapshot", endpoint, 43),
            ]
        )
        assert len(results) == 2
        for result in results:
            assert result.confirmed_endpoint == endpoint
            assert result.anchor_token == 43
            assert len(result.candidate_ids) == 7
            assert all(
                0 <= token < engine.contract.vocab_size
                for token in result.candidate_ids
            )
        torch.testing.assert_close(
            engine.pool.keys[0][a], context_before, rtol=0, atol=0
        )
    finally:
        engine.close()
