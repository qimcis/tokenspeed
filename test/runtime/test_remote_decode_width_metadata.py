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

"""CPU execution of the persistent width/view contract, without GPU kernels.

Only the unmodified metadata methods are compiled from the backend classes;
their modules import optional GPU libraries at import time. The planner is a
shape-sensitive test double. These tests qualify metadata geometry and pointer
lifetime, not DSA kernels or CUDA graph execution.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PAGED = (
    Path(__file__).resolve().parents[2]
    / "python/tokenspeed/runtime/layers/attention/backends/paged"
)


def _compile_class(namespace, filename, name, methods, bases):
    tree = ast.parse((PAGED / filename).read_text())
    source = next(node for node in tree.body if getattr(node, "name", None) == name)
    node = ast.ClassDef(
        name=name,
        bases=[ast.Name(id=base, ctx=ast.Load()) for base in bases],
        keywords=[],
        body=[
            method for method in source.body if getattr(method, "name", None) in methods
        ],
        decorator_list=[],
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(PAGED / filename), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.fixture
def metadata_types():
    def plan(*, seq_lens_2d, page_size, **kwargs):
        assert page_size == 64
        out = kwargs.get("out")
        if out is None:
            return seq_lens_2d.clone()
        assert out.shape == seq_lens_2d.shape
        out.copy_(seq_lens_2d)
        return out

    @dataclasses.dataclass
    class DecodeMetadata:
        num_extends: int
        page_table: torch.Tensor
        max_seq_len_k: int
        seq_lens_k: torch.Tensor
        q_len_per_req: int

    namespace = {
        "__name__": __name__,
        "torch": torch,
        "dsa_plan": plan,
        "TRTLLMMLADecodeMetadata": DecodeMetadata,
    }
    base = _compile_class(
        namespace,
        "base.py",
        "PagedAttentionBackend",
        {
            "prepared_decode_width",
            "prepare_decode_width",
            "verify_floor",
            "block_decode_active",
            "block_decode_expansion",
            "init_cuda_graph_state",
        },
        [],
    )
    base.supports_variable_decode_width = False
    base.draft_block_decode = False
    dense = _compile_class(
        namespace,
        "trtllm_mla.py",
        "TRTLLMMLABackend",
        {"_decode_views", "refresh_decode_metadata", "block_decode_expansion"},
        ["PagedAttentionBackend"],
    )
    dense.supports_variable_decode_width = True
    dsa = _compile_class(
        namespace,
        "dsa.py",
        "DSABackend",
        {
            "forward_decode_metadata",
            "prepare_decode_width",
            "init_cuda_graph_state",
            "refresh_decode_metadata",
            "_refresh_sparse_decode_plan",
        },
        ["PagedAttentionBackend"],
    )
    dsa.supports_variable_decode_width = True
    router = _compile_class(
        namespace,
        "router.py",
        "CacheGroupRouter",
        {
            "_decode_tokens_per_req",
            "_prepare_decode_width",
            "_refresh_decode_locations",
            "_publish_decode_locations",
        },
        [],
    )
    namespace["RouterDecodeWriteLocations"] = SimpleNamespace
    return SimpleNamespace(base=base, dense=dense, dsa=dsa, router=router)


def _make_backend(types):
    dense = types.dense()
    dense.spec_num_tokens = 6
    dense.is_draft = False
    dense.device = "cpu"
    dense.max_num_pages = 4
    dense.max_context_len = 256
    backend = types.dsa()
    backend.spec_num_tokens = 6
    backend.is_draft = False
    backend.device = "cpu"
    backend.kernel_page_size = 64
    backend._dense_backend = dense
    backend.init_cuda_graph_state(4)
    return backend


def _refresh(backend, width, seq, num_extends):
    backend.prepare_decode_width(width)
    backend.refresh_decode_metadata(
        len(seq),
        len(seq),
        torch.tensor(seq, dtype=torch.int32),
        torch.arange(len(seq) * 4, dtype=torch.int32).view(len(seq), 4),
        num_extends=num_extends,
        for_graph_replay=False,
    )
    return backend.forward_decode_metadata


def test_width_one_six_one_restores_same_views_and_plan(metadata_types):
    backend = _make_backend(metadata_types)
    first = _refresh(backend, 1, [3, 10, 20], 0)
    first_ptrs = (first._dsa_seq_lens_2d.data_ptr(), first._dsa_plan.data_ptr())
    six = _refresh(backend, 6, [8, 15, 25], 0)
    assert six is not first
    assert six.q_len_per_req == 6
    assert first.q_len_per_req == 1
    assert tuple(six._dsa_seq_lens_2d.shape) == (18, 1)
    assert six._dsa_plan.flatten().tolist() == [8] * 6 + [15] * 6 + [25] * 6
    restored = _refresh(backend, 1, [4, 11, 21], 0)
    assert restored is first
    assert (
        restored._dsa_seq_lens_2d.data_ptr(),
        restored._dsa_plan.data_ptr(),
    ) == first_ptrs
    assert restored._dsa_plan.flatten().tolist() == [4, 11, 21]
    assert backend.spec_num_tokens == backend._dense_backend.spec_num_tokens == 6


def test_mixed_plan_cannot_replace_pure_decode_capture(metadata_types):
    backend = _make_backend(metadata_types)
    pure = _refresh(backend, 6, [10, 20, 30], 0)
    pure_plan = pure._dsa_plan
    mixed = _refresh(backend, 6, [15, 25, 35], 1)
    assert mixed._dsa_plan is not pure_plan
    assert mixed._dsa_plan.flatten().tolist() == [25] * 6 + [35] * 6
    assert len(backend._dsa_decode_plans) == 1
    assert "_dsa_decode_plans" not in vars(mixed)
    again = _refresh(backend, 6, [16, 26, 36], 0)
    assert again._dsa_plan is pure_plan
    assert pure_plan.flatten().tolist() == [16] * 6 + [26] * 6 + [36] * 6


def test_graph_pointer_snapshot_survives_other_width_and_mixed_plan(metadata_types):
    path = PAGED.parents[3] / "execution/graph_ptr_guard.py"
    spec = importlib.util.spec_from_file_location("width_graph_ptr_guard", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    backend = _make_backend(metadata_types)
    dense = backend._dense_backend
    dense.child_backends = lambda: ()
    _refresh(backend, 6, [10, 20, 30], 0)
    snapshot = guard.snapshot_graph_metadata(dense)
    _refresh(backend, 1, [11, 21, 31], 0)
    _refresh(backend, 6, [12, 22, 32], 1)
    _refresh(backend, 6, [13, 23, 33], 0)
    guard.verify_graph_metadata(dense, snapshot, context="width six restored")


def test_width_one_does_not_apply_six_token_floor(metadata_types):
    backend = _make_backend(metadata_types)
    single = _refresh(backend, 1, [1, 2], 0)
    assert single.seq_lens_k.tolist() == [1, 2]
    assert tuple(single._dsa_seq_lens_2d.shape) == (2, 1)
    assert _refresh(backend, 6, [1, 2], 0).seq_lens_k.tolist() == [6, 6]


@pytest.mark.parametrize("width", [0, 7, -1])
def test_invalid_width_does_not_mutate_dense_view(metadata_types, width):
    backend = _make_backend(metadata_types)
    previous = _refresh(backend, 6, [10, 20], 0)
    with pytest.raises(ValueError):
        backend.prepare_decode_width(width)
    assert backend.forward_decode_metadata is previous
    assert backend.prepared_decode_width == 6


def test_unsupported_leaf_rejects_width_change(metadata_types):
    leaf = metadata_types.base()
    leaf.spec_num_tokens = 6
    leaf.prepare_decode_width(6)
    with pytest.raises(NotImplementedError):
        leaf.prepare_decode_width(1)
    assert leaf.prepared_decode_width == 6


@pytest.mark.parametrize(
    "bs,extends,extend_tokens,kwargs,expected",
    [
        (3, 0, 0, {"num_tokens": 3}, 1),
        (3, 0, 0, {"num_tokens": 18}, 6),
        (3, 1, 7, {"num_tokens": 9}, 1),
        (3, 1, 7, {"num_tokens": 19}, 6),
        (3, 0, 0, {"decode_input_tokens": 1}, 1),
    ],
)
def test_router_uses_actual_decode_geometry(
    metadata_types, bs, extends, extend_tokens, kwargs, expected
):
    router = metadata_types.router()
    router.spec_num_tokens = 6
    router.is_draft = False
    backend = _make_backend(metadata_types)
    router.leaves = {"history": backend}
    assert router._prepare_decode_width(bs, extends, extend_tokens, kwargs) == expected
    assert backend.prepared_decode_width == expected


def test_router_rejects_ragged_or_overcapacity_shape(metadata_types):
    router = metadata_types.router()
    router.spec_num_tokens = 6
    router.is_draft = False
    router.leaves = {}
    for tokens in [0, 5, 21]:
        with pytest.raises(ValueError):
            router._prepare_decode_width(3, 0, 0, {"num_tokens": tokens})


def test_draft_round_window_keeps_native_geometry(metadata_types):
    router = metadata_types.router()
    router.spec_num_tokens = 8
    router.is_draft = True
    router.leaves = {}
    assert router._prepare_decode_width(3, 0, 0, {"decode_input_tokens": 1}) == 8


def test_narrow_write_window_uses_every_requests_anchor(metadata_types):
    # Execute the production CPU slot math used by GroupTableStacks.
    namespace = {"torch": torch}
    tree = ast.parse((PAGED / "write_locations.py").read_text())
    methods = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"_decode_positions", "_gather_slots", "decode_write_locations"}
    ]
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(PAGED / "write_locations.py"),
            "exec",
        ),
        namespace,
    )
    tables = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.int32)
    locations = torch.full((1, 18), -99, dtype=torch.int32)

    class LocationStack:
        def compute_decode_locations(self, bs, seq_lens, width):
            namespace["decode_write_locations"](
                tables,
                torch.tensor([8], dtype=torch.int32),
                seq_lens,
                locations,
                bs,
                width,
            )

        def decode_locations(self, group_id, bs, width):
            assert group_id == "history"
            return locations[0, : bs * width]

    router = metadata_types.router()
    router.stacks = LocationStack()
    router.leaves = {"history": object()}
    router._decode_views = {}
    seq = torch.tensor([9, 10, 1], dtype=torch.int32)
    router._refresh_decode_locations(3, seq, 6)
    wide = router.decode_write_locations
    assert wide.by_group["history"].numel() == 18
    assert wide.by_group["history"].tolist()[-6:] == [0] * 6
    router._refresh_decode_locations(3, seq, 1)
    narrow = router.decode_write_locations
    # Both live requests contribute their own last slot, then null padding.
    assert narrow.by_group["history"].tolist() == [16, 33, 0]
    router._refresh_decode_locations(3, seq + 1, 6)
    assert router.decode_write_locations is wide
    router._refresh_decode_locations(3, seq, 1)
    assert router.decode_write_locations is narrow
