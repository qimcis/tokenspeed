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

"""Real V4.1 padding and MegaMoE packing contracts; no vendor GPU imports."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.models.v41_expert_placement import (
    local_logical_experts,
    pad_expert_tensor,
)


def _packing():
    # Import this dependency-light production module directly: the ops.moe
    # package eagerly loads vendor registrations unavailable on CPU hosts.
    path = (
        Path(__file__).resolve().parents[2]
        / "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_deep_gemm/mega_moe_weights.py"
    )
    spec = importlib.util.spec_from_file_location("v41_placement_packing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
@pytest.mark.parametrize("is_scale", [False, True])
def test_padding_retains_all_bytes_and_zero_lanes(projection, is_scale):
    size, padded, hidden = 128, 512, 256
    divisor = 32 if is_scale else 2
    shape = (
        (hidden, size // divisor) if projection == "w2" else (size, hidden // divisor)
    )
    source = (torch.arange(shape[0] * shape[1]).reshape(shape) % 256).to(torch.uint8)
    result = pad_expert_tensor(
        source,
        projection=projection,
        is_scale=is_scale,
        intermediate_size=size,
        padded_intermediate_size=padded,
    )
    if projection == "w2":
        assert torch.equal(result[:, : shape[1]], source)
        tail = result[:, shape[1] :]
        assert result.shape == (hidden, padded // divisor)
    else:
        assert torch.equal(result[:size], source)
        tail = result[size:]
        assert result.shape == (padded, hidden // divisor)
    assert torch.all(tail == (127 if is_scale else 0))


def test_baseline_slot_ownership_and_no_padding_aliases():
    assert local_logical_experts(
        None, layer_id=0, num_experts=8, ep_rank=1, ep_size=2
    ) == (4, 5, 6, 7)
    source = torch.ones((128, 64), dtype=torch.uint8)
    result = pad_expert_tensor(
        source,
        projection="w1",
        is_scale=False,
        intermediate_size=128,
        padded_intermediate_size=128,
    )
    assert result.data_ptr() == source.data_ptr()


@pytest.mark.parametrize("extra", [-32, 1, 16])
def test_invalid_padding_rejected(extra):
    with pytest.raises(ValueError, match="32-element"):
        pad_expert_tensor(
            torch.zeros(128, 64, dtype=torch.uint8),
            projection="w1",
            is_scale=False,
            intermediate_size=128,
            padded_intermediate_size=128 + extra,
        )


def _processed(packing, slots=3):
    shapes = packing._expected_shapes(slots, 256, 512)
    values = [
        ((torch.arange(torch.tensor(shape).prod()).reshape(shape) + i * 37) % 256).to(
            torch.uint8
        )
        for i, shape in enumerate(shapes)
    ]
    l1 = packing._interleave_gate_up_(values[0].view(torch.int8), 8)
    s1 = packing._reorder_scale_rows_(packing._pack_ue8m0_scale_(values[1]), True, 8)
    s2 = packing._reorder_scale_rows_(packing._pack_ue8m0_scale_(values[3]), False, 8)
    state = packing._DeepGemmMegaMoEState(
        (l1, s1), (values[2].view(torch.int8), s2), torch.device("cpu")
    )
    return values, state


def test_processed_slot_move_preserves_every_byte_and_kernel_pointers():
    packing = _packing()
    canonical, state = _processed(packing)
    kernel = (*state.l1_weights, *state.l2_weights)
    pointers = [(t.data_ptr(), t.stride()) for t in kernel]
    slots = state.routed_expert_weights
    assert all(t[0].is_contiguous() for t in slots)
    before = [t.clone() for t in kernel]
    # The updater stages a cycle and a replicated copy using these exact views.
    staged = [t[[2, 0, 0]].clone() for t in slots]
    for destination, source in zip(slots, staged):
        destination.copy_(source)
    for tensor, old in zip(kernel, before):
        torch.testing.assert_close(tensor, old[[2, 0, 0]])
    assert pointers == [(t.data_ptr(), t.stride()) for t in kernel]
    assert all(
        view.untyped_storage().data_ptr()
        == canonical[index].untyped_storage().data_ptr()
        for index, view in enumerate(slots)
    )


def test_packing_matches_independent_byte_permutation():
    p = _packing()
    original = (torch.arange(2 * 1024 * 8).reshape(2, 1024, 8) % 256).to(torch.uint8)
    packed = p._pack_ue8m0_scale_(original.clone())
    result = p._reorder_scale_rows_(packed, True, 8)
    # Derive row order from semantic gate/up blocks, then UTCCP groups.
    order = torch.arange(1024).reshape(2, 64, 8).transpose(0, 1).reshape(1024)
    order = order.reshape(-1, 4, 32).transpose(1, 2).reshape(1024)
    expected = original[:, order].contiguous().view(torch.int32)
    assert torch.equal(result, expected)


def test_reprocessing_full_refresh_keeps_storage():
    p = _packing()
    canonical, state = _processed(p)
    slots = state.routed_expert_weights
    pointers = [v.data_ptr() for v in slots]
    for i, tensor in enumerate(canonical):
        tensor.fill_(i + 10)
    # Same in-place derivation called after a complete V4.1 checkpoint reload.
    p._interleave_gate_up_(canonical[0].view(torch.int8), 8)
    p._reorder_scale_rows_(p._pack_ue8m0_scale_(canonical[1]), True, 8)
    p._reorder_scale_rows_(p._pack_ue8m0_scale_(canonical[3]), False, 8)
    assert pointers == [v.data_ptr() for v in slots]
    for i, view in enumerate(slots):
        assert torch.all(view.view(torch.uint8) == i + 10)
