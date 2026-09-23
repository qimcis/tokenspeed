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

"""CPU-only checks of the actual optional FlashInfer boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def test_prepare_swaps_both_layouts_and_reuses_workspace(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "python/tokenspeed_kernel/thirdparty/flashinfer/mxfp4.py"
    )
    name = "_tested_optional_mxfp4_boundary"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    vendor = ModuleType("flashinfer")
    vendor.ActivationType = SimpleNamespace(SwigluBias="clamped")
    api = ModuleType("flashinfer.fused_moe")
    calls = []

    def run(*, workspace_buffer, profile_ids, swiglu_limit, **kwargs):
        calls.append((workspace_buffer, profile_ids, swiglu_limit, kwargs))
        kwargs["output"].zero_()

    def workspace_size(**kwargs):
        assert kwargs["num_experts_total"] == 8
        assert kwargs["max_num_tokens"] == 8
        assert kwargs["hidden_size"] == 256
        assert kwargs["intermediate_size"] == 128
        assert kwargs["activation_type"] == "clamped"
        assert kwargs["use_w4_group_scaling"] is True
        return 512

    def interleave_weights(tensor, quant_type):
        assert quant_type == "fp4"
        return tensor.clone()

    def interleave_scales(tensor, group_size):
        assert group_size == 32
        return tensor.clone()

    api.cutlass_fused_moe = run
    api.cutlass_fused_moe_workspace_size = workspace_size
    api.interleave_moe_weights_for_sm90_mixed_gemm = interleave_weights
    api.interleave_moe_scales_for_sm90_mixed_gemm = interleave_scales
    monkeypatch.setitem(sys.modules, "flashinfer", vendor)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", api)
    w13 = (
        torch.arange(256, dtype=torch.uint8).view(1, 256, 1).expand(2, 256, 128).clone()
    )
    w2 = torch.full((2, 256, 64), 23, dtype=torch.uint8)
    s13 = torch.arange(256, dtype=torch.uint8).view(1, 256, 1).expand(2, 256, 8).clone()
    s2 = torch.full((2, 256, 4), 127, dtype=torch.uint8)
    snapshots = tuple(t.clone() for t in (w13, w2, s13, s2))
    state = module.prepare_mxfp4(w13, w2, s13, s2, 6, 4, 1, 8, 10.0)
    assert torch.equal(state.w13, torch.cat((w13[:, 128:], w13[:, :128]), 1))
    assert torch.equal(
        state.scales[0].view(torch.uint8), torch.cat((s13[:, 128:], s13[:, :128]), 1)
    )
    assert torch.equal(state.w2, w2) and torch.equal(
        state.scales[1].view(torch.uint8), s2
    )
    for original, snapshot, transformed in zip(
        (w13, w2, s13, s2), snapshots, (state.w13, state.w2, *state.scales)
    ):
        assert torch.equal(original, snapshot)
        assert original.data_ptr() != transformed.data_ptr()
    assert torch.equal(state.alpha, torch.ones(2))
    assert torch.equal(state.beta, torch.zeros(2))
    assert torch.equal(state.limit, torch.full((2,), 10.0))
    x = torch.ones((1, 256), dtype=torch.bfloat16)
    ids, weights = torch.zeros((1, 6), dtype=torch.int32), torch.full((1, 6), 1 / 6)
    out = torch.empty_like(x)
    for _ in range(2):
        module.run_mxfp4(state, x, weights, ids, out)
    for workspace, profiles, limit, kwargs in calls:
        assert workspace is state.workspace and workspace.numel() == 512
        assert profiles == [-1, -1] and limit is state.limit
        assert kwargs["token_final_scales"] is weights
        assert kwargs["token_selected_experts"] is ids
        assert kwargs["output"] is out
        assert (kwargs["ep_size"], kwargs["ep_rank"]) == (4, 1)
        assert (
            kwargs["swiglu_alpha"] is state.alpha
            and kwargs["swiglu_beta"] is state.beta
        )
        assert kwargs["enable_pdl"] is False
        assert kwargs["use_w4_group_scaling"] is True
