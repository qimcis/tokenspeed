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

"""Hopper W4A16 DeepSeek semantics, independently decoded from canonical MXFP4.

Gates fixed before validation: per-row relative L2 <= .006 and maximum absolute
error <= .002 + .02 * max(abs(reference row)). Zero rows must be exactly zero.
The oracle retains BF16 GEMM1, activation, weighted route and final boundaries.
GPU dependencies are imported only after the CUDA fixture has checked SM90.
"""

from __future__ import annotations

import ast
import copy
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

L2_LIMIT = 0.006
ABS_FLOOR = 0.002
ABS_ROW_FACTOR = 0.02
KERNEL_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = KERNEL_ROOT.parent
OBSERVATIONS = []


def unpack_mxfp4(packed, scales):
    """Low K nibble first; E2M1 magnitude table and unsigned E8M0 exponents."""
    assert packed.dtype == scales.dtype == torch.uint8
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    values = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    exponents = scales.int() - 127
    factors = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), exponents)
    factors = torch.where(scales == 255, float("nan"), factors)
    return values * factors.repeat_interleave(32, dim=-1)


def _pack(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _raw(experts, hidden, intermediate, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        codes = torch.randint(
            0, 16, (experts, n, k), generator=generator, dtype=torch.uint8
        )
        scale_base = 123 if k <= 256 else 121
        scales = torch.randint(
            scale_base - 1,
            scale_base + 2,
            (experts, n, k // 32),
            generator=generator,
            dtype=torch.uint8,
        )
        result.append((_pack(codes), scales))
    return result[0][0], result[1][0], result[0][1], result[1][1]


def _routes(rows, total_experts, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn((rows, total_experts), generator=generator)
    bias = torch.linspace(-0.35, 0.35, total_experts)
    scores = torch.nn.functional.softplus(logits).sqrt()
    ids = (scores + bias).topk(6, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights.float(), ids.to(torch.int32)


def reference_moe(raw, x, weights, ids, ep_rank, limit):
    w13, w2, s13, s2 = (t.detach().cpu() for t in raw)
    x, weights, ids = (t.detach().cpu() for t in (x, weights, ids))
    experts = w13.shape[0]
    intermediate = w2.shape[-1] * 2
    decoded13 = unpack_mxfp4(w13, s13).to(torch.bfloat16).float()
    decoded2 = unpack_mxfp4(w2, s2).to(torch.bfloat16).float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[row, route]) - ep_rank * experts
            if not 0 <= expert < experts or weights[row, route] == 0:
                continue
            gate_up = (x[row].float() @ decoded13[expert].T).to(torch.bfloat16).float()
            gate = gate_up[:intermediate].clamp(max=limit)
            up = gate_up[intermediate:].clamp(-limit, limit)
            activation = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
            weighted = ((activation @ decoded2[expert].T) * weights[row, route]).to(
                torch.bfloat16
            )
            result[row] += weighted.float()
    return result.to(torch.bfloat16)


def assert_correct(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    if not actual.numel():
        return
    error = actual - expected
    norms = expected.norm(dim=1)
    residuals = error.norm(dim=1)
    assert torch.all(residuals[norms == 0] == 0), "zero routes must write exact zero"
    ratios = residuals[norms > 0] / norms[norms > 0]
    max_abs = error.abs().amax(dim=1)
    bounds = ABS_FLOOR + ABS_ROW_FACTOR * expected.abs().amax(dim=1)
    OBSERVATIONS.append(
        {
            "rows": actual.shape[0],
            "hidden": actual.shape[1],
            "max_row_relative_l2": float(ratios.max()) if ratios.numel() else 0.0,
            "max_absolute": float(max_abs.max()),
            "max_absolute_scaled": float((max_abs / bounds).max()),
            "actual": actual.to(torch.bfloat16),
            "reference": expected.to(torch.bfloat16),
        }
    )
    assert torch.all(ratios <= L2_LIMIT), ratios.tolist()
    assert torch.all(max_abs <= bounds), (max_abs.tolist(), bounds.tolist())


@pytest.fixture(scope="module")
def adapter():
    if not torch.cuda.is_available():
        pytest.skip("SM90 GPU required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 GPU required")
    from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4 import (
        hopper_mxfp4_moe,
        prepare_hopper_mxfp4_moe,
    )

    return prepare_hopper_mxfp4_moe, hopper_mxfp4_moe


def _prepare(adapter, raw, ep_size, ep_rank, max_tokens):
    device_raw = tuple(t.cuda() for t in raw)
    return adapter[0](*device_raw, 6, ep_size, ep_rank, max_tokens, 10.0)


def _run(adapter, state, x, weights, ids):
    out = torch.full_like(x, float("nan"))
    returned = adapter[1](state, x, weights, ids, out)
    assert returned is out
    return out


def test_independent_decode_nibbles_signed_zero_and_unsigned_exponents():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2).reshape(1, 1, 32)
    actual = unpack_mxfp4(_pack(codes), torch.full((1, 1, 1), 127, dtype=torch.uint8))
    expected = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    ).repeat(2)
    assert torch.equal(actual.flatten(), expected)
    assert torch.signbit(actual.flatten()[8])
    packed = _pack(torch.ones((1, 1, 128), dtype=torch.uint8))
    extremes = unpack_mxfp4(
        packed, torch.tensor([[[0, 127, 254, 255]]], dtype=torch.uint8)
    )
    assert extremes[0, 0, 0].item() == 2.0**-128
    assert extremes[0, 0, 32].item() == 0.5
    assert extremes[0, 0, 64].item() == 2.0**126
    assert torch.isnan(extremes[0, 0, 96:]).all()


def test_reference_clamp_is_deepseek_not_gpt_oss():
    gates = torch.tensor([-24.0, 24.0])
    ups = torch.tensor([32.0, -32.0])
    gate = gates.clamp(max=10)
    up = ups.clamp(-10, 10)
    actual = gate * torch.sigmoid(gate) * up
    assert actual[0] < 0 and abs(actual[0]) < 1e-7
    assert actual[1] < -99
    assert not torch.equal(actual, gate * torch.sigmoid(1.702 * gate) * (up + 1))
    zero = torch.zeros((1, 2), dtype=torch.bfloat16)
    assert_correct(zero, zero)
    with pytest.raises(AssertionError):
        assert_correct(torch.ones_like(zero), zero)


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_hopper_top6_and_gate_up_layout(adapter, rows):
    raw = _raw(8, 256, 128, 601)
    state = _prepare(adapter, raw, 1, 0, 8)
    generator = torch.Generator(device="cpu").manual_seed(607 + rows)
    x = (0.2 * torch.randn((rows, 256), generator=generator)).to(torch.bfloat16).cuda()
    weights, ids = (t.cuda() for t in _routes(rows, 8, 613 + rows))
    actual = _run(adapter, state, x, weights, ids)
    expected = reference_moe(raw, x, weights, ids, 0, 10)
    assert_correct(actual, expected)
    # Runtime's factor1.5 belongs after experts; adapter receives unscaled weights.
    assert_correct(
        (actual * 1.5).to(torch.bfloat16), (expected * 1.5).to(torch.bfloat16)
    )


def test_production_hidden_intermediate_geometry(adapter):
    raw = _raw(2, 5120, 2304, 617)
    state = _prepare(adapter, raw, 128, 37, 1)
    x = (
        (0.2 * torch.randn((1, 5120), generator=torch.Generator().manual_seed(619)))
        .to(torch.bfloat16)
        .cuda()
    )
    weights = torch.tensor([[0.1, 0.2, 0.15, 0.25, 0.2, 0.1]], device="cuda")
    ids = torch.tensor([[74, 75, 0, 1, 200, 255]], device="cuda", dtype=torch.int32)
    assert_correct(
        _run(adapter, state, x, weights, ids),
        reference_moe(raw, x, weights, ids, 37, 10),
    )


def test_ep_masking_zero_routes_padding_and_no_local_renormalization(adapter):
    raw = _raw(8, 256, 128, 631)
    state = _prepare(adapter, raw, 2, 1, 4)
    x = (
        torch.randn((4, 256), generator=torch.Generator().manual_seed(641))
        .to(torch.bfloat16)
        .cuda()
    )
    ids = torch.tensor(
        [
            [8, 15, 0, 7, -1, 16],
            [0, 1, 2, 3, -1, 99],
            [8, 9, 10, 11, 12, 13],
            [8, 8, 15, -1, 7, 14],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.tensor(
        [
            [0.1, 0.2, 0.15, 0.25, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0, 0, 0, 0, 0, 0],
            [0.1, 0.15, 0.2, 0.2, 0.1, 0.25],
        ],
        device="cuda",
    )
    actual = _run(adapter, state, x, weights, ids)
    assert_correct(actual, reference_moe(raw, x, weights, ids, 1, 10))
    assert torch.count_nonzero(actual[1:3]) == 0


def test_clamp_saturation_and_negative_gate(adapter):
    experts, hidden, intermediate = 2, 256, 128
    c13 = torch.zeros((experts, 2 * intermediate, hidden), dtype=torch.uint8)
    c2 = torch.zeros((experts, hidden, intermediate), dtype=torch.uint8)
    c13[:, :intermediate, 0] = 5
    c13[:, 1:intermediate:2, 0] = 13
    c13[:, intermediate:, 0] = 6
    c13[:, intermediate + 2 :: 4, 0] = 14
    for i in range(intermediate):
        c2[:, i, i] = 2
    raw = (
        _pack(c13),
        _pack(c2),
        torch.full((experts, 2 * intermediate, hidden // 32), 130, dtype=torch.uint8),
        torch.full((experts, hidden, intermediate // 32), 127, dtype=torch.uint8),
    )
    state = _prepare(adapter, raw, 4, 1, 1)
    x = torch.zeros((1, hidden), dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1
    weights = torch.full((1, 6), 1 / 6, device="cuda")
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32, device="cuda")
    actual = _run(adapter, state, x, weights, ids)
    expected = reference_moe(raw, x, weights, ids, 1, 10)
    assert_correct(actual, expected)
    # Inspect a negative-gate coordinate separately so large positive outputs
    # cannot hide a symmetric gate clamp or an OpenAI-style up+1 term.
    torch.testing.assert_close(
        actual[:, 1].cpu().float(), expected[:, 1].float(), rtol=0.02, atol=1e-10
    )


def test_empty_returns_caller_output(adapter):
    state = _prepare(adapter, _raw(2, 256, 128, 643), 4, 0, 8)
    x = torch.empty((0, 256), dtype=torch.bfloat16, device="cuda")
    assert (
        _run(
            adapter,
            state,
            x,
            torch.empty((0, 6), device="cuda"),
            torch.empty((0, 6), dtype=torch.int32, device="cuda"),
        ).shape
        == x.shape
    )


def test_graph_replay_mutates_activations_routes_and_weights(adapter):
    raw = _raw(8, 256, 128, 647)
    state = _prepare(adapter, raw, 2, 1, 4)
    x = torch.zeros((4, 256), dtype=torch.bfloat16, device="cuda")
    weights = torch.full((4, 6), 1 / 6, device="cuda")
    ids = torch.tensor([[8, 9, 10, 11, 12, 13]] * 4, dtype=torch.int32, device="cuda")
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        adapter[1](state, x, weights, ids, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = adapter[1](state, x, weights, ids, out)
    assert result is out
    for epoch in range(3):
        gen = torch.Generator().manual_seed(653 + epoch)
        x.copy_(torch.randn((4, 256), generator=gen).to(torch.bfloat16))
        next_weights, next_ids = _routes(4, 16, 659 + epoch)
        next_ids[0] = torch.tensor([-1, 16, 17, 99, 0, 7], dtype=torch.int32)
        ids.copy_(next_ids)
        weights.copy_(next_weights)
        out.fill_(float("nan"))
        graph.replay()
        assert_correct(out, reference_moe(raw, x, weights, ids, 1, 10))
        assert torch.count_nonzero(out[0]) == 0


def _runtime_swiglu_arg(activation, limit):
    source = ast.parse(
        (REPO_ROOT / "python/tokenspeed/runtime/layers/moe/expert.py").read_text()
    )
    cls = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "MoELayer"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    statements = [
        copy.deepcopy(n) for n in init.body if "self.swiglu_arg" in ast.unparse(n)
    ]
    assert len(statements) == 2
    activation_source = ast.parse(
        (REPO_ROOT / "python/tokenspeed/runtime/layers/activation.py").read_text()
    )
    arg_class = next(
        n
        for n in activation_source.body
        if isinstance(n, ast.ClassDef) and n.name == "SwigluArg"
    )
    instance = SimpleNamespace(activation=activation)
    namespace = {
        "dataclass": dataclass,
        "self": instance,
        "activation_alpha": None,
        "swiglu_limit": limit,
    }
    exec(
        compile(
            ast.Module(body=[arg_class, *statements], type_ignores=[]),
            "expert_metadata",
            "exec",
        ),
        namespace,
    )
    return instance.swiglu_arg


@pytest.mark.parametrize("activation", ["swiglu", "silu", "situ"])
def test_marlin_activation_metadata_regression_cpu(monkeypatch, activation):
    path = KERNEL_ROOT / "python/tokenspeed_kernel/ops/moe/marlin/mxfp4.py"
    nodes = [
        copy.deepcopy(n)
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    ]
    for node in nodes:
        node.decorator_list = []
    calls = []
    fake_module = ModuleType("tokenspeed_kernel.ops.activation.triton")

    def silu(*args, **kwargs):
        calls.append(("silu", kwargs))
        return torch.ones((1, 2), dtype=torch.bfloat16)

    def situ(*args, **kwargs):
        calls.append(("situ", kwargs))
        return torch.ones((1, 2), dtype=torch.bfloat16)

    fake_module.silu_and_mul = silu
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    def gemm(*args, **kwargs):
        return torch.zeros(
            (kwargs["size_m"] * kwargs["top_k"], kwargs["size_n"]), dtype=torch.bfloat16
        )

    namespace = {
        "torch": torch,
        "math": math,
        "MXFP4_BLOCK": 32,
        "situ_and_mul": situ,
        "marlin_make_workspace": lambda device: None,
        "moe_align_block_size": lambda ids, block, experts: (None, None, None),
        "moe_wna16_marlin_gemm": gemm,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    w = SimpleNamespace(
        w13_weight=torch.empty((2, 4, 16)),
        w2_weight=torch.empty((2, 32, 1)),
        w13_weight_scale=None,
        w2_weight_scale=None,
        _marlin_hidden_size=32,
        _marlin_ispp=2,
        num_local_experts=2,
        ep_size=1,
        ep_rank=0,
        activation=activation,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_arg=_runtime_swiglu_arg(activation, 10.0),
        swiglu_beta=None,
    )
    namespace["marlin_mxfp4_precomputed_moe_apply"](
        {"activation": activation},
        torch.zeros((1, 32), dtype=torch.bfloat16),
        w,
        None,
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        None,
        None,
        True,
        False,
    )
    assert len(calls) == 1
    if activation == "situ":
        assert calls[0] == ("situ", {"beta": 4.0, "linear_beta": 25.0})
    else:
        assert calls[0][0] == "silu"
        assert calls[0][1].get("limit") == (10.0 if activation == "swiglu" else None)


@pytest.mark.parametrize(
    "alpha,beta,limit",
    [
        (1.702, None, 10.0),
        (None, 1.0, 10.0),
        (None, None, 0.0),
        (None, None, -10.0),
        (None, None, float("inf")),
        (None, None, float("nan")),
    ],
)
def test_marlin_rejects_unsupported_swiglu_cpu(alpha, beta, limit):
    path = KERNEL_ROOT / "python/tokenspeed_kernel/ops/moe/marlin/mxfp4.py"
    helper = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_swiglu_limit"
    )
    namespace = {"torch": torch, "math": math}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    weight = SimpleNamespace(
        swiglu_arg=SimpleNamespace(alpha=alpha, limit=limit), swiglu_beta=beta
    )
    with pytest.raises(ValueError):
        namespace["_swiglu_limit"](weight, "swiglu")
    # Non-SwiGLU paths must not interpret otherwise irrelevant metadata.
    assert namespace["_swiglu_limit"](weight, "silu") is None
    assert namespace["_swiglu_limit"](weight, "situ") is None


def _extracted_function(path, name, namespace):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def test_actual_runtime_top6_normalization_and_routed_scale_once_cpu():
    select = _extracted_function(
        REPO_ROOT / "python/tokenspeed/runtime/models/deepseek_v41.py",
        "_select_experts",
        {"torch": torch, "F": torch.nn.functional},
    )
    x = torch.eye(2, dtype=torch.bfloat16)
    gate = SimpleNamespace(
        bias_vl=None,
        weight=torch.arange(16).reshape(8, 2).float() / 10,
        e_score_correction_bias=torch.arange(8).float().flip(0) / 3,
    )
    owner = SimpleNamespace(
        gate=gate,
        config=SimpleNamespace(num_experts_per_tok=6, norm_topk_prob=True),
        hash_indices_dtype=torch.int32,
    )
    weights, ids, scores = select(owner, x, None)
    expected_scores = torch.nn.functional.softplus(x.float() @ gate.weight.T).sqrt()
    expected_ids = (expected_scores + gate.e_score_correction_bias).topk(6, -1).indices
    expected_weights = expected_scores.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(-1, keepdim=True)
    assert torch.equal(scores, expected_scores) and torch.equal(ids, expected_ids.int())
    assert torch.equal(weights, expected_weights)
    forward = _extracted_function(
        REPO_ROOT / "python/tokenspeed/runtime/models/deepseek_v4.py",
        "_forward_normal_with_shared",
        {
            "torch": torch,
            "nvtx_range": lambda name: nullcontext(),
            "get_is_capture_mode": lambda: False,
        },
    )
    topk = SimpleNamespace(format=SimpleNamespace(is_bypassed=lambda: False))
    fork = SimpleNamespace(branch=lambda: nullcontext())
    owner._select_experts = lambda hidden, input_ids: (weights, ids, scores)
    owner._make_topk_output = lambda *args: topk
    owner.stream_fork = SimpleNamespace(scope=lambda **kwargs: nullcontext(fork))
    owner.experts = lambda **kwargs: torch.full_like(x, 2)
    owner.routed_scaling_factor = 1.5
    assert torch.equal(
        forward(owner, x, None, 2, 2, lambda hidden: torch.ones_like(x)),
        torch.full_like(x, 4),
    )


@pytest.mark.parametrize(
    "invalid",
    ["x_dtype", "ids_dtype", "weights_dtype", "shape", "stride", "capacity", "alias"],
)
def test_runtime_validation_rejects_before_vendor_cpu(invalid):
    path = KERNEL_ROOT / "python/tokenspeed_kernel/ops/moe/flashinfer/cutlass_mxfp4.py"
    run = _extracted_function(path, "hopper_mxfp4_moe", {"torch": torch})
    state = SimpleNamespace(
        hidden=256, top_k=6, native=SimpleNamespace(max_tokens=2, w13=torch.empty(1))
    )
    x, out = torch.zeros((1, 256), dtype=torch.bfloat16), torch.empty(
        (1, 256), dtype=torch.bfloat16
    )
    weights, ids = torch.ones((1, 6)), torch.zeros((1, 6), dtype=torch.int32)
    if invalid == "x_dtype":
        x = x.float()
    elif invalid == "ids_dtype":
        ids = ids.long()
    elif invalid == "weights_dtype":
        weights = weights.to(torch.bfloat16)
    elif invalid == "shape":
        ids = ids[:, :5]
    elif invalid == "stride":
        x = torch.zeros((1, 512), dtype=torch.bfloat16)[:, ::2]
    elif invalid == "capacity":
        x = torch.zeros((3, 256), dtype=torch.bfloat16)
    else:
        out = x
    with pytest.raises(ValueError):
        run(state, x, weights, ids, out)
