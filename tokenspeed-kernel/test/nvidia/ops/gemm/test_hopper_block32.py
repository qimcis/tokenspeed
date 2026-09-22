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

"""Independent dequantization, roundoff gates, and graph contract for SM90."""

import math

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm import triton as triton_gemm
from tokenspeed_kernel.ops.gemm.hopper_block32 import (
    HopperBlock32Config,
    block32_workspace_shape,
    gemm_fp8_block32,
)
from tokenspeed_kernel.ops.quantization.triton import (
    triton_quantize_fp8_group32_ue8m0,
)
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Hopper Block32 requires SM90",
)


def reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU IEEE-FP32 oracle for the unchanged V4.1 group32 quantizer."""
    values = x.detach().cpu().float().reshape(-1, 32)
    raw = values.abs().amax(-1).clamp_min(1.0e-4) * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    quantized = (values / scale[:, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(x.shape), exponent.byte().reshape(*x.shape[:-1], -1)


def reference_mm(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode scales independently; FP64 oracle and per-output absolute work."""

    def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raw = scale.detach().cpu()
        if raw.dtype == torch.uint8:
            assert bool((raw != 255).all()), "255 is a reserved nonfinite scale"
            decoded = torch.exp2(raw.double() - 127)
        else:
            decoded = raw.double()
        expanded = decoded.repeat_interleave(32, dim=-1)[..., : q.shape[-1]]
        # Production operands are FP8 plus FP32 scales. Use exact FP64 products
        # here so the oracle does not inherit intermediate FP32 underflow.
        return q.detach().cpu().float().double() * expanded

    ad = dequantize(a, a_scale).reshape(-1, a.shape[-1])
    bd = dequantize(b, b_scale)
    shape = (*a.shape[:-1], b.shape[0])
    return (ad @ bd.T).reshape(shape), (ad.abs() @ bd.abs().T).reshape(shape)


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    absolute_work: torch.Tensor,
    k: int,
    split_k: int,
) -> dict[str, float | int | bool]:
    """Frozen local roundoff envelope; no tensor-global absolute tolerance.

    gamma bounds FP32 group-dot plus across-group/reduction accumulation. A
    final half-ULP permits exactly one cast to the requested output format.
    This is a synthetic arithmetic gate, not a model-quality guarantee.
    """
    observed = actual.detach().cpu().double()
    assert torch.isfinite(expected).all() and torch.isfinite(absolute_work).all()
    precision = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}[actual.dtype]
    _, exponent = torch.frexp(expected.abs())
    half_ulp = torch.exp2(exponent.double() - precision - 1)
    min_half_ulp = torch.finfo(actual.dtype).tiny / (2**precision)
    half_ulp = torch.where(expected == 0, min_half_ulp, half_ulp)
    half_ulp = half_ulp.clamp_min(min_half_ulp)
    operations = 32 + math.ceil(k / 32) + split_k + 4
    unit_roundoff = 2.0**-24
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    allowed = gamma * absolute_work + half_ulp + operations * (2.0**-150)
    error = (observed - expected).abs()
    finite = bool(torch.isfinite(observed).all())
    denominator = float(torch.linalg.vector_norm(expected))
    numerator = float(torch.linalg.vector_norm(error))
    return {
        "passed": finite and bool((error <= allowed).all()),
        "finite": finite,
        "elements": observed.numel(),
        "max_abs": float(error.max()) if error.numel() else 0.0,
        "max_scaled_error": float((error / allowed).max()) if error.numel() else 0.0,
        "normalized_l2": numerator / denominator if denominator else numerator,
        "gamma": gamma,
    }


def _config(split_k: int, swap_ab: bool) -> HopperBlock32Config:
    return HopperBlock32Config(16, 64, split_k, swap_ab, 1, 4, 3)


def _inputs(
    shape: tuple[int, ...],
    n: int,
    seed: int,
    scale_dtypes: tuple[torch.dtype, torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(shape, generator=generator).to(torch.float8_e4m3fn).cuda()
    b = torch.randn((n, shape[-1]), generator=generator).to(torch.float8_e4m3fn).cuda()
    groups = math.ceil(shape[-1] / 32)
    scales = []
    for dims, dtype in zip(((*shape[:-1], groups), (n, groups)), scale_dtypes):
        exponents = torch.randint(119, 133, dims, generator=generator)
        scale = (
            exponents.byte()
            if dtype == torch.uint8
            else torch.exp2(exponents.float() - 127)
        )
        if dtype == torch.float32:
            scale *= 1.125  # Exercise non-power-of-two FP32 scale contracts too.
        scales.append(scale.cuda())
    return a, b, scales[0], scales[1]


def _workspace(a: torch.Tensor, b: torch.Tensor, config: HopperBlock32Config):
    shape = block32_workspace_shape(a, b, config)
    return None if shape is None else torch.full(shape, float("nan"), device="cuda")


def _assert_output(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
    config: HopperBlock32Config,
    workspace: torch.Tensor | None,
) -> None:
    returned = gemm_fp8_block32(a, b, sa, sb, out, config, workspace)
    assert returned is out
    expected, work = reference_mm(a, b, sa, sb)
    metrics = error_metrics(out, expected, work, a.shape[-1], config.split_k)
    assert metrics["passed"], metrics


@pytest.mark.parametrize("m,n,k", [(1, 65, 288), (5, 127, 65), (17, 129, 511)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_odd_shapes_and_output_formats(
    m: int, n: int, k: int, dtype: torch.dtype
) -> None:
    a, b, sa, sb = _inputs((m, k), n, 419, (torch.uint8, torch.uint8))
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    _assert_output(a, b, sa, sb, out, _config(1, False), None)


@pytest.mark.parametrize(
    "scale_dtypes",
    [
        (torch.uint8, torch.uint8),
        (torch.float32, torch.uint8),
        (torch.uint8, torch.float32),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize("split_k,swap_ab", [(1, True), (4, False), (16, True)])
def test_scale_formats_split_tail_and_empty_partitions(
    scale_dtypes: tuple[torch.dtype, torch.dtype], split_k: int, swap_ab: bool
) -> None:
    a, b, sa, sb = _inputs((3, 288), 67, 421, scale_dtypes)
    out = torch.empty((3, 67), device="cuda", dtype=torch.bfloat16)
    config = _config(split_k, swap_ab)
    _assert_output(a, b, sa, sb, out, config, _workspace(a, b, config))


def _strided_copy(value: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        tuple(2 * size + 3 for size in value.shape),
        device=value.device,
        dtype=value.dtype,
    )
    view = storage[tuple(slice(1, 1 + 2 * size, 2) for size in value.shape)]
    view.copy_(value)
    return view


def test_batched_strided_inputs_scales_and_output() -> None:
    tensors = _inputs((2, 3, 96), 71, 423, (torch.uint8, torch.uint8))
    a, b, sa, sb = tuple(_strided_copy(value) for value in tensors)
    out = _strided_copy(
        torch.full((2, 3, 71), float("nan"), device="cuda", dtype=torch.bfloat16)
    )
    config = _config(4, True)
    _assert_output(a, b, sa, sb, out, config, _workspace(a, b, config))


@pytest.mark.parametrize(
    "scale_dtypes",
    [
        (torch.uint8, torch.uint8),
        (torch.float32, torch.uint8),
        (torch.uint8, torch.float32),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize("split_k,swap_ab", [(1, False), (4, True), (16, False)])
def test_extreme_reciprocal_scales_and_exact_cancellation(
    split_k: int,
    swap_ab: bool,
    scale_dtypes: tuple[torch.dtype, torch.dtype],
) -> None:
    a = torch.ones((2, 288), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((3, 288), device="cuda")
    b[0, 1::2] = -1
    b[1, :] = 0
    b = b.to(torch.float8_e4m3fn)
    exponents = torch.tensor(
        [0, 1, 63, 119, 127, 135, 191, 253, 254], device="cuda", dtype=torch.uint8
    )
    sa = exponents.repeat(2, 1)
    sb = (254 - exponents).repeat(3, 1)
    if scale_dtypes[0] == torch.float32:
        sa = torch.exp2(sa.cpu().float() - 127).cuda()
    if scale_dtypes[1] == torch.float32:
        sb = torch.exp2(sb.cpu().float() - 127).cuda()
    out = torch.empty((2, 3), device="cuda", dtype=torch.float32)
    config = _config(split_k, swap_ab)
    _assert_output(a, b, sa, sb, out, config, _workspace(a, b, config))
    assert torch.equal(out.cpu(), torch.tensor([[0.0, 0.0, 288.0]]).repeat(2, 1))


def test_graph_replay_reads_live_inputs_and_scales_and_overwrites_scratch() -> None:
    a, b, sa, sb = _inputs((3, 288), 67, 427, (torch.uint8, torch.uint8))
    out = torch.empty((3, 67), device="cuda", dtype=torch.bfloat16)
    config = _config(16, True)
    workspace = _workspace(a, b, config)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemm_fp8_block32(a, b, sa, sb, out, config, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_fp8_block32(a, b, sa, sb, out, config, workspace)
    for seed in (431, 433):
        for destination, source in zip(
            (a, b, sa, sb), _inputs((3, 288), 67, seed, (torch.uint8, torch.uint8))
        ):
            destination.copy_(source)
        workspace.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        expected, work = reference_mm(a, b, sa, sb)
        metrics = error_metrics(out, expected, work, 288, 16)
        assert metrics["passed"], metrics


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_production_quantizer_bytes_and_quant_gemm_chain(dtype: torch.dtype) -> None:
    x = torch.zeros((3, 288), dtype=dtype)
    x[1] = torch.linspace(-448, 448, 288).to(dtype)
    x[2] = torch.tensor(
        [0.0, 1.0e-5, 0.5, -0.5, 448.0, -448.0, 449.0, -449.0] * 36, dtype=dtype
    )
    expected_q, expected_scale = reference_quantize(x)
    a, sa = triton_quantize_fp8_group32_ue8m0(
        x.cuda(), "token_group", 32, "ue8m0", False
    )
    assert torch.equal(a.cpu().view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(sa.cpu(), expected_scale)
    _, b, _, sb = _inputs((3, 288), 67, 439, (torch.uint8, torch.uint8))
    out = torch.empty((3, 67), device="cuda", dtype=torch.bfloat16)
    config = _config(4, False)
    _assert_output(a, b, sa, sb, out, config, _workspace(a, b, config))


@pytest.mark.parametrize(
    "enable_pdl,override", [(False, "triton_mm_fp8_blockscale"), (True, None)]
)
def test_public_dispatch_preserves_output_and_allocates_split_workspace(
    monkeypatch: pytest.MonkeyPatch, enable_pdl: bool, override: str | None
) -> None:
    a, b, sa, sb = _inputs((3, 288), 67, 443, (torch.uint8, torch.uint8))
    out = torch.empty((3, 67), device="cuda", dtype=torch.bfloat16)
    config = _config(4, True)
    calls = []

    def choose(platform, m, n, k):
        assert platform.is_nvidia and (m, n, k) == (3, 67, 288)
        return config

    def candidate(a_arg, b_arg, sa_arg, sb_arg, out_arg, config_arg, workspace):
        assert a_arg is a and b_arg is b and sa_arg is sa and sb_arg is sb
        assert out_arg is out and config_arg is config
        assert workspace.shape == (4, 3, 67) and workspace.dtype == torch.float32
        calls.append(workspace)
        return gemm_fp8_block32(
            a_arg, b_arg, sa_arg, sb_arg, out_arg, config_arg, workspace
        )

    monkeypatch.setattr(triton_gemm, "get_hopper_block32_config", choose)
    monkeypatch.setattr(triton_gemm, "gemm_fp8_block32", candidate)
    previous = pdl_enabled()
    try:
        pdl_enabled(enable_pdl)
        returned = tokenspeed_kernel.mm(
            a,
            b,
            A_scales=sa,
            B_scales=sb,
            bias=None,
            out=out,
            out_dtype=torch.bfloat16,
            alpha=None,
            block_size=[1, 32],
            quant="mxfp8",
            override=override,
            prepacked_scales=False,
        )
    finally:
        pdl_enabled(previous)
    assert returned is out and len(calls) == 1
    expected, work = reference_mm(a, b, sa, sb)
    assert error_metrics(out, expected, work, 288, 4)["passed"]


@pytest.mark.parametrize("mode", ["unmeasured", "float_scales", "float_output"])
def test_wrapper_retains_portable_fallback(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    dtypes = (
        (torch.float32, torch.uint8)
        if mode == "float_scales"
        else (torch.uint8, torch.uint8)
    )
    a, b, sa, sb = _inputs((3, 288), 67, 449, dtypes)
    dtype = torch.float32 if mode == "float_output" else torch.bfloat16
    out = torch.empty((3, 67), device="cuda", dtype=dtype)
    calls = []
    original = triton_gemm._w8a8_block_fp8_matmul

    def choose(platform, m, n, k):
        return None if mode == "unmeasured" else _config(4, True)

    def forbidden(*args):
        raise AssertionError("an unsupported production format reached the candidate")

    class ObservedKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(kwargs)
                return original[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(triton_gemm, "get_hopper_block32_config", choose)
    monkeypatch.setattr(triton_gemm, "gemm_fp8_block32", forbidden)
    monkeypatch.setattr(triton_gemm, "_w8a8_block_fp8_matmul", ObservedKernel())
    returned = triton_gemm.w8a8_block_fp8_matmul_triton(
        a, b, sa, sb, [1, 32], dtype, out
    )
    assert returned is out and len(calls) == 1
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert calls[0] == config
    # This fallback contract preserves the unchanged kernel exactly. Its native
    # FP8 accumulation does not promise the candidate's FP32 roundoff bound.
    expected = torch.empty_like(out)
    m, k = a.shape
    n = b.shape[0]
    original[(((m + 63) // 64) * ((n + 63) // 64),)](
        a,
        b,
        expected,
        sa,
        sb,
        m,
        n,
        k,
        1,
        32,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        expected.stride(0),
        expected.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(1),
        sb.stride(0),
        **config,
    )
    assert torch.isfinite(out).all()
    assert torch.equal(out, expected)
