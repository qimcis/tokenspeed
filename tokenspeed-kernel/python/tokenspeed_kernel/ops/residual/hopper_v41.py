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

"""SM90 V4.1 HC4 epilogues with explicit intermediate BF16 rounding."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, Platform


@triton.jit
def _v41_hc_epilogue(
    x,
    row,
    d,
    mask,
    R,
    POST,
    COMB,
    PRE,
    W,
    HC,
    NORMALIZED,
    Q,
    S,
    H: tl.constexpr,
    EPS: tl.constexpr,
    QUANTIZE: tl.constexpr,
    B: tl.constexpr,
):
    a0 = tl.load(POST + row * 4).to(tl.float32) * x
    a1 = tl.load(POST + row * 4 + 1).to(tl.float32) * x
    a2 = tl.load(POST + row * 4 + 2).to(tl.float32) * x
    a3 = tl.load(POST + row * 4 + 3).to(tl.float32) * x
    for i in tl.static_range(4):
        r = tl.load(R + row * 4 * H + i * H + d, mask, other=0).to(tl.float32)
        a0 += tl.load(COMB + row * 16 + i * 4).to(tl.float32) * r
        a1 += tl.load(COMB + row * 16 + i * 4 + 1).to(tl.float32) * r
        a2 += tl.load(COMB + row * 16 + i * 4 + 2).to(tl.float32) * r
        a3 += tl.load(COMB + row * 16 + i * 4 + 3).to(tl.float32) * r
    # Match the materialized HC residual before the next collapse reads it.
    b0 = a0.to(tl.bfloat16)
    b1 = a1.to(tl.bfloat16)
    b2 = a2.to(tl.bfloat16)
    b3 = a3.to(tl.bfloat16)
    tl.store(HC + row * 4 * H + d, b0, mask)
    tl.store(HC + row * 4 * H + H + d, b1, mask)
    tl.store(HC + row * 4 * H + 2 * H + d, b2, mask)
    tl.store(HC + row * 4 * H + 3 * H + d, b3, mask)
    z = tl.full((B,), 0, tl.float32)
    z += tl.load(PRE + row * 4).to(tl.float32) * b0.to(tl.float32)
    z += tl.load(PRE + row * 4 + 1).to(tl.float32) * b1.to(tl.float32)
    z += tl.load(PRE + row * 4 + 2).to(tl.float32) * b2.to(tl.float32)
    z += tl.load(PRE + row * 4 + 3).to(tl.float32) * b3.to(tl.float32)
    z = tl.where(mask, z.to(tl.bfloat16).to(tl.float32), 0)
    inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
    w = tl.load(W + d, mask, other=0).to(tl.float32)
    normalized = (z * inv * w).to(tl.bfloat16)
    tl.store(NORMALIZED + row * H + d, normalized, mask)
    if QUANTIZE:
        values = tl.reshape(normalized.to(tl.float32), (B // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
        raw = amax * (1.0 / 448.0)
        bits = raw.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        codes = tl.div_rn(values, scale[:, None]).to(tl.float8e4nv)
        tl.store(Q + row * H + d, tl.reshape(codes, (B,)), mask)
        group = tl.arange(0, B // 32)
        tl.store(S + row * (H // 32) + group, exponent.to(tl.uint8), group < H // 32)


@triton.jit
def _v41_post_pre_norm_quant(
    X,
    R,
    POST,
    COMB,
    PRE,
    W,
    HC,
    NORMALIZED,
    Q,
    S,
    H: tl.constexpr,
    EPS: tl.constexpr,
    QUANTIZE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    mask = d < H
    x = tl.load(X + row * H + d, mask, other=0).to(tl.float32)
    _v41_hc_epilogue(
        x,
        row,
        d,
        mask,
        R,
        POST,
        COMB,
        PRE,
        W,
        HC,
        NORMALIZED,
        Q,
        S,
        H,
        EPS,
        QUANTIZE,
        B,
    )


def v41_post_pre_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    quantize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply HC post, previous-pre collapse, RMSNorm and optional exact FP8.

    Args:
        x: Contiguous BF16 sublayer result [tokens, hidden].
        residual: Contiguous BF16 previous HC state [tokens, 4, hidden].
        post: Contiguous FP32 current post coefficients [tokens, 4].
        comb: Contiguous FP32 [tokens, input HC, output HC] coefficients.
        pre: Contiguous FP32 previous-sublayer pre coefficients [tokens, 4].
        norm_weight: Contiguous BF16 or FP32 RMSNorm weights [hidden].
        eps: Finite positive RMSNorm epsilon.
        quantize: Produce row-major per32 E8M0 scales and FP8 E4M3 codes.

    Returns:
        Fresh HC residual, normalized BF16 input, optional FP8 codes/scales.
        No input is mutated and outputs are owned by this invocation, including
        CUDA graph capture. All three unfused BF16 rounding points are retained.
    """
    if not x.is_cuda:
        raise ValueError("V4.1 fused epilogue requires SM90 CUDA")
    platform = Platform.get()
    if not platform.is_nvidia or platform.arch_version != ArchVersion(9, 0):
        raise ValueError("V4.1 fused epilogue requires SM90 CUDA")
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be BF16 [tokens, hidden]")
    rows, hidden = x.shape
    if hidden < 32 or hidden > 8192 or hidden % 32:
        raise ValueError("hidden must be divisible by32 and in [32,8192]")
    if residual.shape != (rows, 4, hidden) or residual.dtype != torch.bfloat16:
        raise ValueError("residual must be BF16 [tokens,4,hidden]")
    for value, shape in ((post, (rows, 4)), (comb, (rows, 4, 4)), (pre, (rows, 4))):
        if value.shape != shape or value.dtype != torch.float32:
            raise ValueError("HC coefficients must be FP32 with matching HC4 shapes")
    if norm_weight.shape != (hidden,) or norm_weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("invalid RMSNorm weight")
    if not math.isfinite(eps) or eps <= 0 or type(quantize) is not bool:
        raise ValueError("require finite positive epsilon and explicit bool quantize")
    tensors = (x, residual, post, comb, pre, norm_weight)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous on the same CUDA device")
    hc = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn) if quantize else None
    scales = (
        torch.empty((rows, hidden // 32), device=x.device, dtype=torch.uint8)
        if quantize
        else None
    )
    if rows:
        _v41_post_pre_norm_quant[(rows,)](
            x,
            residual,
            post,
            comb,
            pre,
            norm_weight,
            hc,
            normalized,
            codes,
            scales,
            hidden,
            eps,
            quantize,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return hc, normalized, codes, scales
