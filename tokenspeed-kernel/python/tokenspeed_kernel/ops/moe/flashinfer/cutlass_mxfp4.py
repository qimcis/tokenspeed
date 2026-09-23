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

"""Explicit, experimental Hopper W4A16 MoE; deliberately not auto-registered."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.thirdparty.flashinfer.mxfp4 import (
    FlashInferMxfp4State,
    prepare_mxfp4,
    run_mxfp4,
)


@dataclass(frozen=True)
class HopperMxfp4MoE:
    """Packed weights and scratch owned by one non-concurrent execution lane."""

    native: FlashInferMxfp4State
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    hidden: int
    experts: int
    top_k: int


@triton.jit
def _sanitize_routes_and_clear(
    IDS,
    WEIGHTS,
    SAFE_IDS,
    SAFE_WEIGHTS,
    OUT,
    TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    OUTPUTS: tl.constexpr,
    GLOBAL_EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < TOKENS:
        cols = tl.arange(0, BLOCK_K)
        ids = tl.load(IDS + pid * TOP_K + cols, cols < TOP_K, other=-1)
        weights = tl.load(WEIGHTS + pid * TOP_K + cols, cols < TOP_K, other=0.0)
        valid = (cols < TOP_K) & (ids >= 0) & (ids < GLOBAL_EXPERTS)
        same = (ids[:, None] == ids[None, :]) & valid[None, :]
        earlier = cols[None, :] < cols[:, None]
        first = valid & (tl.sum((same & earlier).to(tl.int32), axis=1) == 0)
        merged = tl.sum(tl.where(same, weights[None, :], 0.0), axis=1)

        # Native CUTLASS routing requires distinct IDs, even for zero-weight
        # slots. Coalesce each expert into its first slot, then choose distinct
        # fillers from [0, TOP_K) that do not appear in any valid input slot.
        # GLOBAL_EXPERTS >= TOP_K guarantees enough such filler candidates.
        occupied = (
            tl.sum(
                ((cols[:, None] == ids[None, :]) & valid[None, :]).to(tl.int32),
                axis=1,
            )
            > 0
        )
        available = (cols < TOP_K) & ~occupied
        available_rank = tl.cumsum(available.to(tl.int32), axis=0) - 1
        empty_rank = cols - tl.cumsum(first.to(tl.int32), axis=0)
        filler = tl.sum(
            tl.where(
                available[None, :] & (available_rank[None, :] == empty_rank[:, None]),
                cols[None, :],
                0,
            ),
            axis=1,
        )
        tl.store(
            SAFE_IDS + pid * TOP_K + cols,
            tl.where(first, ids, filler),
            cols < TOP_K,
        )
        tl.store(
            SAFE_WEIGHTS + pid * TOP_K + cols,
            tl.where(first, merged, 0.0),
            cols < TOP_K,
        )
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offsets, 0.0, offsets < OUTPUTS)


def prepare_hopper_mxfp4_moe(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    top_k: int,
    ep_size: int,
    ep_rank: int,
    max_tokens: int,
    swiglu_limit: float,
) -> HopperMxfp4MoE:
    """Prepare an explicitly selected SM90 W4A16 adapter before capture.

    ``w13``/``w2`` are contiguous uint8 packed E2M1, low nibble first, with
    shapes [E, 2*I, H/2] ([gate; up]) and [E, H, I/2]. Their uint8 E8M0 scales
    have shapes [E, 2*I, H/32] and [E, H, I/32]. H and I are multiples of128.
    ``top_k`` is the number of global preselected experts (at most eight); ``ep_size`` and
    ``ep_rank`` describe contiguous expert ownership. ``max_tokens`` bounds
    every call, and ``swiglu_limit`` is the positive gate/up clamp.

    The returned state owns transformed packed weights and maximum scratch,
    with no references to canonical or Marlin layouts. Release the caller's
    canonical weights after preparation when selecting this backend. One
    state must not be used concurrently on multiple streams. Warm the normal
    execution path before graph capture; no capture-specific path exists.
    """
    if w2.ndim != 3:
        raise ValueError("w2 must have shape [E, H, I/2]")
    experts, hidden, packed_intermediate = w2.shape
    intermediate = packed_intermediate * 2
    if min(experts, hidden, intermediate, max_tokens, ep_size, top_k) <= 0:
        raise ValueError(
            "MoE dimensions, max_tokens, ep_size and top_k must be positive"
        )
    if hidden % 128 or intermediate % 128:
        raise ValueError("Hopper W4A16 requires H and I divisible by128")
    if not 0 <= ep_rank < ep_size or top_k > min(8, experts * ep_size):
        raise ValueError("Invalid expert-parallel rank or top_k")
    if not math.isfinite(swiglu_limit) or swiglu_limit <= 0:
        raise ValueError("swiglu_limit must be finite and positive")
    shapes = (
        (w13, (experts, 2 * intermediate, hidden // 2)),
        (w2, (experts, hidden, intermediate // 2)),
        (w13_scales, (experts, 2 * intermediate, hidden // 32)),
        (w2_scales, (experts, hidden, intermediate // 32)),
    )
    for tensor, shape in shapes:
        if tensor.shape != shape or tensor.dtype != torch.uint8:
            raise ValueError(
                "Expected canonical packed uint8 MXFP4 weights/E8M0 scales"
            )
        if not tensor.is_contiguous() or tensor.device != w13.device:
            raise ValueError("All weights and scales must be contiguous on one device")
    if not w13.is_cuda or torch.cuda.get_device_capability(w13.device) != (9, 0):
        raise ValueError("This experimental W4A16 adapter supports SM90 CUDA devices")
    native = prepare_mxfp4(
        w13,
        w2,
        w13_scales,
        w2_scales,
        top_k,
        ep_size,
        ep_rank,
        max_tokens,
        swiglu_limit,
    )
    return HopperMxfp4MoE(
        native=native,
        route_ids=torch.empty(
            (max_tokens, top_k), dtype=torch.int32, device=w13.device
        ),
        route_weights=torch.empty(
            (max_tokens, top_k), dtype=torch.float32, device=w13.device
        ),
        hidden=hidden,
        experts=experts,
        top_k=top_k,
    )


def hopper_mxfp4_moe(
    state: HopperMxfp4MoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write one local EP partial into caller-owned BF16 ``out`` and return it.

    ``x``/``out`` are contiguous [M,H] BF16 tensors. Global ``topk_ids`` are
    int32 [M,K], and ``topk_weights`` are FP32 [M,K], already globally
    normalized, with no routed scaling factor. Duplicate expert weights are
    summed into their first occurrence. Invalid IDs contribute zero; nonlocal IDs are handled by native EP routing. No local renormalization or
    DeepSeek routed factor is applied here. All tensors share the prepared
    device. M=0 is supported, and 0<=M<=max_tokens. Inputs and output must not
    alias. The live input/routes are consumed on every eager call or replay.
    """
    if x.ndim != 2 or x.shape[1] != state.hidden:
        raise ValueError("x must have shape [M,H] matching prepared weights")
    tokens = x.shape[0]
    if tokens > state.native.max_tokens:
        raise ValueError("Token count exceeds the prepared workspace capacity")
    for tensor, shape, dtype in (
        (x, (tokens, state.hidden), torch.bfloat16),
        (out, (tokens, state.hidden), torch.bfloat16),
        (topk_ids, (tokens, state.top_k), torch.int32),
        (topk_weights, (tokens, state.top_k), torch.float32),
    ):
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                "Invalid activation, output or precomputed routing shape/dtype"
            )
        if tensor.device != state.native.w13.device or not tensor.is_contiguous():
            raise ValueError(
                "Runtime tensors must be contiguous on the prepared device"
            )
    if tokens == 0:
        return out
    out_begin = out.data_ptr()
    out_end = out_begin + out.numel() * out.element_size()
    for tensor in (x, topk_ids, topk_weights):
        begin = tensor.data_ptr()
        if (
            begin < out_end
            and out_begin < begin + tensor.numel() * tensor.element_size()
        ):
            raise ValueError("Output must not alias activations or routing inputs")
    route_ids = state.route_ids[:tokens]
    route_weights = state.route_weights[:tokens]
    _sanitize_routes_and_clear[(max(tokens, triton.cdiv(out.numel(), 256)),)](
        topk_ids,
        topk_weights,
        route_ids,
        route_weights,
        out,
        TOKENS=tokens,
        TOP_K=state.top_k,
        OUTPUTS=out.numel(),
        GLOBAL_EXPERTS=state.experts * state.native.ep_size,
        BLOCK_K=triton.next_power_of_2(state.top_k),
        BLOCK=256,
    )
    run_mxfp4(state.native, x, route_weights, route_ids, out)
    return out
