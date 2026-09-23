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

"""Explicit Hopper W4A16 MoE for a dedicated prefill execution lane."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.mxfp4 import (
    FlashInferMxfp4State,
    FlashInferMxfp4Workspace,
    create_mxfp4_workspace,
    prepare_mxfp4_with_workspace,
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


@dataclass(frozen=True)
class HopperMxfp4Lane:
    """Fixed-capacity scratch for serialized MoE layers on one execution lane."""

    native: FlashInferMxfp4Workspace
    route_ids: torch.Tensor
    route_weights: torch.Tensor


def create_hopper_mxfp4_lane(
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    ep_size: int,
    ep_rank: int,
    max_tokens: int,
    swiglu_limit: float,
    device: torch.device,
) -> HopperMxfp4Lane:
    """Allocate one SM90 lane before loading weights or capturing graphs.

    ``hidden`` and ``intermediate`` are unpacked dimensions divisible by128;
    ``experts`` is the local count, and ``ep_size``/``ep_rank`` describe global
    contiguous expert ownership. ``top_k`` is at most eight. ``max_tokens``
    bounds every operation, and ``swiglu_limit`` specifies the positive clamp.
    ``device`` must be an indexed SM90 CUDA device.

    Return fixed native/routing buffers shared by layers with exactly this
    geometry. Keep the lane alive for the model lifetime. Calls sharing it must
    be serialized; concurrent execution lanes require separate instances.
    The buffers never grow and do not contain caller outputs.
    """
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
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Hopper W4A16 requires an indexed CUDA device")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("Hopper W4A16 requires an SM90 CUDA device")
    native = create_mxfp4_workspace(
        hidden,
        intermediate,
        experts,
        top_k,
        ep_size,
        ep_rank,
        max_tokens,
        swiglu_limit,
        device,
    )
    return HopperMxfp4Lane(
        native=native,
        route_ids=torch.empty((max_tokens, top_k), dtype=torch.int32, device=device),
        route_weights=torch.empty(
            (max_tokens, top_k), dtype=torch.float32, device=device
        ),
    )


def _prepare_hopper_mxfp4_with_lane(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    lane: HopperMxfp4Lane,
) -> HopperMxfp4MoE:
    geometry = lane.native
    experts, hidden, intermediate = (
        geometry.experts,
        geometry.hidden,
        geometry.intermediate,
    )
    shapes = (
        (w13, (experts, 2 * intermediate, hidden // 2)),
        (w2, (experts, hidden, intermediate // 2)),
        (w13_scales, (experts, 2 * intermediate, hidden // 32)),
        (w2_scales, (experts, hidden, intermediate // 32)),
    )
    for tensor, shape in shapes:
        if tensor.shape != shape or tensor.dtype != torch.uint8:
            raise ValueError(
                "Expected canonical packed uint8 MXFP4 weights/E8M0 scales matching the lane"
            )
        if not tensor.is_contiguous() or tensor.device != geometry.workspace.device:
            raise ValueError(
                "All weights and scales must be contiguous on the lane device"
            )
    native = prepare_mxfp4_with_workspace(w13, w2, w13_scales, w2_scales, geometry)
    return HopperMxfp4MoE(
        native=native,
        route_ids=lane.route_ids,
        route_weights=lane.route_weights,
        hidden=hidden,
        experts=experts,
        top_k=geometry.top_k,
    )


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
    """Prepare standalone packed weights with private scratch before capture.

    Canonical uint8 weights are [E,2I,H/2] in [gate;up] order and [E,H,I/2],
    low nibble first. Raw E8M0 scales are [E,2I,H/32] and [E,H,I/32]. Every
    argument is explicit; parallel ownership, capacity and clamp follow
    :func:`create_hopper_mxfp4_lane`. The returned state retains transformed
    weights only and must not execute concurrently on multiple streams.
    Serving uses a shared model-owned lane through the weight preprocessor.
    """
    if w2.ndim != 3:
        raise ValueError("w2 must have shape [E,H,I/2]")
    experts, hidden, packed_intermediate = w2.shape
    lane = create_hopper_mxfp4_lane(
        hidden,
        packed_intermediate * 2,
        experts,
        top_k,
        ep_size,
        ep_rank,
        max_tokens,
        swiglu_limit,
        w13.device,
    )
    return _prepare_hopper_mxfp4_with_lane(w13, w2, w13_scales, w2_scales, lane)


def cutlass_w4a16_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Transform one loaded layer into FI-only storage using its injected lane."""
    if plan.get("solution") != "cutlass_w4a16":
        raise ValueError(
            "Hopper W4A16 preprocessing requires explicit cutlass_w4a16 selection"
        )
    lane = getattr(w, "_hopper_mxfp4_lane", None)
    if not isinstance(lane, HopperMxfp4Lane):
        raise ValueError("Create and inject _hopper_mxfp4_lane before loading weights")
    if getattr(w, "_hopper_mxfp4_state", None) is not None:
        raise ValueError("Hopper W4A16 weights have already been processed")
    if getattr(w, "_marlin_repacked", False):
        raise ValueError("Hopper W4A16 cannot consume Marlin-repacked weights")
    if (
        plan.get("activation") != "swiglu"
        or getattr(w, "w13_input_layout", "concatenated") != "concatenated"
    ):
        raise ValueError("Hopper W4A16 requires concatenated gate/up SwiGLU weights")
    arg = getattr(w, "swiglu_arg", None)
    if getattr(arg, "alpha", None) not in {None, 1.0} or getattr(
        w, "swiglu_beta", None
    ) not in {None, 0.0}:
        raise ValueError("Hopper W4A16 supports only SwiGLU alpha1/beta0")
    if getattr(arg, "limit", None) != lane.native.swiglu_limit:
        raise ValueError("Layer SwiGLU clamp differs from the prepared lane")
    geometry = lane.native
    if (w.top_k, w.ep_size, w.ep_rank, w.num_local_experts) != (
        geometry.top_k,
        geometry.ep_size,
        geometry.ep_rank,
        geometry.experts,
    ):
        raise ValueError("Layer routing/EP geometry differs from the prepared lane")
    if getattr(w, "tp_size", 1) != 1:
        raise ValueError("Hopper W4A16 expects unsharded local experts (MoE TP1)")
    state = _prepare_hopper_mxfp4_with_lane(
        w.w13_weight,
        w.w2_weight,
        w.w13_weight_scale,
        w.w2_weight_scale,
        lane,
    )
    # Keep Parameter identities so loader-side dictionaries cannot retain the
    # old storage. Native state and parameters share the new packed storage.
    w.w13_weight.data = state.native.w13
    w.w2_weight.data = state.native.w2
    w.w13_weight_scale.data = state.native.scales[0]
    w.w2_weight_scale.data = state.native.scales[1]
    w._hopper_mxfp4_state = state


@register_kernel(
    "moe",
    "apply",
    name="cutlass_w4a16_precomputed_moe_apply",
    solution="cutlass_w4a16",
    weight_preprocessor=cutlass_w4a16_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PERFORMANT,
    tags={"explicit_only", "throughput"},
)
def cutlass_w4a16_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Execute the pinned precomputed-route backend with distinct output storage."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if plan.get("solution") != "cutlass_w4a16" or not do_finalize:
        raise ValueError("Hopper W4A16 requires its explicit plan and finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Hopper W4A16 requires precomputed global top-k routes")
    state = getattr(w, "_hopper_mxfp4_state", None)
    if not isinstance(state, HopperMxfp4MoE):
        raise ValueError("Hopper W4A16 weights were not processed before execution")
    # Scratch may be reused by the next layer; returned results may not. CUDA
    # graph capture retains each call's output, exactly like eager execution.
    out = torch.empty_like(x)
    return hopper_mxfp4_moe(state, x, topk_weights, topk_ids, out)


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
