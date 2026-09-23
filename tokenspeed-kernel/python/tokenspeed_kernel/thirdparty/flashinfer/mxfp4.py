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

"""Optional FlashInfer SM90 mixed-input MXFP4 boundary.

This module imports FlashInfer only when preparing an explicitly selected adapter.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class FlashInferMxfp4State:
    w13: torch.Tensor
    w2: torch.Tensor
    scales: list[torch.Tensor]
    alpha: torch.Tensor
    beta: torch.Tensor
    limit: torch.Tensor
    workspace: torch.Tensor
    activation_type: object
    run: Callable
    ep_size: int
    ep_rank: int
    max_tokens: int


def prepare_mxfp4(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    top_k: int,
    ep_size: int,
    ep_rank: int,
    max_tokens: int,
    swiglu_limit: float,
) -> FlashInferMxfp4State:
    """Prepare vendor layouts and one stream's maximum workspace before capture."""
    try:
        from flashinfer import ActivationType
        from flashinfer.fused_moe import (
            cutlass_fused_moe,
            cutlass_fused_moe_workspace_size,
            interleave_moe_scales_for_sm90_mixed_gemm,
            interleave_moe_weights_for_sm90_mixed_gemm,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Hopper W4A16 requires optional FlashInfer with SM90 mixed-input "
            "weight/scale interleave and persistent CUTLASS workspace APIs"
        ) from exc
    required = {"workspace_buffer", "profile_ids", "swiglu_limit"}
    if not required.issubset(inspect.signature(cutlass_fused_moe).parameters):
        raise RuntimeError(
            "Installed FlashInfer lacks the required persistent W4A16 ABI"
        )

    experts, hidden, packed_intermediate = w2.shape
    intermediate = packed_intermediate * 2
    # TokenSpeed stores [gate; up]; the vendor's gated activation expects
    # [up; gate]. Transform scales identically, without modifying the caller.
    swapped_w13 = torch.cat((w13[:, intermediate:], w13[:, :intermediate]), dim=1)
    prepared_w13 = interleave_moe_weights_for_sm90_mixed_gemm(swapped_w13, "fp4")
    del swapped_w13
    prepared_w2 = interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4")
    swapped_scales = torch.cat(
        (w13_scales[:, intermediate:], w13_scales[:, :intermediate]), dim=1
    )
    prepared_s13 = interleave_moe_scales_for_sm90_mixed_gemm(swapped_scales, 32)
    del swapped_scales
    prepared_s2 = interleave_moe_scales_for_sm90_mixed_gemm(w2_scales, 32)
    # Explicit alpha=1, beta=0 retain DeepSeek SwiGLU and activate the vendor's
    # clamped adaptor: gate=min(gate, limit), up=clamp(up, -limit, limit).
    activation = ActivationType.SwigluBias
    workspace_bytes = cutlass_fused_moe_workspace_size(
        max_num_tokens=max_tokens,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts_total=experts * ep_size,
        top_k=top_k,
        x_dtype=torch.bfloat16,
        weight_dtype=torch.uint8,
        output_dtype=torch.bfloat16,
        activation_type=activation,
        tp_size=1,
        tp_rank=0,
        ep_size=ep_size,
        ep_rank=ep_rank,
        min_latency_mode=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=True,
        use_mxfp8_act_scaling=False,
        use_fused_finalize=True,
        use_packed_weights=False,
        use_wfp4afp8_humming=False,
        device=w13.device,
    )
    return FlashInferMxfp4State(
        w13=prepared_w13,
        w2=prepared_w2,
        scales=[prepared_s13.view(torch.int32), prepared_s2.view(torch.int32)],
        alpha=torch.ones(experts, dtype=torch.float32, device=w13.device),
        beta=torch.zeros(experts, dtype=torch.float32, device=w13.device),
        limit=torch.full(
            (experts,), swiglu_limit, dtype=torch.float32, device=w13.device
        ),
        workspace=torch.empty(workspace_bytes, dtype=torch.uint8, device=w13.device),
        activation_type=activation,
        run=cutlass_fused_moe,
        ep_size=ep_size,
        ep_rank=ep_rank,
        max_tokens=max_tokens,
    )


def run_mxfp4(
    state: FlashInferMxfp4State,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Run the same prepared vendor path in eager execution and CUDA graphs."""
    state.run(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=state.w13,
        fc2_expert_weights=state.w2,
        output_dtype=torch.bfloat16,
        quant_scales=state.scales,
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        input_sf=None,
        swiglu_alpha=state.alpha,
        swiglu_beta=state.beta,
        swiglu_limit=state.limit,
        swizzled_input_sf=False,
        tp_size=1,
        tp_rank=0,
        ep_size=state.ep_size,
        ep_rank=state.ep_rank,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=True,
        use_mxfp8_act_scaling=False,
        use_wfp4afp8_humming=False,
        min_latency_mode=False,
        tune_max_num_tokens=state.max_tokens,
        enable_pdl=False,
        activation_type=state.activation_type,
        output=out,
        use_packed_weights=False,
        use_fused_finalize=True,
        profile_ids=[-1, -1],
        workspace_buffer=state.workspace,
    )
