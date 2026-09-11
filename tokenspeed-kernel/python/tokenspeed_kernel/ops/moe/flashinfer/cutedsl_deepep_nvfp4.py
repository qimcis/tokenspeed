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

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPMode
from tokenspeed_kernel.ops.moe._deepep import apply_bf16_deepep, get_bf16_dispatcher
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty.flashinfer.nvfp4_moe import (
        ActivationType,
        CuteDslMoEWrapper,
        convert_sf_to_mma_layout,
        fp4_quantize,
        grouped_gemm_nt_masked,
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )

    _modes = frozenset(
        {"normal", "low_latency"} if CuteDslMoEWrapper is not None else {"low_latency"}
    )

    def _interleave_gate_up(tensor: torch.Tensor) -> None:
        # Normal and masked GEMMs share one packed weight tensor.
        for expert in tensor.view(torch.uint8):
            shape = expert.shape
            interleaved = expert.view(2, shape[0] // 128, 64, shape[1])
            expert.copy_(interleaved.flip(0).transpose(0, 1).reshape(shape))

    def flashinfer_cutedsl_deepep_nvfp4_moe_weights(plan: dict, w: torch.nn.Module):
        use_normal = plan["deepep_mode"] != "low_latency"
        if use_normal:
            if CuteDslMoEWrapper is None:
                raise ValueError("FlashInfer does not provide the NVFP4 normal MoE API")
            hidden_size = w.w13_weight.shape[2] * 2
            intermediate_size = w.w2_weight.shape[2] * 2
            if hidden_size % 128 or intermediate_size % 64:
                raise ValueError(
                    "CuTe NVFP4 requires hidden size aligned to 128 and "
                    "intermediate size aligned to 64"
                )
            _interleave_gate_up(w.w13_weight.data)
            _interleave_gate_up(w.w13_weight_scale.data)
        plan["_nvfp4_interleaved"] = use_normal
        w13_ws2 = w.w13_weight_scale_2[:, 0]
        w13_input_scale = w.w13_input_scale.max().to(torch.float32)
        w2_input_scale = w.w2_input_scale.max().to(torch.float32)
        w.w13_weight_scale_2 = torch.nn.Parameter(w13_ws2, requires_grad=False)
        w.w13_input_scale_quant = torch.nn.Parameter(
            (1.0 / w13_input_scale).to(torch.float32), requires_grad=False
        )
        w.w2_input_scale_quant = torch.nn.Parameter(
            (1.0 / w2_input_scale).to(torch.float32), requires_grad=False
        )
        w.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
        )
        w.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * w.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        scales = w.w13_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w13_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)

        scales = w.w2_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w2_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)
        if use_normal:
            num_experts = w.w13_weight.shape[0]
            for name in ("w13", "w2"):
                weight = getattr(w, f"{name}_weight")
                scales = getattr(w, f"{name}_blockscale_swizzled")
                mma = convert_sf_to_mma_layout(
                    scales,
                    m=weight.shape[1],
                    k=weight.shape[2] * 2,
                    num_groups=num_experts,
                    sf_vec_size=16,
                )
                setattr(w, f"{name}_blockscale_mma", mma)
            plan["_nvfp4_moe_wrapper"] = CuteDslMoEWrapper(
                num_experts=w.num_experts,
                top_k=w.top_k,
                hidden_size=w.w13_weight.shape[2] * 2,
                intermediate_size=w.w2_weight.shape[2] * 2,
                use_cuda_graph=True,
                max_num_tokens=None,
                num_local_experts=num_experts,
                local_expert_offset=w.ep_rank * num_experts,
                tile_size=128,
                sf_vec_size=16,
                output_dtype=torch.bfloat16,
                device=w.w13_weight.device,
                enable_pdl=False,
                activation_type=ActivationType.Swiglu.value,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
                swiglu_limit=torch.finfo(torch.float32).max,
                situ_beta=None,
                situ_linear_beta=None,
                use_fused_finalize=False,
                quant_mode="w4a4",
            )
        return None

    def flashinfer_cutedsl_nvfp4_experts(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Return weighted BF16 contributions for received rows and global routes.

        The prepared plan owns the persistent wrapper. Invalid and nonlocal
        routes contribute zero; output retains x's row capacity.
        """
        expert_start = w.ep_rank * w.w13_weight.shape[0]
        valid = (topk_ids >= expert_start) & (
            topk_ids < expert_start + w.w13_weight.shape[0]
        )
        valid_rows = valid.any(dim=-1)
        recv_x = torch.where(valid_rows[:, None], x, 0)
        recv_ids = torch.where(valid, topk_ids, -1)
        recv_weights = torch.where(valid, topk_weights, 0)
        wrapper = plan["_nvfp4_moe_wrapper"]
        quantized, scales = fp4_quantize(
            recv_x,
            global_scale=w.w13_input_scale_quant,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
            is_sf_8x4_layout=False,
            is_global_scale_inversed=False,
            enable_pdl=enable_pdl,
            backend="cuda",
        )
        scales = convert_sf_to_mma_layout(
            scales,
            m=recv_x.shape[0],
            k=recv_x.shape[1],
            num_groups=1,
            sf_vec_size=16,
        )
        output = wrapper.run(
            x=quantized,
            x_sf=scales,
            token_selected_experts=recv_ids.to(torch.int32),
            token_final_scales=recv_weights.float(),
            w1_weight=w.w13_weight,
            w1_weight_sf=w.w13_blockscale_mma,
            w1_alpha=w.g1_alphas,
            fc2_input_scale=w.w2_input_scale_quant,
            w2_weight=w.w2_weight,
            w2_weight_sf=w.w2_blockscale_mma,
            w2_alpha=w.g2_alphas,
            tactic=None,
            per_token_scale=None,
        )
        return torch.where(valid_rows[:, None], output, 0)

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        solution="flashinfer_cutedsl",
        weight_preprocessor=flashinfer_cutedsl_deepep_nvfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "deepep_modes": _modes,
            "supports_prefill_graph": frozenset({"normal" in _modes}),
            "ispp_alignment": frozenset({64}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutedsl_deepep_nvfp4_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
        low_latency: bool | None = None,
        overlap_fn: Callable[[], None] | None = None,
    ):
        """Run native NVFP4 experts with normal or masked low-latency DeepEP.

        Args:
            plan: Prepared MoE plan and persistent dispatcher state.
            x: Local BF16 hidden states, shaped [tokens, hidden].
            w: Packed NVFP4 expert weights and calibrated scales.
            router_logits: Logits used when precomputed routes are absent.
            topk_weights: Route weights, applied once by compute or combine.
            topk_ids: Global expert IDs, with -1 marking invalid routes.
            num_tokens_global: Unused.
            max_num_tokens_per_gpu: Unused; capacity comes from the plan.
            do_finalize: Must be true.
            enable_pdl: Quantization PDL setting for normal dispatch.
            low_latency: Collective execution variant selected by the caller.
            overlap_fn: Optional work queued between dispatch phases.

        Returns:
            Combined BF16 output with the same shape as x.
        """
        if not do_finalize:
            raise ValueError("NVFP4 DeepEP does not support deferred finalize")
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        dispatcher = get_bf16_dispatcher(plan, w, x)
        use_low_latency = (
            dispatcher.deepep_mode.resolve(low_latency) == DeepEPMode.low_latency
        )
        if not use_low_latency:

            def compute(recv_x, recv_weights, recv_ids):
                return flashinfer_cutedsl_nvfp4_experts(
                    plan, recv_x, w, recv_weights, recv_ids, enable_pdl
                )

            return apply_bf16_deepep(
                dispatcher,
                x,
                topk_weights,
                topk_ids,
                False,
                w.ep_rank * w.w13_weight.shape[0],
                compute,
                overlap_fn,
            )

        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.float()
        dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=True)
        if overlap_fn is not None:
            overlap_fn()
        recv_hidden, _, _, _, _, _, masked_m = dispatcher.dispatch_b()

        num_local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            recv_hidden,
            masked_m,
            w.w13_input_scale_quant.expand(num_local_experts).contiguous(),
        )
        sf_vec_size = 16
        gateup_output = torch.empty(
            (num_local_experts, recv_hidden.shape[1], w.w2_weight.shape[-1] * 4),
            dtype=torch.bfloat16,
            device=x.device,
        ).permute(1, 2, 0)
        grouped_gemm_nt_masked(
            (a_q, a_q_sf),
            (w.w13_weight.permute(1, 2, 0), w.w13_blockscale_swizzled),
            gateup_output,
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=sf_vec_size,
            alpha=w.g1_alphas.view(1, 1, num_local_experts),
            alpha_dtype="float32",
        )

        gateup_output = gateup_output.permute(2, 0, 1)
        if plan["_nvfp4_interleaved"]:
            shape = gateup_output.shape
            gateup_output = gateup_output.view(*shape[:2], shape[2] // 128, 2, 64)
            gateup_output = gateup_output.transpose(2, 3).flip(2).reshape(shape)
        diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
            gateup_output,
            masked_m,
            w.w2_input_scale_quant.expand(num_local_experts).contiguous(),
        )
        output = torch.empty(
            (num_local_experts, recv_hidden.shape[1], x.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        ).permute(1, 2, 0)
        grouped_gemm_nt_masked(
            (diq, diq_sf),
            (w.w2_weight.permute(1, 2, 0), w.w2_blockscale_swizzled),
            output,
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=sf_vec_size,
            alpha=w.g2_alphas.view(1, 1, num_local_experts),
            alpha_dtype="float32",
        )

        dispatcher.combine_a(
            output.permute(2, 0, 1), topk_ids, topk_weights, low_latency=True
        )
        return dispatcher.combine_b()
