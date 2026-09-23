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


"""Prefill-worker setup for the SM90 weight-only MoE backend."""

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)
BACKEND = "cutlass_w4a16"


def validate_cutlass_w4a16_server(args) -> None:
    """Reject unsupported worker roles before model loading starts."""
    if args.draft_moe_backend == BACKEND:
        raise ValueError("cutlass_w4a16 is a target-only prefill backend")
    if args.moe_backend != BACKEND:
        return
    if args.disaggregation_mode != "prefill":
        raise ValueError("cutlass_w4a16 requires --disaggregation-mode prefill")
    if args.device != "cuda":
        raise ValueError("cutlass_w4a16 requires SM90 CUDA GPUs")
    if args.speculative_algorithm is not None:
        raise ValueError("cutlass_w4a16 does not support speculative workers")
    if args.enable_mixed_batch:
        raise ValueError("cutlass_w4a16 requires prefill-only batches")
    if args.enable_memory_saver:
        raise ValueError("cutlass_w4a16 requires resident weights and workspace")
    if args.enable_eplb or args.ep_num_redundant_experts:
        raise ValueError("cutlass_w4a16 requires fixed contiguous expert ownership")
    if args.all2all_backend != "none":
        raise ValueError("cutlass_w4a16 currently requires --all2all-backend none")
    if args.chunked_prefill_size is None or args.chunked_prefill_size <= 0:
        raise ValueError("cutlass_w4a16 requires positive chunked_prefill_size")
    if args.chunked_prefill_size > args.max_prefill_tokens:
        raise ValueError(
            "cutlass_w4a16 chunked_prefill_size exceeds workspace capacity"
        )
    if args.max_prefill_tokens <= 0:
        raise ValueError("cutlass_w4a16 requires a positive max_prefill_tokens")
    if args.load_format not in {"auto", "safetensors", "instanttensor", "pt"}:
        raise ValueError("cutlass_w4a16 requires a canonical checkpoint loader")


def validate_cutlass_w4a16_model(settings: Mapping, model_config, load_config) -> None:
    """Validate the supported model and parallel layout before construction."""
    if settings["moe_backend"] != BACKEND:
        return
    if settings["disaggregation_mode"] != "prefill":
        raise ValueError("cutlass_w4a16 requires a dedicated prefill worker")
    if load_config.load_format not in {"auto", "safetensors", "instanttensor", "pt"}:
        raise ValueError("cutlass_w4a16 requires canonical checkpoint weights")
    config = model_config.hf_config
    text = getattr(config, "text_config", None)
    if getattr(config, "model_type", None) != "deepseek_v41" or text is None:
        raise ValueError("cutlass_w4a16 currently supports DeepSeek V4.1 Flash only")
    geometry = tuple(
        getattr(text, key, None)
        for key in (
            "hidden_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "swiglu_limit",
        )
    )
    if geometry != (5120, 2304, 384, 6, 10.0):
        raise ValueError(
            "cutlass_w4a16 requires V4.1 Flash H5120/I2304/E384/K6/clamp10"
        )
    if model_config.dtype != torch.bfloat16:
        raise ValueError("cutlass_w4a16 requires bfloat16 model activations")
    mapping = model_config.mapping
    if (
        mapping.pp_size,
        mapping.attn.tp_size,
        mapping.attn.dp_size,
        mapping.moe.tp_size,
        mapping.moe.ep_size,
        mapping.moe.dp_size,
    ) != (1, 8, 1, 1, 8, 1):
        raise ValueError(
            "cutlass_w4a16 currently requires attention TP8 and MoE EP8, DP1/PP1"
        )
    if mapping.attn.cp_size != 1 or mapping.attn.dcp_size != 1:
        raise ValueError("cutlass_w4a16 currently requires CP1 and DCP1")
    if settings["max_prefill_tokens"] <= 0:
        raise ValueError("cutlass_w4a16 workspace capacity must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("cutlass_w4a16 requires SM90 CUDA GPUs")


def prepare_cutlass_w4a16_workspace(model, settings: Mapping) -> None:
    """Attach one fixed-capacity lane to this model's serialized MoE layers.

    Called after construction and before checkpoint loading. Weight conversion
    stays in the ordinary per-module post-load hook, after loader references
    to canonical parameters have been released. Outputs never alias this lane.
    """
    if settings["moe_backend"] != BACKEND:
        return
    import tokenspeed_kernel

    layers = [
        module
        for module in model.modules()
        if getattr(module, "plan", {}).get("solution") == BACKEND
    ]
    if not layers:
        raise ValueError(
            "cutlass_w4a16 selected but no compatible MoE layers were built"
        )
    if hasattr(model, "_cutlass_w4a16_lane"):
        raise RuntimeError("cutlass_w4a16 workspace is already prepared")
    first = layers[0]
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_local_experts",
        "top_k",
        "ep_size",
        "ep_rank",
    )
    geometry = tuple(getattr(first, field) for field in fields)
    for layer in layers:
        if tuple(getattr(layer, field) for field in fields) != geometry:
            raise ValueError(
                "Shared cutlass_w4a16 workspace requires identical expert geometry"
            )
        if layer.w13_weight.device != first.w13_weight.device:
            raise ValueError("Shared cutlass_w4a16 workspace requires one CUDA device")
        if layer._weights_processed or hasattr(layer, "_hopper_mxfp4_lane"):
            raise RuntimeError(
                "Prepare cutlass_w4a16 workspace before weight processing"
            )
    lane = tokenspeed_kernel.create_hopper_mxfp4_lane(
        hidden=first.hidden_size,
        intermediate=first.intermediate_size,
        experts=first.num_local_experts,
        top_k=first.top_k,
        ep_size=first.ep_size,
        ep_rank=first.ep_rank,
        max_tokens=settings["max_prefill_tokens"],
        swiglu_limit=first.swiglu_arg.limit,
        device=first.w13_weight.device,
    )
    model._cutlass_w4a16_lane = lane
    for layer in layers:
        layer._hopper_mxfp4_lane = lane
    logger.info(
        "Prepared shared cutlass_w4a16 workspace for %d layers and %d tokens",
        len(layers),
        settings["max_prefill_tokens"],
    )


def cutlass_w4a16_weight_update_error(server_args) -> str | None:
    """Return why online updates are unsafe for this prepared weight layout."""
    if server_args.moe_backend == BACKEND:
        return (
            "cutlass_w4a16 uses immutable prepared weights; restart with a new "
            "checkpoint instead of updating weights online"
        )
    return None
