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
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPDispatcher, DeepEPMode


def get_bf16_dispatcher(
    plan: dict, w: torch.nn.Module, x: torch.Tensor
) -> DeepEPDispatcher:
    """Return the plan's BF16 dispatcher with fixed transport capacities."""
    dispatcher = plan.get("_deepep_dispatcher")
    if dispatcher is not None:
        return dispatcher
    group = plan["deepep_group"]
    if group is None:
        raise ValueError("DeepEP MoE requires an expert process group")
    mode = DeepEPMode(plan["deepep_mode"])
    capacity = plan["deepep_low_latency_max_num_tokens_per_gpu"]
    if mode.enable_low_latency() and not capacity:
        raise ValueError("DeepEP low latency requires a positive token capacity")
    config = SimpleNamespace(
        top_k=w.top_k,
        num_experts=w.num_experts,
        low_latency_max_num_tokens_per_gpu=capacity,
        hidden_size=x.shape[1],
        world_size=group.size(),
        group=group,
        params_dtype=torch.bfloat16,
    )
    dispatcher = DeepEPDispatcher(
        config,
        deepep_mode=mode,
        async_finish=False,
        return_recv_hook=True,
        use_fp8=False,
    )
    plan["_deepep_dispatcher"] = dispatcher
    return dispatcher


def apply_bf16_deepep(
    dispatcher: DeepEPDispatcher,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    low_latency: bool | None,
    expert_start: int,
    expert_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Apply flat experts between classic dispatch and combine.

    The callable receives BF16 rows, FP32 weights and global expert IDs. It
    returns weighted BF16 contributions; LL uses unit weights here and applies
    the original route weights during combine.
    """
    use_low_latency = (
        dispatcher.deepep_mode.resolve(low_latency) == DeepEPMode.low_latency
    )
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(torch.float32)
    dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=use_low_latency)
    if overlap_fn is not None:
        overlap_fn()
    recv_x, recv_ids, recv_weights, _, _, _, masked_m = dispatcher.dispatch_b()
    if use_low_latency:
        num_experts, capacity, hidden = recv_x.shape
        experts = torch.arange(
            expert_start, expert_start + num_experts, device=x.device, dtype=torch.int32
        )
        valid = torch.arange(capacity, device=x.device)[None, :] < masked_m[:, None]
        recv_ids = torch.where(valid, experts[:, None], -1).reshape(-1, 1)
        recv_weights = valid.to(torch.float32).reshape(-1, 1)
        flat_x = recv_x.reshape(-1, hidden)
    else:
        flat_x = recv_x
    if flat_x.shape[0]:
        valid_rows = (recv_ids >= 0).any(dim=-1)
        flat_x = torch.where(valid_rows[:, None], flat_x, 0)
        recv_weights = torch.where(recv_ids >= 0, recv_weights, 0)
        output = expert_fn(flat_x, recv_weights, recv_ids)
        output = torch.where(valid_rows[:, None], output, 0)
    else:
        output = torch.empty_like(flat_x)
    if use_low_latency:
        output = output.view_as(recv_x)
        combine_ids, combine_weights = topk_ids, topk_weights
    else:
        combine_ids, combine_weights = recv_ids, recv_weights
    dispatcher.combine_a(
        output,
        combine_ids,
        combine_weights,
        low_latency=use_low_latency,
        moe_origin_input=x if use_low_latency else None,
    )
    return dispatcher.combine_b()
