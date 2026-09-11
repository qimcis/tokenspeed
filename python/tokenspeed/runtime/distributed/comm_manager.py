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

from dataclasses import dataclass
from typing import Literal

import torch

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext


@dataclass(frozen=True)
class MoERowLayout:
    """Live source and return rows for one MoE invocation."""

    source_offset: int
    live_rows: int
    output_rows: int
    gather_counts: tuple[int, ...]
    num_global_tokens: int
    max_num_tokens_per_gpu: int


@dataclass(frozen=True)
class MoEInputLayout:
    """Capture-stable placement; live counts come from the current context."""

    comm_manager: CommManager
    placement: Literal["physical", "replicated"]
    physical_rows: int

    def resolve(self, ctx: ForwardContext) -> MoERowLayout:
        return self.comm_manager.moe_row_layout(ctx, self.placement, self.physical_rows)


class CommManager:
    """Manages communication patterns (all_reduce vs RSAG) for each decoder layer."""

    def __init__(
        self,
        mapping: Mapping,
        layer_id: int,
        is_moe: bool,
        prev_is_moe: bool,
        input_layernorm: torch.nn.Module | None = None,
        post_attn_layernorm: torch.nn.Module | None = None,
    ) -> None:
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.prev_is_moe = prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm

    # ---- Scattered token counts ----

    @staticmethod
    def _scatter_count(num_tokens: int, tp_size: int) -> list[int]:
        base, remainder = divmod(num_tokens, tp_size)
        return [base + 1] * remainder + [base] * (tp_size - remainder)

    def get_num_tokens(self, ctx: ForwardContext):
        scattered = self.scattered_num_tokens(ctx)
        return sum(scattered), max(scattered)

    def scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        if global_counts is not None:
            scattered = []
            for attn_dp_rank in range(self.mapping.attn.dp_size):
                # global_counts is indexed by global rank with dp stride
                # tp_size * cp_size; cp peers report the same count.
                num_tokens = global_counts[
                    attn_dp_rank * self.mapping.attn.tp_size * self.mapping.attn.cp_size
                ]
                scattered.extend(
                    self._scatter_count(num_tokens, self.mapping.attn.tp_size)
                )
            return scattered
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        return self._scatter_count(num_tokens, self.mapping.attn.tp_size)

    def attn_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.attn.tp_size * self.mapping.attn.dp_rank
        end = start + self.mapping.attn.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def dense_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.dense.tp_size * self.mapping.dense.dp_rank
        end = start + self.mapping.dense.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def moe_tp_ep_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        tp_ep_size = self.mapping.moe.tp_ep_size
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        # Without DP, all ranks share the batch and the scattered table needs
        # no global metadata, so the lookup below stays valid.
        if global_counts is not None or not self.mapping.attn.has_dp:
            # After post_attn_comm reduce-scatter, each rank holds its
            # scattered share of its attn dp group's tokens, not the raw
            # global count; MoE collectives must size from those rows.
            scattered = self.scattered_num_tokens(ctx)
            return [
                scattered[self.mapping.attn.scatter_index(rank)]
                for rank in self.mapping.moe.tp_ep_group
            ]
        # With DP but no gathered metadata, other dp groups' counts are
        # unknown; only the local rank's contribution can be reported.
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        result = [0] * tp_ep_size
        result[self.mapping.moe.tp_ep_rank] = num_tokens
        return result

    def moe_row_layout(
        self,
        ctx: ForwardContext,
        placement: Literal["physical", "replicated"],
        physical_rows: int,
    ) -> MoERowLayout:
        """Resolve live MoE rows without changing their physical input placement.

        ``physical`` preserves post-attention shards; ``replicated`` balances
        live source rows across attention TP and returns the gathered rows.
        ``physical_rows`` is the input capacity, including graph padding.
        """
        if placement not in ("physical", "replicated"):
            raise ValueError(f"Unknown MoE input placement: {placement}")
        attn = self.mapping.attn
        physical_global = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        physical_local = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        live = ctx.moe_token_counts
        live_global = live.global_num_tokens if live is not None else physical_global
        live_local = live.num_tokens if live is not None else physical_local
        if attn.has_dp and physical_global is None and live is not None:
            raise ValueError(
                "Padded MoE with attention DP requires global token counts"
            )

        all_sources = []
        local_counts = None
        local_capacity = 0
        local_total = 0
        scattered_input = (
            placement == "physical"
            and attn.has_tp
            and not self.use_all_reduce(is_moe=True)
        )
        for dp_rank in range(attn.dp_size):
            rank = dp_rank * attn.tp_size * attn.cp_size
            capacity = (
                physical_global[rank]
                if physical_global is not None
                else physical_local if dp_rank == attn.dp_rank else 0
            )
            count = (
                live_global[rank]
                if live_global is not None
                else live_local if dp_rank == attn.dp_rank else 0
            )
            if not 0 <= count <= capacity:
                raise ValueError(f"MoE live rows {count} exceed capacity {capacity}")
            capacities = self._scatter_count(capacity, attn.tp_size)
            if placement == "replicated":
                sources = self._scatter_count(count, attn.tp_size)
            elif scattered_input:
                offset = 0
                sources = []
                for shard_capacity in capacities:
                    sources.append(min(max(count - offset, 0), shard_capacity))
                    offset += shard_capacity
            else:
                sources = [count] * attn.tp_size
            all_sources.extend(sources)
            if dp_rank == attn.dp_rank:
                local_counts = sources
                local_capacity = (
                    capacities[attn.tp_rank] if scattered_input else capacity
                )
                local_total = count

        if physical_rows != local_capacity:
            raise ValueError(
                f"MoE input has {physical_rows} rows, expected {local_capacity}"
            )
        assert local_counts is not None
        ep_sources = [
            all_sources[attn.scatter_index(rank)] for rank in self.mapping.moe.ep_group
        ]
        replicated = placement == "replicated"
        return MoERowLayout(
            source_offset=sum(local_counts[: attn.tp_rank]) if replicated else 0,
            live_rows=local_counts[attn.tp_rank],
            output_rows=local_total if replicated else local_counts[attn.tp_rank],
            gather_counts=tuple(local_counts),
            num_global_tokens=sum(ep_sources),
            max_num_tokens_per_gpu=max(ep_sources),
        )

    # ---- Communication patterns ----

    def use_all_reduce(self, is_moe: bool):
        if is_moe:
            return self.mapping.attn.tp_size == self.mapping.moe.tp_ep_size
        return self.mapping.attn.tp_size == self.mapping.dense.tp_size

    def pre_attn_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.layer_id == 0:
            return hidden_states

        if not self.mapping.has_attn_tp:
            return hidden_states

        if self.use_all_reduce(self.prev_is_moe):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    def gather_residual(self, residual: torch.Tensor, ctx: ForwardContext):
        """All-gather a residual left scattered by the previous layer's RSAG
        path (e.g. for aux hidden capture); no-op when rows are already full.

        Mirrors the pre_attn_comm gather conditions.
        """
        if self.layer_id == 0:
            return residual
        if not self.mapping.has_attn_tp:
            return residual
        if self.use_all_reduce(self.prev_is_moe):
            return residual
        return token_all_gather(
            residual,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    def post_attn_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual

        if self.use_all_reduce(self.is_moe):
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            # The output residual is expected to have attn_tp_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens.
            # Otherwise, if this layer experiences a RSAG -> AR switch, residual needs allgather.
            if self.layer_id > 0 and not self.use_all_reduce(self.prev_is_moe):
                residual = token_all_gather(
                    residual,
                    group=self.mapping.attn.tp_group,
                    scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
                )
        else:
            token_list = self.attn_tp_group_scattered_num_tokens(ctx)
            hidden_states = token_reduce_scatter(
                hidden_states,
                group=self.mapping.attn.tp_group,
                scattered_num_tokens=token_list,
            )
            # The output residual is expected to have scattered_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens, so needs slice.
            # Otherwise, if this layer experiences a AR -> RSAG switch, residual needs slice.
            if self.layer_id == 0 or self.use_all_reduce(self.prev_is_moe):
                offset = sum(token_list[: self.mapping.attn.tp_rank])
                residual = residual[offset : offset + hidden_states.size(0)]

        return hidden_states, residual

    def pre_mlp_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.is_moe:
            return self.pre_moe_comm(hidden_states, ctx)
        else:
            return self.pre_dense_comm(hidden_states, ctx)

    def pre_dense_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.dense.has_tp:
            return hidden_states

        if self.use_all_reduce(is_moe=False):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )

    def pre_moe_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states

        if self.use_all_reduce(is_moe=True):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )

    def post_mlp_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.is_moe:
            return self.post_moe_comm(hidden_states, residual, ctx)
        else:
            return self.post_dense_comm(hidden_states, residual, ctx)

    def post_dense_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.dense.has_tp:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=False):
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_moe_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=True):
            hidden_states = all_reduce(hidden_states, self.mapping.moe.tp_ep_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_final_norm_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual
        if self.use_all_reduce(self.is_moe):
            return hidden_states, residual
        hidden_states = token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    # ---- Fused allreduce+norm ----

    def use_all_reduce_norm_fusion(self) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce(self.is_moe)
            and self.mapping.has_attn_tp
            and global_server_args_dict.get("enable_allreduce_fusion", False)
            # DeepEP returns complete outputs, with no reduction to defer.
            and global_server_args_dict["all2all_backend"] != "deepep"
        )

    def should_fuse(self, num_tokens: int) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce_norm_fusion()
            and num_tokens > 0
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        )

    def input_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_attn_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.post_attn_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.post_attn_comm(hidden_states, residual, ctx)
            hidden_states, residual = self.post_attn_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_mlp_fused(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual = self.post_mlp_comm(hidden_states, residual, ctx)
        return hidden_states, residual

    def final_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        norm: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if ctx.forward_mode.is_idle():
            return hidden_states, None

        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual_out, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual_out = norm(hidden_states, residual)
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)

        return hidden_states, residual_out
