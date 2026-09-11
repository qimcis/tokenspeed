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

"""A non-layer view of scheduler-owned projected target feature history."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
    PROJECTED_FEATURE_CONSUMER,
    PROJECTED_FEATURE_FIELD,
    PROJECTED_FEATURE_GROUP,
    retained_feature_interval,
)
from tokenspeed.runtime.layers.attention.page_table import (
    group_slot_mapping_from_raw,
    mask_invalid_graph_tokens,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


class ProjectedFeatureCache:
    """Read and write the arena feature field using scheduler block tables.

    This view allocates no persistent request state. The scheduler owns
    allocation, prefix matching, retained rows and snapshot block pins. An
    export caller must retain its scheduler pin until the gather's execution
    completes; a returned tensor owns a copy and never aliases source pages.
    """

    group_id = PROJECTED_FEATURE_GROUP

    def __init__(self, *, pool: CachePool) -> None:
        self.pool = pool
        specs = {spec.group_id: spec for spec in pool.arena.cache_group_specs}
        try:
            spec = specs[self.group_id]
        except KeyError:
            raise ValueError("projected feature cache group is not planned") from None
        fields = tuple(
            sorted(
                (
                    field
                    for field in pool.arena.plan.fields
                    if field.field_id == PROJECTED_FEATURE_FIELD
                    or field.field_id.startswith(PROJECTED_FEATURE_FIELD + ".")
                ),
                key=lambda field: field.field_id,
            )
        )
        if not fields:
            raise ValueError("projected feature fields are not planned")
        if len(fields) > 1 or fields[0].field_id != PROJECTED_FEATURE_FIELD:
            expected = tuple(
                f"{PROJECTED_FEATURE_FIELD}.{index:04d}" for index in range(len(fields))
            )
            if tuple(field.field_id for field in fields) != expected:
                raise ValueError("projected feature chunks must cover ordered columns")
        if (
            spec.family != "history"
            or spec.retention != "sliding_window"
            or spec.entry_stride_tokens != 1
            or spec.sliding_window_tokens is None
            or spec.sliding_window_tokens <= 1
            or any(
                field.group_id != self.group_id
                or field.dtype != "bfloat16"
                or len(field.shape) != 2
                or field.shape[0] != spec.rows_per_page
                or field.page_stride_bytes % 2 != 0
                for field in fields
            )
        ):
            raise ValueError("projected feature cache has incompatible geometry")
        self.block_granularity = spec.block_granularity
        self.hidden_size = sum(field.shape[1] for field in fields)
        self.window_left = spec.sliding_window_tokens - 1
        self._block_count = pool.arena.plan.group(self.group_id).page_count
        self._words = pool.arena.buffer.view(torch.bfloat16)
        bases, block_strides, row_strides = [], [], []
        for field in fields:
            width = field.shape[1]
            base_bytes = pool.arena.plan.field_page_byte_offset(field.field_id, 0)
            if base_bytes % 2:
                raise ValueError("projected feature field is not BF16-aligned")
            bases.extend(base_bytes // 2 + column for column in range(width))
            block_strides.extend([field.page_stride_bytes // 2] * width)
            row_strides.extend([width] * width)
        self._column_bases = torch.tensor(
            bases, dtype=torch.int64, device=self._words.device
        )
        self._block_strides = torch.tensor(
            block_strides, dtype=torch.int64, device=self._words.device
        )
        self._row_strides = torch.tensor(
            row_strides, dtype=torch.int64, device=self._words.device
        )
        # Validate that L2 selection knows this consumer even when L2 is off.
        pool.non_layer_consumer_index(PROJECTED_FEATURE_CONSUMER)

    def _addresses(self, slots: torch.Tensor) -> torch.Tensor:
        """Resolve every feature column in one bounded address matrix.

        Columns can occupy different physical planes. Explicit per-column
        strides keep scatter/gather independent of the number of field chunks;
        no per-chunk accelerator launch is required.
        """
        blocks = slots // self.block_granularity
        rows = slots % self.block_granularity
        addresses = blocks[:, None] * self._block_strides[None, :]
        addresses.addcmul_(rows[:, None], self._row_strides[None, :])
        addresses.add_(self._column_bases[None, :])
        return addresses

    def wait_ready(self) -> None:
        """Order the first field access after every applicable L2 restore."""
        self.pool.wait_for_non_layer_consumer(PROJECTED_FEATURE_CONSUMER)

    def retained_interval(self, endpoint: int) -> tuple[int, int]:
        """The recoverable committed history preceding an anchor position."""
        return retained_feature_interval(endpoint, self.window_left)

    @torch.no_grad()
    def write(
        self,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        projected_features: torch.Tensor,
        is_valid_token: torch.Tensor | None,
    ) -> None:
        """Scatter evaluated rows, using committed endpoints to gate later reads.

        All metadata tensors use the feature arena's device. Positions are
        absolute; ``block_table`` uses absolute scheduler columns. Padding
        writes resolve to the reserved null block, never another request.
        Speculative rows may be written before acceptance, but callers may
        only export rows before the scheduler's committed anchor endpoint.
        """
        if projected_features.shape != (positions.numel(), self.hidden_size):
            raise ValueError("projected feature rows do not match positions/width")
        if projected_features.dtype != torch.bfloat16:
            raise ValueError("projected features must use bfloat16")
        if any(
            tensor.device != self._words.device
            for tensor in (positions, request_indices, block_table, projected_features)
        ):
            raise ValueError(
                "projected feature write tensors must share the arena device"
            )
        self.wait_ready()
        if positions.numel() == 0:
            return
        slots = group_slot_mapping_from_raw(
            positions,
            request_indices,
            block_table,
            self.block_granularity,
            1,
        )
        slots = mask_invalid_graph_tokens(slots, is_valid_token)
        valid = (
            (positions >= 0)
            & (slots >= self.block_granularity)
            & (slots < self._block_count * self.block_granularity)
        )
        slots = torch.where(valid, slots, torch.zeros_like(slots))
        addresses = self._addresses(slots)
        self._words.index_put_(
            (addresses.reshape(-1),), projected_features.reshape(-1), accumulate=False
        )

    @torch.no_grad()
    def gather(
        self,
        block_table: torch.Tensor,
        request_index: int,
        start: int,
        end: int,
        logical_column_offset: int,
    ) -> torch.Tensor:
        """Copy the pinned committed interval to a new device tensor.

        CPU scheduler descriptor tables avoid a device-to-host metadata
        roundtrip. A GPU table is also accepted and validated asynchronously.
        ``logical_column_offset`` is the absolute column at table column zero;
        ordinary scheduler exports use zero. An absent history block is an
        error, never a zero-feature substitute for a prefix cache miss.
        """
        if start < 0 or end < start or end - start > self.window_left:
            raise ValueError(
                "projected feature snapshot is outside the retained window"
            )
        if block_table.ndim != 2 or not 0 <= request_index < block_table.shape[0]:
            raise ValueError("projected feature snapshot has invalid request table")
        if logical_column_offset < 0:
            raise ValueError("logical_column_offset must be non-negative")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("projected feature block table must use integer block ids")
        self.wait_ready()
        if start == end:
            return self._words.new_empty((0, self.hidden_size))
        first_column = start // self.block_granularity - logical_column_offset
        last_column = (end - 1) // self.block_granularity - logical_column_offset
        if first_column < 0 or last_column >= block_table.shape[1]:
            raise ValueError(
                "projected feature snapshot history is absent from block table"
            )
        positions = torch.arange(
            start, end, device=block_table.device, dtype=torch.int64
        )
        columns = positions // self.block_granularity - logical_column_offset
        blocks = block_table[request_index, columns].to(torch.int64)
        present = torch.all((blocks > 0) & (blocks < self._block_count))
        if block_table.device.type == "cpu":
            if not bool(present):
                raise ValueError(
                    "projected feature snapshot contains missing history blocks"
                )
        else:
            torch._assert_async(
                present, "projected feature snapshot contains missing history blocks"
            )
        slots = blocks * self.block_granularity + positions % self.block_granularity
        slots = slots.to(device=self._words.device, non_blocking=True)
        addresses = self._addresses(slots)
        return self._words.index_select(0, addresses.reshape(-1)).view(
            end - start, self.hidden_size
        )
