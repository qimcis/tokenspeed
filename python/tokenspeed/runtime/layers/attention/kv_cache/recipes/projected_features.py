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

"""Projected target features declared as ordinary scheduler-owned history."""

from __future__ import annotations

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheFieldSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
    CacheGroupSpec,
)

PROJECTED_FEATURE_GROUP = "dflash2_projected_features"
PROJECTED_FEATURE_CONSUMER = "dflash2"
PROJECTED_FEATURE_FIELD = "consumer.dflash2.projected_features"
PROJECTED_FEATURE_BLOCKS_PER_PARENT = 4
PROJECTED_FEATURE_EXPORT_SLOTS = 2


def projected_feature_group(
    *, block_granularity: int, hidden_size: int, sliding_window: int
) -> CacheGroupDeclaration:
    """Declare one BF16 feature row per target input token.

    ``sliding_window`` includes the current query, following attention and LCM
    conventions. Thus a window of 2048 retains 2047 rows before the anchor.
    Geometry is independent of attention-layer count and target KV precision.
    """
    if hidden_size <= 0:
        raise ValueError("projected feature hidden_size must be positive")
    if sliding_window <= 1:
        raise ValueError("projected feature sliding_window must exceed one")
    return (
        CacheGroupSpec(
            group_id=PROJECTED_FEATURE_GROUP,
            retention="sliding_window",
            rows_per_page=block_granularity,
            entry_stride_tokens=1,
            sliding_window_tokens=sliding_window,
            family="history",
            transfer_policy=None,
            checkpoint_granularity=None,
        ),
        (
            CacheFieldSpec(
                field_id=PROJECTED_FEATURE_FIELD,
                plane_id=PROJECTED_FEATURE_FIELD,
                shape=(block_granularity, hidden_size),
                dtype="bfloat16",
            ),
        ),
    )


def retained_feature_interval(endpoint: int, window_left: int) -> tuple[int, int]:
    """Return committed input positions before the anchor at ``endpoint``."""
    if endpoint < 0:
        raise ValueError("projected feature endpoint must be non-negative")
    if window_left <= 0:
        raise ValueError("projected feature window_left must be positive")
    return max(0, endpoint - window_left), endpoint


def aliased_projected_feature_group(
    *,
    target_group: CacheGroupDeclaration,
    block_granularity: int,
    hidden_size: int,
    sliding_window: int,
    blocks_per_parent: int,
) -> CacheGroupDeclaration:
    """Pack feature chunks into the target group's existing physical planes.

    The LCM allocator assigns a parent to exactly one cache group at a time.
    Target and feature blocks can therefore interpret the same planes without
    overlapping live storage. Four feature blocks use the bytes otherwise
    occupied by one target block, retaining the target's original parent size.
    """
    if blocks_per_parent <= 0:
        raise ValueError("feature blocks_per_parent must be positive")
    target_spec, target_fields = target_group
    if target_spec.retention != "full_history" or target_spec.family != "history":
        raise ValueError("projected feature aliases require full target history")
    if target_spec.block_granularity != block_granularity:
        raise ValueError("projected feature and target block spans must match")
    planes = [field.plane_id for field in target_fields]
    if len(set(planes)) != len(planes):
        raise ValueError("projected feature aliases require distinct target planes")
    spec, _ = projected_feature_group(
        block_granularity=block_granularity,
        hidden_size=hidden_size,
        sliding_window=sliding_window,
    )
    remaining = hidden_size
    fields = []
    for target_field in sorted(
        target_fields, key=lambda field: (-field.payload_bytes, field.plane_id)
    ):
        width = min(
            remaining,
            target_field.payload_bytes // (block_granularity * 2 * blocks_per_parent),
        )
        if width == 0:
            continue
        fields.append(
            CacheFieldSpec(
                field_id=f"{PROJECTED_FEATURE_FIELD}.{len(fields):04d}",
                plane_id=target_field.plane_id,
                shape=(block_granularity, width),
                dtype="bfloat16",
                exact_page_stride=False,
            )
        )
        remaining -= width
        if remaining == 0:
            return spec, tuple(fields)
    raise ValueError(
        "target cache planes cannot hold the projected feature group at "
        f"packing {blocks_per_parent}: {remaining} feature columns remain"
    )


def projected_feature_workspace_bytes(
    *,
    hidden_size: int,
    window_left: int,
    max_forward_tokens: int,
    max_decode_graph_tokens: int,
    max_prefill_graph_tokens: int,
) -> int:
    """Bound address scratch, device snapshots and immutable address vectors."""
    if min(hidden_size, window_left, max_forward_tokens) <= 0:
        raise ValueError("projected feature workspace dimensions must be positive")
    if min(max_decode_graph_tokens, max_prefill_graph_tokens) < 0:
        raise ValueError("projected feature graph dimensions must be non-negative")
    address_rows = max(window_left, max_forward_tokens)
    # CUDA graph private pools persist while ordinary eager/export work runs.
    # Decode and prefill use distinct shared pools, so charge each once.
    all_address_rows = address_rows + max_decode_graph_tokens + max_prefill_graph_tokens
    address_bytes = all_address_rows * hidden_size * 8
    snapshot_bytes = PROJECTED_FEATURE_EXPORT_SLOTS * window_left * hidden_size * 2
    vector_bytes = 3 * hidden_size * 8
    # Per-row block/offset inputs and masked slot vectors accompany addresses.
    row_metadata_bytes = all_address_rows * 8 * 4
    return address_bytes + snapshot_bytes + vector_bytes + row_metadata_bytes
