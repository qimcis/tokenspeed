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

"""CPU contract tests for scheduler-owned projected feature cache storage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.cache.transfer.layout import (
    layout_from_lcm_plan,
    select_layer_fields,
    select_non_layer_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.projected_features import (
    ProjectedFeatureCache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_field_consumer_id,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
    PROJECTED_FEATURE_FIELD,
    PROJECTED_FEATURE_GROUP,
    aliased_projected_feature_group,
    projected_feature_group,
    projected_feature_workspace_bytes,
    retained_feature_interval,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
    compute_cache_group_page_counts,
)


def _declared_groups(window: int, block: int):
    return (
        (
            CacheGroupSpec(
                group_id="full_attention",
                retention="full_history",
                rows_per_page=block,
                entry_stride_tokens=1,
            ),
            (CacheFieldSpec("layer.0.k", "unit.0", (block, 4), "bfloat16"),),
        ),
        projected_feature_group(
            block_granularity=block, hidden_size=4, sliding_window=window
        ),
    )


class _Pool:
    """A CPU storage fixture; cache math and transfer layout are production code."""

    def __init__(self, window: int, block: int):
        groups = _declared_groups(window, block)
        plan = pack(
            groups,
            prefix_granularity=block,
            cache_blocks_per_lcm_block={spec.group_id: 1 for spec, _ in groups},
            alignment=1,
            max_padding_fraction=1.0,
        ).bind(20)
        buffer = torch.zeros(plan.arena_bytes, dtype=torch.uint8)
        self.rows = buffer.view(torch.bfloat16).as_strided(
            (21 * block, 4),
            (4, 1),
            plan.field_page_byte_offset(PROJECTED_FEATURE_FIELD, 0) // 2,
        )
        self.arena = SimpleNamespace(
            cache_group_specs=tuple(spec for spec, _ in groups),
            plan=plan,
            buffer=buffer,
        )
        self.waits = []

    def non_layer_consumer_index(self, name: str) -> int:
        _, names, _ = select_non_layer_fields(self.arena.plan.fields)
        return 1 + names.index(name)

    def wait_for_non_layer_consumer(self, name: str) -> None:
        self.waits.append(self.non_layer_consumer_index(name))


def test_native_window_counts_only_pre_anchor_history():
    assert retained_feature_interval(0, 2047) == (0, 0)
    assert retained_feature_interval(10, 2047) == (0, 10)
    assert retained_feature_interval(2048, 2047) == (1, 2048)
    assert retained_feature_interval(1_000_000, 2047) == (997_953, 1_000_000)
    spec, fields = projected_feature_group(
        block_granularity=1, hidden_size=6144, sliding_window=2048
    )
    assert fields[0].payload_bytes == 12288
    assert spec.family == "history"
    assert spec.retention == "sliding_window"
    assert spec.sliding_window_tokens == 2048
    counts = compute_cache_group_page_counts(
        (spec,),
        max_total_tokens=0,
        max_live_requests=1,
        max_scheduled_tokens=0,
        max_context_len=4096,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    # 2047 history rows + 1 request boundary + the null block.
    assert counts[spec.group_id] == 2049


@pytest.mark.parametrize(
    "field_id",
    ("consumer..features", "consumer.dflash2.", "unknown.features", "layer.bad.k"),
)
def test_named_consumer_parser_rejects_malformed_fields(field_id):
    with pytest.raises(ValueError, match="cache field"):
        cache_field_consumer_id(field_id)


def test_l2_layout_contains_feature_field_as_non_layer_consumer():
    pool = _Pool(8, 4)
    plan = pool.arena.plan
    layer_fields, layers = select_layer_fields(plan.fields, first_layer=0, num_layers=1)
    named_fields, names, named = select_non_layer_fields(plan.fields)
    assert layers == (("layer.0.k",),)
    assert names == ("dflash2",)
    assert named == ((PROJECTED_FEATURE_FIELD,),)
    assert named_fields.isdisjoint(layer_fields)
    layout = layout_from_lcm_plan(
        plan,
        object(),
        consumers=layers + named,
        group_ids=tuple(spec.group_id for spec in pool.arena.cache_group_specs),
        field_ids=layer_fields | named_fields,
    )
    assert len(layout.consumers) == 2
    assert {group.group_id for group in layout.groups} == {
        "full_attention",
        PROJECTED_FEATURE_GROUP,
    }
    feature = next(
        group for group in layout.groups if group.group_id == PROJECTED_FEATURE_GROUP
    ).fields[0]
    assert feature.payload_bytes == 4 * 4 * 2
    assert feature.block_stride_bytes == feature.payload_bytes


def test_write_and_gather_cross_blocks_in_absolute_order():
    pool = _Pool(8, 4)
    cache = ProjectedFeatureCache(pool=pool)
    table = torch.tensor([[4, 2, 8, 3]], dtype=torch.int32)
    positions = torch.arange(2, 10)
    rows = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
    cache.write(positions, torch.zeros(8, dtype=torch.int32), table, rows, None)
    actual = cache.gather(table, 0, 3, 10, 0)
    torch.testing.assert_close(actual, rows[1:])
    assert cache.retained_interval(10) == (3, 10)
    assert pool.waits == [1, 1]


def test_snapshot_only_exports_committed_inputs_and_owns_copy():
    pool = _Pool(8, 4)
    cache = ProjectedFeatureCache(pool=pool)
    table = torch.tensor([[4, 2]], dtype=torch.int32)
    rows = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    cache.write(torch.arange(6), torch.zeros(6, dtype=torch.int32), table, rows, None)
    # Target evaluated six rows; rejection committed only the first two.
    snapshot = cache.gather(table, 0, 0, 2, 0)
    assert snapshot.shape == (2, 4)
    pool.rows.zero_()
    torch.testing.assert_close(snapshot, rows[:2])


def test_packed_requests_and_graph_padding_do_not_corrupt_other_history():
    pool = _Pool(8, 4)
    cache = ProjectedFeatureCache(pool=pool)
    table = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)
    rows = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    cache.write(
        torch.tensor([0, 1, 0, 1, 0, 1]),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        table,
        rows,
        torch.tensor([True, True, False]),
    )
    torch.testing.assert_close(cache.gather(table, 0, 0, 2, 0), rows[:2])
    torch.testing.assert_close(cache.gather(table, 1, 0, 2, 0), rows[2:4])
    assert torch.count_nonzero(pool.rows[8:12]) == 0


def test_long_context_compact_descriptor_keeps_absolute_positions():
    pool = _Pool(8, 4)
    cache = ProjectedFeatureCache(pool=pool)
    pool.rows[4:16] = torch.arange(48).reshape(12, 4).to(torch.bfloat16)
    table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    # Columns describe absolute positions [1000, 1012).
    actual = cache.gather(table, 0, 1003, 1010, 250)
    torch.testing.assert_close(actual, pool.rows[7:14])


@pytest.mark.parametrize("table", ([[0, 1]], [[-1, 1]], [[21, 1]]))
def test_missing_or_invalid_history_block_is_never_exported(table):
    cache = ProjectedFeatureCache(pool=_Pool(8, 4))
    with pytest.raises(ValueError, match="missing history blocks"):
        cache.gather(torch.tensor(table, dtype=torch.int32), 0, 0, 2, 0)


@pytest.mark.parametrize(
    "start,end,offset", ((0, 8, 0), (-1, 1, 0), (4, 3, 0), (0, 1, 1), (1000, 1001, 0))
)
def test_out_of_window_or_missing_columns_are_rejected(start, end, offset):
    cache = ProjectedFeatureCache(pool=_Pool(8, 4))
    with pytest.raises(ValueError, match="snapshot"):
        cache.gather(torch.tensor([[1, 2]], dtype=torch.int32), 0, start, end, offset)


def test_empty_prompt_history_is_valid_zero_row_snapshot():
    cache = ProjectedFeatureCache(pool=_Pool(8, 4))
    snapshot = cache.gather(torch.tensor([[1]], dtype=torch.int32), 0, 0, 0, 0)
    assert snapshot.shape == (0, 4)
    assert snapshot.dtype == torch.bfloat16


def _glm_groups():
    """Public full GLM-5.3 FP8 cache geometry: 78 x (576 latent + 132 index)."""
    fields = tuple(
        field
        for layer in range(78)
        for field in (
            CacheFieldSpec(
                f"layer.{layer}.latent_kv", f"slot.{layer}", (64, 1, 576), "uint8"
            ),
            CacheFieldSpec(
                f"layer.{layer}.index_k", f"layer.{layer}.index_k", (64, 132), "uint8"
            ),
        )
    )
    target = (
        CacheGroupSpec(
            "full_attention", "full_history", rows_per_page=64, entry_stride_tokens=1
        ),
        fields,
    )
    features = aliased_projected_feature_group(
        target_group=target,
        block_granularity=64,
        hidden_size=6144,
        sliding_window=2048,
        blocks_per_parent=4,
    )
    return target, features


def _glm_layout():
    return pack(
        _glm_groups(),
        prefix_granularity=64,
        cache_blocks_per_lcm_block={"full_attention": 1, PROJECTED_FEATURE_GROUP: 4},
        alignment=1,
        max_padding_fraction=1.0,
    )


class _AliasedPool(_Pool):
    def __init__(self):
        plan = _glm_layout().bind(12)
        self.arena = SimpleNamespace(
            cache_group_specs=tuple(spec for spec, _ in _glm_groups()),
            plan=plan,
            buffer=torch.zeros(plan.arena_bytes, dtype=torch.uint8),
        )
        self.waits = []


def test_full_glm_alias_preserves_target_parent_bytes_and_layer_strides():
    target, (_, features) = _glm_groups()
    baseline = pack(
        (target,),
        prefix_granularity=64,
        cache_blocks_per_lcm_block={"full_attention": 1},
        alignment=1,
        max_padding_fraction=1.0,
    )
    layout = _glm_layout()
    assert layout.lcm_block_bytes == baseline.lcm_block_bytes == 3_534_336
    assert sum(field.shape[1] for field in features) == 6144
    assert len(features) == 111
    assert dict(layout.group_packing)[PROJECTED_FEATURE_GROUP] == 4
    baseline_fields = {field.field_id: field for field in baseline.fields}
    for field in layout.fields:
        if field.field_id in baseline_fields:
            assert field == baseline_fields[field.field_id]
    assert sum(field.payload_bytes for field in features) == 64 * 6144 * 2
    # Flexible index-plane chunks retain planned stride padding.
    assert any(
        field.page_stride_bytes > field.payload_bytes
        for field in layout.fields
        if field.group_id == PROJECTED_FEATURE_GROUP
    )


def test_aliased_scatter_gather_respects_group_parent_ownership_and_all_columns():
    pool = _AliasedPool()
    cache = ProjectedFeatureCache(pool=pool)
    plan, buffer = pool.arena.plan, pool.arena.buffer
    target_ranges = plan.block_byte_segments("full_attention", [1])
    for offset, size in target_ranges:
        buffer[offset : offset + size] = 173
    # Feature block IDs 5..8 belong to parent2, target block1 to parent1.
    table = torch.tensor([[5, 8]], dtype=torch.int32)
    rows = (torch.arange(128 * 6144).reshape(128, 6144) % 997).to(torch.bfloat16)
    cache.write(
        torch.arange(128), torch.zeros(128, dtype=torch.int32), table, rows, None
    )
    torch.testing.assert_close(cache.gather(table, 0, 0, 128, 0), rows)
    for offset, size in target_ranges:
        assert torch.all(buffer[offset : offset + size] == 173)
    column = 0
    for field in sorted(
        (field for field in plan.fields if field.group_id == PROJECTED_FEATURE_GROUP),
        key=lambda field: field.field_id,
    ):
        width = field.shape[1]
        for position in (0, 63, 64, 127):
            block = int(table[0, position // 64])
            offset = (
                plan.field_page_byte_offset(field.field_id, block)
                + (position % 64) * width * 2
            )
            physical = buffer[offset : offset + width * 2].view(torch.bfloat16)
            torch.testing.assert_close(
                physical, rows[position, column : column + width]
            )
        column += width
    assert column == 6144


def test_aliased_l2_transfer_layout_roundtrip_restores_every_feature_chunk():
    pool = _AliasedPool()
    cache = ProjectedFeatureCache(pool=pool)
    plan, buffer = pool.arena.plan, pool.arena.buffer
    table = torch.tensor([[5, 6]], dtype=torch.int32)
    rows = (torch.arange(100 * 6144).reshape(100, 6144) % 443).to(torch.bfloat16)
    cache.write(
        torch.arange(100), torch.zeros(100, dtype=torch.int32), table, rows, None
    )
    layer_fields, layers = select_layer_fields(
        plan.fields, first_layer=0, num_layers=78
    )
    extra_fields, names, extra = select_non_layer_fields(plan.fields)
    layout = layout_from_lcm_plan(
        plan,
        buffer,
        consumers=layers + extra,
        group_ids=("full_attention", PROJECTED_FEATURE_GROUP),
        field_ids=layer_fields | extra_fields,
    )
    assert names == ("dflash2",)
    assert len(layout.consumers[-1]) == 111
    group = next(
        group for group in layout.groups if group.group_id == PROJECTED_FEATURE_GROUP
    )
    assert len(group.fields) == 111
    # Exercise the exact byte ranges the existing L2 executor transfers.
    stored = []
    for block in (5, 6):
        for field in group.fields:
            start = (
                field.device_block_zero_offset_bytes + block * field.block_stride_bytes
            )
            stored.append((start, buffer[start : start + field.payload_bytes].clone()))
    for start, payload in stored:
        buffer[start : start + payload.numel()] = 0
    assert torch.count_nonzero(cache.gather(table, 0, 0, 100, 0)) == 0
    for start, payload in stored:
        buffer[start : start + payload.numel()].copy_(payload)
    torch.testing.assert_close(cache.gather(table, 0, 0, 100, 0), rows)


def test_feature_workspace_charges_snapshot_buffers_and_address_matrix():
    address_rows = 8192 + 128 * 6 + 2048
    expected = (
        (address_rows * 6144 * 8)
        + (2 * 2047 * 6144 * 2)
        + (3 * 6144 * 8)
        + (address_rows * 8 * 4)
    )
    assert (
        projected_feature_workspace_bytes(
            hidden_size=6144,
            window_left=2047,
            max_forward_tokens=8192,
            max_decode_graph_tokens=128 * 6,
            max_prefill_graph_tokens=2048,
        )
        == expected
    )


def test_aliased_window_recovers_at_large_absolute_endpoint_after_page_rotation():
    pool = _AliasedPool()
    cache = ProjectedFeatureCache(pool=pool)
    start, end = cache.retained_interval(1_002_047)
    assert (start, end) == (1_000_000, 1_002_047)
    first_column = start // 64
    last_column = (end - 1) // 64
    blocks = torch.arange(5, 5 + last_column - first_column + 1, dtype=torch.int32)
    table = torch.zeros((1, last_column + 1), dtype=torch.int32)
    table[0, first_column:] = blocks
    positions = torch.arange(start, end)
    # Distinct position and column values expose truncation and chunk swaps.
    rows = ((positions[:, None] % 97) + (torch.arange(6144)[None, :] % 53)).to(
        torch.bfloat16
    )
    cache.write(
        positions, torch.zeros(end - start, dtype=torch.int32), table, rows, None
    )
    snapshot = cache.gather(blocks.unsqueeze(0), 0, start, end, first_column)
    torch.testing.assert_close(snapshot, rows)
    assert snapshot.shape == (2047, 6144)
    assert snapshot.data_ptr() != pool.arena.buffer.data_ptr()
