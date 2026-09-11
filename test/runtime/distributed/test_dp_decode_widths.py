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

"""CPU metadata checks for heterogeneous verification widths and EP sizes."""

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.dp_forward_metadata import (
    common_decode_width,
    planned_decode_width,
)
from tokenspeed.runtime.distributed.mapping import Mapping


def _op(lengths: list[int], num_extends: int, selected_width: int):
    return SimpleNamespace(
        input_lengths=lengths,
        num_extends=lambda: num_extends,
        decode_input_tokens=selected_width,
    )


@pytest.mark.parametrize(
    ("widths", "expected"),
    [
        ([6] * 8, 6),
        ([1] * 8, 1),
        ([1] + [6] * 7, None),
        ([0, 0, 6, 0, 0, 0, 0, 0], 6),
        ([0] * 8, None),
        ([1] * 4 + [6] * 4, None),
        ([0] * 4 + [6] * 4, 6),
    ],
)
def test_common_width_is_only_a_graph_decision(widths, expected):
    original = list(widths)
    assert common_decode_width(widths) == expected
    assert widths == original


@pytest.mark.parametrize("width", [1, 6])
def test_planned_width_uses_reserved_decode_rows(width):
    op = _op([17, width, width], 1, width)
    assert planned_decode_width(op) == width
    assert op.input_lengths == [17, width, width]
    assert op.decode_input_tokens == width


def test_legacy_local_batch_infers_its_already_planned_width():
    op = SimpleNamespace(input_lengths=[6, 6], num_extends=lambda: 0)
    assert planned_decode_width(op) == 6


def test_prefill_and_idle_do_not_vote_for_a_decode_graph_width():
    assert planned_decode_width(None) == 0
    assert planned_decode_width(_op([17, 9], 2, 6)) == 0


@pytest.mark.parametrize(
    "op",
    [_op([1, 6], 0, 6), _op([1, 1], 0, 6), _op([0], 0, 0)],
)
def test_invalid_or_ragged_reservations_fail_instead_of_rewriting_width(op):
    with pytest.raises(ValueError):
        planned_decode_width(op)


def test_negative_gathered_width_is_invalid():
    with pytest.raises(ValueError):
        common_decode_width([6, -1])


def _manager(rank: int, attn_tp: int) -> CommManager:
    mapping = Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=attn_tp,
        attn_cp_size=1,
        dense_tp_size=attn_tp,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    return CommManager(
        mapping=mapping,
        layer_id=1,
        is_moe=True,
        prev_is_moe=True,
        input_layernorm=None,
        post_attn_layernorm=None,
    )


def _context(real_counts: list[int], padded_counts: list[int] | None):
    return SimpleNamespace(
        global_num_tokens=real_counts,
        collective_global_num_tokens=padded_counts,
        collective_num_tokens=None,
        input_num_tokens=real_counts[0],
    )


@pytest.mark.parametrize("rank", range(8))
def test_dpa8_ep8_mixed_width_eager_collectives_use_tokens_not_requests(rank):
    # Two requests per non-idle cohort, with different planned query widths.
    actual_tokens = [2, 12, 0, 6, 1, 0, 18, 6]
    manager = _manager(rank, 1)
    ctx = _context(actual_tokens, None)

    assert manager.scattered_num_tokens(ctx) == actual_tokens
    assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == actual_tokens
    assert manager.get_num_tokens(ctx) == (45, 18)


@pytest.mark.parametrize("rank", range(8))
def test_tp4_dpa2_ep8_does_not_count_tp_replicas_as_more_requests(rank):
    # Cohort A: three width-one rows. Cohort B: two width-six rows.
    # The physical-rank metadata repeats each cohort's full token count.
    actual_tokens = [3] * 4 + [12] * 4
    ctx = _context(actual_tokens, None)
    manager = _manager(rank, 4)
    scattered = [1, 1, 1, 0, 3, 3, 3, 3]

    assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == scattered
    assert manager.get_num_tokens(ctx) == (15, 3)


@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("rank", range(8))
def test_homogeneous_graph_dummy_rows_use_shared_padded_token_geometry(rank, width):
    actual_tokens = [width, 0, width * 2, 0, 0, 0, 0, 0]
    padded_tokens = [4 * width] * 8
    ctx = _context(actual_tokens, padded_tokens)
    manager = _manager(rank, 1)

    assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == padded_tokens
    assert manager.get_num_tokens(ctx) == (32 * width, 4 * width)
    assert ctx.global_num_tokens == actual_tokens


@pytest.mark.parametrize("rank", range(8))
def test_tp4_graph_padding_is_scattered_after_cohort_replication(rank):
    actual_tokens = [0] * 4 + [6] * 4
    ctx = _context(actual_tokens, [24] * 8)
    manager = _manager(rank, 4)

    assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == [6] * 8
    assert manager.get_num_tokens(ctx) == (48, 6)
