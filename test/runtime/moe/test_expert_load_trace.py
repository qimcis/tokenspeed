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

"""Actual per-forward deltas, bounded storage and restart boundaries."""

import os

import pytest
import torch

from tokenspeed.runtime.moe.expert_load_rows import ExpertLoadTrace, LayerExpertLoad


@pytest.mark.parametrize("max_steps", [0, -1, 4097])
def test_trace_rejects_unbounded_window(max_steps):
    with pytest.raises(ValueError, match="between"):
        ExpertLoadTrace(torch.zeros((1, 2), dtype=torch.int64), max_steps=max_steps)


def test_trace_keeps_recent_deltas_in_order_and_snapshot_is_independent():
    counts = torch.zeros((2, 3), dtype=torch.int64)
    trace = ExpertLoadTrace(counts, max_steps=2)
    for delta in (1, 3, 7):
        counts.add_(delta)
        trace.record()
    saved = trace.snapshot()
    assert saved.tolist() == [[[3] * 3] * 2, [[7] * 3] * 2]
    counts.add_(9)
    trace.record()
    assert saved.tolist() == [[[3] * 3] * 2, [[7] * 3] * 2]
    assert trace.snapshot().tolist() == [[[7] * 3] * 2, [[9] * 3] * 2]


def test_reset_rebases_counters_and_drops_previous_placement_history():
    counts = torch.tensor([[20, 30]], dtype=torch.int64)
    trace = ExpertLoadTrace(counts, max_steps=3)
    counts.add_(2)
    trace.record()
    counts.zero_()
    trace.reset()
    assert trace.snapshot().shape == (0, 1, 2)
    counts.add_(5)
    trace.record()
    assert trace.snapshot().tolist() == [[[5, 5]]]


def test_live_row_mask_excludes_padding_and_zero_routes_before_trace():
    # Construct the real row-mask owner without distributed initialization:
    # rows()/reserve() do not need a mapping; only mark_padded does.
    from tokenspeed.runtime.moe.expert_load_rows import ExpertLoadRowMask

    rows = ExpertLoadRowMask(None)
    rows.reserve(4, "cpu")
    rows.mask.copy_(torch.tensor([True, True, False, False]))
    counts = torch.zeros((1, 3), dtype=torch.int64)
    layer = LayerExpertLoad(counts[0], rows)
    trace = ExpertLoadTrace(counts, max_steps=2)
    layer.record(torch.tensor([[0, 1], [1, -1], [2, 2], [0, 2]]))
    trace.record()
    assert trace.snapshot().tolist() == [[[1, 2, 0]]]


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_EXPERT_TRACE_GPU") != "1",
    reason="opt-in GPU trace qualification",
)
def test_cuda_graph_replay_records_distinct_steps_without_capturing_ring_cursor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    counts = torch.zeros((1, 3), dtype=torch.int64, device="cuda")
    trace = ExpertLoadTrace(counts, max_steps=2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts.add_(1)
    counts.zero_()
    trace.reset()
    for _ in range(3):
        graph.replay()
        trace.record()
    assert trace.snapshot().tolist() == [[[1, 1, 1]], [[1, 1, 1]]]


def test_snapshot_projects_replicas_and_switch_invalidates_counter_history():
    from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 0, 1]]),
        2,
        ep_size=2,
        ep_rank=0,
        ep_rank_nodes=(0, 0),
    )
    # The real placement's counters; mask setup belongs to the router tests.
    placement.physical_load = torch.zeros((1, 4), dtype=torch.int64)
    placement.enable_load_trace(max_steps=2)
    initial = placement.placement_snapshot()
    assert initial.ep_rank_nodes == (0, 0)
    assert initial.logical_to_all_physical == (((0, 2, -1), (1, 3, -1)),)
    for row in ([[1, 2, 3, 4]], [[3, 0, 2, 1]]):
        placement.physical_load.add_(torch.tensor(row))
        placement.record_load_step()
    snapshot = placement.snapshot_load()
    assert snapshot.logical_trace.tolist() == [[[4, 6]], [[5, 1]]]
    assert snapshot.logical_count.tolist() == [[9, 7]]
    assert snapshot.placement_epoch == 1
    placement.physical_load.add_(7)
    placement.record_load_step()
    placement.update_layers((0,), torch.tensor([[1, 0, 1, 0]]))
    assert placement.placement_epoch == 2
    assert initial.physical_to_logical == ((0, 1, 0, 1),)
    assert placement.placement_snapshot().physical_to_logical == ((1, 0, 1, 0),)
    fresh = placement.snapshot_load()
    assert fresh.logical_trace.shape == (0, 1, 2)
    assert fresh.logical_count.tolist() == [[0, 0]]
    assert snapshot.logical_trace.tolist() == [[[4, 6]], [[5, 1]]]


def test_record_step_without_trace_preserves_legacy_window():
    from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1]]),
        2,
        ep_size=1,
        ep_rank=0,
        ep_rank_nodes=(0,),
    )
    placement.record_load_step()
    placement.physical_load = torch.tensor([[2, 5]], dtype=torch.int64)
    placement.record_load_step()
    snapshot = placement.snapshot_load()
    assert snapshot.logical_trace is None
    assert snapshot.physical_count.tolist() == [[2, 5]]
