# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Behavioral CPU coverage of trace placement and the real asynchronous controller."""

from dataclasses import replace

import pytest
import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.eplb_algorithms.trace import (
    TracePlacementOptions,
    compute_trace_placement,
    predicted_peak_cost,
)
from tokenspeed.runtime.moe.expert_rebalance import (
    EplbApplyChunk,
    EplbCommit,
    ExpertRebalanceController,
    ExpertRebalanceSpecs,
    PlacementComputeWorker,
    plan_slot_moves,
)

OPTIONS = TracePlacementOptions(
    max_steps=16,
    max_moved_slots=16,
    movement_penalty=0,
    min_relative_improvement=0,
    objective="token_count",
)


def optimize(trace, old, *, ranks=2, options=OPTIONS, costs=None):
    return compute_trace_placement(
        torch.as_tensor(trace),
        old_map=torch.as_tensor(old),
        ep_size=ranks,
        options=options,
        expert_costs=costs,
    )


def test_coactivation_beats_aggregate_counts_with_equal_marginals():
    old = [[0, 1, 2, 3]]
    trace = [[[10, 10, 0, 0]], [[0, 0, 10, 10]]]
    result = optimize(trace, old)
    assert result.before == 40 and result.after == 20
    assert result.physical_to_logical.tolist() == [[0, 2, 1, 3]]
    # Aggregation cannot distinguish these traces; their best pairings differ.
    other = [[[10, 0, 10, 0]], [[0, 10, 0, 10]]]
    assert torch.equal(torch.tensor(trace).sum(0), torch.tensor(other).sum(0))
    assert optimize(other, old).physical_to_logical.tolist() == old


def test_redundancy_is_reallocated_to_hot_expert_without_losing_coverage():
    result = optimize([[[2, 2, 2, 100]]], [[0, 1, 2, 3, 0, 1]])
    counts = torch.bincount(result.physical_to_logical[0].long(), minlength=4)
    assert counts.tolist() == [1, 1, 1, 3]
    assert result.after < result.before
    assert result.moved_slots == int(
        (result.physical_to_logical != torch.tensor([[0, 1, 2, 3, 0, 1]])).sum()
    )


def test_capacity_and_ties_are_deterministic_and_zero_load_stays_put():
    generator = torch.Generator().manual_seed(122)
    trace = torch.randint(0, 100, (8, 3, 12), generator=generator)
    old = torch.arange(12).repeat(3, 1)
    first = optimize(trace, old, ranks=3)
    second = optimize(trace, old.clone(), ranks=3)
    assert torch.equal(first.physical_to_logical, second.physical_to_logical)
    assert first.physical_to_logical.reshape(3, 3, 4).shape[-1] == 4
    for row in first.physical_to_logical:
        assert sorted(row.tolist()) == list(range(12))
    assert torch.equal(
        optimize(torch.zeros_like(trace), old, ranks=3).physical_to_logical, old
    )


def test_movement_budget_is_global_across_layers_and_can_make_partial_progress():
    trace = torch.tensor([[[10, 10, 0, 0]] * 2, [[0, 0, 10, 10]] * 2])
    old = torch.tensor([[0, 1, 2, 3]] * 2)
    result = optimize(trace, old, options=replace(OPTIONS, max_moved_slots=2))
    assert result.moved_slots == 2
    assert result.after == 60 < result.before == 80
    assert result.physical_to_logical.tolist() == [[0, 2, 1, 3], [0, 1, 2, 3]]
    for budget in (0, 1):
        rejected = optimize(
            trace, old, options=replace(OPTIONS, max_moved_slots=budget)
        )
        assert rejected.moved_slots == 0
        assert torch.equal(rejected.physical_to_logical, old)


@pytest.mark.parametrize(
    "options",
    [
        replace(OPTIONS, movement_penalty=11),
        replace(OPTIONS, min_relative_improvement=0.51),
    ],
)
def test_churn_cost_and_hysteresis_reject_unprofitable_change(options):
    result = optimize(
        [[[10, 10, 0, 0]], [[0, 0, 10, 10]]], [[0, 1, 2, 3]], options=options
    )
    assert result.reason == "hysteresis_or_movement_cost"
    assert result.moved_slots == 0 and result.after == result.before


def test_measured_cost_objective_has_explicit_units_and_changes_weighting():
    costs = torch.tensor([[1.0, 1.0, 1.0, 100.0]])
    result = optimize(
        [[[1, 1, 1, 1]]],
        [[0, 1, 2, 3, 0, 1]],
        options=replace(OPTIONS, objective="measured_cost"),
        costs=costs,
    )
    assert (result.physical_to_logical == 3).sum() == 3
    assert result.after < result.before
    with pytest.raises(ValueError, match="requires"):
        optimize(
            [[[1, 1]]], [[0, 1]], options=replace(OPTIONS, objective="measured_cost")
        )
    with pytest.raises(ValueError, match="does not accept"):
        optimize([[[1, 1]]], [[0, 1]], costs=torch.ones(1, 2))


@pytest.mark.parametrize(
    "trace, old",
    [
        (torch.ones(17, 1, 2), [[0, 1]]),
        (torch.empty(0, 1, 2), [[0, 1]]),
        ([[[float("nan"), 1]]], [[0, 1]]),
        ([[[-1, 1]]], [[0, 1]]),
        ([[[1, 1]]], [[0, 0]]),
        ([[[1, 1]]], [[0, 2]]),
        ([[[1, 1]]], [[0.0, 1.0]]),
    ],
)
def test_invalid_or_unbounded_inputs_are_rejected(trace, old):
    with pytest.raises(ValueError):
        optimize(trace, old)


def test_real_async_worker_controller_commits_the_trace_proposal_and_moves_exact_weights():
    old = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    trace = torch.tensor([[[10, 10, 0, 0]], [[0, 0, 10, 10]]])
    specs = ExpertRebalanceSpecs(
        num_layers=1,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        ep_rank=0,
        ep_rank_nodes=(0, 0),
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )
    worker = PlacementComputeWorker()
    controller = ExpertRebalanceController(
        specs,
        rebalance_num_iterations=1,
        layers_per_chunk=1,
        algorithm=EplbAlgorithm.trace,
        commit_delay_forwards=0,
        compute_worker=worker,
        trace_options=OPTIONS,
    )
    try:
        worker.wait_ready(timeout=30)
        controller.request_snapshot()
        controller.on_snapshot(
            trace.sum(0), old, logical_trace=trace, placement_epoch=7
        )
        assert controller.note_round(forwarded=False) == [
            EplbCommit(),
            EplbApplyChunk((0,)),
        ]
        new = controller.placement_for_commit()
        controller.on_commit(new)
        rows, _ = controller.chunk_payload((0,))
        # Execute all source sends from the original bytes, just as staging
        # orders the real updater. Each destination receives its logical weights.
        weights = torch.stack([torch.arange(3) + int(e) * 10 for e in old[0]])
        migrated = weights.clone()
        for rank in range(2):
            moves = plan_slot_moves(
                old[0], rows[0], ep_rank=rank, ep_rank_nodes=(0, 0), num_local=2
            )
            for dst, src_rank in moves.recv:
                logical = int(rows[0, rank * 2 + dst])
                src = next(
                    s for s in range(2) if int(old[0, src_rank * 2 + s]) == logical
                )
                migrated[rank * 2 + dst] = weights[src_rank * 2 + src]
            for dst, src in moves.local_copy:
                migrated[rank * 2 + dst] = weights[rank * 2 + src]
            for dst, src in moves.free_rider:
                migrated[rank * 2 + dst] = migrated[rank * 2 + src]
        assert torch.equal(
            migrated, torch.stack([torch.arange(3) + int(e) * 10 for e in rows[0]])
        )
        assert predicted_peak_cost(trace, new, 2) == 20
        controller.on_chunk_applied((0,))
        assert controller.is_idle and controller.placement_epoch == 8
        assert controller.last_trace_result.reason == "improved"
    finally:
        controller.shutdown()


class _TraceDevice:
    """CPU device boundary: real snapshot tensors; hardware transfers are absent."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.applied = []

    def snapshot_expert_load(self):
        return self.snapshot

    def apply_expert_placement(self, layer_ids, rows, moves):
        self.applied.append((tuple(layer_ids), rows.clone(), moves))


class _ControlQueue:
    def __init__(self):
        self.ops = []
        self.completer = None

    def set_internal_op_completer(self, completer):
        self.completer = completer

    def enqueue_internal_op(self, op):
        self.ops.append(op)


def test_serving_hooks_feed_actual_trace_to_worker_and_publish_only_after_last_chunk():
    from tokenspeed.runtime.engine.eplb_hooks import EplbHooks
    from tokenspeed.runtime.moe.expert_location import ExpertLoadSnapshot

    old = torch.tensor([[0, 1, 2, 3]] * 2, dtype=torch.int32)
    trace = torch.tensor([[[10, 10, 0, 0]] * 2, [[0, 0, 10, 10]] * 2])
    specs = ExpertRebalanceSpecs(
        num_layers=2,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        ep_rank=0,
        ep_rank_nodes=(0, 0),
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )
    worker = PlacementComputeWorker()
    controller = ExpertRebalanceController(
        specs,
        rebalance_num_iterations=1,
        layers_per_chunk=1,
        algorithm=EplbAlgorithm.trace,
        commit_delay_forwards=0,
        compute_worker=worker,
        trace_options=OPTIONS,
    )
    device = _TraceDevice(
        ExpertLoadSnapshot(
            physical_count=trace.sum(0),
            physical_to_logical_map=old,
            logical_trace=trace,
            placement_epoch=5,
        )
    )
    queue = _ControlQueue()
    published = []
    hooks = EplbHooks(
        controller,
        queue,
        device,
        ep_cpu_group=None,
        ep_group_ranks=(0,),
        placement_applied=lambda epoch, rows, local: published.append(
            (epoch, rows, local)
        ),
    )
    try:
        hooks.note_round(forwarded=True)
        hooks.complete(queue.ops.pop(0))
        hooks.note_round(forwarded=False)
        hooks.complete(queue.ops.pop(0))  # commit
        assert published == [] and device.applied == []
        hooks.complete(queue.ops.pop(0))  # first layer
        assert published == [] and len(device.applied) == 1
        hooks.complete(queue.ops.pop(0))  # last layer
        assert len(published) == 1 and published[0][0] == 7
        assert published[0][1].tolist() == [[0, 2, 1, 3]] * 2
        assert published[0][2] == 2
        assert controller.last_trace_result.after == 40
    finally:
        hooks.close()
