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

"""Adversarial CPU checks of placement publication and processed expert moves."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.eplb_algorithms.trace import (
    TracePlacementOptions,
    compute_trace_placement,
)
from tokenspeed.runtime.moe.expert_load_rows import LayerExpertLoad
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata
from tokenspeed.runtime.moe.expert_location_updater import ExpertLocationUpdater
from tokenspeed.runtime.moe.expert_rebalance import (
    ExpertRebalanceController,
    ExpertRebalanceSpecs,
    SlotMoves,
    plan_slot_moves,
)


def _cpu_module(name, relative_path):
    # Parent packages register vendor GPU implementations eagerly. Load these
    # dependency-light production modules without importing those registries.
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def placement(rows, *, ranks=1, rank=0):
    rows = torch.tensor(rows, dtype=torch.int32)
    return ExpertLocationMetadata.from_physical_to_logical_map(
        rows,
        int(rows.max()) + 1,
        ep_size=ranks,
        ep_rank=rank,
        ep_rank_nodes=tuple(range(ranks)),
    )


def updater(metadata, weights):
    # Local moves never use a process group; no communication is simulated.
    return ExpertLocationUpdater(
        metadata,
        weights,
        process_group=None,
        peer_ranks=(0,),
        ep_rank=0,
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )


def test_late_dispatch_materialization_rejects_without_partial_publication():
    metadata = placement([[0, 1, 0, 2]], ranks=2)
    prepared = metadata.prepare_layers((0,), torch.tensor([[2, 0, 1, 0]]))
    tables = (
        metadata.physical_to_logical_map,
        metadata.logical_to_all_physical_map,
        metadata.logical_to_all_physical_map_num_valid,
        metadata.rank_dispatch_map(),
    )
    before = tuple(t.clone() for t in tables)
    snapshot = metadata.placement_snapshot()
    with pytest.raises(RuntimeError, match="materialized"):
        metadata.switch_layer(prepared, 0)
    assert metadata.placement_snapshot() == snapshot
    for actual, expected in zip(tables, before):
        assert torch.equal(actual, expected)


def test_invalid_later_layer_is_rejected_before_any_weight_or_map_write():
    metadata = placement([[0, 1, 2], [2, 0, 1]])
    weights = {
        0: [torch.tensor([[10], [20], [30]])],
        1: [torch.tensor([[30], [10], [20]])],
    }
    before = {layer: tensors[0].clone() for layer, tensors in weights.items()}
    snapshot = metadata.placement_snapshot()
    operation = updater(metadata, weights)
    with pytest.raises(ValueError, match="no physical slot"):
        operation.apply((0, 1), torch.tensor([[1, 0, 2], [1, 1, 1]]), {})
    assert metadata.placement_snapshot() == snapshot
    assert all(torch.equal(weights[layer][0], value) for layer, value in before.items())


def test_staging_failure_leaves_live_slots_and_epoch_unchanged():
    metadata = placement([[0, 1, 2]])
    weights = torch.tensor([[10, 11], [20, 21], [30, 31]])
    before = weights.clone()
    snapshot = metadata.placement_snapshot()
    operation = updater(metadata, {0: [weights]})
    # The first staging copy succeeds; the second source is unavailable.
    moves = SlotMoves(recv=(), send=(), local_copy=((0, 1), (1, 99)), free_rider=())
    with pytest.raises(IndexError):
        operation.apply((0,), torch.tensor([[1, 0, 2]]), {0: moves})
    assert torch.equal(weights, before)
    assert metadata.placement_snapshot() == snapshot


def _packed_weights(logical=(0, 1, 2, 0)):
    module = _cpu_module(
        "expert_adversarial_packing",
        "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_deep_gemm/mega_moe_weights.py",
    )
    shapes = module._expected_shapes(len(logical), 256, 512)
    canonical = []
    for field, shape in enumerate(shapes):
        lanes = torch.arange(torch.tensor(shape[1:]).prod()).reshape(shape[1:])
        canonical.append(
            torch.stack(
                [
                    ((lanes * (2 * slot + 1) + 29 * slot + field * 7) % 256).to(
                        torch.uint8
                    )
                    for slot in logical
                ]
            )
        )
    state = module._DeepGemmMegaMoEState(
        (
            module._interleave_gate_up_(canonical[0].view(torch.int8), 8),
            module._reorder_scale_rows_(
                module._pack_ue8m0_scale_(canonical[1]), True, 8
            ),
        ),
        (
            canonical[2].view(torch.int8),
            module._reorder_scale_rows_(
                module._pack_ue8m0_scale_(canonical[3]), False, 8
            ),
        ),
        torch.device("cpu"),
    )
    return state


def test_real_updater_moves_packed_scales_and_replicas_without_pointer_changes():
    metadata = placement([[0, 1, 2, 0]])
    state = _packed_weights()
    kernel_tensors = (*state.l1_weights, *state.l2_weights)
    original = tuple(t.clone() for t in kernel_tensors)
    addresses = tuple((t.data_ptr(), t.stride()) for t in kernel_tensors)
    operation = updater(metadata, {0: state.routed_expert_weights})
    assert operation.staging_bytes == sum(
        t.numel() * t.element_size() for t in kernel_tensors
    )
    old = metadata.physical_to_logical_map_cpu.clone()
    new = torch.tensor([[2, 0, 0, 1]], dtype=torch.int32)
    moves = plan_slot_moves(old[0], new[0], ep_rank=0, ep_rank_nodes=(0,), num_local=4)
    operation.apply((0,), new, {0: moves})
    for actual, previous in zip(kernel_tensors, original):
        assert torch.equal(actual, previous[[2, 0, 0, 1]])
    assert addresses == tuple((t.data_ptr(), t.stride()) for t in kernel_tensors)
    assert metadata.placement_snapshot().physical_to_logical == ((2, 0, 0, 1),)


def test_replica_routes_and_padding_remain_logical_across_epoch_change():
    metadata = placement([[0, 1, 0, 2]], ranks=2)
    mapping_module = _cpu_module(
        "expert_adversarial_mapping", "python/tokenspeed/runtime/distributed/mapping.py"
    )
    metadata.enable_load_recording(
        mapping_module.Mapping(rank=0, world_size=2, moe_ep_size=2)
    )
    metadata.reserve_load_rows(3)
    metadata.enable_load_trace(max_steps=2)
    metadata.load_rows.mask.copy_(torch.tensor([True, False, True]))
    layer = LayerExpertLoad(metadata.physical_load[0], metadata.load_rows)
    layer.record(torch.tensor([[0, 2], [1, 3], [3, -1]]))
    metadata.record_load_step()
    snapshot = metadata.snapshot_load()
    assert snapshot.logical_trace.tolist() == [[[2, 0, 1]]]
    assert snapshot.logical_count.tolist() == [[2, 0, 1]]
    original_epoch = snapshot.placement_epoch
    # Discard pre-switch physical history; new slot 0 means logical expert 2.
    layer.record(torch.tensor([[0, 1]]))
    metadata.record_load_step()
    metadata.update_layers((0,), torch.tensor([[2, 0, 1, 0]]))
    layer.record(torch.tensor([[0, 1]]))
    metadata.record_load_step()
    fresh = metadata.snapshot_load()
    assert fresh.placement_epoch == original_epoch + 1
    assert fresh.logical_trace.tolist() == [[[1, 0, 1]]]
    assert fresh.logical_count.tolist() == [[1, 0, 1]]
    assert snapshot.logical_trace.tolist() == [[[2, 0, 1]]]


def test_nonleader_commit_rejects_fractional_ids_without_publishing():
    specs = ExpertRebalanceSpecs(
        num_layers=2,
        num_logical_experts=2,
        num_physical_experts=2,
        ep_size=2,
        ep_rank=1,
        ep_rank_nodes=(0, 1),
        all_to_all_ep=True,
        num_groups=None,
        num_nodes=2,
    )
    controller = ExpertRebalanceController(
        specs,
        rebalance_num_iterations=1,
        layers_per_chunk=1,
        algorithm=EplbAlgorithm.deepseek,
        commit_delay_forwards=0,
        compute_worker=None,
    )
    controller.request_snapshot()
    controller.on_snapshot(
        torch.ones((2, 2), dtype=torch.int64),
        torch.tensor([[0, 1], [0, 1]]),
        placement_epoch=12,
    )
    with pytest.raises(ValueError, match="integer"):
        controller.on_commit(torch.tensor([[0.1, 1.1], [0.1, 1.1]]))
    assert not controller.is_applying
    assert controller.placement_epoch == 12
    controller.on_commit(torch.tensor([[1, 0], [1, 0]]))
    with pytest.raises(RuntimeError, match="out of order"):
        controller.chunk_payload((1,))
    controller.on_chunk_applied((0,))
    assert controller.is_applying and controller.placement_epoch == 12
    controller.on_chunk_applied((1,))
    assert controller.is_idle and controller.placement_epoch == 14


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 6])
def test_trace_budget_coverage_and_objective_against_independent_rank_accounting(
    budget,
):
    trace = torch.tensor([[[9, 9, 1, 1]], [[1, 1, 8, 8]], [[2, 5, 2, 4]]])
    old = torch.tensor([[0, 1, 2, 3, 0, 1]])
    costs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    options = TracePlacementOptions(
        max_steps=3,
        max_moved_slots=budget,
        movement_penalty=0.5,
        min_relative_improvement=0.01,
        objective="measured_cost",
    )
    result = compute_trace_placement(
        trace, old_map=old, ep_size=2, options=options, expert_costs=costs
    )
    row = result.physical_to_logical[0].tolist()
    expected = 0.0
    for step in trace[:, 0].tolist():
        ranks = [0.0, 0.0]
        for slot, expert in enumerate(row):
            ranks[slot // 3] += (
                step[expert] * float(costs[0, expert]) / row.count(expert)
            )
        expected += max(ranks)
    assert result.after == pytest.approx(expected)
    assert set(row) == {0, 1, 2, 3}
    assert result.moved_slots == sum(a != b for a, b in zip(row, old[0].tolist()))
    assert result.moved_slots <= budget
    assert result.after + 0.5 * result.moved_slots <= result.before
    assert old.tolist() == [[0, 1, 2, 3, 0, 1]]


def _remote_migration_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        old = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.int32)
        new = torch.tensor([[3, 3, 0, 2, 1, 1]], dtype=torch.int32)
        metadata = placement(old.tolist(), ranks=2, rank=rank)
        state = _packed_weights(tuple(old[0, rank * 3 : rank * 3 + 3].tolist()))
        kernel = (*state.l1_weights, *state.l2_weights)
        addresses = tuple((t.data_ptr(), t.stride()) for t in kernel)
        expected = _packed_weights(tuple(new[0, rank * 3 : rank * 3 + 3].tolist()))
        operation = ExpertLocationUpdater(
            metadata,
            {0: state.routed_expert_weights},
            process_group=dist.group.WORLD,
            peer_ranks=(0, 1),
            ep_rank=rank,
            all_to_all_ep=True,
            num_groups=None,
            num_nodes=2,
        )
        moves = plan_slot_moves(
            old[0], new[0], ep_rank=rank, ep_rank_nodes=(0, 1), num_local=3
        )
        operation.apply((0,), new, {0: moves})
        for actual, wanted in zip(kernel, (*expected.l1_weights, *expected.l2_weights)):
            assert torch.equal(actual, wanted)
        assert addresses == tuple((t.data_ptr(), t.stride()) for t in kernel)
        assert metadata.placement_epoch == 2
        assert torch.equal(metadata.physical_to_logical_map_cpu, new)
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gloo_migration_preserves_remote_replica_scale_bytes(tmp_path):
    mp.spawn(
        _remote_migration_rank,
        args=((tmp_path / "gloo").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_finite_trace_that_overflows_aggregate_objective_is_rejected():
    options = TracePlacementOptions(
        max_steps=2,
        max_moved_slots=2,
        movement_penalty=0,
        min_relative_improvement=0,
        objective="token_count",
    )
    with pytest.raises(ValueError, match="finite|overflow"):
        compute_trace_placement(
            torch.full((2, 1, 2), 1e308, dtype=torch.float64),
            old_map=torch.tensor([[0, 1]]),
            ep_size=2,
            options=options,
        )


def test_fractional_prepared_rows_are_rejected_without_mutating_metadata():
    metadata = placement([[0, 1, 0, 2]])
    before = metadata.placement_snapshot()
    with pytest.raises(ValueError, match="integer"):
        metadata.update_layers((0,), torch.tensor([[2.5, 0.5, 1.5, 0.5]]))
    assert metadata.placement_snapshot() == before
