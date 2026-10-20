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

"""CPU behavior tests for bounded capture of real computed top-k IDs."""

import pytest
import torch

from tokenspeed.runtime.moe.route_capture import RouteCapture
from tokenspeed.runtime.moe.route_feedback import (
    RouteCaptureCadence,
    RouteObservation,
    observations_from_rows,
)


def capture(*, interval=2):
    return RouteCapture(
        layer_widths=((2, 2), (7, 1)),
        interval=interval,
        max_requests=3,
        pin_memory=False,
    )


def finish(value):
    return value.finish(
        request_ids=("a", "b"), generations=(11, 12), computed_frontiers=(33, 42)
    )


def test_capture_reads_precomputed_routes_and_preserves_identity():
    value = capture()
    logits = torch.tensor([[0.0, 8.0, 2.0, 5.0], [9.0, 1.0, 0.0, 7.0]])
    ids = logits.topk(2).indices
    value.begin(num_extends=0, input_lengths=(1, 1))
    value.observe(2, ids)
    value.observe(7, torch.tensor([[3], [-1]]))
    # Copies own storage; a router scratch reuse cannot rewrite this observation.
    ids.fill_(0)
    result = finish(value)
    assert result.observations() == (
        RouteObservation("a", 11, 33, ((2 << 32) | 1, (2 << 32) | 3, (7 << 32) | 3)),
        RouteObservation("b", 12, 42, ((2 << 32) | 0, (2 << 32) | 3)),
    )


def test_cadence_ignores_prefill_idle_and_bounds_decode():
    value = RouteCaptureCadence(interval=3, max_requests=2)
    assert not value.begin(num_extends=1, input_lengths=(500,))
    assert not value.begin(num_extends=0, input_lengths=())
    assert [value.begin(num_extends=0, input_lengths=(1,)) for _ in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="single-token"):
        value.begin(num_extends=0, input_lengths=(2,))
    with pytest.raises(ValueError, match="bounded"):
        value.begin(num_extends=0, input_lengths=(1, 1, 1))


def test_unsampled_forward_does_not_touch_buffers():
    value = capture()
    value.begin(num_extends=0, input_lengths=(1, 1))
    value.observe(2, torch.tensor([[1, 2], [3, 4]]))
    value.observe(7, torch.tensor([[5], [6]]))
    before = finish(value).observations()
    value.begin(num_extends=0, input_lengths=(1, 1))
    # Invalid shape is intentionally ignored outside a sampled forward.
    value.observe(2, torch.empty((0, 0)))
    assert finish(value) is None
    assert value.buffers[2][:2].tolist() == [[1, 2], [3, 4]]
    assert len(before) == 2


def test_capture_reuses_bounded_storage_after_completion():
    value = capture(interval=1)
    pointers = [tensor.data_ptr() for tensor in value.buffers.values()]
    for expert in range(5):
        value.begin(num_extends=0, input_lengths=(1, 1))
        value.observe(2, torch.full((2, 2), expert))
        value.observe(7, torch.full((2, 1), expert))
        assert all(
            (key & 0xFFFFFFFF) == expert
            for row in finish(value).observations()
            for key in row.expert_keys
        )
        assert pointers == [tensor.data_ptr() for tensor in value.buffers.values()]


def test_missing_router_and_repeated_router_fail_instead_of_silent_noop():
    value = capture()
    value.begin(num_extends=0, input_lengths=(1, 1))
    value.observe(2, torch.ones((2, 2), dtype=torch.int64))
    with pytest.raises(RuntimeError, match="twice"):
        value.observe(2, torch.ones((2, 2), dtype=torch.int64))
    with pytest.raises(RuntimeError, match="all selected"):
        finish(value)


def test_sharded_or_padded_rows_are_rejected():
    value = capture()
    value.begin(num_extends=0, input_lengths=(1, 1))
    with pytest.raises(RuntimeError, match="replicated decode"):
        value.observe(2, torch.ones((1, 2), dtype=torch.int64))


@pytest.mark.parametrize("layers", [(), ((0, 257),), ((0, 1), (0, 1)), ((-1, 1),)])
def test_capture_memory_and_key_bounds(layers):
    with pytest.raises(ValueError):
        RouteCapture(layer_widths=layers, interval=1, max_requests=2, pin_memory=False)


def test_padding_empty_support_and_duplicate_experts():
    rows = observations_from_rows(
        request_ids=("a", "b"),
        generations=(1, 2),
        computed_frontiers=(10, 20),
        layer_rows=((3, ((7, 7, -1), (-1, -1, -1))),),
    )
    assert rows == (RouteObservation("a", 1, 10, ((3 << 32) | 7,)),)


@pytest.mark.parametrize(
    "generations,frontiers,layers",
    [
        ((1,), (1, 2), ((0, ((1,), (2,))),)),
        ((1, 2), (1, 2), ((0, ((1,),)),)),
        ((1, 2), (1, 2), ((0, ((2**32,), (2,))),)),
        ((0, 2), (1, 2), ((0, ((1,), (2,))),)),
    ],
)
def test_invalid_identity_or_support_is_rejected(generations, frontiers, layers):
    with pytest.raises(ValueError):
        observations_from_rows(
            request_ids=("a", "b"),
            generations=generations,
            computed_frontiers=frontiers,
            layer_rows=layers,
        )


def _replica_feedback_worker(rank, rendezvous):
    import torch.distributed as dist

    from tokenspeed.runtime.engine.route_feedback_hooks import RouteFeedbackHooks

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        hooks = RouteFeedbackHooks(
            interval=2,
            max_requests=3,
            leader=rank == 0,
            group=dist.group.WORLD,
            source_rank=0,
        )
        # Both ranks see identical schedule metadata; only rank0 owns routes.
        value = capture()
        value.begin(num_extends=0, input_lengths=(1, 1))
        value.observe(2, torch.tensor([[8, 4], [1, 9]]))
        value.observe(7, torch.tensor([[2], [5]]))
        captured = finish(value) if rank == 0 else None
        hooks.collect(num_extends=0, input_lengths=(1, 1), captured=captured)
        expected = (
            RouteObservation(
                "a", 11, 33, ((2 << 32) | 4, (2 << 32) | 8, (7 << 32) | 2)
            ),
            RouteObservation(
                "b", 12, 42, ((2 << 32) | 1, (2 << 32) | 9, (7 << 32) | 5)
            ),
        )
        assert hooks.drain() == expected
        assert hooks.drain() == ()
        hooks.collect(num_extends=0, input_lengths=(1, 1), captured=None)
        assert hooks.drain() == ()
        hooks.collect(num_extends=1, input_lengths=(500,), captured=None)
        assert hooks.drain() == ()
    finally:
        dist.destroy_process_group()


def test_completed_routes_are_identical_on_real_cpu_replicas(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _replica_feedback_worker,
        args=(str(tmp_path / "gloo_routes"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("placement", [False, True])
def test_captured_routes_drive_real_scheduler_and_reject_old_lifetimes(placement):
    ts = pytest.importorskip("tokenspeed_scheduler")
    config = ts.SchedulerConfig()
    config.role = ts.SchedulerConfig.Role.Fused
    config.prefix_granularity = 1
    config.num_device_pages = 256
    config.num_host_pages = 0
    config.num_snapshot_pages = 1
    config.max_retracted_requests = 0
    config.max_scheduled_tokens = 32
    config.max_batch_size = 4
    config.max_decode_batch_size = 2
    config.overlap_schedule_depth = 0
    config.decode_input_tokens = 1
    config.disable_l2_cache = True
    config.disable_prefix_cache = True
    config.enable_l3_storage = False
    config.l3_prefetch_min_pages = 0
    config.cache_groups = [ts.CacheGroupConfig("full", 1, 256, ts.AttnKind.Full)]
    scheduler = ts.Scheduler(config)
    scheduler.configure_expert_affinity(4, 1, 8)

    def submit(rid):
        request = ts.RequestSpec()
        request.request_id = rid
        request.tokens = [10, 20]
        request.max_new_tokens = 32
        scheduler.submit_requests([request])

    def complete(batch):
        events = ts.ExecutionEvent()
        for rid in batch.request_ids:
            event = ts.ForwardEvent.ExtendResult()
            event.request_id = rid
            event.tokens = [99]
            events.add_event(event)
        scheduler.advance(events)

    for rid in ("a", "b", "c"):
        submit(rid)
    prefill = scheduler.next_execution_plan().forward[0]
    assert prefill.num_extends() == 3
    complete(prefill)
    capture = RouteCapture(
        layer_widths=((2, 1),), interval=1, max_requests=4, pin_memory=False
    )
    old_observation = None
    for expected in (("a", "b"), ("c", "a")):
        batch = scheduler.next_execution_plan().forward[0]
        assert tuple(batch.request_ids) == expected
        capture.begin(
            num_extends=batch.num_extends(), input_lengths=tuple(batch.input_lengths)
        )
        logits = torch.tensor(
            [[0.0, 9.0] if rid == "c" else [9.0, 0.0] for rid in expected]
        )
        capture.observe(2, logits.topk(1).indices)
        routes = capture.finish(
            request_ids=tuple(batch.request_ids),
            generations=tuple(batch.request_generations),
            computed_frontiers=tuple(batch.computed_frontiers),
        ).observations()
        for row in routes:
            assert not scheduler.update_expert_route(
                row.request_id,
                row.generation,
                row.computed_frontier,
                list(row.expert_keys),
            )
        complete(batch)
        for row in routes:
            assert scheduler.update_expert_route(
                row.request_id,
                row.generation,
                row.computed_frontier,
                list(row.expert_keys),
            )
            if row.request_id == "a":
                old_observation = row
    if placement:
        from tokenspeed.runtime.engine.route_feedback_hooks import (
            PlacementFeedbackHooks,
        )

        publisher = PlacementFeedbackHooks(
            scheduler, layer_ids=(2,), ep_rank_nodes=(0, 0)
        )
        publisher.publish(
            1,
            ((0, 0, 1, 1),),
            2,
            logical_to_all_physical=(((1, 0), (3, 2)),),
        )
        with pytest.raises(ValueError):
            publisher.publish(2, ((0, 0, 2, 2),), 2)
        assert scheduler.expert_placement_epoch() == 1
        publisher.publish(2, torch.tensor([[1, 1, 0, 0]]), 2)
        assert scheduler.expert_placement_epoch() == 2
    grouped = scheduler.next_execution_plan().forward[0]
    assert tuple(grouped.request_ids) == (("b", "c") if placement else ("b", "a"))
    complete(grouped)
    events = ts.ExecutionEvent()
    finish_event = ts.ForwardEvent.Abort()
    finish_event.request_id = "a"
    events.add_event(finish_event)
    scheduler.advance(events)
    scheduler.next_execution_plan()  # The planner releases the aborted request.
    submit("a")
    assert scheduler.request_generation("a") != old_observation.generation
    assert not scheduler.update_expert_route(
        "a",
        old_observation.generation,
        old_observation.computed_frontier,
        list(old_observation.expert_keys),
    )


@pytest.mark.parametrize(
    ("nodes", "replicas", "expected"),
    [
        (
            (0, 0, 0, 0),
            [[0, 1, 6], [2, 3], [4, 5, 7]],
            [[3, 0, 0, 1], [0, 4, 0, 0], [0, 0, 3, 1]],
        ),
        (
            (0, 0, 1, 1),
            [[0, 1, 6], [2, 3], [4, 5, 7]],
            [[2, 0, 0, 2], [0, 4, 0, 0], [0, 0, 3, 1]],
        ),
        (
            (0, 0, 0, 0),
            [[6, 1, 0], [2, 3], [4, 5, 7]],
            [[1, 0, 0, 3], [0, 4, 0, 0], [0, 0, 3, 1]],
        ),
    ],
)
def test_static_placement_work_counts_sources_not_uniform_replicas(
    nodes, replicas, expected
):
    from tokenspeed.runtime.engine.route_feedback_hooks import (
        _replicated_static_rank_work,
    )

    actual = _replicated_static_rank_work([replicas], 2, nodes)
    assert actual == expected
    assert all(sum(row) == len(nodes) for row in actual)
    # Two copies on rank zero do not execute twice for a single source.
    assert actual[0][0] != 4 * 2 / 3
