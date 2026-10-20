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

"""CPU-only pool planning and lifecycle tests; process boundaries use fake workers."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tokenspeed.runtime.load_balancing.actuator import (
    DrainObservation,
    GPUInventory,
    LocalProcessBackend,
    PoolController,
    ProcessStatus,
    RegistryAdmissionControl,
    WorkerIdentity,
    WorkerPhase,
)
from tokenspeed.runtime.load_balancing.pool import PoolAllocator, PoolBounds, PoolDemand
from tokenspeed.runtime.load_balancing.registry import WorkerRegistry
from tokenspeed.runtime.load_balancing.types import Worker, Workload


def allocator(
    *, stable=1, cooldown=0, movement=2, minimum=0, maximum=4, total=4, spare=0
):
    return PoolAllocator(
        {"decode": PoolBounds(minimum, maximum, 1)},
        total_gpus=total,
        min_spare_gpus=spare,
        max_spare_gpus=total,
        target_seconds=1,
        telemetry_ttl=5,
        hysteresis_fraction=0.15,
        stable_samples=stable,
        cooldown_seconds=cooldown,
        max_moved_gpus=movement,
    )


def demand(work, now, *, reserved=0, capacity=1):
    return {"decode": PoolDemand(work, reserved, capacity, now)}


def test_targets_account_reserved_work_measured_capacity_and_movement():
    policy = allocator(movement=1)
    plan = policy.plan({"decode": 0}, demand(1, 0, reserved=3, capacity=2), now=0)
    assert plan.targets == {"decode": 1}
    assert policy.plan(
        plan.targets, demand(1, 1, reserved=3, capacity=2), now=1
    ).targets == {"decode": 2}


def test_hysteresis_requires_distinct_reports_and_cooldown():
    policy = allocator(stable=2, cooldown=10)
    assert policy.plan({"decode": 1}, demand(3, 0), now=0).targets == {"decode": 1}
    assert policy.plan({"decode": 1}, demand(3, 0), now=0).targets == {"decode": 1}
    assert policy.plan({"decode": 1}, demand(3, 1), now=1).targets == {"decode": 3}
    assert policy.plan({"decode": 3}, demand(0, 2), now=2).targets == {"decode": 3}
    assert policy.plan({"decode": 3}, demand(0, 3), now=3).targets == {"decode": 3}
    assert policy.plan({"decode": 3}, demand(0, 11), now=11).targets == {"decode": 1}


def test_oscillating_demand_does_not_move_pools():
    policy = allocator(stable=2)
    for now in range(10):
        assert policy.plan(
            {"decode": 2}, demand(1 if now % 2 else 3, now), now=now
        ).targets == {"decode": 2}


def test_small_demand_changes_inside_deadband_do_not_move():
    policy = allocator()
    assert policy.plan({"decode": 2}, demand(2.1, 0), now=0).targets == {"decode": 2}


@pytest.mark.parametrize("observed", [0, 12])
def test_stale_or_future_measurements_cannot_resize(observed):
    policy = allocator()
    assert policy.plan({"decode": 2}, demand(0, observed), now=10).targets == {
        "decode": 2
    }


def test_epd_budget_and_spare_are_preserved_for_indivisible_tp_workers():
    policy = PoolAllocator(
        {
            "encoder": PoolBounds(1, 2, 1),
            "prefill": PoolBounds(1, 3, 2),
            "decode": PoolBounds(1, 3, 2),
        },
        total_gpus=8,
        min_spare_gpus=1,
        max_spare_gpus=3,
        target_seconds=1,
        telemetry_ttl=5,
        hysteresis_fraction=0,
        stable_samples=1,
        cooldown_seconds=0,
        max_moved_gpus=4,
    )
    current = {"encoder": 1, "prefill": 1, "decode": 1}
    demands = {r: PoolDemand(100 if r == "prefill" else 1, 0, 1, 0) for r in current}
    plan = policy.plan(current, demands, now=0)
    assert plan.targets == {"encoder": 1, "prefill": 2, "decode": 1}
    assert plan.spare_gpus == 1


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf")])
def test_invalid_measurement_rejected(capacity):
    with pytest.raises(ValueError):
        PoolDemand(1, 0, capacity, 0)


def test_atomic_inventory_cannot_double_allocate_and_release_is_idempotent():
    inventory = GPUInventory([0, 1, 2, 3])
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda n: inventory.reserve(str(n), 2, spare=0), range(8))
        )
    assigned = [gpu for result in results if result is not None for gpu in result]
    assert len(assigned) == 4 and len(set(assigned)) == 4
    inventory.release("0")
    inventory.release("0")
    assert all(owner != "0" for owner in inventory.owners.values())


class FakeBackend:
    """A fake process/network boundary with explicit outstanding work."""

    def __init__(self):
        self.states = {}
        self.admitting = {}
        self.pending = {}
        self.stop_failures = 0
        self.fail_start = False
        self.fail_close = False
        self.stops = []
        self.starts = []

    def start(self, worker):
        self.starts.append(worker)
        self.states[worker.incarnation] = ProcessStatus.STARTING
        self.pending[worker.incarnation] = 0
        self.admitting[worker.incarnation] = False
        if self.fail_start:
            raise OSError("partial start")

    def status(self, worker):
        return self.states[worker.incarnation]

    def set_admission(self, worker, *, admitting):
        if not admitting and self.fail_close:
            raise OSError("admission control unavailable")
        self.admitting[worker.incarnation] = admitting

    def drained(self, worker):
        return (
            not self.admitting[worker.incarnation]
            and self.pending[worker.incarnation] == 0
        )

    def stop(self, worker):
        assert self.drained(worker)
        if self.stop_failures:
            self.stop_failures -= 1
            raise OSError("stop failed")
        self.stops.append(worker.incarnation)
        self.states[worker.incarnation] = ProcessStatus.STOPPED


def controller(backend, *, movement=2, maximum=4):
    return PoolController(
        allocator(movement=movement, maximum=maximum),
        GPUInventory(range(4)),
        backend,
        startup_timeout=2,
        drain_timeout=3,
    )


def make_ready(control, backend, count=1):
    control.reconcile({"decode": count}, now=0)
    for worker in backend.starts:
        backend.states[worker.incarnation] = ProcessStatus.READY
    control.reconcile({"decode": count}, now=1)
    return list(control.workers.values())


def test_tick_actually_starts_worker_and_ready_admission_is_explicit():
    backend = FakeBackend()
    control = controller(backend)
    assert control.tick(demand(1, 0), now=0).targets == {"decode": 1}
    assert len(backend.starts) == 1
    identity = backend.starts[0]
    assert not backend.admitting[identity.incarnation]
    backend.states[identity.incarnation] = ProcessStatus.READY
    control.tick(demand(1, 1), now=1)
    assert backend.admitting[identity.incarnation]


def test_drain_timeout_blocks_without_killing_inflight_then_completes():
    backend = FakeBackend()
    control = controller(backend)
    worker = make_ready(control, backend)[0]
    backend.pending[worker.identity.incarnation] = 1
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 0}, now=6)
    assert worker.phase == WorkerPhase.BLOCKED
    assert not backend.stops and len(control.inventory.owners) == 1
    backend.pending[worker.identity.incarnation] = 0
    control.reconcile({"decode": 0}, now=7)
    assert worker.phase == WorkerPhase.STOPPED
    assert not control.inventory.owners
    control.reconcile({"decode": 0}, now=8)
    assert len(backend.stops) == 1


def test_reverse_target_resumes_drain_before_stop():
    backend = FakeBackend()
    control = controller(backend)
    worker = make_ready(control, backend)[0]
    backend.pending[worker.identity.incarnation] = 1
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 1}, now=3)
    assert worker.phase == WorkerPhase.READY
    assert backend.admitting[worker.identity.incarnation]
    assert len(backend.starts) == 1 and not backend.stops


def test_partial_start_retains_gpu_until_confirmed_cleanup_and_uses_new_incarnation():
    backend = FakeBackend()
    backend.fail_start = True
    control = controller(backend)
    control.reconcile({"decode": 1}, now=0)
    first = backend.starts[0]
    assert first.incarnation in control.inventory.owners.values()
    backend.pending[first.incarnation] = 1
    control.reconcile({"decode": 0}, now=1)
    assert first.incarnation in control.inventory.owners.values()
    backend.pending[first.incarnation] = 0
    backend.fail_start = False
    control.reconcile({"decode": 1}, now=2)
    assert first.incarnation not in control.inventory.owners.values()
    assert backend.starts[-1].incarnation != first.incarnation


def test_failed_admission_closure_never_permits_stop():
    backend = FakeBackend()
    control = controller(backend)
    worker = make_ready(control, backend)[0]
    backend.fail_close = True
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 0}, now=9)
    assert worker.phase == WorkerPhase.BLOCKED and not backend.stops
    assert len(control.inventory.owners) == 1
    backend.fail_close = False
    control.reconcile({"decode": 0}, now=10)
    assert worker.phase == WorkerPhase.STOPPED


def test_stop_failure_keeps_ownership_and_retries():
    backend = FakeBackend()
    control = controller(backend)
    worker = make_ready(control, backend)[0]
    backend.stop_failures = 1
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 0}, now=3)
    assert worker.phase == WorkerPhase.STOPPING and control.inventory.owners
    control.reconcile({"decode": 0}, now=4)
    assert worker.phase == WorkerPhase.STOPPED and not control.inventory.owners


def test_lost_worker_with_outstanding_work_remains_owned():
    backend = FakeBackend()
    control = controller(backend, maximum=1)
    worker = make_ready(control, backend)[0]
    backend.pending[worker.identity.incarnation] = 1
    backend.states[worker.identity.incarnation] = ProcessStatus.FAILED
    control.reconcile({"decode": 1}, now=2)
    control.reconcile({"decode": 1}, now=6)
    assert len(backend.starts) == 1
    assert control.inventory.owners and not backend.stops


def test_repeated_same_tick_does_not_bypass_movement_bound():
    backend = FakeBackend()
    control = controller(backend, movement=1)
    for _ in range(5):
        control.reconcile({"decode": 4}, now=0)
    assert len(backend.starts) == 1
    control.reconcile({"decode": 4}, now=1)
    assert len(backend.starts) == 2


def test_startup_timeout_drains_before_stopping():
    backend = FakeBackend()
    control = controller(backend)
    control.reconcile({"decode": 1}, now=0)
    worker = next(iter(control.workers.values()))
    backend.pending[worker.identity.incarnation] = 1
    control.reconcile({"decode": 0}, now=3)
    assert not backend.stops and worker.failure == "startup timeout"


def registry_control():
    registry = WorkerRegistry(
        freshness_seconds=10, max_workers=4, max_leases=10, history_seconds=10
    )
    identity = WorkerIdentity("d", "a", "decode", (0,))
    worker = Worker(
        "d", "a", "decode", "http://localhost:1234", Workload(requests=2), 1
    )
    registry.upsert(worker)
    observations = {"value": DrainObservation("a", 2, 0, 0)}
    clock = {"now": 2}
    admission = RegistryAdmissionControl(
        registry,
        lambda _: observations["value"],
        clock=lambda: clock["now"],
        freshness_seconds=3,
    )
    return registry, identity, worker, admission, observations, clock


def test_registry_drain_requires_no_leases_and_fresh_engine_ack_after_closure():
    registry, identity, _, admission, observations, clock = registry_control()
    lease = registry.reserve(
        "request", "decode", lambda _: 0, Workload(requests=1), now=2
    )
    admission.set_admission(identity, admitting=False)
    assert not admission.drained(identity)
    registry.release(lease.lease_id, now=2)
    observations["value"] = DrainObservation("a", 1, 0, 0)
    assert not admission.drained(identity)
    observations["value"] = DrainObservation("a", 2, 0, 1)
    assert not admission.drained(identity)
    observations["value"] = DrainObservation("a", 2, 0, 0)
    assert admission.drained(identity)
    clock["now"] = 6
    assert not admission.drained(identity)


def test_worker_reincarnation_cannot_be_drained_or_resumed_by_old_owner():
    registry, identity, worker, admission, _, _ = registry_control()
    admission.set_admission(identity, admitting=False)
    registry.upsert(replace(worker, incarnation="new", observed_at=2))
    with pytest.raises(ValueError, match="incarnation"):
        admission.set_admission(identity, admitting=True)
    with pytest.raises(ValueError, match="incarnation"):
        admission.drained(identity)
    assert not registry.get("d").worker.draining


class LocalAdmission:
    def __init__(self):
        self.admitting = {}
        self.pending = 0

    def ready(self, worker):
        return True

    def set_admission(self, worker, *, admitting):
        self.admitting[worker.incarnation] = admitting

    def drained(self, worker):
        return not self.admitting.get(worker.incarnation, True) and not self.pending


@pytest.mark.skipif(os.name != "posix", reason="local POSIX process groups")
def test_executable_local_controller_drains_real_cpu_process(tmp_path):
    admission = LocalAdmission()

    def commands(worker):
        return [
            {
                "role": "decode",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "env": {},
            }
        ]

    backend = LocalProcessBackend(
        commands, admission, tmp_path, apply=True, shutdown_grace=0.2
    )
    control = controller(backend)
    try:
        control.tick(demand(1, 0), now=0)
        control.tick(demand(1, 1), now=1)
        worker = next(iter(control.workers.values()))
        process = backend.groups[worker.identity.incarnation].processes[0][1]
        assert process.poll() is None and worker.phase == WorkerPhase.READY
        admission.pending = 1
        control.reconcile({"decode": 0}, now=2)
        control.reconcile({"decode": 0}, now=9)
        assert process.poll() is None
        admission.pending = 0
        control.reconcile({"decode": 0}, now=10)
        assert process.poll() is not None and not control.inventory.owners
    finally:
        for group in backend.groups.values():
            group.close()


@pytest.mark.skipif(os.name != "posix", reason="local POSIX process groups")
def test_local_partial_start_cleanup_covers_already_started_rank(tmp_path):
    admission = LocalAdmission()

    def commands(worker):
        return [
            {
                "role": "rank0",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "env": {},
            },
            {"role": "rank1", "argv": [str(tmp_path / "absent")], "env": {}},
        ]

    backend = LocalProcessBackend(
        commands, admission, tmp_path, apply=True, shutdown_grace=0.2
    )
    control = controller(backend)
    try:
        control.reconcile({"decode": 1}, now=0)
        worker = next(iter(control.workers.values()))
        process = backend.groups[worker.identity.incarnation].processes[0][1]
        assert process.poll() is None and control.inventory.owners
        control.reconcile({"decode": 0}, now=1)
        assert process.poll() is not None and not control.inventory.owners
    finally:
        for group in backend.groups.values():
            group.close()


def test_process_backend_requires_explicit_apply_and_preserves_gpu_assignment(tmp_path):
    admission = LocalAdmission()
    backend = LocalProcessBackend(
        lambda _: [], admission, tmp_path, apply=False, shutdown_grace=1
    )
    with pytest.raises(RuntimeError, match="opt-in"):
        backend.start(WorkerIdentity("d", "a", "decode", (0,)))


def test_indivisible_spare_budget_avoids_greedy_dead_end():
    policy = PoolAllocator(
        {"prefill": PoolBounds(0, 1, 3), "decode": PoolBounds(0, 2, 2)},
        total_gpus=5,
        min_spare_gpus=0,
        max_spare_gpus=0,
        target_seconds=1,
        telemetry_ttl=5,
        hysteresis_fraction=0,
        stable_samples=1,
        cooldown_seconds=0,
        max_moved_gpus=5,
    )
    plan = policy.plan(
        {"prefill": 0, "decode": 0},
        {"prefill": PoolDemand(0, 0, 1, 0), "decode": PoolDemand(100, 0, 1, 0)},
        now=0,
    )
    assert plan.targets == {"prefill": 1, "decode": 1}
    assert plan.spare_gpus == 0


def test_out_of_order_measurements_do_not_count_toward_stability():
    policy = allocator(stable=2)
    assert policy.plan({"decode": 1}, demand(4, 2), now=2).targets == {"decode": 1}
    assert (
        policy.plan({"decode": 1}, demand(4, 1), now=3).reason
        == "regressed telemetry: hold"
    )
    assert policy.plan({"decode": 1}, demand(4, 4), now=4).targets == {"decode": 1}


def test_drain_timeout_can_be_reversed_without_migrating_work():
    backend = FakeBackend()
    control = controller(backend)
    worker = make_ready(control, backend)[0]
    backend.pending[worker.identity.incarnation] = 1
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 0}, now=9)
    control.reconcile({"decode": 1}, now=10)
    assert worker.phase == WorkerPhase.READY and worker.failure is None
    assert len(backend.starts) == 1 and not backend.stops


def test_direct_target_cannot_reduce_below_minimum():
    backend = FakeBackend()
    control = PoolController(
        allocator(minimum=1),
        GPUInventory(range(4)),
        backend,
        startup_timeout=2,
        drain_timeout=3,
    )
    make_ready(control, backend)
    with pytest.raises(ValueError, match="minimum"):
        control.reconcile({"decode": 0}, now=2)
    assert not backend.stops


def test_finite_inputs_cannot_overflow_total_work():
    with pytest.raises(ValueError, match="outstanding"):
        PoolDemand(1e308, 1e308, 1, 0)
    policy = allocator()
    assert policy.plan(
        {"decode": 0}, demand(1e308, 0, capacity=1e-308), now=0
    ).targets == {"decode": 2}


def test_completed_history_is_bounded_without_discarding_active_ownership():
    backend = FakeBackend()
    control = controller(backend)
    for step in range(132):
        now = step * 3
        control.reconcile({"decode": 1}, now=now)
        control.reconcile({"decode": 0}, now=now + 1)
        control.reconcile({"decode": 0}, now=now + 2)
    assert len(control.workers) == 128
    assert not control.inventory.owners
    control.reconcile({"decode": 1}, now=400)
    assert len(control.workers) == 129 and len(control.inventory.owners) == 1
