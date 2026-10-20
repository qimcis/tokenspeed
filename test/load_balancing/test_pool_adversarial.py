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

"""CPU adversarial tests for physical ownership and independent drain proof."""

import os
import sys
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
from tokenspeed.runtime.load_balancing.pool import PoolAllocator, PoolBounds
from tokenspeed.runtime.load_balancing.registry import WorkerRegistry
from tokenspeed.runtime.load_balancing.types import Worker, Workload


def policy(bounds, *, total=5, spare=1, movement=2):
    return PoolAllocator(
        bounds,
        total_gpus=total,
        min_spare_gpus=spare,
        max_spare_gpus=total,
        target_seconds=1,
        telemetry_ttl=5,
        hysteresis_fraction=0,
        stable_samples=1,
        cooldown_seconds=0,
        max_moved_gpus=movement,
    )


class ProcessBoundary:
    """Fake process boundary; each start remains owned until explicit exit."""

    def __init__(self):
        self.states = {}
        self.admission = {}
        self.pending = {}
        self.stopped = []
        self.finish_stop = True

    def start(self, worker):
        self.states[worker.incarnation] = ProcessStatus.STARTING
        self.admission[worker.incarnation] = False
        self.pending[worker.incarnation] = 0

    def status(self, worker):
        return self.states[worker.incarnation]

    def set_admission(self, worker, *, admitting):
        self.admission[worker.incarnation] = admitting

    def drained(self, worker):
        return (
            not self.admission[worker.incarnation]
            and not self.pending[worker.incarnation]
        )

    def stop(self, worker):
        assert self.drained(worker)
        self.stopped.append(worker.incarnation)
        if self.finish_stop:
            self.states[worker.incarnation] = ProcessStatus.STOPPED


def test_fixed_minimum_tp_pack_must_fit_movement_budget():
    with pytest.raises(ValueError, match="indivisible"):
        policy({"decode": PoolBounds(1, 1, 4)}, movement=2)


def test_cross_role_resize_charges_starting_and_draining_tp_packs_until_exit():
    backend = ProcessBoundary()
    control = PoolController(
        policy({"decode": PoolBounds(0, 2, 2), "prefill": PoolBounds(0, 2, 2)}),
        GPUInventory(range(5)),
        backend,
        startup_timeout=20,
        drain_timeout=3,
    )
    control.reconcile({"decode": 2, "prefill": 0}, now=0)
    control.reconcile({"decode": 2, "prefill": 0}, now=1)
    old_workers = list(control.workers.values())
    assert len(old_workers) == 2
    assert len(control.inventory.owners) == 4
    for worker in old_workers:
        backend.states[worker.identity.incarnation] = ProcessStatus.READY
        backend.pending[worker.identity.incarnation] = 1
    control.reconcile({"decode": 0, "prefill": 2}, now=2)
    control.reconcile({"decode": 0, "prefill": 2}, now=3)
    control.reconcile({"decode": 0, "prefill": 2}, now=9)
    assert all(w.phase == WorkerPhase.BLOCKED for w in old_workers)
    assert not backend.stopped
    assert len(control.workers) == 2
    assert len(control.inventory.owners) == 4
    backend.pending[old_workers[0].identity.incarnation] = 0
    control.reconcile({"decode": 0, "prefill": 2}, now=10)
    new_workers = [w for w in control.workers.values() if w.identity.role == "prefill"]
    assert len(new_workers) == 1
    assert new_workers[0].phase == WorkerPhase.STARTING
    assert set(new_workers[0].identity.gpu_ids) == set(old_workers[0].identity.gpu_ids)
    assert len(control.inventory.owners) == 4
    assert old_workers[1].phase == WorkerPhase.BLOCKED


def test_stop_return_without_confirmed_exit_does_not_release_or_replace():
    backend = ProcessBoundary()
    backend.finish_stop = False
    control = PoolController(
        policy({"decode": PoolBounds(0, 1, 2)}),
        GPUInventory(range(5)),
        backend,
        startup_timeout=20,
        drain_timeout=3,
    )
    control.reconcile({"decode": 1}, now=0)
    worker = next(iter(control.workers.values()))
    backend.states[worker.identity.incarnation] = ProcessStatus.READY
    control.reconcile({"decode": 0}, now=1)
    control.reconcile({"decode": 0}, now=2)
    control.reconcile({"decode": 1}, now=3)
    assert worker.phase == WorkerPhase.STOPPING
    assert len(control.workers) == 1
    assert len(control.inventory.owners) == 2
    backend.finish_stop = True
    control.reconcile({"decode": 1}, now=4)
    assert worker.phase == WorkerPhase.STOPPED
    assert len(control.workers) == 2
    assert len(control.inventory.owners) == 2


def registry_admission():
    registry = WorkerRegistry(
        freshness_seconds=5, max_workers=4, max_leases=10, history_seconds=10
    )
    identity = WorkerIdentity("decode", "boot-1", "decode", (0, 1))
    worker = Worker(
        "decode", "boot-1", "decode", "http://localhost:1234", Workload(requests=2), 10
    )
    registry.upsert(worker)
    state = {"now": 10, "observation": None}
    admission = RegistryAdmissionControl(
        registry,
        lambda _: state["observation"],
        clock=lambda: state["now"],
        freshness_seconds=3,
    )
    return registry, identity, worker, state, admission


@pytest.mark.parametrize(
    "observation",
    [
        None,
        DrainObservation("boot-1", 9, 0, 0),
        DrainObservation("boot-1", 11, 0, 0),
        DrainObservation("boot-0", 10, 0, 0),
        DrainObservation("boot-1", 10, 1, 0),
        DrainObservation("boot-1", 10, 0, 1),
    ],
)
def test_cancelled_lease_cannot_substitute_for_engine_drain_proof(observation):
    registry, identity, _, state, admission = registry_admission()
    lease = registry.reserve(
        "request", "decode", lambda _: 0, Workload(requests=1), now=10
    )
    admission.set_admission(identity, admitting=False)
    registry.release(lease.lease_id, now=10)
    state["observation"] = observation
    assert not registry.active_leases(identity.worker_id, identity.incarnation)
    assert not admission.drained(identity)
    state["observation"] = DrainObservation("boot-1", 10, 0, 0)
    assert admission.drained(identity)


def test_reclose_requires_new_engine_ack_after_reopening():
    _, identity, _, state, admission = registry_admission()
    admission.set_admission(identity, admitting=False)
    state["observation"] = DrainObservation("boot-1", 10, 0, 0)
    assert admission.drained(identity)
    admission.set_admission(identity, admitting=True)
    state["now"] = 11
    admission.set_admission(identity, admitting=False)
    assert not admission.drained(identity)
    state["observation"] = DrainObservation("boot-1", 11, 0, 0)
    assert admission.drained(identity)


def test_telemetry_refresh_does_not_reopen_closed_admission():
    registry, identity, worker, _, admission = registry_admission()
    admission.set_admission(identity, admitting=False)
    registry.upsert(
        replace(worker, observed_at=11, sequence=worker.sequence + 1, draining=False)
    )
    assert registry.get(identity.worker_id).worker.draining


@pytest.mark.skipif(os.name != "posix", reason="local POSIX process groups")
def test_partial_process_start_with_missing_drain_ack_retains_ownership(tmp_path):
    registry, _, template, state, admission = registry_admission()

    def commands(worker):
        registry.upsert(
            replace(
                template,
                worker_id=worker.worker_id,
                incarnation=worker.incarnation,
                draining=True,
            )
        )
        return [
            {
                "role": "rank0",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "env": {},
            },
            {"role": "rank1", "argv": [str(tmp_path / "missing")], "env": {}},
        ]

    backend = LocalProcessBackend(
        commands, admission, tmp_path, apply=True, shutdown_grace=0.2
    )
    control = PoolController(
        policy({"decode": PoolBounds(0, 1, 2)}),
        GPUInventory(range(5)),
        backend,
        startup_timeout=2,
        drain_timeout=3,
    )
    try:
        control.reconcile({"decode": 1}, now=10)
        worker = next(iter(control.workers.values()))
        process = backend.groups[worker.identity.incarnation].processes[0][1]
        state["now"] = 11
        control.reconcile({"decode": 0}, now=11)
        state["now"] = 20
        control.reconcile({"decode": 0}, now=20)
        assert worker.phase == WorkerPhase.BLOCKED
        assert process.poll() is None
        assert len(control.inventory.owners) == 2
        state["observation"] = DrainObservation(worker.identity.incarnation, 20, 0, 0)
        control.reconcile({"decode": 0}, now=20)
        assert worker.phase == WorkerPhase.STOPPED
        assert process.poll() is not None
        assert not control.inventory.owners
    finally:
        for group in backend.groups.values():
            group.close()
