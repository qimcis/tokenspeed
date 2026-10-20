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

"""Local pool actuation with exclusive GPU ownership and explicit drain proof.

The backend must close the sole admission path and prove both leases and engine
work have drained. Missing/stale drain proof blocks resizing indefinitely; there
is deliberately no active decoder migration or forced drain timeout.
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from tokenspeed.runtime.load_balancing.pool import PoolAllocator, PoolDemand, PoolPlan
from tokenspeed.runtime.load_balancing.registry import WorkerRegistry
from tokenspeed.runtime.load_balancing.types import Worker

_COMPLETED_HISTORY_LIMIT = 128


class WorkerPhase(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    BLOCKED = "blocked"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ProcessStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    incarnation: str
    role: str
    gpu_ids: tuple[int, ...]


@dataclass
class ManagedWorker:
    identity: WorkerIdentity
    phase: WorkerPhase
    changed_at: float
    failure: str | None = None
    admission_closed: bool = False


class WorkerBackend(Protocol):
    def start(self, worker: WorkerIdentity) -> None: ...
    def status(self, worker: WorkerIdentity) -> ProcessStatus: ...
    def set_admission(self, worker: WorkerIdentity, *, admitting: bool) -> None: ...
    def drained(self, worker: WorkerIdentity) -> bool: ...
    def stop(self, worker: WorkerIdentity) -> None: ...


class GPUInventory:
    """Atomic ownership within one controller process, never a fleet-wide lock."""

    def __init__(self, gpu_ids: Sequence[int]) -> None:
        if (
            not gpu_ids
            or len(set(gpu_ids)) != len(gpu_ids)
            or any(type(g) is not int or g < 0 for g in gpu_ids)
        ):
            raise ValueError("GPU inventory must contain distinct nonnegative IDs")
        self.gpu_ids = tuple(sorted(gpu_ids))
        self._owners: dict[int, str] = {}
        self._lock = threading.RLock()

    def reserve(self, owner: str, count: int, *, spare: int) -> tuple[int, ...] | None:
        if type(count) is not int or count <= 0 or type(spare) is not int or spare < 0:
            raise ValueError("invalid reservation size or spare count")
        with self._lock:
            existing = tuple(g for g in self.gpu_ids if self._owners.get(g) == owner)
            if existing:
                if len(existing) != count:
                    raise ValueError("owner already has a different allocation")
                return existing
            free = tuple(g for g in self.gpu_ids if g not in self._owners)
            if len(free) - count < spare:
                return None
            assigned = free[:count]
            self._owners.update((g, owner) for g in assigned)
            return assigned

    def release(self, owner: str) -> None:
        with self._lock:
            self._owners = {g: o for g, o in self._owners.items() if o != owner}

    @property
    def owners(self) -> dict[int, str]:
        with self._lock:
            return dict(self._owners)


class PoolController:
    """Reconcile desired counts and drive real start/drain/stop operations.

    A complete worker is one allocation regardless of its TP process count.
    Call ``tick`` with measured demand to execute allocator decisions. Caller
    owns scheduling ticks and supplying trustworthy engine drain observations.
    """

    def __init__(
        self,
        allocator: PoolAllocator,
        inventory: GPUInventory,
        backend: WorkerBackend,
        *,
        startup_timeout: float,
        drain_timeout: float,
    ) -> None:
        if len(inventory.gpu_ids) != allocator.total_gpus:
            raise ValueError("allocator budget differs from physical inventory")
        if any(
            not math.isfinite(t) or t <= 0 for t in (startup_timeout, drain_timeout)
        ):
            raise ValueError("startup and drain timeouts must be positive and finite")
        self.allocator = allocator
        self.inventory = inventory
        self.backend = backend
        self.startup_timeout = startup_timeout
        self.drain_timeout = drain_timeout
        self.workers: dict[str, ManagedWorker] = {}
        self._lock = threading.RLock()
        self._last_now = -math.inf
        self._movement_remaining = allocator.max_moved_gpus

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                role: sum(
                    w.identity.role == role
                    and w.phase in (WorkerPhase.STARTING, WorkerPhase.READY)
                    for w in self.workers.values()
                )
                for role in self.allocator.bounds
            }

    def tick(self, demands: Mapping[str, PoolDemand], *, now: float) -> PoolPlan:
        with self._lock:
            plan = self.allocator.plan(self.counts(), demands, now=now)
            self.reconcile(plan.targets, now=now)
            return plan

    def _drain(self, worker: ManagedWorker, now: float) -> None:
        self.backend.set_admission(worker.identity, admitting=False)
        worker.admission_closed = True
        worker.phase = WorkerPhase.DRAINING
        worker.changed_at = now

    def _advance(self, worker: ManagedWorker, now: float) -> None:
        if worker.phase == WorkerPhase.STOPPED:
            return
        identity = worker.identity
        status = self.backend.status(identity)
        if status in (ProcessStatus.FAILED, ProcessStatus.STOPPED) and worker.phase in (
            WorkerPhase.STARTING,
            WorkerPhase.READY,
        ):
            worker.failure = "worker process lost; wait for lease/engine cleanup"
            self._drain(worker, now)
        if worker.phase == WorkerPhase.STARTING:
            if status == ProcessStatus.READY:
                self.backend.set_admission(identity, admitting=True)
                worker.admission_closed = False
                worker.phase, worker.changed_at = WorkerPhase.READY, now
            elif now - worker.changed_at >= self.startup_timeout:
                worker.failure = "startup timeout"
                self._drain(worker, now)
        if worker.phase in (WorkerPhase.DRAINING, WorkerPhase.BLOCKED):
            if not worker.admission_closed:
                self._drain(worker, now)
            if self.backend.drained(identity):
                worker.phase, worker.changed_at = WorkerPhase.STOPPING, now
            elif now - worker.changed_at >= self.drain_timeout:
                worker.phase = WorkerPhase.BLOCKED
                worker.failure = (
                    worker.failure
                    or "drain blocked: active work or missing drain proof"
                )
        if worker.phase == WorkerPhase.STOPPING:
            self.backend.stop(identity)
            if self.backend.status(identity) == ProcessStatus.STOPPED:
                self.inventory.release(identity.incarnation)
                worker.phase, worker.changed_at = WorkerPhase.STOPPED, now

    def reconcile(self, targets: Mapping[str, int], *, now: float) -> None:
        """Apply a bounded target idempotently; failures retain GPU ownership."""
        with self._lock:
            if not math.isfinite(now) or now < 0 or now < self._last_now:
                raise ValueError("controller clock must be monotonic and finite")
            if set(targets) != set(self.allocator.bounds):
                raise ValueError("targets must cover exactly the configured roles")
            if any(
                type(targets[r]) is not int or not 0 <= targets[r] <= b.maximum
                for r, b in self.allocator.bounds.items()
            ):
                raise ValueError("invalid worker target")
            current = self.counts()
            if any(
                targets[r] < min(current[r], b.minimum)
                for r, b in self.allocator.bounds.items()
            ):
                raise ValueError("target would reduce pool below its minimum")
            if (
                sum(
                    targets[r] * b.gpus_per_worker
                    for r, b in self.allocator.bounds.items()
                )
                > self.allocator.total_gpus - self.allocator.min_spare_gpus
            ):
                raise ValueError("target exceeds GPU budget")
            if now > self._last_now:
                self._movement_remaining = self.allocator.max_moved_gpus
            self._last_now = now
            # A reversed target may resume a healthy draining worker before stop.
            for role in self.allocator.bounds:
                need = targets[role] - self.counts()[role]
                for worker in self.workers.values():
                    if need <= 0:
                        break
                    if (
                        worker.identity.role == role
                        and worker.phase in (WorkerPhase.DRAINING, WorkerPhase.BLOCKED)
                        and (
                            worker.failure is None
                            or worker.failure.startswith("drain blocked:")
                        )
                    ):
                        try:
                            if (
                                self.backend.status(worker.identity)
                                == ProcessStatus.READY
                            ):
                                self.backend.set_admission(
                                    worker.identity, admitting=True
                                )
                                worker.admission_closed = False
                                worker.failure = None
                                worker.phase, worker.changed_at = WorkerPhase.READY, now
                                need -= 1
                        except Exception as exc:
                            worker.failure = str(exc)
            for worker in list(self.workers.values()):
                try:
                    self._advance(worker, now)
                except Exception as exc:
                    worker.failure = str(exc)
                    # Failed admission closure must never be mistaken for drain.
                    if worker.phase != WorkerPhase.STOPPING:
                        worker.phase = WorkerPhase.BLOCKED
            movement = self._movement_remaining
            for role, bounds in self.allocator.bounds.items():
                excess = self.counts()[role] - targets[role]
                candidates = [
                    w
                    for w in self.workers.values()
                    if w.identity.role == role
                    and w.phase in (WorkerPhase.READY, WorkerPhase.STARTING)
                ]
                for worker in reversed(candidates):
                    if excess <= 0 or movement < bounds.gpus_per_worker:
                        break
                    try:
                        self._drain(worker, now)
                    except Exception as exc:
                        worker.failure, worker.phase = str(exc), WorkerPhase.BLOCKED
                    movement -= bounds.gpus_per_worker
                    excess -= 1
            for role, bounds in self.allocator.bounds.items():
                for _ in range(max(0, targets[role] - self.counts()[role])):
                    if movement < bounds.gpus_per_worker:
                        break
                    owned = sum(
                        w.identity.role == role and w.phase != WorkerPhase.STOPPED
                        for w in self.workers.values()
                    )
                    if owned >= bounds.maximum:
                        break
                    incarnation = uuid.uuid4().hex
                    devices = self.inventory.reserve(
                        incarnation,
                        bounds.gpus_per_worker,
                        spare=self.allocator.min_spare_gpus,
                    )
                    if devices is None:
                        break
                    identity = WorkerIdentity(
                        f"{role}-{incarnation}", incarnation, role, devices
                    )
                    worker = ManagedWorker(
                        identity, WorkerPhase.STARTING, now, admission_closed=True
                    )
                    self.workers[identity.worker_id] = worker
                    movement -= bounds.gpus_per_worker
                    try:
                        self.backend.start(identity)
                    except Exception as exc:
                        worker.failure = f"start failed: {exc}"
                        # Backend may own a partial process group. Keep GPUs until
                        # admission closure, quiescence and confirmed group exit.
                        worker.phase = WorkerPhase.BLOCKED
                        worker.admission_closed = False
            self._movement_remaining = movement
            completed = sorted(
                (w for w in self.workers.values() if w.phase == WorkerPhase.STOPPED),
                key=lambda w: w.changed_at,
            )
            for worker in completed[:-_COMPLETED_HISTORY_LIMIT]:
                del self.workers[worker.identity.worker_id]


class AdmissionControl(Protocol):
    """Implementations must fence incarnations and own the sole ingress path."""

    def ready(self, worker: WorkerIdentity) -> bool: ...
    def set_admission(self, worker: WorkerIdentity, *, admitting: bool) -> None: ...
    def drained(self, worker: WorkerIdentity) -> bool: ...


class LocalProcessBackend:
    """Execute supplied worker commands using the deployment ProcessGroup.

    ``apply`` is explicit: false rejects start/stop, with no silent simulation.
    Commands and drain/readiness contracts must be supplied by the deployment;
    arbitrary gateways do not gain those capabilities by selecting this backend.
    """

    def __init__(
        self,
        commands: Callable[[WorkerIdentity], Sequence[dict]],
        admission: AdmissionControl,
        directory: Path,
        *,
        apply: bool,
        shutdown_grace: float,
    ) -> None:
        from tokenspeed.deployment.deepseek_v41.launch import ProcessGroup

        if not math.isfinite(shutdown_grace) or shutdown_grace <= 0:
            raise ValueError("shutdown_grace must be positive and finite")
        self.commands = commands
        self.admission = admission
        self.directory = directory
        self.apply = apply
        self.shutdown_grace = shutdown_grace
        self.groups: dict[str, ProcessGroup] = {}
        self._stopping: set[str] = set()

    def start(self, worker: WorkerIdentity) -> None:
        from tokenspeed.deployment.deepseek_v41.launch import ProcessGroup

        if not self.apply:
            raise RuntimeError("local process apply requires explicit opt-in")
        if worker.incarnation in self.groups:
            return
        commands = tuple(self.commands(worker))
        if not commands:
            raise ValueError("worker requires at least one process")
        self.admission.set_admission(worker, admitting=False)
        directory = self.directory / worker.incarnation
        directory.mkdir(parents=True, exist_ok=False)
        group = ProcessGroup(shutdown_grace=self.shutdown_grace)
        self.groups[worker.incarnation] = group
        for command in commands:
            env = dict(command["env"])
            assigned = ",".join(map(str, worker.gpu_ids))
            if (
                "CUDA_VISIBLE_DEVICES" in env
                and env["CUDA_VISIBLE_DEVICES"] != assigned
            ):
                raise ValueError("command GPU assignment differs from owned devices")
            env["CUDA_VISIBLE_DEVICES"] = assigned
            group.start({**command, "env": env}, directory)

    def status(self, worker: WorkerIdentity) -> ProcessStatus:
        group = self.groups.get(worker.incarnation)
        if group is None:
            return ProcessStatus.STOPPED
        if worker.incarnation in self._stopping:
            # ProcessGroup.close reaps leaders; independently confirm descendants
            # are gone before allowing the allocation to be reused.
            for _, process in group.processes:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                    return ProcessStatus.FAILED
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return ProcessStatus.FAILED
            del self.groups[worker.incarnation]
            self._stopping.remove(worker.incarnation)
            return ProcessStatus.STOPPED
        try:
            group.check()
        except RuntimeError:
            return ProcessStatus.FAILED
        return (
            ProcessStatus.READY
            if self.admission.ready(worker)
            else ProcessStatus.STARTING
        )

    def set_admission(self, worker: WorkerIdentity, *, admitting: bool) -> None:
        self.admission.set_admission(worker, admitting=admitting)

    def drained(self, worker: WorkerIdentity) -> bool:
        return self.admission.drained(worker)

    def stop(self, worker: WorkerIdentity) -> None:
        if not self.apply:
            raise RuntimeError("local process apply requires explicit opt-in")
        group = self.groups.get(worker.incarnation)
        if group is None:
            return
        if not self.admission.drained(worker):
            raise RuntimeError("refusing to stop worker without drain proof")
        self._stopping.add(worker.incarnation)
        group.close()


@dataclass(frozen=True)
class DrainObservation:
    """Engine acknowledgment after ingress closure, including outstanding transfers."""

    incarnation: str
    observed_at: float
    pending_requests: int
    pending_transfers: int

    def __post_init__(self) -> None:
        if (
            not self.incarnation
            or not math.isfinite(self.observed_at)
            or self.observed_at < 0
        ):
            raise ValueError("invalid drain observation identity/time")
        if any(
            type(n) is not int or n < 0
            for n in (self.pending_requests, self.pending_transfers)
        ):
            raise ValueError("pending work must be nonnegative integer counts")


class RegistryAdmissionControl:
    """Fence shared-registry admission and require fresh engine quiescence.

    Deployment registers every managed incarnation as draining before launch.
    The registry must be the sole request ingress. A lease reaching zero does
    not prove cancellation finished on the engine or that transfers completed.
    The caller provides that independent acknowledgment through ``observe``.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        observe: Callable[[WorkerIdentity], DrainObservation | None],
        *,
        clock: Callable[[], float],
        freshness_seconds: float,
    ) -> None:
        if not math.isfinite(freshness_seconds) or freshness_seconds <= 0:
            raise ValueError("drain freshness must be positive and finite")
        self.registry = registry
        self.observe = observe
        self.clock = clock
        self.freshness_seconds = freshness_seconds
        self._closed_at: dict[str, float] = {}

    def _worker(self, worker: WorkerIdentity) -> Worker:
        snapshot = self.registry.get(worker.worker_id)
        if snapshot.worker.incarnation != worker.incarnation:
            raise ValueError("worker incarnation changed; refuse lifecycle action")
        return snapshot.worker

    def ready(self, worker: WorkerIdentity) -> bool:
        current = self._worker(worker)
        age = self.clock() - current.observed_at
        return (
            current.ready
            and current.healthy
            and 0 <= age <= self.registry.freshness_seconds
        )

    def set_admission(self, worker: WorkerIdentity, *, admitting: bool) -> None:
        self._worker(worker)
        if admitting and not self.ready(worker):
            raise RuntimeError("worker is not freshly ready")
        self.registry.set_draining(worker.worker_id, worker.incarnation, not admitting)
        if admitting:
            self._closed_at.pop(worker.incarnation, None)
        else:
            current = {
                snapshot.worker.incarnation for snapshot in self.registry.snapshot()
            }
            self._closed_at = {
                incarnation: at
                for incarnation, at in self._closed_at.items()
                if incarnation in current
            }
            self._closed_at.setdefault(worker.incarnation, self.clock())

    def drained(self, worker: WorkerIdentity) -> bool:
        current = self._worker(worker)
        if not current.draining or worker.incarnation not in self._closed_at:
            return False
        if self.registry.active_leases(worker.worker_id, worker.incarnation):
            return False
        observation = self.observe(worker)
        if observation is None or observation.incarnation != worker.incarnation:
            return False
        age = self.clock() - observation.observed_at
        return (
            0 <= age <= self.freshness_seconds
            and observation.observed_at >= self._closed_at[worker.incarnation]
            and observation.pending_requests == 0
            and observation.pending_transfers == 0
        )
