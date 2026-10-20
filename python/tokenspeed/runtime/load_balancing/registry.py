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

"""Atomic admission and incarnation-scoped lease accounting."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace

from tokenspeed.runtime.load_balancing.types import (
    CompatiblePath,
    Lease,
    Worker,
    WorkerSnapshot,
    Workload,
)


class NoCapacity(RuntimeError):
    pass


class RequestFinished(RuntimeError):
    pass


class WorkerRegistry:
    def __init__(
        self,
        *,
        freshness_seconds: float,
        max_workers: int,
        max_leases: int,
        history_seconds: float,
    ):
        for value in (freshness_seconds, history_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("time bounds must be finite and positive")
        if (
            type(max_workers) is not int
            or type(max_leases) is not int
            or min(max_workers, max_leases) <= 0
        ):
            raise ValueError("registry size bounds must be positive integers")
        self.freshness_seconds = freshness_seconds
        self.history_seconds = history_seconds
        self.max_workers = max_workers
        self.max_leases = max_leases
        self._workers: dict[str, Worker] = {}
        self._leases: dict[str, Lease] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._finished: dict[tuple[str, str], float] = {}
        self._retired: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _now(now: float | None) -> float:
        value = time.monotonic() if now is None else now
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("now must be finite and nonnegative")
        return value

    def upsert(self, worker: Worker) -> None:
        with self._lock:
            old = self._workers.get(worker.worker_id)
            if (worker.worker_id, worker.incarnation) in self._retired:
                raise ValueError("retired worker incarnation cannot return")
            if old is None and len(self._workers) >= self.max_workers:
                raise NoCapacity("worker registry full")
            if old is not None:
                if old.incarnation == worker.incarnation:
                    if (
                        worker.sequence <= old.sequence
                        or worker.observed_at < old.observed_at
                    ):
                        raise ValueError("stale worker report")
                    if worker.role != old.role or worker.endpoint != old.endpoint:
                        raise ValueError(
                            "worker identity changed without a new incarnation"
                        )
                    # Telemetry cannot undo an operator's drain decision.
                    worker = replace(worker, draining=old.draining)
                else:
                    if len(self._retired) >= self.max_leases:
                        raise NoCapacity("incarnation history full; reconcile registry")
                    self._retired.add((old.worker_id, old.incarnation))
            self._workers[worker.worker_id] = worker

    def _snapshot(self, worker: Worker) -> WorkerSnapshot:
        reserved = Workload()
        for lease in self._leases.values():
            if (lease.worker_id, lease.incarnation) == (
                worker.worker_id,
                worker.incarnation,
            ):
                reserved = reserved + lease.workload
        return WorkerSnapshot(worker, reserved)

    def get(self, worker_id: str) -> WorkerSnapshot:
        with self._lock:
            return self._snapshot(self._workers[worker_id])

    def snapshot(
        self, role: str | None = None, now: float | None = None
    ) -> tuple[WorkerSnapshot, ...]:
        self._now(now)
        with self._lock:
            return tuple(
                self._snapshot(w)
                for w in sorted(self._workers.values(), key=lambda w: w.worker_id)
                if role is None or w.role == role
            )

    def set_draining(self, worker_id: str, incarnation: str, draining: bool) -> None:
        if type(draining) is not bool:
            raise ValueError("draining must be boolean")
        with self._lock:
            worker = self._workers[worker_id]
            if worker.incarnation != incarnation:
                raise ValueError("worker incarnation changed")
            self._workers[worker_id] = replace(worker, draining=draining)

    def active_leases(self, worker_id: str, incarnation: str) -> tuple[Lease, ...]:
        with self._lock:
            return tuple(
                lease
                for lease in self._leases.values()
                if (lease.worker_id, lease.incarnation) == (worker_id, incarnation)
            )

    def remove(self, worker_id: str, incarnation: str) -> None:
        with self._lock:
            worker = self._workers[worker_id]
            if worker.incarnation != incarnation:
                raise ValueError("worker incarnation changed")
            if self.active_leases(worker_id, incarnation):
                raise ValueError("worker still has active leases")
            if len(self._retired) >= self.max_leases:
                raise NoCapacity("incarnation history full; reconcile registry")
            self._retired.add((worker_id, incarnation))
            del self._workers[worker_id]

    def reserve(
        self,
        request_id: str,
        role: str,
        score: Callable[[WorkerSnapshot], float],
        workload: Workload | Callable[[WorkerSnapshot], Workload],
        now: float | None = None,
    ) -> Lease:
        now = self._now(now)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("request_id must be a 1..256 character string")
        key = (request_id, role)
        with self._lock:
            self._finished = {
                k: t
                for k, t in self._finished.items()
                if now - t < self.history_seconds
            }
            if key in self._finished:
                raise RequestFinished(
                    "request already finished within idempotency retention"
                )
            if key in self._requests:
                existing = self._leases[self._requests[key]]
                current = self._workers.get(existing.worker_id)
                if current is None or current.incarnation != existing.incarnation:
                    raise ValueError("request belongs to a retired worker incarnation")
                if not callable(workload) and workload != existing.workload:
                    raise ValueError("idempotency key reused with different work")
                return existing
            if len(self._leases) + len(self._finished) >= self.max_leases:
                raise NoCapacity("lease/history registry full")
            candidates = []
            for worker in self._workers.values():
                if (
                    worker.role != role
                    or not worker.ready
                    or not worker.healthy
                    or worker.draining
                ):
                    continue
                if not 0 <= now - worker.observed_at <= self.freshness_seconds:
                    continue
                snapshot = self._snapshot(worker)
                cost = score(snapshot)
                if math.isnan(cost):
                    raise ValueError("policy produced NaN")
                if not math.isfinite(cost):
                    continue
                amount = workload(snapshot) if callable(workload) else workload
                if not isinstance(amount, Workload):
                    raise ValueError("policy must return Workload")
                if not (snapshot.total + amount).fits(worker.capacity):
                    continue
                candidates.append((cost, worker.worker_id, worker, amount))
            if not candidates:
                raise NoCapacity("no fresh ready worker has capacity")
            _, _, worker, amount = min(candidates, key=lambda c: (c[0], c[1]))
            lease = Lease(
                uuid.uuid4().hex,
                request_id,
                worker.worker_id,
                worker.incarnation,
                amount,
                now,
            )
            self._leases[lease.lease_id] = lease
            self._requests[key] = lease.lease_id
            return lease

    def update_lease(self, lease_id: str, workload: Workload) -> Lease:
        if not isinstance(workload, Workload):
            raise ValueError("workload must be Workload")
        with self._lock:
            lease = self._leases[lease_id]
            worker = self._workers.get(lease.worker_id)
            if worker is None or worker.incarnation != lease.incarnation:
                raise ValueError("lease belongs to a retired incarnation")
            reserved = worker.reported + workload
            for other in self._leases.values():
                if other.lease_id != lease_id and (
                    other.worker_id,
                    other.incarnation,
                ) == (lease.worker_id, lease.incarnation):
                    reserved = reserved + other.workload
            if not reserved.fits(worker.capacity):
                raise NoCapacity("lease update exceeds worker capacity")
            updated = replace(lease, workload=workload)
            self._leases[lease_id] = updated
            return updated

    def release(self, lease_id: str, now: float | None = None) -> bool:
        now = self._now(now)
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return False
            key = next(
                key for key, value in self._requests.items() if value == lease_id
            )
            del self._requests[key]
            self._finished[key] = now
            return True

    def reserve_path(
        self,
        request_id: str,
        paths: tuple[CompatiblePath, ...],
        demands: dict[
            str,
            tuple[
                Callable[[WorkerSnapshot], float], Callable[[WorkerSnapshot], Workload]
            ],
        ],
        now: float | None = None,
    ) -> tuple[CompatiblePath, tuple[Lease, ...]]:
        """Choose a compatible path and reserve every stage under one lock.

        Shared physical workers retain one registry identity across frontends;
        path selection therefore sees reservations from every competing path.
        """
        now = self._now(now)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("request_id must be a 1..256 character string")
        with self._lock:
            if any(key[0] == request_id for key in self._requests):
                raise ValueError("request already has active leases")
            self._finished = {
                k: t
                for k, t in self._finished.items()
                if now - t < self.history_seconds
            }
            if any(key[0] == request_id for key in self._finished):
                raise RequestFinished("request already finished")
            if len(self._leases) + len(self._finished) + len(demands) > self.max_leases:
                raise NoCapacity("lease/history registry full")
            candidates = []
            for path in paths:
                stages = []
                roles = set()
                total_score = 0.0
                for worker_id, incarnation in path.workers:
                    worker = self._workers.get(worker_id)
                    if (
                        worker is None
                        or worker.incarnation != incarnation
                        or worker.role not in demands
                    ):
                        break
                    if worker.role in roles:
                        raise ValueError(
                            "a compatible path must contain exactly one worker per stage"
                        )
                    if (
                        not worker.ready
                        or not worker.healthy
                        or worker.draining
                        or not 0 <= now - worker.observed_at <= self.freshness_seconds
                    ):
                        break
                    score, workload = demands[worker.role]
                    snapshot = self._snapshot(worker)
                    cost = score(snapshot)
                    if math.isnan(cost):
                        raise ValueError("policy produced NaN")
                    if not math.isfinite(cost):
                        break
                    amount = workload(snapshot)
                    if not (snapshot.total + amount).fits(worker.capacity):
                        break
                    roles.add(worker.role)
                    stages.append((worker, amount))
                    total_score += cost
                else:
                    if roles == set(demands):
                        candidates.append((total_score, path.path_id, path, stages))
            if not candidates:
                raise NoCapacity("no compatible path has capacity")
            _, _, path, stages = min(candidates, key=lambda c: (c[0], c[1]))
            leases = tuple(
                Lease(
                    uuid.uuid4().hex,
                    request_id,
                    worker.worker_id,
                    worker.incarnation,
                    amount,
                    now,
                )
                for worker, amount in stages
            )
            for (worker, _), lease in zip(stages, leases):
                self._leases[lease.lease_id] = lease
                self._requests[(request_id, worker.role)] = lease.lease_id
            return path, leases
