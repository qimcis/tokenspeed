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

"""Bounded pool targets from measured service capacity and outstanding work.

Work and service capacity use the same caller-declared units per role. Targets
are worker counts; a TP group consumes ``gpus_per_worker`` indivisible devices.
No synthetic capacity values or demand ratios are supplied by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


def _integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: float, name: str, *, positive: bool) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


@dataclass(frozen=True)
class PoolBounds:
    minimum: int
    maximum: int
    gpus_per_worker: int

    def __post_init__(self) -> None:
        _integer(self.minimum, "minimum", 0)
        _integer(self.maximum, "maximum", self.minimum)
        _integer(self.gpus_per_worker, "gpus_per_worker", 1)


@dataclass(frozen=True)
class PoolDemand:
    """Outstanding role work and measured capacity of one complete worker."""

    queued_work: float
    reserved_work: float
    service_work_per_second: float
    observed_at: float

    def __post_init__(self) -> None:
        for name in ("queued_work", "reserved_work", "observed_at"):
            _finite(getattr(self, name), name, positive=False)
        _finite(self.service_work_per_second, "service_work_per_second", positive=True)
        _finite(self.work, "total outstanding work", positive=False)

    @property
    def work(self) -> float:
        return self.queued_work + self.reserved_work


@dataclass(frozen=True)
class PoolPlan:
    targets: dict[str, int]
    reason: str
    spare_gpus: int


class PoolAllocator:
    """Stateful hysteresis followed by a budget-safe, movement-bounded plan.

    ``max_spare_gpus`` is a target on converged plans; movement limits and stale
    evidence may temporarily leave additional unused devices. ``min_spare_gpus``
    and the total GPU budget are hard constraints on every plan.
    """

    def __init__(
        self,
        bounds: Mapping[str, PoolBounds],
        *,
        total_gpus: int,
        min_spare_gpus: int,
        max_spare_gpus: int,
        target_seconds: float,
        telemetry_ttl: float,
        hysteresis_fraction: float,
        stable_samples: int,
        cooldown_seconds: float,
        max_moved_gpus: int,
    ) -> None:
        if not bounds or set(bounds) - {"prefill", "decode", "encoder"}:
            raise ValueError("pool bounds must name prefill, decode or encoder")
        _integer(total_gpus, "total_gpus", 1)
        _integer(min_spare_gpus, "min_spare_gpus", 0)
        _integer(max_spare_gpus, "max_spare_gpus", min_spare_gpus)
        _integer(stable_samples, "stable_samples", 1)
        _integer(max_moved_gpus, "max_moved_gpus", 1)
        for name, value in (
            ("target_seconds", target_seconds),
            ("telemetry_ttl", telemetry_ttl),
        ):
            _finite(value, name, positive=True)
        _finite(cooldown_seconds, "cooldown_seconds", positive=False)
        _finite(hysteresis_fraction, "hysteresis_fraction", positive=False)
        if hysteresis_fraction >= 1 or max_spare_gpus > total_gpus:
            raise ValueError("invalid hysteresis or spare bounds")
        self.bounds = dict(sorted(bounds.items()))
        self.total_gpus = total_gpus
        self.min_spare_gpus = min_spare_gpus
        self.max_spare_gpus = max_spare_gpus
        self.target_seconds = target_seconds
        self.telemetry_ttl = telemetry_ttl
        self.hysteresis_fraction = hysteresis_fraction
        self.stable_samples = stable_samples
        self.cooldown_seconds = cooldown_seconds
        self.max_moved_gpus = max_moved_gpus
        self._candidate: dict[str, int] | None = None
        self._samples = 0
        self._last_move = -math.inf
        self._last_observation: tuple[float, ...] | None = None
        self._last_now = -math.inf
        minimum = sum(b.minimum * b.gpus_per_worker for b in bounds.values())
        maximum = sum(b.maximum * b.gpus_per_worker for b in bounds.values())
        if (
            minimum > total_gpus - min_spare_gpus
            or maximum < total_gpus - max_spare_gpus
        ):
            raise ValueError("pool bounds cannot satisfy GPU and spare budgets")
        if any(
            b.gpus_per_worker > max_moved_gpus and b.maximum > 0
            for b in bounds.values()
        ):
            raise ValueError("movement bound cannot move an indivisible worker")
        if not self._spares_feasible({r: b.minimum for r, b in self.bounds.items()}):
            raise ValueError("indivisible worker sizes cannot satisfy spare budget")

    def _spares_feasible(self, counts: Mapping[str, int]) -> bool:
        budget = self.total_gpus - self.min_spare_gpus
        reachable = {self._used(counts)}
        for role, bounds in self.bounds.items():
            reachable = {
                used + extra * bounds.gpus_per_worker
                for used in reachable
                for extra in range(
                    min(
                        bounds.maximum - counts[role],
                        (budget - used) // bounds.gpus_per_worker,
                    )
                    + 1
                )
            }
        return any(self.total_gpus - used <= self.max_spare_gpus for used in reachable)

    def _used(self, counts: Mapping[str, int]) -> int:
        return sum(counts[role] * b.gpus_per_worker for role, b in self.bounds.items())

    def plan(
        self,
        current: Mapping[str, int],
        demands: Mapping[str, PoolDemand],
        *,
        now: float,
    ) -> PoolPlan:
        """Return targets; caller supplies current desired counts including starts.

        Minima may be temporarily unmet during initial startup or worker loss.
        Restoration remains subject to the same GPU movement budget.
        """
        _finite(now, "now", positive=False)
        if now < self._last_now:
            raise ValueError("controller clock must be monotonic")
        self._last_now = now
        if set(current) != set(self.bounds) or set(demands) != set(self.bounds):
            raise ValueError(
                "counts and demand must cover exactly the configured roles"
            )
        for role, count in current.items():
            _integer(count, f"{role} count", 0)
            if count > self.bounds[role].maximum:
                raise ValueError("current count exceeds maximum")
        if self._used(current) > self.total_gpus - self.min_spare_gpus:
            raise ValueError("current pool violates GPU/spare budget")

        def result(counts: Mapping[str, int], reason: str) -> PoolPlan:
            return PoolPlan(dict(counts), reason, self.total_gpus - self._used(counts))

        if any(
            d.observed_at > now or now - d.observed_at > self.telemetry_ttl
            for d in demands.values()
        ):
            self._candidate = None
            self._samples = 0
            return result(current, "stale telemetry: hold")
        observation = tuple(demands[r].observed_at for r in self.bounds)
        if self._last_observation is not None and any(
            current_time < previous_time
            for current_time, previous_time in zip(observation, self._last_observation)
        ):
            self._candidate = None
            self._samples = 0
            return result(current, "regressed telemetry: hold")
        desired = {}
        for role, b in self.bounds.items():
            demand = demands[role]
            need = demand.work / demand.service_work_per_second / self.target_seconds
            count = current[role]
            if count >= b.minimum and count * (
                1 - self.hysteresis_fraction
            ) <= need <= count * (1 + self.hysteresis_fraction):
                desired[role] = count
            else:
                desired[role] = max(b.minimum, math.ceil(min(b.maximum, need)))
        # Allocate scarce devices to the largest residual service-time pressure.
        target = {role: b.minimum for role, b in self.bounds.items()}
        while True:
            used = self._used(target)
            eligible = [
                r
                for r, b in self.bounds.items()
                if target[r] < b.maximum
                and used + b.gpus_per_worker <= self.total_gpus - self.min_spare_gpus
                and self._spares_feasible({**target, r: target[r] + 1})
                and (
                    target[r] < desired[r]
                    or self.total_gpus - used > self.max_spare_gpus
                )
            ]
            if not eligible:
                break
            role = max(
                eligible,
                key=lambda r: (
                    demands[r].work
                    / (demands[r].service_work_per_second * (target[r] + 1)),
                    -self.bounds[r].gpus_per_worker,
                    r,
                ),
            )
            target[role] += 1
        if self.total_gpus - self._used(target) > self.max_spare_gpus:
            raise ValueError("indivisible worker sizes cannot satisfy spare budget")
        if target != self._candidate:
            self._candidate, self._samples = target, 0
        if observation != self._last_observation:
            self._samples += 1
            self._last_observation = observation
        if (
            self._samples < self.stable_samples
            or now - self._last_move < self.cooldown_seconds
        ):
            return result(current, "hysteresis: hold")
        bounded = dict(current)
        remaining = self.max_moved_gpus
        for growing in (False, True):
            for role, b in self.bounds.items():
                delta = target[role] - bounded[role]
                if (delta > 0) != growing:
                    continue
                steps = min(abs(delta), remaining // b.gpus_per_worker)
                if growing:
                    steps = min(
                        steps,
                        (self.total_gpus - self.min_spare_gpus - self._used(bounded))
                        // b.gpus_per_worker,
                    )
                bounded[role] += steps if growing else -steps
                remaining -= steps * b.gpus_per_worker
        if bounded != dict(current):
            self._last_move = now
        return result(bounded, "resize" if bounded != dict(current) else "balanced")
