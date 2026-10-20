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

"""Prefill latency estimates in milliseconds; coefficients require offline calibration.

The runtime has worker-local Device/Host cache indexes and an optional shared L3
store. L3 presence is not residency: L3CacheHooks probes and prefetches, and a
short landing becomes a shorter hit. Consequently unknown locality is a miss,
and a known remote hit retains its transfer estimate.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tokenspeed.runtime.load_balancing.types import (
    Lease,
    Worker,
    WorkerSnapshot,
    Workload,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.load_balancing.registry import WorkerRegistry

Tokens = Annotated[int, Field(ge=0, le=2**31 - 1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PrefillCostModel(BaseModel):
    """C(q,c)=a*q+b*q*c+d*q² ms. q includes replay; c excludes replay."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    linear_ms_per_token: Nonnegative
    context_ms_per_token_squared: Nonnegative
    quadratic_ms_per_token_squared: Nonnegative

    @model_validator(mode="after")
    def positive_model(self) -> PrefillCostModel:
        if not any(self.coefficients):
            raise ValueError("at least one cost coefficient must be positive")
        return self

    @property
    def coefficients(self) -> tuple[float, float, float]:
        return (
            self.linear_ms_per_token,
            self.context_ms_per_token_squared,
            self.quadratic_ms_per_token_squared,
        )

    def predict_ms(self, uncached_tokens: int, cached_prefix_tokens: int) -> float:
        """Predict work, not a measured speedup or a service guarantee."""
        _tokens(uncached_tokens)
        _tokens(cached_prefix_tokens)
        a, b, d = self.coefficients
        q, c = uncached_tokens, cached_prefix_tokens
        result = a * q + b * q * c + d * q * q
        if not math.isfinite(result):
            raise ValueError("prefill cost overflow")
        return result

    def tokens_for_budget(
        self, *, budget_ms: float, cached_prefix_tokens: int, token_cap: int
    ) -> int:
        """Largest integer q within budget and cap; zero means no feasible work.

        Native admission additionally enforces cache/replay quanta and grants its
        minimum progress quantum when an otherwise empty iteration cannot fit.
        This estimator never hides that overrun or treats replay tokens as free.
        """
        _finite_nonnegative(budget_ms, "budget_ms")
        _tokens(cached_prefix_tokens)
        _tokens(token_cap)
        low, high = 0, token_cap
        while low < high:
            middle = (low + high + 1) // 2
            if self.predict_ms(middle, cached_prefix_tokens) <= budget_ms:
                low = middle
            else:
                high = middle - 1
        return low


class PrefillObservation(BaseModel):
    """One measured prefill duration; same worker/model/execution configuration."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    uncached_tokens: Annotated[int, Field(gt=0, le=2**31 - 1)]
    cached_prefix_tokens: Tokens
    elapsed_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PrefillFit(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    cost_model: PrefillCostModel
    training_samples: int
    heldout_samples: int
    heldout_residuals_ms: tuple[float, ...]
    heldout_mean_absolute_error_ms: Nonnegative
    heldout_root_mean_squared_error_ms: Nonnegative
    heldout_max_absolute_error_ms: Nonnegative


def fit_prefill_cost(
    training: Sequence[PrefillObservation], heldout: Sequence[PrefillObservation]
) -> PrefillFit:
    """Fit nonnegative least squares offline, then report untouched holdout errors.

    Three columns permit exact active-set enumeration without a runtime SciPy
    dependency. Full-rank training is required to identify all coefficients.
    Callers must separate workloads/measurements; no synthetic calibration is
    supplied by this module. Residuals are predicted minus observed milliseconds.
    """
    import numpy as np

    if len(training) < 3 or not heldout:
        raise ValueError("need at least three training samples and a heldout set")
    features = np.asarray(
        [
            [
                x.uncached_tokens,
                x.uncached_tokens * x.cached_prefix_tokens,
                x.uncached_tokens**2,
            ]
            for x in training
        ],
        dtype=np.float64,
    )
    targets = np.asarray([x.elapsed_ms for x in training], dtype=np.float64)
    scales = np.maximum(np.linalg.norm(features, axis=0), 1.0)
    normalized = features / scales
    if np.linalg.matrix_rank(normalized) != 3:
        raise ValueError("training samples cannot identify three cost coefficients")
    best = np.zeros(3)
    best_error = float(targets @ targets)
    for size in range(1, 4):
        for subset in itertools.combinations(range(3), size):
            active, *_ = np.linalg.lstsq(normalized[:, subset], targets, rcond=None)
            if np.any(active < 0):
                continue
            coefficients = np.zeros(3)
            coefficients[list(subset)] = active
            residual = normalized @ coefficients - targets
            error = float(residual @ residual)
            if error < best_error:
                best, best_error = coefficients, error
    a, b, d = (float(x) for x in best / scales)
    model = PrefillCostModel(
        linear_ms_per_token=a,
        context_ms_per_token_squared=b,
        quadratic_ms_per_token_squared=d,
    )
    residuals = tuple(
        model.predict_ms(x.uncached_tokens, x.cached_prefix_tokens) - x.elapsed_ms
        for x in heldout
    )
    absolute = [abs(x) for x in residuals]
    return PrefillFit(
        cost_model=model,
        training_samples=len(training),
        heldout_samples=len(heldout),
        heldout_residuals_ms=residuals,
        heldout_mean_absolute_error_ms=sum(absolute) / len(absolute),
        heldout_root_mean_squared_error_ms=math.hypot(*residuals)
        / math.sqrt(len(residuals)),
        heldout_max_absolute_error_ms=max(absolute),
    )


class PrefixAvailability(BaseModel):
    """Request-specific, trusted evidence for one worker incarnation.

    Unknown evidence must have zero tokens and zero transfer cost. A shared
    storage hit is remote until the worker reports it resident. Storage probe
    results are estimates and do not bypass scheduler admission or landing.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    kind: Literal["unknown", "miss", "resident", "remote"]
    cached_prefix_tokens: Tokens
    transfer_ms: Nonnegative
    observed_at: Nonnegative
    worker_incarnation: str

    @model_validator(mode="after")
    def valid_hit(self) -> PrefixAvailability:
        if self.kind in ("unknown", "miss") and (
            self.cached_prefix_tokens or self.transfer_ms
        ):
            raise ValueError("unknown cache and misses cannot claim cached work")
        if self.kind == "resident" and self.transfer_ms:
            raise ValueError("resident cache has no transfer cost")
        return self


def _tokens(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 2**31 - 1:
        raise ValueError("token counts must be nonnegative int32 integers")


def _finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


class WorkerPrefix(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    worker_id: str
    availability: PrefixAvailability


class PrefillRequest(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    prompt_tokens: Annotated[int, Field(gt=0, le=2**31 - 1)]
    replay_tokens: Tokens
    queued_at: Nonnegative
    prefixes: tuple[WorkerPrefix, ...] = ()

    @model_validator(mode="after")
    def unique_workers(self) -> PrefillRequest:
        ids = [entry.worker_id for entry in self.prefixes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate worker prefix evidence")
        if any(
            entry.availability.cached_prefix_tokens > self.prompt_tokens
            for entry in self.prefixes
        ):
            raise ValueError("cached prefix cannot exceed prompt length")
        return self


class PrefillPolicy(BaseModel):
    """Atomically route/reserve predicted milliseconds on a complete P worker.

    Health, telemetry freshness, incarnation and multidimensional admission are
    enforced by WorkerRegistry under its lock. Cache evidence has an independent
    TTL. Context tiers are preferences, relaxed after max_tier_wait_seconds;
    worker context_limit remains a hard bound even after that wait.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    cost_model: PrefillCostModel
    cache_max_age_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    context_tiers: tuple[Annotated[int, Field(gt=0, le=2**31 - 1)], ...]
    max_tier_wait_seconds: Nonnegative

    @model_validator(mode="after")
    def ordered_tiers(self) -> PrefillPolicy:
        if tuple(sorted(set(self.context_tiers))) != self.context_tiers:
            raise ValueError("context tiers must be strictly increasing")
        return self

    def predicted_work_ms(
        self, request: PrefillRequest, worker: Worker, *, now: float
    ) -> float:
        """Use only fresh, incarnation-matched request-prefix evidence."""
        _finite_nonnegative(now, "now")
        cached, transfer = 0, 0.0
        for entry in request.prefixes:
            evidence = entry.availability
            if (
                entry.worker_id == worker.worker_id
                and evidence.worker_incarnation == worker.incarnation
                and 0 <= now - evidence.observed_at <= self.cache_max_age_seconds
            ):
                cached = evidence.cached_prefix_tokens
                transfer = evidence.transfer_ms
                break
        # Even a complete prefix hit needs a final prompt token for logits.
        replay = min(cached, max(1, request.replay_tokens))
        retained = cached - replay
        work = (
            self.cost_model.predict_ms(request.prompt_tokens - retained, retained)
            + transfer
        )
        if not math.isfinite(work):
            raise ValueError("prefill work including transfer overflow")
        return work

    def workload(
        self, request: PrefillRequest, snapshot: WorkerSnapshot, *, now: float
    ) -> Workload:
        return Workload(
            requests=1,
            prefill_work=self.predicted_work_ms(request, snapshot.worker, now=now),
        )

    def score(
        self, request: PrefillRequest, snapshot: WorkerSnapshot, *, now: float
    ) -> float:
        _finite_nonnegative(now, "now")
        if request.queued_at > now:
            raise ValueError("request queue time is in the future")
        preferred = next(
            (tier for tier in self.context_tiers if request.prompt_tokens <= tier), None
        )
        worker = snapshot.worker
        if (
            worker.context_limit is not None
            and request.prompt_tokens > worker.context_limit
        ):
            return math.inf
        if (
            preferred is not None
            and now - request.queued_at < self.max_tier_wait_seconds
            and worker.context_limit != preferred
        ):
            return math.inf
        # Both measured remaining work and pending leases are milliseconds.
        return (
            snapshot.total.prefill_work
            + self.workload(request, snapshot, now=now).prefill_work
        )

    def reserve(
        self,
        registry: WorkerRegistry,
        *,
        request_id: str,
        request: PrefillRequest,
        now: float,
    ) -> Lease:
        """Reserve work before dispatch; release/update through the registry lease."""
        _finite_nonnegative(now, "now")
        if request.queued_at > now:
            raise ValueError("request queue time is in the future")
        return registry.reserve(
            request_id=request_id,
            role="prefill",
            score=lambda snapshot: self.score(request, snapshot, now=now),
            workload=lambda snapshot: self.workload(request, snapshot, now=now),
            now=now,
        )
