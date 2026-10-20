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

"""CPU experiment analysis. Estimates and replay results never establish GPU gains."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import OrderedDict
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from tokenspeed.deployment.deepseek_v41.config import StrictConfig

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Outcome(StrictConfig):
    task_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    admitted: bool
    success: bool
    attempts: Annotated[int, Field(ge=1)]
    committed_tokens: Annotated[int, Field(ge=0)]
    latency_ms: Nonnegative
    ttft_ms: Nonnegative | None

    @model_validator(mode="after")
    def validate_outcome(self):
        if not self.admitted and (
            self.success or self.committed_tokens or self.ttft_ms is not None
        ):
            raise ValueError(
                "rejected tasks cannot succeed, produce tokens or have TTFT"
            )
        if self.ttft_ms is not None and self.ttft_ms > self.latency_ms:
            raise ValueError("TTFT exceeds task latency")
        return self


class ResourceCost(StrictConfig):
    role: Literal[
        "aggregate", "encode", "prefill", "decode", "host", "network", "storage"
    ]
    cost_usd: Nonnegative
    allocated_gpus: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_cost(self):
        compute = self.role in ("aggregate", "encode", "prefill", "decode")
        if compute != (self.allocated_gpus > 0):
            raise ValueError("GPU allocations must belong to compute roles")
        return self


class Experiment(StrictConfig):
    workload_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sampling_id: str = Field(min_length=1)
    mode: Literal["aggregate", "pd", "epd"]
    duration_seconds: PositiveFloat
    offered_task_ids: list[str] = Field(min_length=1)
    resources: list[ResourceCost]
    outcomes: list[Outcome]

    @model_validator(mode="after")
    def validate_population(self):
        offered = set(self.offered_task_ids)
        actual = [row.task_id for row in self.outcomes]
        if len(offered) != len(self.offered_task_ids) or len(set(actual)) != len(
            actual
        ):
            raise ValueError(
                "task IDs must be unique; aggregate retries in one outcome"
            )
        if offered != set(actual):
            raise ValueError(
                "every offered task, including rejection/failure, needs an outcome"
            )
        roles = [resource.role for resource in self.resources]
        compute = {
            "aggregate": {"aggregate"},
            "pd": {"prefill", "decode"},
            "epd": {"encode", "prefill", "decode"},
        }[self.mode]
        if len(set(roles)) != len(roles) or set(roles) != compute | {
            "host",
            "network",
            "storage",
        }:
            raise ValueError(
                "itemize every compute role plus host, network and storage exactly once"
            )
        if sum(resource.cost_usd for resource in self.resources) <= 0:
            raise ValueError("total experiment cost must be positive")
        return self


def summarize(experiment: Experiment) -> dict:
    count = len(experiment.outcomes)
    success = sum(row.success for row in experiment.outcomes)
    cost = sum(item.cost_usd for item in experiment.resources)
    gpus = sum(item.allocated_gpus for item in experiment.resources)
    ttft = [row.ttft_ms for row in experiment.outcomes if row.ttft_ms is not None]
    latencies = [row.latency_ms for row in experiment.outcomes]
    admitted_latencies = [row.latency_ms for row in experiment.outcomes if row.admitted]
    rejected = sum(not row.admitted for row in experiment.outcomes)
    return {
        "offered_tasks": count,
        "successful_tasks": success,
        "rejected_tasks": rejected,
        "rejection_rate": rejected / count,
        "attempts": sum(row.attempts for row in experiment.outcomes),
        "success_rate": success / count,
        "total_cost_usd": cost,
        "allocated_gpus": gpus,
        "cost_per_success_usd": cost / success if success else None,
        "successful_tasks_per_gpu": success / gpus,
        "successful_tasks_per_gpu_second": success
        / (gpus * experiment.duration_seconds),
        "committed_tokens_per_second": sum(
            row.committed_tokens for row in experiment.outcomes
        )
        / experiment.duration_seconds,
        "latency_p50_ms": float(np.quantile(latencies, 0.5)),
        "latency_p99_ms": float(np.quantile(latencies, 0.99)),
        "admitted_latency_p99_ms": (
            float(np.quantile(admitted_latencies, 0.99)) if admitted_latencies else None
        ),
        "admitted_latency_observations": len(admitted_latencies),
        "ttft_p99_ms": float(np.quantile(ttft, 0.99)) if ttft else None,
        "ttft_observations": len(ttft),
    }


def compare(
    baseline: Experiment,
    candidate: Experiment,
    *,
    margin: float,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict:
    """Paired cluster bootstrap plus a conservative finite-sample quality bound.

    The bound treats independent clusters as the independent observations and
    weights them by their offered-task counts. Unlike a percentile bootstrap,
    it retains uncertainty when every observed pair agrees. It is deliberately
    conservative; neither bound proves hardware correctness or service parity.
    Costs are conditioned on the observed total allocations: no cost variability
    across runs is estimated. Independent clusters and fixed cluster sizes are
    assumptions, not properties this function can verify from identifiers.
    """
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("non-inferiority margin must be finite and in [0, 1)")
    if not 0.5 < confidence < 1 or type(resamples) is not int or resamples < 100:
        raise ValueError("confidence must be in (0.5, 1), with at least 100 resamples")
    if (baseline.workload_id, baseline.model_revision, baseline.sampling_id) != (
        candidate.workload_id,
        candidate.model_revision,
        candidate.sampling_id,
    ):
        raise ValueError("paired runs must match workload, checkpoint and sampling")
    left = {row.task_id: row for row in baseline.outcomes}
    right = {row.task_id: row for row in candidate.outcomes}
    if left.keys() != right.keys():
        raise ValueError("paired runs must contain identical offered tasks")
    clusters = {}
    for task_id, old in left.items():
        new = right[task_id]
        if old.cluster_id != new.cluster_id:
            raise ValueError("cluster assignments differ between runs")
        values = clusters.setdefault(old.cluster_id, [0, 0, 0])
        values[0] += 1
        values[1] += old.success
        values[2] += new.success
    data = np.asarray(list(clusters.values()), dtype=np.float64)
    n, old_success, new_success = data.sum(axis=0)
    difference = (new_success - old_success) / n
    alpha = 1 - confidence
    # Cluster i's contribution lies in [-n_i/N, n_i/N]. Hoeffding's bound
    # applies to the sum, without treating correlated tasks as independent.
    squared_weights = float(np.square(data[:, 0] / n).sum())
    half_width = math.sqrt(2 * squared_weights * math.log(1 / alpha))
    conservative_lower = max(-1.0, difference - half_width)
    rng = np.random.default_rng(seed)
    differences, ratios = [], []
    old_summary, new_summary = summarize(baseline), summarize(candidate)
    cost_ratio = new_summary["total_cost_usd"] / old_summary["total_cost_usd"]
    for _ in range(resamples):
        sample = data[rng.integers(0, len(data), size=len(data))].sum(axis=0)
        differences.append((sample[2] - sample[1]) / sample[0])
        ratios.append(
            cost_ratio * sample[1] / sample[2] if sample[2] and sample[1] else math.inf
        )
    bootstrap_lower = float(np.quantile(differences, alpha))
    # Use an order statistic so infinities cannot produce NaN during interpolation.
    bootstrap_ratio_upper = sorted(ratios)[
        min(resamples - 1, math.ceil(confidence * resamples) - 1)
    ]
    # Simultaneous one-sided success-rate bounds retain unseen-disagreement
    # uncertainty even when bootstrap resamples all have the same outcomes.
    rate_error = math.sqrt(0.5 * squared_weights * math.log(2 / alpha))
    old_rate_upper = min(1.0, old_success / n + rate_error)
    new_rate_lower = max(0.0, new_success / n - rate_error)
    conservative_ratio_upper = (
        cost_ratio * old_rate_upper / new_rate_lower if new_rate_lower else math.inf
    )
    ratio_upper = max(bootstrap_ratio_upper, conservative_ratio_upper)
    lower = min(conservative_lower, bootstrap_lower)
    return {
        "baseline": old_summary,
        "candidate": new_summary,
        "independent_clusters": len(data),
        "confidence": confidence,
        "success_rate_difference": float(difference),
        "bootstrap_quality_lower": bootstrap_lower,
        "conservative_quality_lower": conservative_lower,
        "quality_lower": lower,
        "noninferiority_margin": margin,
        "quality_gate": bool(lower > -margin),
        "bootstrap_cost_ratio_upper": (
            bootstrap_ratio_upper if math.isfinite(bootstrap_ratio_upper) else None
        ),
        "conservative_cost_ratio_upper": (
            conservative_ratio_upper
            if math.isfinite(conservative_ratio_upper)
            else None
        ),
        "cost_ratio_upper": ratio_upper if math.isfinite(ratio_upper) else None,
        "cost_gate": bool(ratio_upper < 1),
        "uncertainty_scope": "independent_fixed_size_clusters_conditional_on_observed_total_costs",
        "release_approved": False,
        "remaining_gates": [
            "service constraints",
            "numerical and transfer correctness",
            "hardware evidence and provenance",
        ],
    }


class Step(StrictConfig):
    phase: Literal["encode", "prefill", "decode"]
    expert_bytes: Nonnegative
    compute_units: Nonnegative
    transfer_bytes: Nonnegative
    observed_ms: PositiveFloat

    def features(self) -> list[float]:
        return [
            1.0,
            self.expert_bytes / 1e9,
            self.compute_units / 1e9,
            self.transfer_bytes / 1e9,
        ]


def calibrate(training: list[Step], held_out: list[Step], *, tolerance: float) -> dict:
    """Fit per-phase linear screens; reject unidentified or nonphysical fits."""
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("relative-error tolerance must be finite and in (0, 1)")
    result = {}
    phases = {row.phase for row in training} | {row.phase for row in held_out}
    if not phases:
        raise ValueError("calibration requires training and held-out observations")
    for phase in sorted(phases):
        train = [row for row in training if row.phase == phase]
        test = [row for row in held_out if row.phase == phase]
        if len(train) < 4 or not test:
            raise ValueError(
                f"{phase} needs at least four training rows and held-out data"
            )
        x = np.asarray([row.features() for row in train])
        y = np.asarray([row.observed_ms for row in train])
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != x.shape[1]:
            raise ValueError(
                f"{phase} calibration features are not independently identified"
            )
        if np.any(coefficients < -1e-8):
            raise ValueError(
                f"{phase} fit has negative costs; redesign the mechanism model"
            )
        coefficients = np.maximum(coefficients, 0)
        features = np.asarray([row.features() for row in test])
        predictions = features @ coefficients
        errors = np.abs(predictions - [row.observed_ms for row in test]) / [
            row.observed_ms for row in test
        ]
        extrapolated = np.any(
            (features < x.min(axis=0)) | (features > x.max(axis=0)), axis=1
        )
        result[phase] = {
            "coefficients": coefficients.tolist(),
            "feature_units": [
                "intercept",
                "expert_GB",
                "compute_Gunits",
                "transfer_GB",
            ],
            "held_out_predictions_ms": predictions.tolist(),
            "held_out_relative_errors": errors.tolist(),
            "max_relative_error": float(errors.max()),
            "extrapolated_rows": int(extrapolated.sum()),
            "calibration_gate": bool(
                np.all(errors <= tolerance) and not np.any(extrapolated)
            ),
        }
    return {
        "status": "offline_screen_only",
        "phases": result,
        "release_approved": False,
    }


class PipelineRequest(StrictConfig):
    request_id: str = Field(min_length=1)
    arrival_ms: Nonnegative
    encode_ms: Nonnegative
    prefill_ms: Nonnegative
    decode_ms: Nonnegative


def replay(
    requests: list[PipelineRequest],
    *,
    encode_workers: int,
    prefill_workers: int,
    decode_workers: int,
) -> dict:
    """FCFS queue screen from calibrated service demands, without batching claims."""
    if any(
        type(count) is not int
        for count in (encode_workers, prefill_workers, decode_workers)
    ):
        raise ValueError("worker capacities must be integers")
    if prefill_workers < 1 or decode_workers < 1 or encode_workers < 0:
        raise ValueError("P/D need capacity and E capacity cannot be negative")
    if not encode_workers and any(row.encode_ms for row in requests):
        raise ValueError("image work requires encoder capacity")
    if len({row.request_id for row in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    arrivals = [(row.arrival_ms, index) for index, row in enumerate(requests)]
    queue_ms = {row.request_id: 0.0 for row in requests}
    for name, count in (
        ("encode", encode_workers),
        ("prefill", prefill_workers),
        ("decode", decode_workers),
    ):
        free = [0.0] * count
        completed = []
        # A downstream queue follows readiness order, not original arrival order.
        for ready, index in sorted(arrivals):
            duration = getattr(requests[index], f"{name}_ms")
            if name == "encode" and not duration:
                completed.append((ready, index))
                continue
            available = heapq.heappop(free)
            start = max(ready, available)
            queue_ms[requests[index].request_id] += start - ready
            end = start + duration
            heapq.heappush(free, end)
            completed.append((end, index))
        arrivals = completed
    return {
        "status": "FCFS_screen_no_batching_or_admission_model",
        "requests": [
            {
                "request_id": requests[i].request_id,
                "latency_ms": end - requests[i].arrival_ms,
                "queue_ms": queue_ms[requests[i].request_id],
            }
            for end, i in sorted(arrivals, key=lambda item: item[1])
        ],
        "release_approved": False,
    }


def route_union_oracle(routes: list[list[list[int]]], *, batch_size: int) -> dict:
    """Exact small-window oracle, using future routes; never an online scheduler."""
    n = len(routes)
    if (
        type(batch_size) is not int
        or not 1 <= n <= 12
        or batch_size < 1
        or n % batch_size
    ):
        raise ValueError("use 1..12 token rows divisible by the batch size")
    layers = len(routes[0])
    if layers == 0 or any(len(row) != layers for row in routes):
        raise ValueError("every row must have the same nonempty layer set")
    for row in routes:
        for support in row:
            if (
                len(support) != 6
                or len(set(support)) != 6
                or any(type(x) is not int or not 0 <= x < 384 for x in support)
            ):
                raise ValueError(
                    "each layer needs six distinct logical experts in [0, 384)"
                )

    def cost(group):
        return sum(
            len(set().union(*(routes[i][layer] for i in group)))
            for layer in range(layers)
        )

    memo = {}

    def solve(remaining):
        if not remaining:
            return 0, []
        if remaining in memo:
            return memo[remaining]
        first, *tail = remaining
        best = (math.inf, [])
        for others in itertools.combinations(tail, batch_size - 1):
            group = (first, *others)
            rest = tuple(i for i in tail if i not in others)
            subtotal, groups = solve(rest)
            proposal = cost(group) + subtotal
            if proposal < best[0]:
                best = (proposal, [list(group), *groups])
        memo[remaining] = best
        return best

    baseline = sum(cost(range(i, i + batch_size)) for i in range(0, n, batch_size))
    optimum, groups = solve(tuple(range(n)))
    return {
        "status": "future_route_oracle_not_a_speedup",
        "baseline_union_sum": baseline,
        "minimum_union_sum": optimum,
        "maximum_union_reduction": 1 - optimum / baseline,
        "groups": groups,
    }


def locality(rows: list[int], capacities: list[int]) -> dict:
    """Exact LRU hit curves; hash addresses do not imply uniform access frequency."""
    if not rows or any(type(row) is not int or row < 0 for row in rows):
        raise ValueError("provide nonnegative row IDs")
    if not capacities or any(type(n) is not int or n < 0 for n in capacities):
        raise ValueError("capacities must be nonnegative integers")
    result = {}
    for capacity in sorted(set(capacities)):
        cache, hits = OrderedDict(), 0
        for row in rows:
            if row in cache:
                hits += 1
                cache.move_to_end(row)
            elif capacity:
                cache[row] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
        result[capacity] = {
            "hits": hits,
            "misses": len(rows) - hits,
            "hit_rate": hits / len(rows),
        }
    return {
        "accesses": len(rows),
        "distinct_rows": len(set(rows)),
        "capacities": result,
    }


class CacheReuseEstimate(StrictConfig):
    """Explicit inputs for one cache hit versus recomputation, with no overlap.

    Bytes must reflect physical payloads including scales, packing and replicas.
    Costs are caller-supplied incremental costs for each path, not inferred from
    latency. Include contention in effective bandwidth and queue observations.
    """

    physical_transfer_bytes: Annotated[int, Field(ge=0)]
    effective_bandwidth_bytes_per_second: PositiveFloat
    remote_lookup_ms: Nonnegative
    remote_restore_replay_ms: Nonnegative
    remote_queue_ms: Nonnegative
    local_recompute_ms: Nonnegative
    local_queue_ms: Nonnegative
    remote_cost_usd: Nonnegative
    local_cost_usd: Nonnegative


def cache_economics(estimate: CacheReuseEstimate) -> dict:
    """Compare additive critical paths for a cache hit; never infer a token cutoff.

    This screen excludes misses, overlap, admission changes and fleet allocation
    effects. A positive saving is a hypothesis requiring workload validation,
    not evidence for adding a remote cache or reducing an allocated GPU count.
    """
    transfer_ms = (
        1000
        * estimate.physical_transfer_bytes
        / estimate.effective_bandwidth_bytes_per_second
    )
    remote_ms = (
        estimate.remote_lookup_ms
        + transfer_ms
        + estimate.remote_restore_replay_ms
        + estimate.remote_queue_ms
    )
    local_ms = estimate.local_recompute_ms + estimate.local_queue_ms
    return {
        "status": "serial_cache_hit_screen_not_workload_economics",
        "transfer_ms": transfer_ms,
        "remote_path_ms": remote_ms,
        "local_path_ms": local_ms,
        "latency_saving_ms": local_ms - remote_ms,
        "incremental_cost_saving_usd": estimate.local_cost_usd
        - estimate.remote_cost_usd,
        "remote_faster": remote_ms < local_ms,
        "remote_cheaper": estimate.remote_cost_usd < estimate.local_cost_usd,
        "release_approved": False,
    }
