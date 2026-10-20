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

"""Trace mechanism counters and explicitly authorized paired GPU microbenchmarks.

CPU reports count supplied observations. They neither run the scheduler nor
predict latency. GPU reports time actual runtime entrypoints, including their
collectives, and are not evidence of serving goodput or graph qualification.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from tokenspeed.deployment.deepseek_v41.config import StrictConfig

Nonnegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]


class ExpertBatchObservation(StrictConfig):
    # IDs include request generation; eligible means capacity/admission eligible.
    eligible: list[str]
    selected: list[str]
    predicted_experts: dict[str, list[Nonnegative]]
    actual_experts: dict[str, list[Nonnegative]]

    @model_validator(mode="after")
    def validate_membership(self):
        for ids in (self.eligible, self.selected):
            if len(ids) != len(set(ids)) or any(not item for item in ids):
                raise ValueError("request generation IDs must be nonempty and unique")
        if not set(self.selected) <= set(self.eligible):
            raise ValueError("selected requests must be admission eligible")
        if not set(self.predicted_experts) <= set(self.eligible):
            raise ValueError("predictions must refer to eligible requests")
        if not set(self.actual_experts) <= set(self.selected):
            raise ValueError("actual routes must refer to selected requests")
        return self


class ExpertScreen(StrictConfig):
    # Scheduler priority-promotion threshold, not a universal service bound.
    max_deferrals: Positive
    baseline: list[ExpertBatchObservation]
    candidate: list[ExpertBatchObservation]


class EngramScreen(StrictConfig):
    layer_id: Nonnegative
    model_revision: str = Field(min_length=1)
    row_start: Nonnegative
    row_end: Positive
    embedding_dim: Literal[256]
    capacity: Nonnegative
    hot_global_ids: list[Nonnegative]
    lookup_global_ids: list[Nonnegative]

    @model_validator(mode="after")
    def validate_shard(self):
        if self.row_end <= self.row_start:
            raise ValueError("Engram shard must contain at least one row")
        if len(set(self.hot_global_ids)) != len(self.hot_global_ids):
            raise ValueError("hot IDs must be unique")
        if len(self.hot_global_ids) > self.capacity:
            raise ValueError("hot IDs exceed cache capacity")
        if any(not self.row_start <= row < self.row_end for row in self.hot_global_ids):
            raise ValueError("hot IDs must be owned by the reported shard")
        return self


class DSparkScreen(StrictConfig):
    rows: Nonnegative
    vocabulary_tiles: Positive
    block_size: Positive
    tp_size: Positive


class MechanismScreen(StrictConfig):
    evidence: Literal["synthetic", "recorded"]
    source: str = Field(min_length=1)
    experts: ExpertScreen
    engram: EngramScreen
    dspark: DSparkScreen


def _expert_counters(batches: list[ExpertBatchObservation], max_deferrals: int) -> dict:
    deferred: dict[str, int] = {}
    maximum = 0
    exceedances: set[str] = set()
    rows = []
    for batch in batches:
        selected = set(batch.selected)
        # An absent generation is no longer eligible; it accrues no deferrals.
        deferred = {
            key: value for key, value in deferred.items() if key in batch.eligible
        }
        for key in batch.eligible:
            deferred[key] = 0 if key in selected else deferred.get(key, 0) + 1
            maximum = max(maximum, deferred[key])
            if deferred[key] > max_deferrals:
                exceedances.add(key)
        row = {"selected": len(selected), "eligible": len(batch.eligible)}
        for name, routes in (
            ("predicted", batch.predicted_experts),
            ("actual", batch.actual_experts),
        ):
            known = selected.intersection(routes)
            union = {expert for key in known for expert in routes[key]}
            row[f"{name}_known_requests"] = len(known)
            row[f"{name}_known_union"] = len(union)
            # Missing hints cannot be scored as a zero-expert request.
            row[f"{name}_complete_union"] = len(union) if known == selected else None
        rows.append(row)
    return {
        "batches": rows,
        "selected_total": sum(len(batch.selected) for batch in batches),
        "maximum_consecutive_eligible_deferrals": maximum,
        "deferral_promotion_threshold": max_deferrals,
        "deferral_threshold_exceedances": sorted(exceedances),
        "end_of_trace_pending": dict(
            sorted((key, value) for key, value in deferred.items() if value)
        ),
    }


def dspark_payload_counts(config: DSparkScreen) -> dict:
    """Count all-gather tensor payloads, not estimated network wire traffic."""
    calls = config.block_size if config.tp_size > 1 and config.rows > 0 else 0
    baseline = config.rows * config.vocabulary_tiles * 8 * calls
    candidate = config.rows * 8 * calls
    return {
        "dtype": "int64",
        "collective": "all_gather",
        "collective_calls": calls,
        "baseline_per_rank_input_bytes": baseline,
        "candidate_per_rank_input_bytes": candidate,
        "baseline_per_rank_output_bytes": baseline * config.tp_size,
        "candidate_per_rank_output_bytes": candidate * config.tp_size,
        "candidate_local_reduction_elements": config.rows
        * config.vocabulary_tiles
        * config.block_size,
        "network_wire_bytes": None,
    }


def run_mechanism_screen(raw: dict) -> dict:
    """Validate a JSON-compatible trace and return counters without GPU access.

    Expert keys are layer-qualified integers (layer_id << 32 | expert_id).
    Request IDs must include generation, and supplied predictions must come
    only from earlier completed work. No trace provenance is inferred here.
    """
    config = MechanismScreen.model_validate(raw)
    engram = config.engram
    hot = set(engram.hot_global_ids)
    local = [
        row
        for row in engram.lookup_global_ids
        if engram.row_start <= row < engram.row_end
    ]
    hits = sum(row in hot for row in local)
    row_bytes = engram.embedding_dim + engram.embedding_dim // 32
    same_batch_count = len(config.experts.baseline) == len(config.experts.candidate)
    paired = list(zip(config.experts.baseline, config.experts.candidate))
    comparison = {
        "batch_counts_match": same_batch_count,
        "paired_eligible_sets_match": same_batch_count
        and all(
            set(base.eligible) == set(candidate.eligible) for base, candidate in paired
        ),
        "paired_selected_counts_match": same_batch_count
        and all(
            len(base.selected) == len(candidate.selected) for base, candidate in paired
        ),
    }
    return {
        "status": "cpu_mechanism_screen_only",
        "evidence": config.evidence,
        "source": config.source,
        "configuration": config.model_dump(),
        "hardware": "CPU counters; no model or scheduler execution",
        "baseline": "FIFO observations / uncached host rows / tiled candidate all-gather",
        "candidate": "expert-affinity observations / static hot rows / compact candidate all-gather",
        "experts": {
            "comparison": comparison,
            "baseline": _expert_counters(
                config.experts.baseline, config.experts.max_deferrals
            ),
            "candidate": _expert_counters(
                config.experts.candidate, config.experts.max_deferrals
            ),
        },
        "engram": {
            "dtype": "FP8 E4M3 codes + uint8 E8M0 scales; BF16 output",
            "local_lookups": len(local),
            "nonlocal_zero_lookups": len(engram.lookup_global_ids) - len(local),
            "hits": hits,
            "misses": len(local) - hits,
            "hit_rate": hits / len(local) if local else None,
            "logical_baseline_host_row_bytes": len(local) * row_bytes,
            "logical_candidate_host_row_bytes": (len(local) - hits) * row_bytes,
            "resident_hot_bytes": len(hot) * (row_bytes + 8),
        },
        "dspark": dspark_payload_counts(config.dspark),
        "latency_ms": None,
        "speedup": None,
        "limitations": [
            "Counters do not establish GPU correctness, latency, HBM traffic or serving goodput.",
            "Deferral exceedances are observations beyond the promotion threshold, not fairness violations; backlog can delay promoted requests.",
            "Deferrals cover supplied eligible observations only; pending requests are censored at trace end.",
            "Expert comparisons require matched load and admission capacity; equal observed counts alone do not establish these.",
            "Engram bytes are logical accesses, excluding transaction amplification and staging.",
            "All-gather payloads exclude transport topology and protocol overhead.",
        ],
    }


class GPUExperiment(StrictConfig):
    allow_gpu: bool
    warmup: Positive
    repetitions: Annotated[int, Field(ge=2)]
    source: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)


def _require_gpu(config: GPUExperiment) -> None:
    if not config.allow_gpu:
        raise ValueError("GPU measurement requires explicit allow_gpu=True")


def _paired_gpu(
    baseline, candidate, *, device, group, config: GPUExperiment, metadata: dict
) -> dict:
    """Synchronized complete-path wall time, worst rank, alternating pair order."""
    _require_gpu(config)
    import statistics
    import time

    import torch
    import torch.distributed as dist

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("paired GPU measurements require a CUDA device")
    distributed = len(group) > 1
    if distributed and not dist.is_initialized():
        raise ValueError("the supplied process group must be initialized")

    process_group = None
    if distributed:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        process_group = process_group_manager.get_device_process_group(group)

    def parity():
        reference = baseline().clone()
        result = candidate()
        torch.cuda.synchronize(device)
        # All ranks agree before anyone proceeds to the next collective.
        matches = torch.isclose(reference, result, rtol=0, atol=0, equal_nan=True).all()
        flag = matches.to(dtype=torch.int32).reshape(1)
        if distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=process_group)
        if not bool(flag.item()):
            raise AssertionError("paired GPU outputs differ on at least one rank")

    with torch.inference_mode():
        parity()
        for _ in range(config.warmup):
            baseline()
            candidate()
        torch.cuda.synchronize(device)
        samples = {"baseline": [], "candidate": []}
        operations = {"baseline": baseline, "candidate": candidate}
        for pair in range(config.repetitions):
            order = (
                ("baseline", "candidate")
                if pair % 2 == 0
                else ("candidate", "baseline")
            )
            for name in order:
                if distributed:
                    dist.barrier(group=process_group)
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                operations[name]()
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if distributed:
                    # Outside the timed region; report the slowest participating rank.
                    elapsed = torch.tensor(
                        elapsed_ms, dtype=torch.float64, device=device
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=process_group)
                    elapsed_ms = elapsed.item()
                samples[name].append(elapsed_ms)
        parity()
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "status": "gpu_paired_microbenchmark_measured",
        "configuration": config.model_dump(),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "world_size": len(group),
        },
        "measurement": metadata,
        "timing": "synchronized wall clock including launches and collectives; worst participating rank",
        "samples_ms": samples,
        "median_ms": medians,
        "paired_baseline_over_candidate": [
            a / b
            for a, b in zip(samples["baseline"], samples["candidate"], strict=True)
        ],
        "parity": "exact values (NaNs equal), before warmup and after timing",
        "limitations": [
            "Eager microbenchmark only; no CUDA graph or concurrent-request qualification.",
            "Repeated inputs warm device and host caches; this is not a cold-system measurement.",
            "No end-to-end serving speedup, quality or service-level claim follows from this result.",
        ],
    }


def benchmark_dspark_gpu(
    local_base_logits,
    bonus_token_ids,
    markov_head,
    lm_head,
    tp_group,
    *,
    config: GPUExperiment,
) -> dict:
    """Pair the complete tiled and compact greedy block sampler on real tensors.

    The caller initializes the CUDA device and tensor-parallel group and loads
    the same BF16 Markov head for both runs. Scoring, local compact reduction,
    every block-step all-gather, and final resolution are inside the timer.
    Workspace allocation and compilation/warmup are outside it. All group
    ranks must enter this function with matching shapes and configuration.
    """
    _require_gpu(config)
    import torch

    from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
        dspark_greedy_workspace,
        sample_dspark_block_greedy,
    )

    if local_base_logits.ndim != 3 or local_base_logits.dtype != torch.float32:
        raise ValueError(
            "DSpark benchmark logits must be [rows, block, local_vocab] FP32"
        )
    if (
        markov_head.embedding.weight.dtype != torch.bfloat16
        or markov_head.projection.weight.dtype != torch.bfloat16
    ):
        raise ValueError("DSpark benchmark requires BF16 Markov weights")
    rows, block, local_vocab = local_base_logits.shape
    if min(rows, block, local_vocab) < 1:
        raise ValueError("DSpark benchmark requires nonempty shapes")
    tp_size = len(tp_group)
    if lm_head.tp_size != tp_size:
        raise ValueError("DSpark head and benchmark process group sizes differ")
    operations = {}
    for mode in ("tiles", "compact"):
        candidates, partials, compact = dspark_greedy_workspace(
            tp_size,
            rows,
            local_vocab,
            local_base_logits.device,
            candidate_communication=mode,
        )
        output = torch.empty(
            (rows, block), dtype=torch.int32, device=local_base_logits.device
        )

        def operation(
            mode=mode,
            candidates=candidates,
            partials=partials,
            compact=compact,
            output=output,
        ):
            return sample_dspark_block_greedy(
                local_base_logits,
                bonus_token_ids,
                markov_head,
                lm_head,
                tp_group,
                candidates,
                partials,
                output,
                candidate_communication=mode,
                compact_partials=compact,
            )

        operations[mode] = operation
    return _paired_gpu(
        operations["tiles"],
        operations["compact"],
        device=local_base_logits.device,
        group=tp_group,
        config=config,
        metadata={
            "mechanism": "DSpark complete greedy block sampler",
            "baseline": "tiles",
            "candidate": "compact",
            "logit_shape": list(local_base_logits.shape),
            "logit_dtype": str(local_base_logits.dtype),
            "embedding_shape": list(markov_head.embedding.weight.shape),
            "projection_shape": list(markov_head.projection.weight.shape),
            "weight_dtype": "torch.bfloat16",
            "candidate_dtype": "torch.int64",
            "includes": [
                "FP32 scoring",
                "local candidate reduction",
                "all-gather",
                "resolve",
            ],
        },
    )


def benchmark_engram_gpu(
    baseline_embedding, candidate_embedding, indices, *, config: GPUExperiment
) -> dict:
    """Pair real embedding forwards, including UVA misses and shard all-reduce.

    Both embeddings must share canonical registered CPU table storage. The
    candidate's immutable hot set must already be loaded. Allocation, lookup,
    dequantization, shard masking and the actual runtime collective are timed.
    Pass representative indices containing misses; hit-only inputs are rejected
    because they cannot measure the complete host fallback path.
    """
    _require_gpu(config)
    import torch

    baseline, candidate = baseline_embedding, candidate_embedding
    if not baseline.host_table or not candidate.host_table:
        raise ValueError("Engram comparison requires canonical host tables")
    if baseline.hot_cache_runtime is not None or candidate.hot_cache_runtime is None:
        raise ValueError(
            "Engram comparison requires uncached baseline and cached candidate"
        )
    if (
        baseline.layer_id,
        baseline.row_start,
        baseline.row_end,
        baseline.host_layout,
        baseline.tp_group,
    ) != (
        candidate.layer_id,
        candidate.row_start,
        candidate.row_end,
        candidate.host_layout,
        candidate.tp_group,
    ):
        raise ValueError("Engram baseline and candidate ownership must match")
    if (
        baseline.weight.data_ptr() != candidate.weight.data_ptr()
        or baseline.scale.data_ptr() != candidate.scale.data_ptr()
    ):
        raise ValueError(
            "Engram baseline and candidate must share canonical table storage"
        )
    if not indices.is_cuda or indices.numel() == 0:
        raise ValueError("Engram benchmark requires nonempty CUDA indices")
    cache = candidate.hot_cache_runtime.require_ready()
    plan = candidate.hot_cache_runtime.plan
    if plan.model_revision != config.model_revision:
        raise ValueError("Engram benchmark revision differs from the hot set")
    local = (indices >= candidate.row_start) & (indices < candidate.row_end)
    hits = local & torch.isin(indices - candidate.row_start, cache.local_row_ids)
    local_count, hit_count = int(local.sum().item()), int(hits.sum().item())
    global_counts = torch.tensor(
        [
            local_count,
            hit_count,
            len(plan.local_row_ids),
            plan.resident_nbytes,
            plan.staging_nbytes,
        ],
        device=indices.device,
        dtype=torch.int64,
    )
    if len(candidate.tp_group) > 1:
        import torch.distributed as dist

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        dist.all_reduce(
            global_counts,
            op=dist.ReduceOp.SUM,
            group=process_group_manager.get_device_process_group(candidate.tp_group),
        )
    global_local, global_hits, global_hot_rows, global_resident, global_staging = (
        global_counts.tolist()
    )
    if global_local == global_hits:
        raise ValueError(
            "Engram complete-path benchmark needs at least one local miss across the measured group"
        )
    return _paired_gpu(
        lambda: baseline(indices),
        lambda: candidate(indices),
        device=indices.device,
        group=candidate.tp_group,
        config=config,
        metadata={
            "mechanism": "Engram complete embedding forward",
            "baseline": "canonical host gather/dequantize",
            "candidate": "fused immutable hot cache with canonical host misses",
            "index_shape": list(indices.shape),
            "index_dtype": str(indices.dtype),
            "table_shape": list(candidate.weight.shape),
            "code_dtype": str(candidate.weight.dtype),
            "scale_dtype": str(candidate.scale.dtype),
            "output_dtype": "torch.bfloat16",
            "layout": candidate.host_layout,
            "layer_id": candidate.layer_id,
            "row_start": candidate.row_start,
            "row_end": candidate.row_end,
            "rank_hot_rows": len(plan.local_row_ids),
            "rank_resident_bytes": plan.resident_nbytes,
            "rank_staging_bytes_bound": plan.staging_nbytes,
            "group_hot_row_copies": global_hot_rows,
            "group_resident_bytes": global_resident,
            "group_staging_bytes_bound": global_staging,
            "group_counter_scope": "sum of rank-local accesses and memory; shared-table replicas count separately",
            "local_lookups": local_count,
            "hits": hit_count,
            "misses": local_count - hit_count,
            "group_local_lookups": global_local,
            "group_hits": global_hits,
            "group_misses": global_local - global_hits,
            "includes": [
                "lookup",
                "UVA misses",
                "dequantization",
                "mask",
                "runtime shard all-reduce when sharded",
            ],
        },
    )
