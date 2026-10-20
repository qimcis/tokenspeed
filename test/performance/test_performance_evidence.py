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

"""Adversarial CPU accounting checks; none establishes GPU performance."""

import pytest

from tokenspeed.deployment.deepseek_v41.analysis import Experiment, summarize
from tokenspeed.deployment.deepseek_v41.performance_bench import (
    DSparkScreen,
    dspark_payload_counts,
    run_mechanism_screen,
)
from tokenspeed.deployment.deepseek_v41.workload import (
    Evaluation,
    MeasurementRun,
    join_evaluations,
)
from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotSetLayer,
    EngramHotSetManifest,
    plan_engram_hot_cache,
)


def observations():
    batch = {
        "eligible": ["first:1", "second:1"],
        "selected": ["first:1", "second:1"],
        "predicted_experts": {"first:1": [7], "second:1": [(1 << 32) | 7]},
        "actual_experts": {"first:1": [7], "second:1": [(1 << 32) | 7]},
    }
    return {
        "evidence": "synthetic",
        "source": "adversarial CPU accounting fixture",
        "experts": {"max_deferrals": 2, "baseline": [batch], "candidate": [batch]},
        "engram": {
            "layer_id": 1,
            "model_revision": "fixture",
            "row_start": 4,
            "row_end": 8,
            "embedding_dim": 256,
            "capacity": 2,
            "hot_global_ids": [4, 6],
            "lookup_global_ids": [3, 4, 4, 5, 5, 6, 8],
        },
        "dspark": {"rows": 3, "vocabulary_tiles": 8, "block_size": 4, "tp_size": 2},
    }


def test_empty_dspark_batch_has_no_collectives_or_reduction():
    counts = dspark_payload_counts(
        DSparkScreen(rows=0, vocabulary_tiles=8, block_size=4, tp_size=8)
    )
    assert counts["collective_calls"] == 0
    assert counts["candidate_local_reduction_elements"] == 0
    assert counts["baseline_per_rank_input_bytes"] == 0
    assert counts["candidate_per_rank_output_bytes"] == 0


def test_single_tile_compaction_keeps_payload_and_adds_local_reduction():
    counts = dspark_payload_counts(
        DSparkScreen(rows=3, vocabulary_tiles=1, block_size=4, tp_size=8)
    )
    assert (
        counts["baseline_per_rank_input_bytes"]
        == counts["candidate_per_rank_input_bytes"]
    )
    assert (
        counts["baseline_per_rank_output_bytes"]
        == counts["candidate_per_rank_output_bytes"]
    )
    assert counts["candidate_local_reduction_elements"] == 12
    assert counts["network_wire_bytes"] is None


def test_larger_tensor_parallel_group_is_not_normalized_out_of_received_payload():
    def counts(tp):
        return dspark_payload_counts(
            DSparkScreen(rows=3, vocabulary_tiles=8, block_size=4, tp_size=tp)
        )

    small, large = counts(2), counts(8)
    assert (
        large["candidate_per_rank_input_bytes"]
        == small["candidate_per_rank_input_bytes"]
    )
    assert (
        large["candidate_per_rank_output_bytes"]
        == 4 * small["candidate_per_rank_output_bytes"]
    )
    assert (
        large["candidate_local_reduction_elements"]
        == small["candidate_local_reduction_elements"]
    )


@pytest.mark.parametrize("evidence", ["synthetic", "recorded"])
def test_recorded_routes_still_cannot_establish_measured_speedup(evidence):
    trace = observations()
    trace["evidence"] = evidence
    result = run_mechanism_screen(trace)
    assert result["status"] == "cpu_mechanism_screen_only"
    assert result["latency_ms"] is None
    assert result["speedup"] is None
    # Matching expert numbers in different layers are separate experts.
    assert result["experts"]["candidate"]["batches"][0]["actual_complete_union"] == 2


def test_pending_never_served_request_remains_in_fairness_evidence():
    trace = observations()
    trace["experts"]["candidate"] = [
        {
            "eligible": ["never-served:1"],
            "selected": [],
            "predicted_experts": {},
            "actual_experts": {},
        }
    ] * 3
    result = run_mechanism_screen(trace)["experts"]["candidate"]
    assert result["selected_total"] == 0
    assert result["deferral_threshold_exceedances"] == ["never-served:1"]
    assert result["end_of_trace_pending"] == {"never-served:1": 3}


def test_repeated_hits_and_misses_are_logical_accesses_not_unique_rows():
    counts = run_mechanism_screen(observations())["engram"]
    assert counts["local_lookups"] == 5
    assert counts["hits"] == 3
    assert counts["misses"] == 2
    assert counts["nonlocal_zero_lookups"] == 2
    assert counts["logical_baseline_host_row_bytes"] == 5 * 264
    assert counts["logical_candidate_host_row_bytes"] == 2 * 264
    assert counts["resident_hot_bytes"] == 2 * 272


def test_shared_cache_replication_cost_exceeds_sharded_cost():
    manifest = EngramHotSetManifest(
        1, "fixture", (EngramHotSetLayer(1, 16, (0, 5, 10, 15)),)
    )

    def plan(start, end):
        return plan_engram_hot_cache(
            manifest,
            layer_id=1,
            model_revision="fixture",
            total_rows=16,
            row_start=start,
            row_end=end,
            capacity_rows=4,
            embedding_dim=256,
        )

    shared = [plan(0, 16) for _ in range(4)]
    sharded = [plan(rank * 4, (rank + 1) * 4) for rank in range(4)]
    assert sum(row.resident_nbytes for row in shared) == 4 * 4 * 272
    assert sum(row.resident_nbytes for row in sharded) == 4 * 272
    assert sum(row.staging_nbytes for row in shared) == 4 * 4 * 272


def experiment(gpus, duration):
    return Experiment.model_validate(
        {
            "workload_id": "fixed",
            "model_revision": "a" * 40,
            "sampling_id": "greedy",
            "mode": "pd",
            "duration_seconds": duration,
            "offered_task_ids": ["success", "failed", "rejected"],
            "resources": [
                {"role": "prefill", "cost_usd": 2.0, "allocated_gpus": gpus},
                {"role": "decode", "cost_usd": 3.0, "allocated_gpus": gpus},
                {"role": "host", "cost_usd": 4.0, "allocated_gpus": 0},
                {"role": "network", "cost_usd": 5.0, "allocated_gpus": 0},
                {"role": "storage", "cost_usd": 6.0, "allocated_gpus": 0},
            ],
            "outcomes": [
                {
                    "task_id": name,
                    "cluster_id": name,
                    "admitted": admitted,
                    "success": success,
                    "attempts": attempts,
                    "committed_tokens": tokens,
                    "latency_ms": 100.0,
                    "ttft_ms": ttft,
                }
                for name, admitted, success, attempts, tokens, ttft in [
                    ("success", True, True, 1, 10, 20.0),
                    ("failed", True, False, 3, 2, 20.0),
                    ("rejected", False, False, 1, 0, None),
                ]
            ],
        }
    )


def test_cost_retains_failed_attempts_all_resource_roles_and_idle_allocation():
    report = summarize(experiment(4, 10.0))
    assert report["offered_tasks"] == 3
    assert report["success_rate"] == pytest.approx(1 / 3)
    assert report["attempts"] == 5
    assert report["total_cost_usd"] == report["cost_per_success_usd"] == 20
    assert report["allocated_gpus"] == 8
    assert report["successful_tasks_per_gpu_second"] == 1 / 80
    assert summarize(experiment(8, 10.0))["successful_tasks_per_gpu_second"] == 1 / 160
    assert summarize(experiment(4, 20.0))["successful_tasks_per_gpu_second"] == 1 / 160


def test_evaluation_latency_includes_client_queue_time():
    run = MeasurementRun.model_validate(
        {
            "workload_id": "fixed",
            "model": "model",
            "sampling_id": "greedy",
            "duration_seconds": 2.0,
            "requests": [
                {
                    "task_id": "queued",
                    "cluster_id": "queued",
                    "status": "completed",
                    "http_status": 200,
                    "admitted": True,
                    "queue_ms": 1000.0,
                    "latency_ms": 100.0,
                    "ttft_ms": 25.0,
                    "completion_tokens": 2,
                    "response": {},
                    "events": [],
                    "error": None,
                }
            ],
        }
    )
    (result,) = join_evaluations(
        run,
        [
            Evaluation(
                task_id="queued",
                success=True,
                admitted=True,
                committed_tokens=2,
            )
        ],
    )
    assert result.latency_ms == 1100
    assert result.ttft_ms == 1025
