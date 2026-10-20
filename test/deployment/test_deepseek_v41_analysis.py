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

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.analysis import (
    CacheReuseEstimate,
    Experiment,
    PipelineRequest,
    Step,
    cache_economics,
    calibrate,
    compare,
    locality,
    replay,
    route_union_oracle,
    summarize,
)


def experiment(successes, *, cost=100.0, clusters=None):
    return Experiment.model_validate(
        {
            "workload_id": "held-out",
            "model_revision": "a" * 40,
            "sampling_id": "greedy",
            "mode": "aggregate",
            "duration_seconds": 10.0,
            "offered_task_ids": [str(i) for i in range(len(successes))],
            "resources": [
                {
                    "role": role,
                    "cost_usd": cost if role == "aggregate" else 0.0,
                    "allocated_gpus": 8 if role == "aggregate" else 0,
                }
                for role in ("aggregate", "host", "network", "storage")
            ],
            "outcomes": [
                {
                    "task_id": str(i),
                    "cluster_id": str(i if clusters is None else clusters[i]),
                    "admitted": True,
                    "success": success,
                    "attempts": 1,
                    "committed_tokens": 5,
                    "latency_ms": 2.0,
                    "ttft_ms": 1.0,
                }
                for i, success in enumerate(successes)
            ],
        }
    )


def test_failed_tasks_and_retries_are_not_removed_from_cost():
    data = experiment([True, False]).model_dump()
    data["outcomes"][1].update(
        admitted=False, committed_tokens=0, ttft_ms=None, attempts=2
    )
    report = summarize(Experiment.model_validate(data))
    assert report["cost_per_success_usd"] == 100
    assert report["offered_tasks"] == 2 and report["rejected_tasks"] == 1
    assert report["attempts"] == 3
    assert report["committed_tokens_per_second"] == 0.5
    assert report["ttft_observations"] == 1
    assert report["rejection_rate"] == 0.5
    assert report["admitted_latency_observations"] == 1
    assert report["successful_tasks_per_gpu_second"] == 1 / 80


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["outcomes"].pop(),
        lambda d: d["outcomes"].append(d["outcomes"][0]),
        lambda d: d["resources"].pop(),
        lambda d: d["outcomes"][0].update(admitted=False),
        lambda d: d["outcomes"][0].update(latency_ms=math.nan),
    ],
)
def test_incomplete_or_invalid_experiments_are_rejected(edit):
    data = experiment([True, False]).model_dump()
    edit(data)
    with pytest.raises(ValidationError):
        Experiment.model_validate(data)


def test_all_agree_does_not_remove_finite_sample_uncertainty():
    baseline = experiment([True] * 4)
    report = compare(
        baseline,
        experiment([True] * 4, cost=50.0),
        margin=0.01,
        confidence=0.95,
        resamples=200,
        seed=7,
    )
    assert report["bootstrap_quality_lower"] == 0
    assert report["quality_lower"] < 0
    assert not report["quality_gate"]
    assert report["bootstrap_cost_ratio_upper"] == 0.5
    assert report["conservative_cost_ratio_upper"] > 1
    assert not report["cost_gate"]
    assert not report["release_approved"]


def test_cost_saving_gate_requires_independent_success_evidence():
    options = dict(margin=0.1, confidence=0.95, resamples=100, seed=9)
    report = compare(
        experiment([True] * 1000), experiment([True] * 1000, cost=50.0), **options
    )
    assert 0.5 < report["cost_ratio_upper"] < 1
    assert report["cost_gate"]
    dependent = compare(
        experiment([True] * 1000, clusters=[0] * 1000),
        experiment([True] * 1000, cost=50.0, clusters=[0] * 1000),
        **options,
    )
    assert dependent["cost_ratio_upper"] is None
    assert not dependent["cost_gate"]


def test_cost_gates_fail_when_baseline_has_no_successes():
    report = compare(
        experiment([False] * 100),
        experiment([True] * 100, cost=50.0),
        margin=0.1,
        confidence=0.95,
        resamples=100,
        seed=9,
    )
    assert report["baseline"]["cost_per_success_usd"] is None
    assert report["cost_ratio_upper"] is None
    assert not report["cost_gate"]


def test_cluster_dependence_is_preserved():
    many = experiment([True] * 1000)
    single = experiment([True] * 1000, clusters=[0] * 1000)
    options = dict(margin=0.1, confidence=0.95, resamples=100, seed=3)
    assert compare(many, many, **options)["quality_gate"]
    assert not compare(single, single, **options)["quality_gate"]


def test_regression_and_zero_success_do_not_pass():
    report = compare(
        experiment([True] * 100),
        experiment([False] * 100),
        margin=0.1,
        confidence=0.95,
        resamples=100,
        seed=0,
    )
    assert not report["quality_gate"] and not report["cost_gate"]
    assert report["cost_ratio_upper"] is None
    assert report["candidate"]["cost_per_success_usd"] is None


def test_pairing_mismatch_fails():
    original = experiment([True, False])
    altered = original.model_dump()
    altered["outcomes"][0]["cluster_id"] = "other"
    with pytest.raises(ValueError, match="cluster"):
        compare(
            original,
            Experiment.model_validate(altered),
            margin=0.1,
            confidence=0.95,
            resamples=100,
            seed=0,
        )


def step(x, phase="decode", multiplier=1):
    return Step(
        phase=phase,
        expert_bytes=x[0] * 1e9,
        compute_units=x[1] * 1e9,
        transfer_bytes=x[2] * 1e9,
        observed_ms=float(multiplier * (1 + 2 * x[0] + 3 * x[1] + 4 * x[2])),
    )


def test_calibration_requires_held_out_evidence_and_reports_bad_fit():
    training = [
        step(x) for x in [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 2)]
    ]
    good = calibrate(training, [step((1, 1, 1))], tolerance=0.1)
    assert good["phases"]["decode"]["calibration_gate"]
    assert not good["release_approved"]
    bad = calibrate(training, [step((1, 1, 1), multiplier=2)], tolerance=0.1)
    assert not bad["phases"]["decode"]["calibration_gate"]
    outside = calibrate(training, [step((3, 0, 0))], tolerance=0.1)
    assert outside["phases"]["decode"]["extrapolated_rows"] == 1
    assert not outside["phases"]["decode"]["calibration_gate"]
    with pytest.raises(ValueError, match="identified"):
        calibrate([step((1, 1, 1))] * 4, [step((1, 1, 1))], tolerance=0.1)


def test_queue_replay_uses_downstream_readiness_and_text_bypasses_encode():
    rows = [
        PipelineRequest(
            request_id="image",
            arrival_ms=0.0,
            encode_ms=10.0,
            prefill_ms=1.0,
            decode_ms=1.0,
        ),
        PipelineRequest(
            request_id="text",
            arrival_ms=1.0,
            encode_ms=0.0,
            prefill_ms=1.0,
            decode_ms=1.0,
        ),
    ]
    report = replay(rows, encode_workers=1, prefill_workers=1, decode_workers=1)
    assert report["requests"][1]["latency_ms"] == 2
    assert report["requests"][0]["latency_ms"] == 12
    with pytest.raises(ValueError, match="encoder capacity"):
        replay(rows, encode_workers=0, prefill_workers=1, decode_workers=1)


def test_queue_replay_accumulates_downstream_waits_and_uses_parallel_capacity():
    rows = [
        PipelineRequest(
            request_id=str(i),
            arrival_ms=0.0,
            encode_ms=0.0,
            prefill_ms=1.0,
            decode_ms=3.0,
        )
        for i in range(3)
    ]
    serial = replay(rows, encode_workers=0, prefill_workers=1, decode_workers=1)
    parallel = replay(rows, encode_workers=0, prefill_workers=1, decode_workers=2)
    assert [row["latency_ms"] for row in serial["requests"]] == [4, 7, 10]
    assert [row["queue_ms"] for row in serial["requests"]] == [0, 3, 6]
    assert [row["latency_ms"] for row in parallel["requests"]] == [4, 5, 7]
    with pytest.raises(ValueError, match="integers"):
        replay(rows, encode_workers=0, prefill_workers=1.5, decode_workers=1)


def test_oracle_finds_exact_reordering_and_labels_it_as_future_information():
    a, b = list(range(6)), list(range(6, 12))
    report = route_union_oracle([[a], [b], [a], [b]], batch_size=2)
    assert report["baseline_union_sum"] == 24
    assert report["minimum_union_sum"] == 12
    assert report["groups"] == [[0, 2], [1, 3]]
    assert "not_a_speedup" in report["status"]
    assert route_union_oracle([[a]] * 4, batch_size=2)["maximum_union_reduction"] == 0


def test_locality_uses_access_frequency_not_hash_space_density():
    result = locality([123456, 999999, 123456, 999999], [0, 1, 2])
    assert result["capacities"][0]["hits"] == 0
    assert result["capacities"][1]["hits"] == 0
    assert result["capacities"][2]["hits"] == 2


def test_cache_hit_screen_accounts_for_transfer_replay_and_queue_costs():
    inputs = dict(
        physical_transfer_bytes=1_000_000_000,
        effective_bandwidth_bytes_per_second=10_000_000_000.0,
        remote_lookup_ms=2.0,
        remote_restore_replay_ms=8.0,
        remote_queue_ms=5.0,
        local_recompute_ms=150.0,
        local_queue_ms=0.0,
        remote_cost_usd=0.1,
        local_cost_usd=0.05,
    )
    report = cache_economics(CacheReuseEstimate(**inputs))
    assert report["transfer_ms"] == 100
    assert report["remote_path_ms"] == 115
    assert report["latency_saving_ms"] == 35
    assert report["remote_faster"] and not report["remote_cheaper"]
    assert report["incremental_cost_saving_usd"] == -0.05
    assert not report["release_approved"]
    inputs["effective_bandwidth_bytes_per_second"] = 1_000_000_000.0
    assert not cache_economics(CacheReuseEstimate(**inputs))["remote_faster"]
    inputs["effective_bandwidth_bytes_per_second"] = 0.0
    with pytest.raises(ValidationError):
        CacheReuseEstimate(**inputs)
