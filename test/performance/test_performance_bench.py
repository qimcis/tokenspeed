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

"""Counter correctness; no scheduler simulation or GPU performance claims."""

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.performance_bench import (
    DSparkScreen,
    GPUExperiment,
    benchmark_dspark_gpu,
    benchmark_engram_gpu,
    dspark_payload_counts,
    run_mechanism_screen,
)


def trace():
    baseline = {
        "eligible": ["a:1", "b:1", "c:1"],
        "selected": ["a:1", "b:1"],
        "predicted_experts": {"a:1": [0, 1], "b:1": [2, 3], "c:1": [0, 1]},
        "actual_experts": {"a:1": [0, 1], "b:1": [2, 3]},
    }
    candidate = {
        **baseline,
        "selected": ["a:1", "c:1"],
        "actual_experts": {"a:1": [0, 1], "c:1": [1, 4]},
    }
    return {
        "evidence": "synthetic",
        "source": "unit-test fixture",
        "experts": {
            "max_deferrals": 2,
            "baseline": [baseline],
            "candidate": [candidate],
        },
        "engram": {
            "layer_id": 1,
            "model_revision": "fixture",
            "row_start": 4,
            "row_end": 8,
            "embedding_dim": 256,
            "capacity": 2,
            "hot_global_ids": [4, 6],
            "lookup_global_ids": [3, 4, 4, 5, 6, 8],
        },
        "dspark": {"rows": 3, "vocabulary_tiles": 8, "block_size": 4, "tp_size": 2},
    }


def test_mechanism_counters_do_not_claim_measured_performance():
    result = run_mechanism_screen(trace())
    assert result["status"] == "cpu_mechanism_screen_only"
    assert result["evidence"] == "synthetic"
    assert result["latency_ms"] is None and result["speedup"] is None
    baseline = result["experts"]["baseline"]["batches"][0]
    candidate = result["experts"]["candidate"]["batches"][0]
    assert baseline["predicted_complete_union"] == 4
    assert candidate["predicted_complete_union"] == 2
    assert candidate["actual_complete_union"] == 3
    engram = result["engram"]
    assert (engram["hits"], engram["misses"], engram["nonlocal_zero_lookups"]) == (
        3,
        1,
        2,
    )
    assert engram["logical_candidate_host_row_bytes"] == 264
    assert engram["logical_baseline_host_row_bytes"] == 4 * 264
    assert engram["resident_hot_bytes"] == 2 * 272
    payload = result["dspark"]
    assert payload["baseline_per_rank_input_bytes"] == 3 * 8 * 8 * 4
    assert payload["candidate_per_rank_input_bytes"] == 3 * 8 * 4
    assert payload["candidate_per_rank_output_bytes"] == 3 * 8 * 4 * 2
    assert payload["network_wire_bytes"] is None


def test_missing_hints_are_not_counted_as_a_complete_union():
    data = trace()
    del data["experts"]["candidate"][0]["predicted_experts"]["c:1"]
    result = run_mechanism_screen(data)["experts"]["candidate"]["batches"][0]
    assert result["predicted_complete_union"] is None
    assert result["predicted_known_union"] == 2
    assert result["predicted_known_requests"] == 1


def test_deferral_counts_reset_on_service_absence_and_request_generation_change():
    data = trace()
    deferred = {
        "eligible": ["a:1", "b:1"],
        "selected": ["a:1"],
        "predicted_experts": {},
        "actual_experts": {},
    }
    data["experts"]["candidate"] = [deferred] * 3
    report = run_mechanism_screen(data)["experts"]["candidate"]
    assert report["maximum_consecutive_eligible_deferrals"] == 3
    assert report["deferral_threshold_exceedances"] == ["b:1"]
    assert report["end_of_trace_pending"] == {"b:1": 3}
    service = {**deferred, "selected": ["b:1"]}
    data["experts"]["candidate"] = [deferred, service, deferred]
    report = run_mechanism_screen(data)["experts"]["candidate"]
    assert report["maximum_consecutive_eligible_deferrals"] == 1
    assert report["deferral_threshold_exceedances"] == []
    generation = {**deferred, "eligible": ["a:1", "b:2"]}
    data["experts"]["candidate"] = [deferred, generation]
    assert run_mechanism_screen(data)["experts"]["candidate"][
        "end_of_trace_pending"
    ] == {"b:2": 1}


def test_empty_cache_and_nonlocal_inputs_have_defined_rates():
    data = trace()
    data["engram"].update(capacity=0, hot_global_ids=[], lookup_global_ids=[4, 5])
    report = run_mechanism_screen(data)["engram"]
    assert report["hits"] == 0 and report["misses"] == 2
    assert report["hit_rate"] == 0
    data["engram"]["lookup_global_ids"] = [0, 9]
    assert run_mechanism_screen(data)["engram"]["hit_rate"] is None


def test_single_rank_has_no_collective_bytes_but_still_local_reduction():
    result = dspark_payload_counts(
        DSparkScreen(rows=3, vocabulary_tiles=8, block_size=4, tp_size=1)
    )
    assert result["collective_calls"] == 0
    assert (
        result["baseline_per_rank_input_bytes"]
        == result["candidate_per_rank_input_bytes"]
        == 0
    )
    assert result["candidate_local_reduction_elements"] == 96


@pytest.mark.parametrize(
    "change",
    [
        {"hot_global_ids": [4, 4]},
        {"hot_global_ids": [3]},
        {"capacity": 1},
        {"row_end": 4},
        {"embedding_dim": 128},
    ],
)
def test_invalid_cache_trace_is_rejected(change):
    data = trace()
    data["engram"].update(change)
    with pytest.raises(ValidationError):
        run_mechanism_screen(data)


def test_invalid_membership_and_coercions_rejected():
    data = trace()
    data["experts"]["candidate"][0]["selected"] = ["not-eligible"]
    with pytest.raises(ValidationError):
        run_mechanism_screen(data)
    data = trace()
    data["dspark"]["rows"] = "3"
    with pytest.raises(ValidationError):
        run_mechanism_screen(data)


def test_gpu_entrypoints_reject_without_opt_in_before_loading_runtime():
    config = GPUExperiment(
        allow_gpu=False,
        warmup=1,
        repetitions=2,
        source="test",
        model_revision="fixture",
    )
    with pytest.raises(ValueError, match="allow_gpu=True"):
        benchmark_dspark_gpu(None, None, None, None, None, config=config)
    with pytest.raises(ValueError, match="allow_gpu=True"):
        benchmark_engram_gpu(None, None, None, config=config)


def test_expert_comparison_flags_smaller_candidate_batches_and_different_eligibility():
    data = trace()
    comparison = run_mechanism_screen(data)["experts"]["comparison"]
    assert all(comparison.values())
    data["experts"]["candidate"][0]["selected"] = ["a:1"]
    data["experts"]["candidate"][0]["actual_experts"] = {"a:1": [0, 1]}
    comparison = run_mechanism_screen(data)["experts"]["comparison"]
    assert not comparison["paired_selected_counts_match"]
    assert comparison["paired_eligible_sets_match"]
    data["experts"]["candidate"] = []
    assert not any(run_mechanism_screen(data)["experts"]["comparison"].values())


def test_empty_dspark_batch_has_no_collective_or_reduction_work():
    result = dspark_payload_counts(
        DSparkScreen(rows=0, vocabulary_tiles=8, block_size=4, tp_size=2)
    )
    assert result["collective_calls"] == 0
    assert result["candidate_local_reduction_elements"] == 0
    assert (
        result["baseline_per_rank_input_bytes"]
        == result["candidate_per_rank_input_bytes"]
        == 0
    )
