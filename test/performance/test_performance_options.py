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

from dataclasses import replace

import pytest

from tokenspeed.runtime.performance_options import (
    PerformanceCapabilities,
    PerformanceOptions,
    validate_performance_options,
)


def capabilities(**changes):
    return replace(
        PerformanceCapabilities(
            model_is_deepseek_v41=True,
            nvidia_cuda=True,
            role="aggregate",
            speculation=None,
            speculative_sampling=False,
            pipeline_size=1,
            attention_dp=1,
            attention_tp=8,
            world_size=8,
            query_context_size=1,
            kv_parallel_size=1,
            eager=True,
            prefill_graph_disabled=True,
            overlap_disabled=True,
            mixed_batch=False,
            expert_rebalancing=False,
            custom_expert_placement=False,
            routing_simulation=False,
            host_engram=True,
            host_engram_layout="sharded",
            load_format="auto",
        ),
        **changes
    )


@pytest.mark.parametrize("role", ["aggregate", "decode"])
def test_affinity_accepts_replicated_input_multi_gpu(role):
    validate_performance_options(
        PerformanceOptions(expert_affinity=True), capabilities(role=role)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"role": "prefill"},
        {"role": "encode"},
        {"speculation": "DSPARK"},
        {"eager": False},
        {"prefill_graph_disabled": False},
        {"overlap_disabled": False},
        {"mixed_batch": True},
        {"pipeline_size": 2},
        {"attention_dp": 2},
        {"attention_tp": 4},
        {"query_context_size": 2},
        {"kv_parallel_size": 2},
        {"expert_rebalancing": True},
        {"custom_expert_placement": True},
        {"routing_simulation": True},
        {"nvidia_cuda": False},
        {"model_is_deepseek_v41": False},
    ],
)
def test_affinity_rejects_unsupported_capabilities(change):
    with pytest.raises(ValueError):
        validate_performance_options(
            PerformanceOptions(expert_affinity=True), capabilities(**change)
        )


def test_baseline_does_not_require_experimental_capabilities():
    validate_performance_options(
        PerformanceOptions(),
        capabilities(nvidia_cuda=False, model_is_deepseek_v41=False),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"expert_affinity_lookahead": 1},
        {"expert_affinity_max_layers": 33},
        {"expert_affinity_capture_interval": 0},
        {"max_decode_batch_size": -1},
        {"dspark_candidate_communication": "typo"},
        {"engram_hot_cache_capacity": True},
        {"engram_hot_cache_capacity": 2},
        {"engram_hot_cache_manifest": "ignored.json"},
    ],
)
def test_invalid_knobs_never_silently_pass(changes):
    with pytest.raises(ValueError):
        PerformanceOptions(**changes).validate()


def test_inert_affinity_tuning_is_rejected():
    with pytest.raises(ValueError, match="requires expert_affinity"):
        PerformanceOptions(expert_affinity_max_deferrals=4).validate()


def test_compact_supports_dp_but_not_sampling_or_pipeline():
    options = PerformanceOptions(dspark_candidate_communication="compact")
    validate_performance_options(
        options, capabilities(speculation="DSPARK", attention_dp=2, attention_tp=4)
    )
    for change in (
        {"speculation": None},
        {"speculative_sampling": True},
        {"pipeline_size": 2},
    ):
        with pytest.raises(ValueError, match="compact"):
            validate_performance_options(
                options, replace(capabilities(speculation="DSPARK"), **change)
            )


@pytest.mark.parametrize(
    "change",
    [{"host_engram": False}, {"host_engram_layout": "auto"}, {"load_format": "dummy"}],
)
def test_cache_requires_real_host_tables_and_loader(change):
    options = PerformanceOptions(
        engram_hot_cache_capacity=1,
        engram_hot_cache_manifest="hot.json",
        engram_hot_cache_model_revision="pinned",
    )
    with pytest.raises(ValueError, match="Engram"):
        validate_performance_options(options, capabilities(**change))


def test_old_native_bindings_remain_valid_for_baseline_but_reject_opt_ins():
    from tokenspeed.runtime.performance_options import (
        require_scheduler_performance_support,
    )

    # Stand in only for the unavailable older third-party native extension.
    class OldConfig:
        pass

    class OldScheduler:
        pass

    require_scheduler_performance_support(PerformanceOptions(), OldConfig, OldScheduler)
    for options in (
        PerformanceOptions(max_decode_batch_size=8),
        PerformanceOptions(expert_affinity=True),
    ):
        with pytest.raises(ValueError, match="build/install"):
            require_scheduler_performance_support(options, OldConfig, OldScheduler)
