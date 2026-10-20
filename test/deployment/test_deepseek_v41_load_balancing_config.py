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


from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.config import Deployment, build_manifest
from tokenspeed.runtime.performance_options import EplbTraceOptions, PrefillWorkOptions

ROOT = Path(__file__).resolve().parents[2]


def candidate():
    return Deployment.load(ROOT / "recipes/deepseek-v41-b300/load-balancing.toml")


def test_complete_frontend_proxy_has_real_service_and_distinct_ports():
    manifest = build_manifest(
        candidate(), snapshot="/snapshot", output_directory=Path("/output")
    )
    commands = {command["role"]: command for command in manifest["commands"]}
    assert commands["gateway"]["port"] == 18002
    assert commands["router"]["port"] == 18000
    assert "tokenspeed.runtime.load_balancing.service" in commands["router"]["argv"]
    settings = manifest["generated_files"]["routing.json"]
    assert settings["workers"][0]["endpoint"] == "http://127.0.0.1:18002"
    assert settings["forward_role"] == "complete"
    for role in ("prefill", "decode"):
        args = commands[role]["argv"]
        assert args[args.index("--ep-num-redundant-experts") + 1] == "4"
        assert "--enforce-eager" in args
        assert "--disable-overlap-schedule" in args
        assert "--disable-prefill-graph" in args


def test_quota_is_forwarded_only_to_prefill_and_requires_all_coefficients():
    data = candidate().model_dump()
    # Synthetic CPU fixture, not a calibration recommendation.
    data["prefill_quota"] = dict(
        budget_ms=4.0,
        linear_ms_per_token=0.1,
        context_ms_per_token_squared=0.0,
        quadratic_ms_per_token_squared=0.0,
    )
    manifest = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    commands = {command["role"]: command["argv"] for command in manifest["commands"]}
    assert "--prefill-work-budget-ms" in commands["prefill"]
    assert "--prefill-work-budget-ms" not in commands["decode"]
    del data["prefill_quota"]["linear_ms_per_token"]
    with pytest.raises(ValidationError):
        Deployment.model_validate(data)


def test_replica_weight_ledger_accounts_physical_copies():
    data = candidate().model_dump()
    with_replicas = build_manifest(
        Deployment.model_validate(data), snapshot="/snapshot"
    )
    for worker in data["workers"]:
        worker["experts"] = None
    without = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    for candidate_row, control_row in zip(
        with_replicas["memory"]["workers"], without["memory"]["workers"], strict=True
    ):
        a = candidate_row["components_per_rank"]["routed_expert_values"]
        b = control_row["components_per_rank"]["routed_expert_values"]
        assert a * 384 == b * 388


def test_trace_flags_reach_only_selected_role():
    data = candidate().model_dump()
    data["workers"][0]["experts"].update(
        algorithm="trace",
        rebalance=True,
        rebalance_num_iterations=64,
        rebalance_layers_per_chunk=1,
        trace=dict(
            window=64,
            max_moved_slots=4,
            movement_penalty=0.0,
            min_relative_improvement=0.01,
            objective="token_count",
        ),
    )
    manifest = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    commands = {command["role"]: command["argv"] for command in manifest["commands"]}
    assert "--eplb-trace-window" in commands["prefill"]
    assert "--eplb-trace-window" not in commands["decode"]
    assert "--enable-eplb" in commands["prefill"]


@pytest.mark.parametrize(
    "change",
    [
        dict(dispatch="fake"),
        dict(redundant_experts=1),
        dict(rebalance=True),
        dict(algorithm="trace"),
        dict(execution="default"),
    ],
)
def test_unsupported_placement_recipes_fail_early(change):
    data = candidate().model_dump()
    data["workers"][0]["experts"].update(change)
    with pytest.raises(ValidationError):
        Deployment.model_validate(data)


def test_old_scheduler_quota_default_is_compatible_but_opt_in_errors():
    class OldScheduler:
        pass

    PrefillWorkOptions().require_scheduler_support(OldScheduler)
    with pytest.raises(ValueError, match="build/install"):
        PrefillWorkOptions(1.0, 1.0, 0.0, 0.0).require_scheduler_support(OldScheduler)


@pytest.mark.parametrize(
    "values",
    [
        (1.0, None, 0.0, 0.0),
        (float("nan"), 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, -1.0, 0.0, 0.0),
        (True, 1.0, 0.0, 0.0),
    ],
)
def test_invalid_or_partial_calibrations_fail(values):
    with pytest.raises(ValueError):
        PrefillWorkOptions(*values).validate(role="prefill")


def test_quota_rejects_decode_but_baseline_remains_valid():
    PrefillWorkOptions().validate(role="decode")
    with pytest.raises(ValueError, match="prefill role"):
        PrefillWorkOptions(1.0, 1.0, 0.0, 0.0).validate(role="decode")


@pytest.mark.parametrize(
    "change",
    [
        dict(window=0),
        dict(max_moved_slots=-1),
        dict(movement_penalty=float("inf")),
        dict(min_relative_improvement=1.0),
        dict(objective="measured_cost"),
        dict(expert_costs="unused.json"),
    ],
)
def test_invalid_trace_options_fail(change):
    values = dict(
        window=64,
        max_moved_slots=4,
        movement_penalty=0.0,
        min_relative_improvement=0.01,
        objective="token_count",
    )
    values.update(change)
    with pytest.raises(ValueError):
        EplbTraceOptions(**values).validate()


def test_placement_affinity_requires_opt_in_and_atomic_rebalance():
    data = candidate().model_dump()
    data["performance"] = dict(
        expert_affinity=True,
        expert_affinity_placement=True,
        execution="eager_serial",
        dspark_candidate_communication="tiles",
    )
    manifest = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    decode = next(c["argv"] for c in manifest["commands"] if c["role"] == "decode")
    assert "--expert-affinity-placement" in decode
    assert decode[decode.index("--expert-affinity-placement") + 1] != "True"
    data["workers"][1]["experts"].update(
        rebalance=True, rebalance_num_iterations=64, rebalance_layers_per_chunk=1
    )
    with pytest.raises(ValidationError, match="coherent"):
        Deployment.model_validate(data)
    data["workers"][1]["experts"]["rebalance_layers_per_chunk"] = 40
    Deployment.model_validate(data)
    data["performance"]["expert_affinity_placement"] = False
    with pytest.raises(ValidationError, match="EPLB"):
        Deployment.model_validate(data)


def test_generated_routing_config_accepted_by_real_service():
    import asyncio

    import httpx

    from tokenspeed.runtime.load_balancing.service import RoutingService

    manifest = build_manifest(candidate(), snapshot="/snapshot")
    client = httpx.AsyncClient()
    try:
        service = RoutingService(
            manifest["generated_files"]["routing.json"], client=client
        )
        workers = service.registry.snapshot()
        assert len(workers) == 1
        assert not workers[0].worker.ready
        assert workers[0].worker.endpoint == "http://127.0.0.1:18002"
    finally:
        asyncio.run(client.aclose())


def test_trace_and_migration_ledger_counts_persistent_device_buffers():
    data = candidate().model_dump()
    data["workers"][0]["experts"].update(
        algorithm="trace",
        rebalance=True,
        rebalance_num_iterations=64,
        rebalance_layers_per_chunk=1,
        trace=dict(
            window=64,
            max_moved_slots=4,
            movement_penalty=0.0,
            min_relative_improvement=0.01,
            objective="token_count",
        ),
    )
    manifest = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    prefill, decode = manifest["memory"]["workers"]
    components = prefill["components_per_rank"]
    counter_bytes = 40 * 388 * 8
    assert components["expert_route_counters"] == counter_bytes
    assert components["expert_trace_ring"] == 64 * counter_bytes
    assert components["expert_trace_previous_counters"] == counter_bytes
    assert (
        components["expert_migration_value_staging"]
        == components["routed_expert_values"] // 40
    )
    assert "expert_trace_ring" not in decode["components_per_rank"]
    assert "expert_migration_value_staging" not in decode["components_per_rank"]
