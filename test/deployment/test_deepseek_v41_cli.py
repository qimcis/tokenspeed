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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tokenspeed.deployment.deepseek_v41"


def invoke(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    return subprocess.run(
        [sys.executable, "-m", MODULE, *map(str, args)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize("mode", ["aggregate", "pd", "epd"])
def test_plan_is_a_cpu_only_json_entrypoint(mode):
    result = invoke("plan", ROOT / "recipes/deepseek-v41-b300" / f"{mode}.toml")
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["config"]["mode"] == mode
    assert manifest["revision"] == "1bae0e1088993b9b82de08f59de5be5a34e42170"
    assert not manifest["hardware_validated"]


def test_invalid_cli_never_falls_through_to_launch():
    result = invoke("plan", ROOT / "recipes/deepseek-v41-b300/epd.toml", "--typo")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_help_does_not_import_gpu_runtime():
    result = invoke("--help")
    assert result.returncode == 0
    assert "cache-economics" in result.stdout
    assert "launch" in result.stdout


def test_oracle_and_cache_economics_emit_serializable_reports(tmp_path):
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps([[list(range(6))]] * 4))
    result = invoke("route-oracle", routes, "--batch-size", "2")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["minimum_union_sum"] == 12
    estimate = tmp_path / "cache.json"
    estimate.write_text(
        json.dumps(
            {
                "physical_transfer_bytes": 1000000,
                "effective_bandwidth_bytes_per_second": 1000000000.0,
                "remote_lookup_ms": 1.0,
                "remote_restore_replay_ms": 1.0,
                "remote_queue_ms": 0.0,
                "local_recompute_ms": 10.0,
                "local_queue_ms": 0.0,
                "remote_cost_usd": 0.001,
                "local_cost_usd": 0.002,
            }
        )
    )
    result = invoke("cache-economics", estimate)
    assert result.returncode == 0, result.stderr
    assert not json.loads(result.stdout)["release_approved"]


def test_grade_requires_external_evaluations_and_full_resource_costs(tmp_path):
    raw = {
        "workload_id": "cpu-fixture",
        "model": "deepseek-v41-flash",
        "sampling_id": "greedy",
        "duration_seconds": 1.0,
        "quality": "unknown",
        "release_approved": False,
        "requests": [
            {
                "task_id": "one",
                "cluster_id": "conversation",
                "status": "completed",
                "http_status": 200,
                "admitted": True,
                "success": None,
                "queue_ms": 2.0,
                "latency_ms": 10.0,
                "ttft_ms": 4.0,
                "completion_tokens": 3,
                "response": {"choices": [{"message": {"content": "wrong answer"}}]},
                "events": [],
                "error": None,
            }
        ],
    }
    files = {
        "raw": raw,
        "evaluations": [
            {
                "task_id": "one",
                "success": False,
                "admitted": True,
                "committed_tokens": 3,
            }
        ],
        "costs": [
            {
                "role": role,
                "cost_usd": 1.0 if role == "aggregate" else 0.0,
                "allocated_gpus": 8 if role == "aggregate" else 0,
            }
            for role in ("aggregate", "host", "network", "storage")
        ],
    }
    for name, data in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    result = invoke(
        "grade",
        tmp_path / "raw.json",
        tmp_path / "evaluations.json",
        "--costs",
        tmp_path / "costs.json",
        "--mode",
        "aggregate",
        "--model-revision",
        "a" * 40,
    )
    assert result.returncode == 0, result.stderr
    experiment = json.loads(result.stdout)
    assert experiment["outcomes"][0]["success"] is False
    assert experiment["outcomes"][0]["latency_ms"] == 12.0
    assert experiment["outcomes"][0]["ttft_ms"] == 6.0
    assert experiment["offered_task_ids"] == ["one"]
    (tmp_path / "evaluations.json").write_text("[]")
    failed = invoke(
        "grade",
        tmp_path / "raw.json",
        tmp_path / "evaluations.json",
        "--costs",
        tmp_path / "costs.json",
        "--mode",
        "aggregate",
        "--model-revision",
        "a" * 40,
    )
    assert failed.returncode != 0
    assert "every measured task" in failed.stderr
