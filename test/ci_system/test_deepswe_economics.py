# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "ci" / "deepswe" / "economics.py"
SPEC = importlib.util.spec_from_file_location("deepswe_economics", MODULE_PATH)
assert SPEC and SPEC.loader
economics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(economics)


def trial(name: str, reward: int | None, completion: str, exception: str | None):
    return {
        "trial_name": name,
        "started_at": "2030-01-01T00:01:00",
        "finished_at": "2030-01-01T00:50:00",
        "agent_execution": {"finished_at": completion},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": exception} if exception else None,
    }


def task(task_id: str, names: list[str]):
    return {
        "task_id": task_id,
        "trial_names": names,
        "submitted_at": "2030-01-01T00:00:00Z",
        "deadline_at": "2030-01-01T00:30:00Z",
    }


def manifest(tasks):
    return {
        "pier_timezone": "UTC",
        "tasks": tasks,
        "allocations": [
            {
                "name": "target",
                "started_at": "2030-01-01T00:00:00Z",
                "finished_at": "2030-01-01T01:00:00Z",
                "hourly_cost_usd": 8,
            },
            {
                "name": "worker",
                "started_at": "2030-01-01T00:00:00Z",
                "finished_at": "2030-01-01T01:00:00Z",
                "hourly_cost_usd": 1,
            },
        ],
        "additional_costs": [{"name": "network", "cost_usd": 1}],
    }


def test_retries_count_once_and_all_cancelled_retry_cost_remains():
    rows = [
        trial("cancelled", None, "2030-01-01T00:05:00", "CancelledError"),
        trial("retry", 1, "2030-01-01T00:20:00", None),
        trial("repeat", 1, "2030-01-01T00:25:00", None),
    ]
    data = {"trial_results": rows, "stats": {"cost_usd": 0.001}}
    report = economics.build_report(
        [data], manifest([task("one", ["cancelled", "retry", "repeat"])])
    )

    assert report["offered_tasks"] == 1
    assert report["deadline_qualified_tasks"] == 1
    assert report["total_cost_usd"] == 10
    assert report["deadline_qualified_tasks_per_usd"] == 0.1
    assert report["tasks"][0]["attempts"][0]["outcome"] == "cancelled"


def test_failed_late_missing_and_unadmitted_tasks_do_not_erase_cost():
    rows = [
        trial("late", 1, "2030-01-01T00:30:00.000001", None),
        trial("incorrect", 0, "2030-01-01T00:20:00", None),
        trial("failed", 1, "2030-01-01T00:20:00", "RuntimeError"),
        trial("boundary", 1, "2030-01-01T00:30:00", None),
    ]
    tasks = [
        task(name, [name])
        for name in ("late", "incorrect", "failed", "boundary", "missing")
    ]
    tasks.append(task("not-admitted", []))
    report = economics.build_report([{"trial_results": rows}], manifest(tasks))

    assert report["offered_tasks"] == 6
    assert report["correct_tasks"] == 2
    assert report["deadline_qualified_tasks"] == 1
    assert report["deadline_attainment"] == pytest.approx(1 / 6)
    assert report["total_cost_usd"] == 10


def test_unfinished_success_is_not_counted():
    row = trial("one", 1, "2030-01-01T00:20:00", None)
    row["finished_at"] = None
    report = economics.build_report(
        [{"trial_results": [row]}], manifest([task("one", ["one"])])
    )
    assert report["deadline_qualified_tasks"] == 0
    assert report["total_cost_usd"] == 10


def test_join_refuses_unassigned_or_reused_trials():
    data = {"trial_results": [trial("one", 1, "2030-01-01T00:20:00", None)]}
    with pytest.raises(ValueError, match="missing from task manifest"):
        economics.build_report([data], manifest([task("unrelated", [])]))
    with pytest.raises(ValueError, match="assigned trial_name"):
        economics.build_report(
            [data], manifest([task("one", ["one"]), task("two", ["one"])])
        )
    with pytest.raises(ValueError, match="trial_name"):
        economics.build_report([data, data], manifest([task("one", ["one"])]))


@pytest.mark.parametrize("cost", [-1, float("inf"), float("nan"), True])
def test_invalid_cost_cannot_produce_favorable_economics(cost):
    ledger = manifest([task("missing", [])])
    ledger["allocations"][0]["hourly_cost_usd"] = cost
    with pytest.raises(ValueError, match="finite nonnegative"):
        economics.build_report([], ledger)


def test_missing_completion_or_truncated_billing_fails_closed():
    row = trial("one", 1, "2030-01-01T00:20:00", None)
    data = {"trial_results": [row]}
    ledger = manifest([task("one", ["one"])])
    row["agent_execution"] = None
    with pytest.raises(ValueError, match="agent_execution.finished_at"):
        economics.build_report([data], ledger)
    row["agent_execution"] = {"finished_at": "2030-01-01T00:20:00"}
    row["finished_at"] = "2030-01-01T01:00:01"
    with pytest.raises(ValueError, match="outside the allocation ledger"):
        economics.build_report([data], ledger)


def test_equivalent_timezone_offsets_do_not_change_accounting():
    ledger = manifest([task("one", ["one"])])
    ledger["allocations"][1]["started_at"] = "2029-12-31T19:00:00-05:00"
    ledger["allocations"][1]["finished_at"] = "2029-12-31T20:00:00-05:00"
    row = trial("one", 1, "2029-12-31T19:20:00-05:00", None)
    report = economics.build_report([{"trial_results": [row]}], ledger)
    assert report["deadline_qualified_tasks"] == 1
    assert report["total_cost_usd"] == 10


def test_cli_joins_existing_pier_artifact_without_runtime_dependencies(tmp_path):
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "economic-report.json"
    result_path.write_text(
        json.dumps({"trial_results": [trial("one", 1, "2030-01-01T00:20:00", None)]}),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest([task("one", ["one"])])), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(result_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "1/1 correct tasks by deadline" in result.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["total_cost_usd"] == 10
