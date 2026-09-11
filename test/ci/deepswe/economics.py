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

"""Join existing Pier results with workload deadlines and full allocation costs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def _timestamp(value: Any, label: str, naive_timezone: ZoneInfo | None) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO 8601 timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO 8601 timestamp") from error
    if result.tzinfo is None:
        if naive_timezone is None:
            raise ValueError(f"{label} must include a timezone")
        result = result.replace(tzinfo=naive_timezone)
    return result


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _unique_name(value: Any, seen: set[str], label: str) -> str:
    if not isinstance(value, str) or not value or value in seen:
        raise ValueError(f"{label} must be a nonempty unique string: {value!r}")
    seen.add(value)
    return value


def build_report(
    results: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Return task goodput per dollar from finalized Pier results and a ledger.

    ``results`` contains existing Pier job result objects. ``manifest`` supplies
    the complete offered task set, logical-task to trial-name mapping, absolute
    deadlines, the timezone of naive Pier timestamps, and all allocation costs.
    Retries can satisfy a logical task only once. Missing, cancelled, failed and
    late work never removes allocation cost. Invalid or ambiguous joins raise
    ``ValueError`` instead of silently producing a favorable report.
    """
    timezone = ZoneInfo(manifest["pier_timezone"])
    trials: dict[str, dict[str, Any]] = {}
    trial_names: set[str] = set()
    for result in results:
        for trial in result["trial_results"]:
            name = _unique_name(trial["trial_name"], trial_names, "trial_name")
            trials[name] = trial

    allocation_rows = []
    allocation_names: set[str] = set()
    for allocation in manifest["allocations"]:
        name = _unique_name(allocation["name"], allocation_names, "allocation name")
        start = _timestamp(allocation["started_at"], f"{name}.started_at", None)
        finish = _timestamp(allocation["finished_at"], f"{name}.finished_at", None)
        if finish <= start:
            raise ValueError(f"{name}: allocation must have positive duration")
        rate = _nonnegative(allocation["hourly_cost_usd"], f"{name}.hourly_cost_usd")
        allocation_rows.append(
            {
                "name": name,
                "started_at": start.isoformat(),
                "finished_at": finish.isoformat(),
                "cost_usd": (finish - start).total_seconds() / 3600 * rate,
            }
        )
    if not allocation_rows:
        raise ValueError("at least one complete allocation interval is required")
    # Compare aware datetimes, not ISO strings with potentially different offsets.
    accounting_start_time = min(
        _timestamp(row["started_at"], "allocation start", None)
        for row in allocation_rows
    )
    accounting_finish_time = max(
        _timestamp(row["finished_at"], "allocation finish", None)
        for row in allocation_rows
    )
    extra_rows = []
    extra_names: set[str] = set()
    for item in manifest["additional_costs"]:
        name = _unique_name(item["name"], extra_names, "additional cost name")
        extra_rows.append(
            {"name": name, "cost_usd": _nonnegative(item["cost_usd"], name)}
        )
    total_cost = math.fsum(row["cost_usd"] for row in allocation_rows + extra_rows)
    if not math.isfinite(total_cost) or total_cost <= 0:
        raise ValueError("total allocation cost must be finite and positive")

    task_names: set[str] = set()
    assigned_trials: set[str] = set()
    task_rows = []
    for task in manifest["tasks"]:
        task_id = _unique_name(task["task_id"], task_names, "task_id")
        submitted = _timestamp(task["submitted_at"], f"{task_id}.submitted_at", None)
        deadline = _timestamp(task["deadline_at"], f"{task_id}.deadline_at", None)
        if deadline < submitted:
            raise ValueError(f"{task_id}: deadline precedes submission")
        if not accounting_start_time <= submitted <= accounting_finish_time:
            raise ValueError(f"{task_id}: submission is outside the allocation ledger")
        successes = []
        attempts = []
        for name in task["trial_names"]:
            _unique_name(name, assigned_trials, "assigned trial_name")
            trial = trials.get(name)
            if trial is None:
                attempts.append({"trial_name": name, "outcome": "missing"})
                continue
            start_value = trial.get("started_at")
            finish_value = trial.get("finished_at")
            start = (
                _timestamp(start_value, f"{name}.started_at", timezone)
                if start_value is not None
                else None
            )
            finish = (
                _timestamp(finish_value, f"{name}.finished_at", timezone)
                if finish_value is not None
                else None
            )
            for instant in (start, finish):
                if instant is not None and not (
                    submitted <= instant <= accounting_finish_time
                ):
                    raise ValueError(
                        f"{name}: execution is outside the allocation ledger"
                    )
            if start is not None and finish is not None and finish < start:
                raise ValueError(f"{name}: trial finish precedes start")
            exception = trial.get("exception_info")
            reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get(
                "reward"
            )
            if exception is not None:
                outcome = (
                    "cancelled"
                    if exception.get("exception_type") == "CancelledError"
                    else "failed"
                )
            elif finish is None:
                outcome = "unfinished"
            elif isinstance(reward, bool) or reward not in (0, 1):
                raise ValueError(f"{name}: finalized trial lacks a binary reward")
            elif reward == 0:
                outcome = "incorrect"
            else:
                completion = _timestamp(
                    (trial.get("agent_execution") or {}).get("finished_at"),
                    f"{name}.agent_execution.finished_at",
                    timezone,
                )
                if start is None or not start <= completion <= finish:
                    raise ValueError(
                        f"{name}: agent completion is outside trial execution"
                    )
                successes.append(completion)
                outcome = "on_time" if completion <= deadline else "late"
            attempts.append({"trial_name": name, "outcome": outcome})
        completion = min(successes) if successes else None
        task_rows.append(
            {
                "task_id": task_id,
                "submitted_at": submitted.isoformat(),
                "deadline_at": deadline.isoformat(),
                "correct": completion is not None,
                "deadline_qualified": completion is not None and completion <= deadline,
                "completion_at": completion.isoformat() if completion else None,
                "attempts": attempts,
            }
        )
    if not task_rows:
        raise ValueError("the complete offered task set must not be empty")
    unassigned = trial_names - assigned_trials
    if unassigned:
        raise ValueError(
            f"Pier trials missing from task manifest: {sorted(unassigned)}"
        )
    correct = sum(row["correct"] for row in task_rows)
    qualified = sum(row["deadline_qualified"] for row in task_rows)
    return {
        "offered_tasks": len(task_rows),
        "correct_tasks": correct,
        "deadline_qualified_tasks": qualified,
        "deadline_attainment": qualified / len(task_rows),
        "total_cost_usd": total_cost,
        "deadline_qualified_tasks_per_usd": qualified / total_cost,
        "accounting_started_at": accounting_start_time.isoformat(),
        "accounting_finished_at": accounting_finish_time.isoformat(),
        "allocations": allocation_rows,
        "additional_costs": extra_rows,
        "tasks": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        results = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.results
        ]
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = build_report(results, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{report['deadline_qualified_tasks']}/{report['offered_tasks']} correct tasks "
        f"by deadline; ${report['total_cost_usd']:.6f} allocated; "
        f"{report['deadline_qualified_tasks_per_usd']:.6f} tasks/USD"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
