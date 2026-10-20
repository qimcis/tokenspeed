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

"""Pinned deployment recipes and offline experiment tools for DeepSeek V4.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tokenspeed.deployment.deepseek_v41.config import Deployment, build_manifest


def _read(path: Path):
    from tokenspeed.deployment.deepseek_v41.workload import strict_json

    return strict_json(path.read_bytes())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "launch"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "launch":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--startup-timeout", type=float, required=True)
            command.add_argument("--shutdown-grace", type=float, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--margin", type=float, required=True)
    compare.add_argument("--confidence", type=float, required=True)
    compare.add_argument("--resamples", type=int, required=True)
    compare.add_argument("--seed", type=int, required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("observations", type=Path)
    calibrate.add_argument("--tolerance", type=float, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("workload", type=Path)
    for role in ("encode", "prefill", "decode"):
        replay.add_argument(f"--{role}-workers", type=int, required=True)
    oracle = commands.add_parser("route-oracle")
    oracle.add_argument("routes", type=Path)
    oracle.add_argument("--batch-size", type=int, required=True)
    locality = commands.add_parser("locality")
    locality.add_argument("rows", type=Path)
    locality.add_argument("--capacities", type=int, nargs="+", required=True)
    cache = commands.add_parser("cache-economics")
    cache.add_argument("estimate", type=Path)
    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("directory", type=Path)
    hotset = commands.add_parser("hotset")
    hotset.add_argument("frequencies", type=Path)
    hotset.add_argument("--capacity-rows", type=int, required=True)
    screen = commands.add_parser("performance-screen")
    screen.add_argument("observations", type=Path)
    measure = commands.add_parser("measure")
    measure.add_argument("workload", type=Path)
    measure.add_argument("--endpoint", required=True)
    measure.add_argument("--concurrency", type=int, required=True)
    measure.add_argument("--connect-timeout", type=float, required=True)
    measure.add_argument("--request-timeout", type=float, required=True)
    measure.add_argument("--max-response-bytes", type=int, required=True)
    grade = commands.add_parser("grade")
    grade.add_argument("measurements", type=Path)
    grade.add_argument("evaluations", type=Path)
    grade.add_argument("--costs", type=Path, required=True)
    grade.add_argument("--mode", choices=("aggregate", "pd", "epd"), required=True)
    grade.add_argument("--model-revision", required=True)
    args = parser.parse_args(argv)
    if args.command in ("plan", "launch"):
        config = Deployment.load(args.config)
        if args.command == "launch":
            from tokenspeed.deployment.deepseek_v41.launch import run

            run(
                config,
                args.output,
                startup_timeout=args.startup_timeout,
                shutdown_grace=args.shutdown_grace,
            )
            return
        result = build_manifest(config, snapshot=f"<snapshot:{config.revision}>")
    elif args.command == "checkpoint":
        from tokenspeed.deployment.deepseek_v41.checkpoint import audit_checkpoint

        result = audit_checkpoint(args.directory)
    elif args.command == "hotset":
        from tokenspeed.deployment.deepseek_v41.hotset import build_hotset

        result = build_hotset(_read(args.frequencies), capacity_rows=args.capacity_rows)
    elif args.command == "performance-screen":
        from tokenspeed.deployment.deepseek_v41.performance_bench import (
            run_mechanism_screen,
        )

        result = run_mechanism_screen(_read(args.observations))
    elif args.command == "measure":
        from tokenspeed.deployment.deepseek_v41.workload import (
            ReplayConfig,
            load_workload,
            run_workload,
        )

        result = run_workload(
            load_workload(args.workload),
            ReplayConfig(
                endpoint=args.endpoint,
                concurrency=args.concurrency,
                connect_timeout_seconds=args.connect_timeout,
                request_timeout_seconds=args.request_timeout,
                max_response_bytes=args.max_response_bytes,
            ),
            api_key=os.environ.get("TOKENSPEED_BENCH_API_KEY"),
        ).model_dump()
    elif args.command == "grade":
        from tokenspeed.deployment.deepseek_v41.analysis import Experiment, ResourceCost
        from tokenspeed.deployment.deepseek_v41.workload import (
            Evaluation,
            MeasurementRun,
            join_evaluations,
        )

        run = MeasurementRun.model_validate(_read(args.measurements))
        outcomes = join_evaluations(
            run, [Evaluation.model_validate(row) for row in _read(args.evaluations)]
        )
        result = Experiment(
            workload_id=run.workload_id,
            model_revision=args.model_revision,
            sampling_id=run.sampling_id,
            mode=args.mode,
            duration_seconds=run.duration_seconds,
            offered_task_ids=[row.task_id for row in run.requests],
            resources=[ResourceCost.model_validate(row) for row in _read(args.costs)],
            outcomes=outcomes,
        ).model_dump()
    else:
        from tokenspeed.deployment.deepseek_v41 import analysis

        if args.command == "compare":
            result = analysis.compare(
                analysis.Experiment.model_validate(_read(args.baseline)),
                analysis.Experiment.model_validate(_read(args.candidate)),
                margin=args.margin,
                confidence=args.confidence,
                resamples=args.resamples,
                seed=args.seed,
            )
        elif args.command == "calibrate":
            data = _read(args.observations)
            result = analysis.calibrate(
                [analysis.Step.model_validate(row) for row in data["training"]],
                [analysis.Step.model_validate(row) for row in data["held_out"]],
                tolerance=args.tolerance,
            )
        elif args.command == "replay":
            result = analysis.replay(
                [
                    analysis.PipelineRequest.model_validate(row)
                    for row in _read(args.workload)
                ],
                encode_workers=args.encode_workers,
                prefill_workers=args.prefill_workers,
                decode_workers=args.decode_workers,
            )
        elif args.command == "cache-economics":
            result = analysis.cache_economics(
                analysis.CacheReuseEstimate.model_validate(_read(args.estimate))
            )
        elif args.command == "route-oracle":
            result = analysis.route_union_oracle(
                _read(args.routes), batch_size=args.batch_size
            )
        else:
            result = analysis.locality(_read(args.rows), args.capacities)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
