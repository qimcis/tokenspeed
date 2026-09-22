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

"""Bounded H20 TP8 allreduce/HC/norm/quant complete-chain benchmark.

Launch eight fresh workers (one visible-device ordinal per rank)::

    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        tokenspeed-kernel/benchmarks/communication/bench_hopper_mhc.py \
        --output result.json --seconds 35 --round-mode bf16_step

Requires eight idle SM90 H20 GPUs and installed TokenSpeed kernel dependencies.
Precompile the kernel cache first when startup plus compilation cannot fit the
explicit 35-second maximum. Every worker includes imports, CUDA/collective
initialization, correctness, capture, warmup and timing in its deadline. Use an
outer process-group watchdog to bound torchrun startup and drain failed workers.
Existing output files are never reused. Partial sweeps cannot produce a summary.

GPU events measure hot-input graphs containing eight complete calls and two
replays. Three alternating pairs retain every rank's timings; summary medians
use each pair's slowest rank. These are kernel-chain timings, not model latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
import traceback
from pathlib import Path

WORLD = 8
CALLS = 8
REPLAYS = 2
PAIRS = 3
L2_LIMIT = 0.006
CASES = tuple((rows, quantize) for rows in (1, 2, 4, 8) for quantize in (False, True))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, torch):
    reports = []
    for name, a, e in zip(("hc", "normalized"), actual[:2], expected[:2]):
        a, e = (a.detach().cpu().double(), e.detach().cpu().double())
        assert (
            a.shape == e.shape and torch.isfinite(a).all() and torch.isfinite(e).all()
        )
        error = (a - e).flatten(1).norm(dim=1)
        norm = e.flatten(1).norm(dim=1)
        assert torch.all(error[norm == 0] == 0), "nonzero error against zero reference"
        ratios = torch.where(norm > 0, error / norm, torch.zeros_like(error))
        assert torch.all(ratios <= L2_LIMIT), (name, ratios.tolist())
        reports.append(
            {
                "output": name,
                "row_relative_l2": ratios.tolist(),
                "max_absolute_error": (a - e).abs().max().item(),
            }
        )
    return reports


def _progress(phase, report, path, started, deadline):
    report["status"] = phase
    report.setdefault("startup_phases", []).append(
        {"phase": phase, "wall_seconds": time.monotonic() - started}
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    if time.monotonic() >= deadline:
        raise TimeoutError(f"worker deadline after {phase}")


def worker(rank, directory, rendezvous, round_mode, started, deadline):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    report = {"rank": rank, "pid": os.getpid(), "status": "starting", "cases": []}
    path = Path(directory) / f"rank{rank}.json"
    initialized = False
    save(path, report)
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        assert torch.cuda.get_device_capability(rank) == (9, 0)
        assert torch.cuda.get_device_name(rank) == "NVIDIA H20"
        _progress("cuda_initialized", report, path, started, deadline)
        dist.init_process_group(
            "nccl",
            init_method=rendezvous,
            rank=rank,
            world_size=WORLD,
            timeout=timedelta(seconds=8),
            device_id=torch.device("cuda", rank),
        )
        _progress("process_group_initialized", report, path, started, deadline)
        initialized = True
        from tokenspeed_kernel.ops.communication import trtllm as ar
        from tokenspeed_kernel.ops.communication.hopper_mhc import (
            _v41_allreduce_epilogue_kernel,
            prepare_v41_allreduce,
            v41_allreduce_post_pre_norm_quant,
        )
        from tokenspeed_kernel.ops.quantization.triton import (
            triton_quantize_fp8_group32_ue8m0,
        )
        from tokenspeed_kernel.ops.residual.triton import (
            mhc_pre_layer_norm_hc4,
            triton_mhc_post,
        )

        assert ar.ensure_workspace_initialized(rank, dist.group.WORLD, 8, 5120, False)
        _progress("baseline_workspace_initialized", report, path, started, deadline)
        manager = ar._manager_for_group(dist.group.WORLD)
        workspace = prepare_v41_allreduce(
            dist.group.WORLD, torch.device("cuda", rank), 8, round_mode, 1000000000, 8.0
        )
        _progress("candidate_workspace_initialized", report, path, started, deadline)
        report.update(
            status="ready",
            ready_wall_seconds=time.monotonic() - started,
            torch_version=torch.__version__,
            cuda_build=torch.version.cuda,
            device=torch.cuda.get_device_name(rank),
            round_mode=round_mode,
            workspace={
                "max_rows": 8,
                "hidden": 5120,
                "ipc_available": manager.workspace_tensor is not None,
                "mnnvl_available": manager.mnnvl_workspace is not None,
                "use_fp32_lamport": manager.use_fp32_lamport,
            },
            sources={
                str(Path(fn.__code__.co_filename)): digest(fn.__code__.co_filename)
                for fn in (
                    ar.trtllm_workspace_allreduce,
                    v41_allreduce_post_pre_norm_quant,
                    triton_mhc_post,
                    mhc_pre_layer_norm_hc4,
                    triton_quantize_fp8_group32_ue8m0,
                )
            },
        )
        save(path, report)
        for case_index, (rows, quantize) in enumerate(CASES):
            assert time.monotonic() < deadline, "whole sweep wall deadline"
            seed = 1201 + case_index
            common = torch.Generator(device="cpu").manual_seed(seed)
            local = torch.Generator(device="cpu").manual_seed(seed * 17 + rank)
            x = torch.randn((rows, 5120), generator=local).to(torch.bfloat16).cuda(rank)
            residual = (
                torch.randn((rows, 4, 5120), generator=common)
                .to(torch.bfloat16)
                .cuda(rank)
            )
            post = torch.rand((rows, 4), generator=common).cuda(rank)
            comb = torch.rand((rows, 4, 4), generator=common).cuda(rank)
            pre = torch.rand((rows, 4), generator=common).cuda(rank)
            weight = (1 + 0.1 * torch.randn((5120,), generator=common)).cuda(rank)

            def baseline():
                reduced = ar.trtllm_workspace_allreduce(x, dist.group.WORLD)
                assert (
                    reduced is not None
                ), "baseline declined; no silent NCCL substitution"
                hc = triton_mhc_post(reduced, residual, post, comb)
                normalized = torch.empty_like(x)
                mhc_pre_layer_norm_hc4(pre, hc, weight, normalized, eps=1e-06)
                codes, scales = (
                    triton_quantize_fp8_group32_ue8m0(
                        normalized, "token_group", 32, "ue8m0", False
                    )
                    if quantize
                    else (None, None)
                )
                return (hc, normalized, codes, scales)

            def candidate():
                return v41_allreduce_post_pre_norm_quant(
                    workspace, x, residual, post, comb, pre, weight, 1e-06, quantize
                )

            dispatch = []
            original = ar.trtllm_allreduce_fusion

            def trace_dispatch(*arguments, **keywords):
                selected = keywords["workspace_ptrs"]
                dispatch.append(
                    {
                        "backend": (
                            "mnnvl" if selected is manager.mnnvl_workspace else "ipc"
                        ),
                        "workspace_class": type(selected).__name__,
                        "fp32_acc": keywords["fp32_acc"],
                        "use_oneshot": keywords["use_oneshot"],
                        "trigger_completion_at_end": keywords[
                            "trigger_completion_at_end"
                        ],
                    }
                )
                return original(*arguments, **keywords)

            ar.trtllm_allreduce_fusion = trace_dispatch
            try:
                expected = baseline()
            finally:
                ar.trtllm_allreduce_fusion = original
            assert len(dispatch) == 1 and dispatch[0]["fp32_acc"] is False
            compiled = _v41_allreduce_epilogue_kernel.warmup(
                x,
                residual,
                post,
                comb,
                pre,
                weight,
                expected[0],
                expected[1],
                expected[2],
                expected[3],
                workspace.data_ptrs,
                workspace.signal_ptrs,
                8,
                rank,
                round_mode,
                1000000000,
                1e-06,
                quantize,
                num_warps=8,
                grid=(rows,),
            )
            compiled._init_handles()
            dist.barrier()
            actual = candidate()
            case = {
                "rows": rows,
                "quantize": quantize,
                "seed": seed,
                "dispatch": dispatch[0],
                "status": "checking",
                "pairs": [],
            }
            report["cases"].append(case)
            case["eager_correctness"] = compare(actual, expected, torch)
            if quantize:
                codes, scales = triton_quantize_fp8_group32_ue8m0(
                    actual[1], "token_group", 32, "ue8m0", False
                )
                assert torch.equal(codes.view(torch.uint8), actual[2].view(torch.uint8))
                assert torch.equal(scales, actual[3])
                case["exact_candidate_quantization"] = True
            graphs = []
            for fn in (baseline, candidate):
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(CALLS):
                        output = fn()
                graphs.append((graph, output))
            for graph, _ in graphs:
                graph.replay()
            torch.cuda.synchronize()
            case["graph_correctness"] = compare(graphs[1][1], graphs[0][1], torch)
            if quantize:
                codes, scales = triton_quantize_fp8_group32_ue8m0(
                    graphs[1][1][1], "token_group", 32, "ue8m0", False
                )
                assert torch.equal(
                    codes.view(torch.uint8), graphs[1][1][2].view(torch.uint8)
                )
                assert torch.equal(scales, graphs[1][1][3])
            for pair in range(PAIRS):
                values = {}
                order = (0, 1) if (pair + case_index) % 2 == 0 else (1, 0)
                for arm in order:
                    assert time.monotonic() < deadline, "whole sweep wall deadline"
                    dist.barrier()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(REPLAYS):
                        graphs[arm][0].replay()
                    end.record()
                    end.synchronize()
                    values[("baseline", "candidate")[arm]] = (
                        begin.elapsed_time(end) * 1000 / (CALLS * REPLAYS)
                    )
                case["pairs"].append(
                    {"order": list(order), "us_per_complete_call": values}
                )
            case["status"] = "complete"
            case["wall_seconds_since_start"] = time.monotonic() - started
            save(path, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
            traceback=traceback.format_exc(),
        )
        if "actual" in locals() and "expected" in locals():
            raw = Path(directory) / f"rank{rank}-failed-outputs.pt"
            torch.save(
                {
                    name: [t.detach().cpu() if t is not None else None for t in values]
                    for (name, values) in (("actual", actual), ("expected", expected))
                },
                raw,
            )
            report["failed_raw_outputs"] = {"path": str(raw), "sha256": digest(raw)}
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        save(path, report)
        if initialized and report["status"] == "complete":
            dist.destroy_process_group()


def summarize(rank_reports):
    assert len(rank_reports) == WORLD
    assert {r["rank"] for r in rank_reports} == set(range(WORLD))
    assert all(
        (
            r["status"] == "complete" and len(r["cases"]) == len(CASES)
            for r in rank_reports
        )
    )
    summaries = []
    for index, (rows, quantize) in enumerate(CASES):
        cases = [rank["cases"][index] for rank in rank_reports]
        assert all(
            (
                c["rows"] == rows
                and c["quantize"] == quantize
                and (c["status"] == "complete")
                and (len(c["pairs"]) == PAIRS)
                for c in cases
            )
        )
        assert all((c["dispatch"] == cases[0]["dispatch"] for c in cases))
        worst_rank_pairs = {
            arm: [
                max((c["pairs"][p]["us_per_complete_call"][arm] for c in cases))
                for p in range(PAIRS)
            ]
            for arm in ("baseline", "candidate")
        }
        medians = {
            arm: statistics.median(values) for (arm, values) in worst_rank_pairs.items()
        }
        summaries.append(
            {
                "rows": rows,
                "quantize": quantize,
                "dispatch": cases[0]["dispatch"],
                "per_pair_max_rank_us": worst_rank_pairs,
                "median_max_rank_us": medians,
                "baseline_over_candidate": medians["baseline"] / medians["candidate"],
            }
        )
    return summaries


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--round-mode", choices=("bf16_step", "fp32"), required=True)
    args = parser.parse_args()
    assert 0 < args.seconds <= 35
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD and 0 <= rank < WORLD
    assert (
        int(os.environ["LOCAL_RANK"]) == rank
    ), "single host/shared visible ordinals required"
    directory = args.output.with_suffix(".ranks")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank{rank}.entry.json").open("x") as stream:
        json.dump(
            {
                "rank": rank,
                "pid": os.getpid(),
                "status": "entering",
                "benchmark_sha256": digest(__file__),
                "seconds_limit": args.seconds,
            },
            stream,
        )
    assert not (directory / f"rank{rank}.json").exists()
    deadline = started + args.seconds
    finished = threading.Event()

    def watchdog():
        if not finished.wait(max(0, deadline - time.monotonic())):
            marker = directory / f"rank{rank}.watchdog.json"
            marker.write_text(
                json.dumps(
                    {
                        "status": "worker_timeout",
                        "rank": rank,
                        "pid": os.getpid(),
                        "wall_seconds": time.monotonic() - started,
                        "drain_proven": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    owns_output = False
    try:
        if rank == 0:
            with args.output.open("x") as stream:
                owns_output = True
                json.dump(
                    {
                        "status": "running",
                        "benchmark_sha256": digest(__file__),
                        "seconds_limit_per_rank": args.seconds,
                        "calls_per_graph": CALLS,
                        "replays": REPLAYS,
                        "pairs": PAIRS,
                        "rank_count": WORLD,
                        "round_mode": args.round_mode,
                        "relative_l2_limit": L2_LIMIT,
                        "timing_scope": "paired complete-chain CUDA graphs with hot inputs; per-pair maximum rank latency",
                    },
                    stream,
                )
        worker(rank, str(directory), "env://", args.round_mode, started, deadline)
        report_path = directory / f"rank{rank}.json"
        with (directory / f"rank{rank}.done.json").open("x") as stream:
            json.dump(
                {
                    "rank": rank,
                    "worker_returned": True,
                    "report_sha256": digest(report_path),
                },
                stream,
            )
        if rank == 0:
            markers = [directory / f"rank{peer}.done.json" for peer in range(WORLD)]
            while not all(marker.exists() for marker in markers):
                assert time.monotonic() < deadline, "all-rank completion deadline"
                assert not list(directory.glob("rank*.watchdog.json")), "peer timed out"
                assert not list(directory.glob("rank*.entry-error.json")), "peer failed"
                time.sleep(0.01)
            ranks = []
            for peer, marker in enumerate(markers):
                report_path = directory / f"rank{peer}.json"
                assert json.loads(marker.read_text())["report_sha256"] == digest(
                    report_path
                )
                ranks.append(json.loads(report_path.read_text()))
            record = json.loads(args.output.read_text())
            record.update(
                status="complete",
                summary=summarize(ranks),
                ranks=ranks,
                wall_seconds=time.monotonic() - started,
                rank_artifacts={
                    p.name: digest(p) for p in directory.glob("rank*.json")
                },
            )
            save(args.output, record)
    except BaseException as error:
        (directory / f"rank{rank}.entry-error.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "wall_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
        if rank == 0 and owns_output:
            record = json.loads(args.output.read_text())
            record.update(
                status="failed",
                error={"type": type(error).__name__, "message": str(error)},
                wall_seconds=time.monotonic() - started,
            )
            save(args.output, record)
        raise
    finally:
        finished.set()


if __name__ == "__main__":
    main()
