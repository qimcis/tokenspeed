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

"""Owned TP8 correctness stress. Authoring/importing this module launches no GPU work.

Run only through the task's charged controller, using torchrun --nproc-per-node=8.
This is a correctness worker, not a performance benchmark or runtime activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

MODES = ("bf16_step", "fp32")
EAGER_ROWS = (1, 8, 2, 4, 8, 1, 4, 2)
GRAPH_ROWS = (1, 8, 1, 8, 8, 1)
LIMIT = 0.006


def relative_l2_rows(reference, actual):
    """Strict per-row relative errors, with zero reference handled explicitly."""
    rows = reference.shape[0] if reference.ndim > 1 else 1
    expected = reference.double().reshape(rows, -1)
    observed = actual.double().reshape(rows, -1)
    numerators = (observed - expected).norm(dim=1).tolist()
    denominators = expected.norm(dim=1).tolist()
    return [
        (
            numerator / denominator
            if denominator
            else (0.0 if numerator == 0 else float("inf"))
        )
        for numerator, denominator in zip(numerators, denominators)
    ]


def relative_l2(reference, actual):
    return max(relative_l2_rows(reference, actual))


def rank_input(torch, rows, rank, epoch):
    """CPU rank-known BF16 inputs; every rank independently derives every peer."""
    values = torch.arange(rows * 5120, dtype=torch.float32).reshape(rows, 5120)
    values = ((values + epoch * 13) % 97 - 48) * (1.0 / 128.0)
    return (values + (rank - 3.5) * (1.0 / 16.0)).to(torch.bfloat16)


def expected_sum(torch, rows, epoch, mode):
    if mode not in MODES:
        raise ValueError("unsupported round mode")
    value = rank_input(torch, rows, 0, epoch).float()
    for peer in range(1, 8):
        value = value + rank_input(torch, rows, peer, epoch).float()
        if mode == "bf16_step":
            value = value.to(torch.bfloat16).float()
    return value.to(torch.bfloat16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--epilogue-sha", required=True)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 25:
        raise ValueError("worker deadline must be within [5,25] seconds")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("this worker requires torchrun with exactly eight ranks")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rank-{rank}.json"
    with output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    finished = threading.Event()

    def watchdog():
        if not finished.wait(args.seconds):
            (args.output_dir / f"rank-{rank}-timeout.json").write_text(
                json.dumps(
                    {
                        "status": "worker_timeout",
                        "rank": rank,
                        "pid": os.getpid(),
                        "elapsed_seconds": time.monotonic() - started,
                        "drain_proven": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    record = {
        "status": "running",
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "worker_seconds_cap": args.seconds,
        "gate": {"finite": True, "normalized_l2_max_per_row": LIMIT},
        "eager_rows": EAGER_ROWS,
        "graph_rows": GRAPH_ROWS,
        "checks": [],
        "gpu_drain_proven": False,
        "rank_skew": {
            "rank": 7,
            "delay_seconds": 0.030,
            "mode": "bf16_step",
            "epoch": 0,
        },
        "missing_peer_timeout_tested": False,
    }

    def save():
        record["wall_seconds"] = time.monotonic() - started
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    try:
        import torch
        import torch.distributed as dist
        from tokenspeed_kernel.ops.communication import hopper_mhc as candidate
        from tokenspeed_kernel.ops.quantization.triton import (
            triton_quantize_fp8_group32_ue8m0,
        )
        from tokenspeed_kernel.ops.residual import hopper_v41 as epilogue
        from tokenspeed_kernel.ops.residual.triton import (
            mhc_pre_layer_norm_hc4,
            triton_mhc_post,
        )

        torch.set_num_threads(1)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("SM90 required")
        if sha(candidate.__file__) != args.source_sha:
            raise RuntimeError("collective source SHA differs from frozen input")
        if sha(epilogue.__file__) != args.epilogue_sha:
            raise RuntimeError("epilogue source SHA differs from frozen input")
        record["sources"] = {
            "worker": sha(__file__),
            "collective": args.source_sha,
            "epilogue": args.epilogue_sha,
            "original_residual": sha(triton_mhc_post.__code__.co_filename),
            "original_quantization": sha(
                triton_quantize_fp8_group32_ue8m0.__code__.co_filename
            ),
        }
        record["torch"] = torch.__version__
        record["device_name"] = torch.cuda.get_device_name(device)
        save()
        dist.init_process_group("nccl", timeout=timedelta(seconds=12))
        generator = torch.Generator(device="cpu").manual_seed(9037)
        residual = (
            (torch.randn((8, 4, 5120), generator=generator) * 0.2)
            .to(torch.bfloat16)
            .to(device)
        )
        post = (torch.rand((8, 4), generator=generator) * 0.2).to(device)
        comb = (torch.rand((8, 4, 4), generator=generator) * 0.2).to(device)
        pre = (torch.rand((8, 4), generator=generator) * 0.25).to(device)
        weight = (
            torch.linspace(0.5, 1.5, 5120, dtype=torch.float32)
            .to(torch.bfloat16)
            .to(device)
        )
        weight32 = torch.linspace(0.5, 1.5, 5120, dtype=torch.float32).to(device)
        inputs = {
            rows: torch.empty((rows, 5120), dtype=torch.bfloat16, device=device)
            for rows in (1, 2, 4, 8)
        }

        def original(reduced, rows, norm_weight):
            hc = triton_mhc_post(reduced, residual[:rows], post[:rows], comb[:rows])
            norm = torch.empty_like(reduced)
            mhc_pre_layer_norm_hc4(pre[:rows], hc, norm_weight, norm, eps=1e-6)
            q, s = triton_quantize_fp8_group32_ue8m0(
                norm, "token_group", 32, "ue8m0", False
            )
            return hc, norm, q, s

        def fused(workspace, rows, norm_weight):
            return candidate.v41_allreduce_post_pre_norm_quant(
                workspace,
                inputs[rows],
                residual[:rows],
                post[:rows],
                comb[:rows],
                pre[:rows],
                norm_weight,
                1e-6,
                True,
            )

        def validate(actual, expected, metadata, exact):
            a = actual.float().cpu()
            e = expected.float().cpu()
            finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
            row_errors = relative_l2_rows(e, a) if finite else []
            error = max(row_errors) if finite else float("inf")
            delta = (a - e).abs().flatten()
            worst = int(delta.argmax())
            item = {
                **metadata,
                "finite": finite,
                "normalized_l2": error if error != float("inf") else None,
                "per_row_normalized_l2": [
                    value if value != float("inf") else None for value in row_errors
                ],
                "max_abs": float(delta[worst]) if finite else None,
                "worst_flat_index": worst,
                "reference_at_worst": float(e.flatten()[worst]) if finite else None,
                "actual_at_worst": float(a.flatten()[worst]) if finite else None,
                "exact": bool(torch.equal(e, a)),
                "actual_first8": a.flatten()[:8].tolist() if finite else [],
                "reference_first8": e.flatten()[:8].tolist() if finite else [],
            }
            record["checks"].append(item)
            save()
            if not finite or error > LIMIT or (exact and not item["exact"]):
                raise AssertionError(f"numerical failure: {metadata}")

        def validate_chain(actual, expected, metadata):
            validate(actual[0], expected[0], {**metadata, "output": "hc"}, False)
            validate(
                actual[1], expected[1], {**metadata, "output": "normalized"}, False
            )
            q, scales = triton_quantize_fp8_group32_ue8m0(
                actual[1], "token_group", 32, "ue8m0", False
            )
            exact = bool(torch.equal(q.view(torch.uint8), actual[2].view(torch.uint8)))
            exact = exact and bool(torch.equal(scales, actual[3]))
            record["checks"].append(
                {
                    **metadata,
                    "output": "quantized_actual_normalized",
                    "exact": exact,
                    "baseline_code_mismatches": int(
                        torch.count_nonzero(
                            expected[2].view(torch.uint8) != actual[2].view(torch.uint8)
                        )
                    ),
                    "baseline_scale_mismatches": int(
                        torch.count_nonzero(expected[3] != actual[3])
                    ),
                }
            )
            save()
            if not exact:
                raise AssertionError(
                    "group32 quantization differs on identical normalized input"
                )

        for mode in MODES:
            workspace = candidate.prepare_v41_allreduce(
                dist.group.WORLD, device, 8, mode, 1_000_000_000, 10.0
            )
            # Compile every collective specialization before peers enter a
            # handshake. warmup compiles/loads only, without launching kernels.
            warm = original(expected_sum(torch, 8, 0, mode).to(device), 8, weight)
            candidate._v41_allreduce_kernel.warmup(
                inputs[8],
                warm[1],
                workspace.data_ptrs,
                workspace.signal_ptrs,
                8,
                rank,
                mode,
                1_000_000_000,
                num_warps=8,
                grid=(8,),
            )
            candidate._v41_allreduce_epilogue_kernel.warmup(
                inputs[8],
                residual,
                post,
                comb,
                pre,
                weight,
                warm[0],
                warm[1],
                warm[2],
                warm[3],
                workspace.data_ptrs,
                workspace.signal_ptrs,
                8,
                rank,
                mode,
                1_000_000_000,
                1e-6,
                True,
                num_warps=8,
                grid=(8,),
            )
            candidate._v41_allreduce_epilogue_kernel.warmup(
                inputs[8],
                residual,
                post,
                comb,
                pre,
                weight32,
                warm[0],
                warm[1],
                warm[2],
                warm[3],
                workspace.data_ptrs,
                workspace.signal_ptrs,
                8,
                rank,
                mode,
                1_000_000_000,
                1e-6,
                True,
                num_warps=8,
                grid=(8,),
            )
            torch.cuda.synchronize()
            dist.barrier()
            for epoch, rows in enumerate(EAGER_ROWS):
                inputs[rows].copy_(rank_input(torch, rows, rank, epoch))
                expected = expected_sum(torch, rows, epoch, mode).to(device)
                reference = original(expected, rows, weight)
                torch.cuda.synchronize()
                dist.barrier()
                if mode == "bf16_step" and epoch == 0 and rank == 7:
                    time.sleep(0.030)
                actual = candidate.v41_allreduce(workspace, inputs[rows])
                validate(
                    actual,
                    expected,
                    {
                        "kind": "eager",
                        "mode": mode,
                        "rows": rows,
                        "epoch": epoch,
                        "output": "allreduce",
                    },
                    True,
                )
                dist.barrier()
                result = fused(workspace, rows, weight)
                validate_chain(
                    result,
                    reference,
                    {
                        "kind": "eager",
                        "mode": mode,
                        "rows": rows,
                        "epoch": epoch,
                    },
                )

            for rows in (1, 8):
                epoch = 50 + rows
                inputs[rows].copy_(rank_input(torch, rows, rank, epoch))
                reference = original(
                    expected_sum(torch, rows, epoch, mode).to(device), rows, weight32
                )
                torch.cuda.synchronize()
                dist.barrier()
                result = fused(workspace, rows, weight32)
                validate_chain(
                    result,
                    reference,
                    {
                        "kind": "eager_fp32_weight",
                        "mode": mode,
                        "rows": rows,
                        "epoch": epoch,
                        "norm_weight_dtype": "float32",
                    },
                )

            graphs = {}
            graph_outputs = {}
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            for rows in (1, 8):
                graph = torch.cuda.CUDAGraph()
                # Warmup already compiled this exact kernel; capture records
                # only. Replay on the default stream is externally ordered
                # after synchronized capture and never overlaps another graph.
                with torch.cuda.graph(graph, stream=capture_stream):
                    graph_outputs[rows] = fused(workspace, rows, weight)
                graphs[rows] = graph
            torch.cuda.synchronize()
            dist.barrier()
            for index, rows in enumerate(GRAPH_ROWS):
                epoch = 100 + index
                inputs[rows].copy_(rank_input(torch, rows, rank, epoch))
                reference = original(
                    expected_sum(torch, rows, epoch, mode).to(device), rows, weight
                )
                torch.cuda.synchronize()
                dist.barrier()
                graphs[rows].replay()
                validate_chain(
                    graph_outputs[rows],
                    reference,
                    {
                        "kind": "graph_mutated_input",
                        "mode": mode,
                        "rows": rows,
                        "epoch": epoch,
                    },
                )
            torch.cuda.synchronize()
            dist.barrier()
            if torch.count_nonzero(workspace.signals).item() != 0:
                raise AssertionError(
                    "signals did not return to zero after serial calls"
                )
            del graphs, graph_outputs, workspace

        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
        record["status"] = "passed"
        record["owned_cuda_work_synchronized"] = True
        save()
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        finished.set()


if __name__ == "__main__":
    main()
