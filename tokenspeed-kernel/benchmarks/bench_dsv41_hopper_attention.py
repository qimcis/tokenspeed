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

"""Compare padded and compact V4.1 attention chains on synthetic Hopper inputs.

This is an attention-chain benchmark, not a model or serving benchmark. Both
arms include an identical raw-query clone to stand in for a fresh projection
output and prevent in-place RoPE accumulating across CUDA graph replays.
Run from the repository root with PYTHONPATH=python:tokenspeed-kernel/python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import triton as portable

_DIM = 512
_ROPE_DIM = 64
_WINDOW = 128
_GLOBAL = 512
_ARM_NAMES = ("padded", "compact")


@dataclass
class Case:
    definition: dict
    arms: dict[str, Callable[[], torch.Tensor]]
    reference: Callable[[], torch.Tensor]
    input_query: torch.Tensor


@dataclass
class Replay:
    call: Callable[[], object]
    result: Callable[[], torch.Tensor]


def _csv_ints(text: str) -> list[int]:
    result = [int(value) for value in text.split(",")]
    if not result or any(value < 1 for value in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def _csv_choices(text: str, allowed: tuple[str, ...]) -> list[str]:
    result = text.split(",")
    if not result or any(value not in allowed for value in result):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values in {allowed}"
        )
    return result


def _rope_tables(max_position: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    inverse_freq = 10000.0 ** (
        -torch.arange(0, _ROPE_DIM, 2, device=device, dtype=torch.float32) / _ROPE_DIM
    )
    angles = torch.arange(max_position, device=device)[:, None] * inverse_freq
    return (
        torch.cat((angles.cos(), angles.sin()), dim=-1),
        torch.cat((angles.cos(), -angles.sin()), dim=-1),
    )


def _mask_slots(slots: torch.Tensor, pattern: str) -> torch.Tensor:
    lengths = torch.full(
        (slots.shape[0],), slots.shape[1], device=slots.device, dtype=torch.int32
    )
    if pattern == "ragged":
        lengths = (
            torch.arange(slots.shape[0], device=slots.device, dtype=torch.int32) * 31
            + slots.shape[1] // 2
        ) % (slots.shape[1] + 1)
        slots[:, ::11] = -1
    elif pattern == "empty":
        lengths.zero_()
        slots.fill_(-1)
    elif pattern != "full":
        raise ValueError(f"unknown visibility pattern: {pattern}")
    return lengths


def _paged_segment(
    tokens: int, width: int, cache_format: str, pattern: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Separate physical rows per query avoid a single all-shared hot-cache case.
    rows = torch.randn(tokens * width, _DIM, dtype=torch.bfloat16, device=device)
    row_bytes = 528 if cache_format == "swa" else 288
    cache = torch.zeros(
        ((rows.shape[0] + 63) // 64 + 1, 64, row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(64, 64 + rows.shape[0], device=device, dtype=torch.int64)
    dsv41.cache_scatter(rows, cache, slots, cache_format)
    slots = slots.reshape(tokens, width)
    return cache, slots, _mask_slots(slots, pattern)


def _rotate_query(
    source: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, padded: bool
) -> torch.Tensor:
    query = source.clone()
    if padded:
        return dsv41.rope_pad_query(query, positions, table, "triton")
    return dsv41.rope_inplace(query, positions, table, "triton")


def _finish(
    output: torch.Tensor, heads: int, positions: torch.Tensor, inverse: torch.Tensor
) -> torch.Tensor:
    # Preserve the strided slice the model passes to inverse RoPE.
    return dsv41.rope_inplace(output[:, :heads], positions, inverse, "triton")


def _softmax_reference(
    query: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    scores = torch.bmm(query.float(), kv.float().transpose(1, 2)) * _DIM**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    sink_rows = sink[None, :, None].expand(query.shape[0], -1, 1)
    probabilities = torch.cat((scores, sink_rows), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.bmm(probabilities, kv.float()).to(query.dtype)


def _target_case(
    tokens: int,
    heads: int,
    with_global: bool,
    pattern: str,
    query_chunk_size: int,
    device: torch.device,
) -> Case:
    source = torch.randn(tokens, heads, _DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(tokens, device=device, dtype=torch.int64) + _WINDOW
    tables = _rope_tables(tokens + _WINDOW + 1, device)
    sink = torch.linspace(-3, 3, heads, dtype=torch.float32, device=device)
    padded_heads = 64 if heads <= 64 else 128
    padded_sink = F.pad(sink, (0, padded_heads - heads), value=-float("inf"))
    swa, swa_slots, swa_lengths = _paged_segment(
        tokens, _WINDOW, "swa", pattern, device
    )
    glob, global_slots, global_lengths = (
        _paged_segment(tokens, _GLOBAL, "global", pattern, device)
        if with_global
        else (None, None, None)
    )

    def forward(padded: bool) -> torch.Tensor:
        query = _rotate_query(source, positions, tables[0], padded)
        # Pin the registration: a native Blackwell donor must not silently
        # replace either arm when this script is run on another architecture.
        output = portable.selected_attention(
            query,
            swa,
            swa_slots,
            swa_lengths,
            glob,
            global_slots,
            global_lengths,
            padded_sink if padded else sink,
            _DIM**-0.5,
            None,
            query_chunk_size,
            None,
            None,
            None,
        )
        return _finish(output, heads, positions, tables[1])

    def reference() -> torch.Tensor:
        parts, masks = [], []
        segments = [(swa, swa_slots, swa_lengths, "swa")]
        if with_global:
            segments.append((glob, global_slots, global_lengths, "global"))
        for cache, slots, lengths, fmt in segments:
            parts.append(dsv41.cache_gather(cache, slots, fmt, None))
            masks.append(
                (torch.arange(slots.shape[1], device=device) < lengths[:, None])
                & (slots >= 0)
                & (slots < cache.shape[0] * 64)
            )
        query = _rotate_query(source, positions, tables[0], False)
        output = _softmax_reference(
            query, torch.cat(parts, dim=1), torch.cat(masks, dim=1), sink
        )
        return _finish(output, heads, positions, tables[1])

    return Case(
        definition={
            "operation": "target_selected",
            "tokens": tokens,
            "live_heads": heads,
            "padded_heads": padded_heads,
            "head_dim": _DIM,
            "swa_width": _WINDOW,
            "global_width": _GLOBAL if with_global else 0,
            "visibility": pattern,
            "query_chunk_size": query_chunk_size,
            "registration": "triton_dsv41_selected_attention",
            "chain": "query_clone,query_rope,gather_dequant,attention,live_slice,inverse_rope",
        },
        arms={"padded": lambda: forward(True), "compact": lambda: forward(False)},
        reference=reference,
        input_query=source,
    )


def _dspark_case(
    batch: int, heads: int, block: int, pattern: str, device: torch.device
) -> Case:
    # Import the real runtime chain only when explicitly requested. Missing
    # runtime dependencies fail the case, rather than silently timing a copy.
    from tokenspeed.runtime.models.deepseek_v41_dspark import (
        _quantized_kv,
        _window_rows,
        _WindowAttention,
    )

    tokens = batch * block
    source = torch.randn(tokens, heads, _DIM, dtype=torch.bfloat16, device=device)
    positions = (
        torch.arange(block, device=device, dtype=torch.int64)[None, :]
        .expand(batch, -1)
        .reshape(-1)
        + _WINDOW
    )
    tables = _rope_tables(_WINDOW + block + 1, device)
    sink = torch.linspace(-3, 3, heads, dtype=torch.float32, device=device)
    padded_heads = 64 if heads <= 64 else 128
    padded_sink = F.pad(sink, (0, padded_heads - heads), value=-float("inf"))
    window = torch.randn(batch * 2 + 1, 64, _DIM, dtype=torch.bfloat16, device=device)
    window[0].zero_()
    slots = torch.arange(64, 64 + batch * _WINDOW, device=device).reshape(
        batch, _WINDOW
    )
    lengths = _mask_slots(slots, pattern)
    slots.masked_fill_(
        torch.arange(_WINDOW, device=device)[None, :] >= lengths[:, None], -1
    )
    swa = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
    backend = _WindowAttention(positions, window, slots, block)

    def forward(padded: bool) -> torch.Tensor:
        query = _rotate_query(source, positions, tables[0], padded)
        output = backend.forward_v41(
            query,
            swa,
            layer_id=0,
            positions=positions,
            request_indices=backend.meta.request_indices,
            forward_mode=None,
            index_q=None,
            index_weights=None,
            attn_sink=padded_sink if padded else sink,
            softmax_scale=_DIM**-0.5,
            index_process_group=None,
            swa_rope_cache=tables[0],
        )
        return _finish(output, heads, positions, tables[1])

    def reference() -> torch.Tensor:
        current = dsv41.rope_inplace(swa.clone(), positions, tables[0], "triton")
        kv = torch.cat(
            (
                _window_rows(window, slots),
                _quantized_kv(current).view(batch, block, -1),
            ),
            dim=1,
        ).repeat_interleave(block, dim=0)
        valid = torch.cat(
            (slots >= 0, torch.ones(batch, block, device=device, dtype=torch.bool)),
            dim=1,
        ).repeat_interleave(block, dim=0)
        query = _rotate_query(source, positions, tables[0], False)
        output = _softmax_reference(query, kv, valid, sink)
        return _finish(output, heads, positions, tables[1])

    return Case(
        definition={
            "operation": "dspark_window",
            "tokens": tokens,
            "batch": batch,
            "block": block,
            "live_heads": heads,
            "padded_heads": padded_heads,
            "head_dim": _DIM,
            "history_width": _WINDOW,
            "visibility": pattern,
            "registration": "runtime._WindowAttention.forward_v41",
            "chain": "query_clone,query_rope,history_gather,current_kv_rope_quant,attention,live_slice,inverse_rope",
            "empty_visibility": "history only; current noncausal proposal block remains visible",
        },
        arms={"padded": lambda: forward(True), "compact": lambda: forward(False)},
        reference=reference,
        input_query=source,
    )


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _capture(forward: Callable[[], torch.Tensor], warmup: int) -> Replay:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = forward()
    torch.cuda.current_stream().wait_stream(stream)
    return Replay(call=graph.replay, result=lambda: output)


def _validate(case: Case, args: argparse.Namespace) -> dict:
    original = case.input_query.clone()
    expected = case.reference()
    outputs = {name: case.arms[name]() for name in _ARM_NAMES}
    errors = {}
    for name, output in outputs.items():
        torch.testing.assert_close(output, expected, rtol=args.rtol, atol=args.atol)
        errors[f"{name}_reference_max_abs"] = _max_error(output, expected)
    target_case = case.definition["operation"] == "target_selected"
    torch.testing.assert_close(
        outputs["compact"],
        outputs["padded"],
        rtol=0 if target_case else args.rtol,
        atol=0 if target_case else args.atol,
    )
    errors["between_arms_contract"] = (
        "exact_retained_heads" if target_case else "configured_bf16_tolerance"
    )
    errors["between_arms_max_abs"] = _max_error(outputs["compact"], outputs["padded"])
    if "graph" in args.modes:
        for name in _ARM_NAMES:
            replay = _capture(case.arms[name], 2)
            # Replaying unchanged inputs twice catches cumulative in-place RoPE.
            for _ in range(2):
                replay.call()
                torch.testing.assert_close(
                    replay.result(), expected, rtol=args.rtol, atol=args.atol
                )
            # A graph must consume live query bytes, not a stale captured value.
            case.input_query.mul_(0.75)
            changed_reference = case.reference()
            replay.call()
            torch.testing.assert_close(
                replay.result(), changed_reference, rtol=args.rtol, atol=args.atol
            )
            case.input_query.copy_(original)
            del replay
    torch.testing.assert_close(case.input_query, original, rtol=0, atol=0)
    torch.cuda.synchronize()
    return errors


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "samples_us": values,
        "median_us": median,
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p10_us": ordered[round((len(ordered) - 1) * 0.1)],
        "p90_us": ordered[round((len(ordered) - 1) * 0.9)],
        "relative_mad": statistics.median(abs(value - median) for value in values)
        / median,
    }


def _measure(
    case: Case, mode: str, args: argparse.Namespace, rng: random.Random
) -> dict:
    if mode == "graph":
        replays = {name: _capture(case.arms[name], args.warmup) for name in _ARM_NAMES}
        calls = {name: replay.call for name, replay in replays.items()}
    else:
        calls = case.arms
    for _ in range(args.warmup):
        for name in _ARM_NAMES:
            calls[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in _ARM_NAMES}
    orders = []
    for _ in range(args.samples):
        order = list(_ARM_NAMES)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.calls_per_sample):
                calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(
                start.elapsed_time(end) * 1000.0 / args.calls_per_sample
            )
    summaries = {name: _summary(values) for name, values in samples.items()}
    return {
        "mode": mode,
        "arm_order": orders,
        "arms": summaries,
        "median_speedup": summaries["padded"]["median_us"]
        / summaries["compact"]["median_us"],
        "paired_speedups": [
            baseline / candidate
            for baseline, candidate in zip(
                samples["padded"], samples["compact"], strict=True
            )
        ],
    }


def _command(command: list[str], root: Path) -> str:
    try:
        return subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def _metadata(args: argparse.Namespace, device: torch.device) -> dict:
    root = Path(__file__).resolve().parents[2]
    props = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("torch", "tokenspeed-kernel", "tokenspeed-triton", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "command": sys.argv,
        "revision": _command(["git", "rev-parse", "HEAD"], root),
        "git_status": _command(["git", "status", "--short"], root),
        "tracked_diff_sha256": hashlib.sha256(
            _command(["git", "diff", "HEAD", "--"], root).encode()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_compute_capability": [props.major, props.minor],
        "gpu_memory_bytes": props.total_memory,
        "gpu_sm_count": props.multi_processor_count,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu",
                "--format=csv,noheader",
            ],
            root,
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scope": "synthetic attention-chain A/B; excludes Q/KV and output projections, target KV writes, MoE, TP communication, scheduler, model weights and token generation",
        "cache_mode": "warm repeated inputs; no cache flush; not a cold-HBM bandwidth benchmark",
        "timer": "CUDA events; eager includes CPU launch starvation; graph captures one whole chain per replay",
        "arm_contract": "padded uses FlashMLA 64/128-head RoPE; compact uses live-head in-place RoPE; both include identical input clone, attention, live-head slice, inverse RoPE",
        "projection_contract": "both arms start from the same live-head Q projection result; zero-padding occurs after projection only in the padded arm; neither arm measures a Q projection GEMM",
        "result_units": "microseconds per attention-chain invocation; no model tokens/sec or serving throughput is reported",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--operations",
        type=lambda x: _csv_choices(x, ("target", "dspark")),
        default=["target", "dspark"],
    )
    parser.add_argument(
        "--modes",
        type=lambda x: _csv_choices(x, ("eager", "graph")),
        default=["eager", "graph"],
    )
    parser.add_argument("--tokens", type=_csv_ints, default=[1, 5, 6, 24, 96, 384])
    parser.add_argument("--dspark-batches", type=_csv_ints, default=[1, 2, 4, 16, 64])
    parser.add_argument("--heads", type=_csv_ints, default=[8])
    parser.add_argument("--dspark-block", type=int, default=5)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--visibility", choices=("full", "ragged", "empty"), default="full"
    )
    parser.add_argument(
        "--validation-patterns",
        type=lambda x: _csv_choices(x, ("full", "ragged", "empty")),
        default=["full", "ragged", "empty"],
    )
    parser.add_argument("--validation-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--calls-per-sample", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.008)
    parser.add_argument("--rtol", type=float, default=0.008)
    parser.add_argument("--allow-non-hopper", action="store_true")
    args = parser.parse_args()
    for name in (
        "dspark_block",
        "query_chunk_size",
        "validation_runs",
        "warmup",
        "samples",
        "calls_per_sample",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(head > 128 for head in args.heads):
        parser.error("heads must be within [1,128]")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances cannot be negative")
    return args


def _save(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("this benchmark requires an NVIDIA CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0) and not args.allow_non_hopper:
        raise RuntimeError(
            "expected Hopper SM90; pass --allow-non-hopper to label a comparison on another architecture"
        )
    report = {
        "schema_version": 1,
        "metadata": _metadata(args, device),
        "cases": [],
        "status": "running",
    }
    _save(args.output, report)
    factories = []
    for heads in args.heads:
        if "target" in args.operations:
            for tokens in args.tokens:
                for with_global in (False, True):
                    factories.append(("target", (tokens, heads, with_global)))
        if "dspark" in args.operations:
            for batch in args.dspark_batches:
                factories.append(("dspark", (batch, heads, args.dspark_block)))
    rng = random.Random(args.seed)
    try:
        for operation, parameters in factories:

            def make_case(pattern: str) -> Case:
                if operation == "target":
                    return _target_case(
                        *parameters, pattern, args.query_chunk_size, device
                    )
                return _dspark_case(*parameters, pattern, device)

            checks = []
            for run in range(args.validation_runs):
                for pattern in args.validation_patterns:
                    torch.manual_seed(args.seed + run)
                    validation_case = make_case(pattern)
                    checks.append(
                        {
                            "seed": args.seed + run,
                            "visibility": pattern,
                            **_validate(validation_case, args),
                        }
                    )
                    del validation_case
            torch.manual_seed(args.seed)
            case = make_case(args.visibility)
            measurements = [_measure(case, mode, args, rng) for mode in args.modes]
            result = {
                "definition": case.definition,
                "correctness": checks,
                "measurements": measurements,
            }
            report["cases"].append(result)
            _save(args.output, report)
            print(
                json.dumps(
                    {
                        "case": case.definition,
                        "timings": [
                            {
                                "mode": entry["mode"],
                                "padded_us": entry["arms"]["padded"]["median_us"],
                                "compact_us": entry["arms"]["compact"]["median_us"],
                                "speedup": entry["median_speedup"],
                            }
                            for entry in measurements
                        ],
                    }
                ),
                flush=True,
            )
            del case
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        _save(args.output, report)


if __name__ == "__main__":
    main()
