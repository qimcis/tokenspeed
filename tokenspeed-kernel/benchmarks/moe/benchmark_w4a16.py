#!/usr/bin/env python3
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

"""Bounded paired local-expert chain benchmark with validation and artifact pins.

This is synthetic single-layer, single-GPU EP-partial evidence, not serving or
distributed throughput. Mapping/routing, both GEMMs, clamp10 SwiGLU and weighted
finalization are timed. Gate/top-k selection, EP communication, shared experts,
and the model's routed factor1.5 are outside both arms. Preparation is charged
to worker wall time but excluded from graph replay latency.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import signal
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SHAPES = (1, 2, 4, 8, 128, 1024)
EVICTION_BYTES = 256 * 1024 * 1024
COLD_REPEATS = 5
SEMANTICS = dict(
    hidden=5120,
    intermediate=2304,
    top_k=6,
    swiglu_limit=10,
    local_experts=48,
    ep_size=8,
    routed_scale_inside_adapter=1.0,
)
TOLERANCES = dict(
    row_relative_l2=0.006,
    max_absolute_base=0.002,
    max_absolute_row_scale=0.02,
    exact_zero_reference_rows=True,
)
SOURCES = {
    "candidate": "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/cutlass_mxfp4.py",
    "candidate_native": "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mxfp4.py",
    "marlin": "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/marlin/mxfp4.py",
    "baseline_activation": "tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/triton.py",
    "marlin_binding": "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/marlin_moe.py",
    "tests": "tokenspeed-kernel/test/nvidia/ops/moe/test_hopper_w4a16.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_evidence(path, expected_sha, source_hashes):
    if sha(path) != expected_sha:
        raise ValueError("Validation evidence SHA differs")
    evidence = json.loads(path.read_text())
    if evidence.get("status") != "passed":
        raise ValueError("Independent oracle validation must pass before timing")
    if (
        evidence.get("semantics") != SEMANTICS
        or evidence.get("tolerances") != TOLERANCES
    ):
        raise ValueError(
            "Validation semantics/tolerances differ from the frozen contract"
        )
    if evidence.get("source_sha256") != {
        name: source_hashes[name]
        for name in (
            "candidate",
            "candidate_native",
            "marlin",
            "baseline_activation",
            "tests",
            "driver",
        )
    }:
        raise ValueError("Validated production source differs from benchmark source")
    return evidence


def validate_build(path, expected_sha, library):
    if sha(path) != expected_sha:
        raise ValueError("Prebuilt native artifact evidence SHA differs")
    evidence = json.loads(path.read_text())
    stat = library.stat()
    if not (
        evidence.get("status") == "ready"
        and evidence.get("path") == str(library.resolve())
        and evidence.get("size") == stat.st_size
        and evidence.get("mtime_ns") == stat.st_mtime_ns
        and len(evidence.get("sha256", "")) == 64
    ):
        raise ValueError("Native library differs from the CPU-frozen artifact")
    return evidence


def route_ids(rows, rank):
    # Uniform global-expert schedule, six distinct choices per token. Its first
    # token has one local route; a random all-nonlocal M1 is not representative.
    return [
        [(rank * 48 + row * 37 + slot * 61) % 384 for slot in range(6)]
        for row in range(rows)
    ]


def paired_orders(rng, repeats):
    arms = ["marlin", "hopper_w4a16"]
    orders = [
        arms.copy() if repeat % 2 == 0 else list(reversed(arms))
        for repeat in range(repeats)
    ]
    rng.shuffle(orders)
    return orders


def measure_replays(torch, graph, inner, eviction):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Same current CUDA stream: this entire random-buffer read/write precedes
    # the begin event. It is charged wall time, excluded from GPU latency.
    if eviction is not None:
        if inner != 1:
            raise ValueError("Cold-cache samples require exactly one graph replay")
        eviction.add_(1)
    begin.record()
    for _ in range(inner):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / inner


def summarize_pairs(pairs):
    return dict(
        median_us={
            name: statistics.median(pair["us"][name] for pair in pairs)
            for name in ("marlin", "hopper_w4a16")
        },
        median_paired_speedup=statistics.median(
            pair["marlin_over_candidate"] for pair in pairs
        ),
    )


def parity(torch, reference, actual):
    reference, actual = reference.float(), actual.float()
    error = (actual - reference).abs()
    norm = torch.linalg.vector_norm(reference, dim=1)
    l2 = torch.linalg.vector_norm(error, dim=1) / norm.clamp_min(1e-30)
    absmax = error.amax(dim=1)
    permitted = 0.002 + 0.02 * reference.abs().amax(dim=1)
    zero = norm == 0
    finite = torch.isfinite(actual).all() & torch.isfinite(reference).all()
    valid = (
        finite
        & ((l2 <= 0.006) | zero).all()
        & (absmax <= permitted).all()
        & ((absmax == 0) | ~zero).all()
    )
    worst_l2, worst_abs = float(l2.max().item()), float(absmax.max().item())
    return dict(
        passed=bool(valid.item()),
        finite=bool(finite.item()),
        max_row_relative_l2=worst_l2 if math.isfinite(worst_l2) else None,
        max_absolute_error=worst_abs if math.isfinite(worst_abs) else None,
        zero_reference_rows=int(zero.sum().item()),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--validation-sha256", required=True)
    p.add_argument("--native-library", type=Path, required=True)
    p.add_argument("--build-evidence", type=Path, required=True)
    p.add_argument("--build-evidence-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ep-rank", type=int, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--wall-seconds", type=float, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--warmup-replays", type=int, required=True)
    p.add_argument("--target-sample-ms", type=float, required=True)
    p.add_argument("--max-inner", type=int, required=True)
    a = p.parse_args()
    if not (
        a.wall_seconds == 24
        and a.repeats == 5
        and a.warmup_replays == 2
        and a.target_sample_ms == 2
        and a.max_inner == 32
        and 0 <= a.ep_rank < 8
    ):
        p.error(
            "Frozen protocol requires wall24/repeats5/warmup2/target2ms/max-inner32"
        )
    if a.output.exists():
        p.error("Fresh output path required; never overwrite prior attempts")
    if not a.native_library.is_file() or a.native_library.stat().st_size == 0:
        p.error("Native fused_moe_90 shared library must already be built")
    return a


def main():
    started = time.monotonic()
    a = parse_args()
    deadline = started + a.wall_seconds
    result = dict(
        schema=2,
        status="running",
        metric="complete local-expert CUDA graph chain microseconds",
        primary_measurement="cold_cache",
        secondary_measurement="hot_cache",
        cold_cache=dict(
            eviction_bytes=EVICTION_BYTES,
            eviction_dtype="torch.int32",
            eviction_operation="add_(1)",
            initialization="independent-seed uniform random int32",
            repeats=COLD_REPEATS,
            replays_per_sample=1,
            flush_inside_timed_interval=False,
            surrogate_note="Cache-cold surrogate, not guaranteed physical L2 state; incompressible256MiB buffer exceeds4x nominal H20 L2 capacity",
        ),
        semantics=SEMANTICS,
        tolerances=TOLERANCES,
        shapes=list(SHAPES),
        seed=a.seed,
        ep_rank=a.ep_rank,
        benchmark_sha256=sha(Path(__file__)),
        wall_cap_seconds=a.wall_seconds,
        repeats=a.repeats,
        warmup_replays=a.warmup_replays,
        target_sample_ms=a.target_sample_ms,
        max_inner=a.max_inner,
        phases={},
        rows=[],
        exclusions=[
            "gate/top-k selection",
            "EP communication",
            "shared experts",
            "model routed factor1.5",
        ],
        validation_coverage_note="Independent oracle uses bounded local-expert cases, including actual H/I with E2/EP128. This benchmark additionally checks both production arms at E48/EP8 for every measured M; arm parity is not an independent oracle.",
        native_library=dict(
            path=str(a.native_library.resolve()),
            size=a.native_library.stat().st_size,
            mtime_ns=a.native_library.stat().st_mtime_ns,
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["worker_wall_seconds"] = time.monotonic() - started
        temp = a.output.with_suffix(a.output.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(a.output)

    def check_time():
        if time.monotonic() > deadline - 0.5:
            raise TimeoutError(
                "Bounded worker deadline; incomplete shapes cannot establish a win"
            )

    def expired(signum, frame):
        raise TimeoutError("Worker wall deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    save()
    try:
        result["source_sha256"] = {
            key: sha(a.source_root / path) for key, path in SOURCES.items()
        }
        result["source_sha256"]["driver"] = sha(
            Path(__file__).with_name("validate_w4a16.py")
        )
        result["validation"] = dict(
            path=str(a.validation.resolve()),
            sha256=a.validation_sha256,
            evidence=validate_evidence(
                a.validation, a.validation_sha256, result["source_sha256"]
            ),
        )
        result["native_build_evidence"] = dict(
            path=str(a.build_evidence.resolve()),
            sha256=a.build_evidence_sha256,
            artifact=validate_build(
                a.build_evidence, a.build_evidence_sha256, a.native_library
            ),
        )
        sys.path.insert(0, str(a.source_root / "tokenspeed-kernel/python"))
        import torch

        torch.set_num_threads(1)
        torch.cuda.set_device(a.device)
        if torch.cuda.get_device_capability(a.device) != (9, 0):
            raise RuntimeError("This experiment requires SM90")
        props = torch.cuda.get_device_properties(a.device)
        result["hardware"] = dict(
            name=props.name,
            total_memory=props.total_memory,
            multiprocessors=props.multi_processor_count,
            capability=[9, 0],
            torch_version=torch.__version__,
            cuda_build=torch.version.cuda,
        )
        result["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("flashinfer-python", "tokenspeed-triton")
        }
        candidate = importlib.import_module(
            "tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4"
        )
        native = importlib.import_module(
            "tokenspeed_kernel.thirdparty.flashinfer.mxfp4"
        )
        marlin = importlib.import_module("tokenspeed_kernel.ops.moe.marlin.mxfp4")
        for name, module in [
            ("candidate", candidate),
            ("candidate_native", native),
            ("marlin", marlin),
        ]:
            if (
                Path(module.__file__).resolve()
                != (a.source_root / SOURCES[name]).resolve()
            ):
                raise RuntimeError("Imported source differs from frozen source root")
        result["phases"]["imports_seconds"] = time.monotonic() - started
        check_time()
        device = torch.device("cuda", a.device)
        generator = torch.Generator(device=device).manual_seed(a.seed)
        prepared_at = time.monotonic()
        canonical = {
            "w13": torch.randint(
                0,
                256,
                (48, 4608, 2560),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ),
            "w2": torch.randint(
                0,
                256,
                (48, 5120, 1152),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ),
            "s13": torch.randint(
                120,
                124,
                (48, 4608, 160),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ),
            "s2": torch.randint(
                119,
                123,
                (48, 5120, 72),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ),
        }
        # Tiny byte samples identify deterministic fixture generation without
        # copying/hashing almost a gigabyte back to the CPU under a25s lease.
        result["canonical_weights"] = {
            key: dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                bytes=value.numel(),
                prefix256_sha256=hashlib.sha256(
                    value.flatten()[:256].cpu().numpy().tobytes()
                ).hexdigest(),
            )
            for key, value in canonical.items()
        }
        state = candidate.prepare_hopper_mxfp4_moe(
            canonical["w13"],
            canonical["w2"],
            canonical["s13"],
            canonical["s2"],
            6,
            8,
            a.ep_rank,
            1024,
            10.0,
        )
        module = torch.nn.Module()
        for field, key in [
            ("w13_weight", "w13"),
            ("w2_weight", "w2"),
            ("w13_weight_scale", "s13"),
            ("w2_weight_scale", "s2"),
        ]:
            setattr(
                module, field, torch.nn.Parameter(canonical[key], requires_grad=False)
            )
        module.activation = "swiglu"
        module.swiglu_arg = SimpleNamespace(alpha=1.0, limit=10.0)
        module.swiglu_beta = 0.0
        module.num_local_experts = 48
        module.ep_size = 8
        module.ep_rank = a.ep_rank
        plan = {"activation": "swiglu"}
        if marlin._swiglu_limit(module, "swiglu") != 10.0:
            raise RuntimeError("Corrected Marlin clamp10 baseline required")
        marlin.marlin_mxfp4_moe_weights(plan, module)
        del canonical
        torch.cuda.synchronize()
        result["phases"]["weight_preparation_seconds"] = time.monotonic() - prepared_at
        result["canonical_source_released"] = True
        result["candidate_workspace_bytes"] = (
            state.native.workspace.numel() * state.native.workspace.element_size()
        )
        state_tensors = dict(
            w13=state.native.w13,
            w2=state.native.w2,
            s13=state.native.scales[0],
            s2=state.native.scales[1],
            alpha=state.native.alpha,
            beta=state.native.beta,
            limit=state.native.limit,
            workspace=state.native.workspace,
            route_ids=state.route_ids,
            route_weights=state.route_weights,
        )
        storages = {
            value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
            for value in state_tensors.values()
        }
        result["candidate_persistent_state"] = dict(
            tensors={
                key: dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    logical_bytes=value.numel() * value.element_size(),
                )
                for key, value in state_tensors.items()
            },
            distinct_storage_bytes=sum(storages.values()),
            scratch_bytes=sum(
                state_tensors[key].numel() * state_tensors[key].element_size()
                for key in ("workspace", "route_ids", "route_weights")
            ),
            warning="One-layer allocation only. Full-model integration must budget all layers and share serialized scratch; multiplying workspace by40 is not validated.",
        )
        del state_tensors, storages
        check_time()
        all_x = torch.randn(
            (1024, 5120), dtype=torch.bfloat16, device=device, generator=generator
        )
        all_weights = torch.rand(
            (1024, 6), dtype=torch.float32, device=device, generator=generator
        )
        all_weights /= all_weights.sum(dim=1, keepdim=True)
        ids_cpu = route_ids(1024, a.ep_rank)
        all_ids = torch.tensor(ids_cpu, dtype=torch.int32, device=device)
        rng = random.Random(a.seed ^ 0x514A)
        cold_rng = random.Random(a.seed ^ 0xC01D)
        # A separate generator preserves the already declared canonical weight
        # and input sequence. Integer addition preserves random data entropy;
        # repeated zero writes could be compressed in the cache hierarchy.
        eviction_seed = a.seed ^ 0x4E564C32
        eviction_generator = torch.Generator(device=device).manual_seed(eviction_seed)
        eviction = torch.randint(
            -(2**31),
            2**31 - 1024,
            (EVICTION_BYTES // 4,),
            dtype=torch.int32,
            device=device,
            generator=eviction_generator,
        )
        result["cold_cache"]["eviction_seed"] = eviction_seed
        for rows in SHAPES:
            check_time()
            x, weights, ids = all_x[:rows], all_weights[:rows], all_ids[:rows]
            out = torch.empty_like(x)

            def baseline():
                return marlin.marlin_mxfp4_precomputed_moe_apply(
                    plan, x, module, None, weights, ids, None, 1024, True, False
                )

            def proposed():
                return candidate.hopper_mxfp4_moe(state, x, weights, ids, out)

            reference = baseline()
            actual = proposed()
            row = dict(
                m=rows,
                correctness=parity(torch, reference, actual),
                local_routes=sum(
                    a.ep_rank * 48 <= expert < (a.ep_rank + 1) * 48
                    for choices in ids_cpu[:rows]
                    for expert in choices
                ),
                global_routes=rows * 6,
                pairs=[],
            )
            result["rows"].append(row)
            save()
            if not row["correctness"]["passed"]:
                raise RuntimeError(
                    f"Actual E48/EP8 arm parity failed at M={rows}; no timing for this case"
                )
            graphs, retained_outputs = {}, {}
            for name, fn in [("marlin", baseline), ("hopper_w4a16", proposed)]:
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    retained_outputs[name] = fn()
                graphs[name] = graph
                for _ in range(a.warmup_replays):
                    graph.replay()
            torch.cuda.synchronize()
            row["graph_correctness"] = parity(
                torch, retained_outputs["marlin"], retained_outputs["hopper_w4a16"]
            )
            if not row["graph_correctness"]["passed"]:
                raise RuntimeError(f"Graph arm parity failed at M={rows}")
            check_time()
            cold = dict(replays_per_sample=1, pairs=[])
            row["cold_cache"] = cold
            for repeat, order in enumerate(paired_orders(cold_rng, COLD_REPEATS)):
                pair = dict(repeat=repeat, order=order, us={})
                for name in order:
                    check_time()
                    pair["us"][name] = measure_replays(torch, graphs[name], 1, eviction)
                pair["marlin_over_candidate"] = (
                    pair["us"]["marlin"] / pair["us"]["hopper_w4a16"]
                )
                cold["pairs"].append(pair)
            cold.update(summarize_pairs(cold["pairs"]))
            cold["status"] = "complete"
            save()
            # Rewarm after the eviction experiment. Existing top-level row
            # samples/medians are retained as the explicitly secondary hot data.
            for graph in graphs.values():
                for _ in range(a.warmup_replays):
                    graph.replay()
            torch.cuda.synchronize()
            pilot = {
                name: measure_replays(torch, graphs[name], 1, None) for name in graphs
            }
            row["pilot_us"] = pilot
            inner = max(
                1,
                min(
                    a.max_inner,
                    math.ceil(a.target_sample_ms * 1000 / max(pilot.values())),
                ),
            )
            row["replays_per_sample"] = inner
            for repeat, order in enumerate(paired_orders(rng, a.repeats)):
                check_time()
                pair = dict(repeat=repeat, order=order, us={})
                for name in order:
                    pair["us"][name] = measure_replays(torch, graphs[name], inner, None)
                pair["marlin_over_candidate"] = (
                    pair["us"]["marlin"] / pair["us"]["hopper_w4a16"]
                )
                row["pairs"].append(pair)
            row.update(summarize_pairs(row["pairs"]))
            row["status"] = "complete"
            save()
            del graphs, retained_outputs, reference, actual
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["source_unchanged"] = all(
            sha(a.source_root / path) == result["source_sha256"][key]
            for key, path in SOURCES.items()
        )
        result["source_unchanged"] &= (
            sha(Path(__file__).with_name("validate_w4a16.py"))
            == result["source_sha256"]["driver"]
        )
        validate_build(a.build_evidence, a.build_evidence_sha256, a.native_library)
        if not result["source_unchanged"]:
            raise RuntimeError("Frozen source changed during benchmark")
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save()


if __name__ == "__main__":
    main()
