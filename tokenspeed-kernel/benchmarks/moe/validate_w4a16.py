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


"""Bounded one-GPU validation with source and output artifacts.

The caller owns GPU reservation, timeout and process cleanup. All source pins,
per-case outcomes and actual/reference output tensors are retained. This driver
performs no benchmark timing and makes no accounting-ledger mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

STARTED = time.monotonic()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 25:
        raise ValueError("The frozen validation worker maximum is25 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.with_suffix(".artifacts")
    directory.mkdir(exist_ok=False)
    with args.output.open("x") as stream:
        json.dump({"status": "starting"}, stream)
    kernel = args.repo / "tokenspeed-kernel"
    paths = {
        "candidate": kernel
        / "python/tokenspeed_kernel/ops/moe/flashinfer/cutlass_mxfp4.py",
        "candidate_native": kernel
        / "python/tokenspeed_kernel/thirdparty/flashinfer/mxfp4.py",
        "marlin": kernel / "python/tokenspeed_kernel/ops/moe/marlin/mxfp4.py",
        "baseline_activation": kernel
        / "python/tokenspeed_kernel/ops/activation/triton.py",
        "tests": kernel / "test/nvidia/ops/moe/test_hopper_w4a16.py",
        "driver": Path(__file__).resolve(),
    }
    pins = {name: digest(path) for name, path in paths.items()}
    results = []
    state = {
        "status": "running",
        "source_sha256": pins,
        "cases": results,
        "semantics": {
            "hidden": 5120,
            "intermediate": 2304,
            "local_experts": 48,
            "ep_size": 8,
            "top_k": 6,
            "swiglu_limit": 10.0,
            "routed_scale_inside_adapter": 1.0,
        },
        "coverage": {
            "oracle_small": {"hidden": 256, "intermediate": 128, "local_experts": 8},
            "oracle_actual_geometry": {
                "hidden": 5120,
                "intermediate": 2304,
                "local_experts": 2,
                "ep_size": 128,
            },
            "actual_e48_ep8_oracle": False,
            "note": "Actual E48/EP8 complete-arm parity is additionally required by the microdriver.",
        },
        "tolerances": {
            "row_relative_l2": 0.006,
            "max_absolute_base": 0.002,
            "max_absolute_row_scale": 0.02,
            "exact_zero_reference_rows": True,
        },
    }
    lock = threading.Lock()
    completed = threading.Event()

    def write():
        state["wall_seconds"] = time.monotonic() - STARTED
        temporary = args.output.with_suffix(".pending.json")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)

    def deadline():
        if not completed.wait(max(0, args.seconds - (time.monotonic() - STARTED))):
            with lock:
                state["status"] = "timeout"
                write()
            os._exit(124)

    threading.Thread(target=deadline, daemon=True).start()
    with lock:
        write()
    try:
        import pytest
        import torch

        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError(
                "A visible SM90 GPU is required; skipped GPU tests cannot pass validation"
            )

        class Reports:
            module = None

            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_makereport(self, item, call):
                outcome = yield
                report = outcome.get_result()
                self.module = item.module
                if report.when == "call" or (
                    report.when == "setup" and report.outcome != "passed"
                ):
                    entry = {
                        "nodeid": report.nodeid,
                        "phase": report.when,
                        "outcome": report.outcome,
                        "seconds": report.duration,
                    }
                    if report.failed:
                        entry["failure"] = str(report.longrepr)
                    with lock:
                        results.append(entry)
                        write()

        plugin = Reports()
        exit_code = pytest.main(
            ["--noconftest", "-q", "-x", str(paths["tests"])], plugins=[plugin]
        )
        observations = []
        if plugin.module is not None:
            for index, observed in enumerate(plugin.module.OBSERVATIONS):
                tensor_path = directory / f"output-{index:03d}.pt"
                torch.save(
                    {key: observed[key] for key in ("actual", "reference")}, tensor_path
                )
                observations.append(
                    {
                        key: value
                        for key, value in observed.items()
                        if key not in ("actual", "reference")
                    }
                    | {"path": str(tensor_path), "sha256": digest(tensor_path)}
                )
        torch.cuda.synchronize()
        unchanged = all(digest(path) == pins[name] for name, path in paths.items())
        gpu_passed = [
            entry
            for entry in results
            if entry["outcome"] == "passed"
            and not entry["nodeid"].endswith("_cpu")
            and "test_independent_decode" not in entry["nodeid"]
            and "test_reference_clamp" not in entry["nodeid"]
            and "_cpu[" not in entry["nodeid"]
        ]
        passed = (
            exit_code == 0 and unchanged and len(gpu_passed) == 9 and len(results) == 28
        )
        with lock:
            state.update(
                status="passed" if passed else "failed",
                pytest_exit_code=int(exit_code),
                source_unchanged=unchanged,
                gpu_cases_passed=len(gpu_passed),
                observations=observations,
                cuda_device=torch.cuda.get_device_name(),
                cuda_capability=list(torch.cuda.get_device_capability()),
            )
            write()
        return 0 if passed else 1
    except BaseException as exc:
        with lock:
            state.update(status="failed", exception=repr(exc))
            write()
        raise
    finally:
        completed.set()


if __name__ == "__main__":
    sys.exit(main())
