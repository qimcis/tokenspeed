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

"""Explicit hardware launch, with pinned snapshots and owned process groups."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from tokenspeed.deployment.deepseek_v41.config import Deployment, build_manifest


def gpu_inventory() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return [
        {
            "index": int(row[0]),
            "name": row[1].strip(),
            "uuid": row[2].strip(),
            "memory_bytes": int(row[3]) * 2**20,
            "compute_capability": row[4].strip(),
        }
        for row in csv.reader(result.stdout.splitlines())
    ]


def validate_hardware(config: Deployment, inventory: list[dict]) -> None:
    by_id = {gpu["index"]: gpu for gpu in inventory}
    from tokenspeed.deployment.deepseek_v41.config import memory_lower_bound

    ledger = memory_lower_bound(config)
    for worker in ledger["workers"]:
        for index in worker["gpus"]:
            if index not in by_id:
                raise ValueError(f"GPU {index} is unavailable")
            gpu = by_id[index]
            if "B300" not in gpu["name"].split() or gpu["compute_capability"] != "10.3":
                raise ValueError(f"GPU {index} is not a B300: {gpu['name']}")
            budget = gpu["memory_bytes"] * config.engine.gpu_memory_utilization
            if worker["known_device_bytes_per_rank"] >= budget:
                raise ValueError(
                    f"GPU {index}: known payloads already exceed the memory budget"
                )


def validate_snapshot(config: Deployment, snapshot: Path) -> None:
    """Check ledger assumptions; snapshot_download establishes revision identity."""
    checkpoint = json.loads((snapshot / "config.json").read_text())
    text = checkpoint.get("text_config", {})
    expected = {
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "head_dim": 512,
        "sliding_window": 128,
        "moe_intermediate_size": 2304,
        "index_head_dim": 128,
        "engram_head_dim": 256,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "engram_num_embeddings": [384006168, 384016682],
    }
    if checkpoint.get("model_type") != "deepseek_v41":
        raise ValueError("snapshot is not a DeepSeek V4.1 checkpoint")
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"checkpoint geometry mismatch for {key}")
    quantization = checkpoint.get("quantization_config", {})
    if quantization.get("expert_dtype") != "fp4":
        raise ValueError("checkpoint expert_dtype does not match FP4 memory accounting")
    if config.engine.speculation == "dspark":
        for key, value in {
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [37, 38, 39],
        }.items():
            if text.get(key) != value:
                raise ValueError(f"checkpoint geometry mismatch for {key}")
    if config.mode == "epd" and not checkpoint.get("vision_config"):
        raise ValueError("EPD requires the checkpoint's vision configuration")


def snapshot_fingerprints(snapshot: Path) -> dict[str, str]:
    """Record small model/processor/tokenizer artifacts without rereading weights."""
    paths = [
        path
        for path in snapshot.iterdir()
        if path.is_file() and path.suffix in (".json", ".jinja", ".py")
    ]
    encoding = snapshot / "encoding"
    if encoding.is_dir():
        paths.extend(path for path in encoding.rglob("*") if path.is_file())
    fingerprints = {}
    for path in sorted(paths):
        with path.open("rb") as source:
            fingerprints[str(path.relative_to(snapshot))] = hashlib.file_digest(
                source, "sha256"
            ).hexdigest()
    return fingerprints


def validate_ports(manifest: dict) -> None:
    """Fail before model loading when a planned local listener is already busy.

    This is a preflight, not a reservation: another process can still bind before
    workers start. Upstream also chooses dynamic NCCL and Mooncake rank ports.
    """
    ports = [manifest["ports"]["gateway"], manifest["ports"]["gateway_metrics"]]
    if "router" in manifest["ports"]:
        ports.append(manifest["ports"]["router"])
    for worker in manifest["ports"]["workers"].values():
        ports.extend([worker["grpc"], *worker["control_cluster"]])
        if worker["bootstrap"] is not None:
            ports.append(worker["bootstrap"])
    if len(ports) != len(set(ports)):
        raise ValueError("configured listeners overlap")
    for port in ports:
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                raise ValueError(f"planned local port {port} is unavailable") from error


def provenance() -> dict:
    packages = {}
    for package in (
        "tokenspeed",
        "tokenspeed-kernel",
        "tokenspeed-scheduler",
        "tokenspeed-mooncake",
        "tokenspeed-smg",
        "tokenspeed-smg-grpc-servicer",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[4]
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "source_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "packages": packages,
        "python": platform.python_version(),
    }


class ProcessGroup:
    """Own only the process groups this launcher creates, including their children."""

    def __init__(self, *, shutdown_grace: float):
        if not math.isfinite(shutdown_grace) or shutdown_grace <= 0:
            raise ValueError("shutdown grace must be finite and positive")
        self.shutdown_grace = shutdown_grace
        self.closed = False
        self.processes: list[tuple[str, subprocess.Popen]] = []
        self.logs = []

    def start(self, command: dict, directory: Path) -> None:
        if self.closed:
            raise RuntimeError("process group is closed")
        log = (directory / f"{command['role']}.log").open("xb")
        env = os.environ.copy()
        # The recipe must not inherit a forced transport with different semantics.
        env.pop("MC_INTRANODE_NVLINK", None)
        env.pop("MC_INTRA_NVLINK", None)
        for name in ("SLURM_STEP_NUM_NODES", "SLURM_NODEID", "SLURM_STEP_NODELIST"):
            env.pop(name, None)
        env.update(command["env"])
        try:
            process = subprocess.Popen(
                command["argv"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        self.logs.append(log)
        self.processes.append((command["role"], process))

    def check(self) -> None:
        for role, process in self.processes:
            status = process.poll()
            if status is not None:
                raise RuntimeError(
                    f"{role} exited with status {status}; see {role}.log"
                )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # A leader may have exited while its descendants still own GPU memory.
        pending = [process for _, process in reversed(self.processes)]
        try:
            for process in pending:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + self.shutdown_grace
            while pending and time.monotonic() < deadline:
                alive = []
                for process in pending:
                    process.poll()  # Reap owned leaders before probing the group.
                    try:
                        os.killpg(process.pid, 0)
                        alive.append(process)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        # Some POSIX hosts return EPERM while the leader is
                        # exiting. Retain the group until it can be reaped.
                        alive.append(process)
                pending = alive
                if pending:
                    time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            for process in pending:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for _, process in self.processes:
                process.wait()
        finally:
            for log in self.logs:
                log.close()


async def _wait_workers(
    commands: list[dict], group: ProcessGroup, timeout: float
) -> None:
    from tokenspeed.cli._proc import wait_grpc_serving

    tasks = [
        asyncio.create_task(
            wait_grpc_serving(f"127.0.0.1:{command['port']}", timeout=timeout)
        )
        for command in commands
    ]
    try:
        while tasks:
            group.check()
            done, pending = await asyncio.wait(
                tasks, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            tasks = list(pending)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def run(
    config: Deployment,
    directory: Path,
    *,
    startup_timeout: float,
    shutdown_grace: float,
) -> None:
    if platform.system() != "Linux":
        raise ValueError(
            "hardware launch requires Linux; use plan for CPU-only validation"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (startup_timeout, shutdown_grace)
    ):
        raise ValueError("startup and shutdown bounds must be finite and positive")
    directory = directory.resolve()
    validate_ports(
        build_manifest(config, snapshot="<preflight>", output_directory=directory)
    )
    inventory = gpu_inventory()
    validate_hardware(config, inventory)
    # Never overwrite a previous experiment's evidence.
    directory.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(config.model, revision=config.revision))
    validate_snapshot(config, snapshot)
    manifest = build_manifest(
        config, snapshot=str(snapshot), output_directory=directory
    )
    manifest["snapshot_fingerprints"] = snapshot_fingerprints(snapshot)
    for name, content in manifest["generated_files"].items():
        (directory / name).write_text(json.dumps(content, indent=2) + "\n")
    manifest["provenance"] = provenance()
    manifest["gpu_inventory"] = inventory
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    group = ProcessGroup(shutdown_grace=shutdown_grace)
    old_handlers = {}

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        workers = [
            command
            for command in manifest["commands"]
            if command["role"] not in ("gateway", "router")
        ]
        for command in workers:
            group.start(command, directory)
        asyncio.run(_wait_workers(workers, group, startup_timeout))
        gateway = next(
            command for command in manifest["commands"] if command["role"] == "gateway"
        )
        group.start(gateway, directory)
        deadline = time.monotonic() + startup_timeout
        while True:
            group.check()
            try:
                with urlopen(
                    f"http://127.0.0.1:{gateway['port']}/v1/models", timeout=2
                ) as response:
                    models = json.load(response)["data"]
                if any(model["id"] == config.served_model_name for model in models):
                    break
            except (URLError, TimeoutError, KeyError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("gateway did not serve the configured model")
            time.sleep(0.2)
        router = next(
            (
                command
                for command in manifest["commands"]
                if command["role"] == "router"
            ),
            None,
        )
        if router is not None:
            group.start(router, directory)
            deadline = time.monotonic() + startup_timeout
            while True:
                group.check()
                try:
                    with urlopen(
                        f"http://127.0.0.1:{router['port']}/v1/models", timeout=2
                    ) as response:
                        models = json.load(response)["data"]
                    if any(model["id"] == config.served_model_name for model in models):
                        break
                except (URLError, TimeoutError, KeyError, json.JSONDecodeError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "routing proxy did not serve the configured model"
                    )
                time.sleep(0.2)
        (directory / "ready.json").write_text(
            json.dumps({"model": config.served_model_name, "port": config.base_port})
            + "\n"
        )
        while True:
            group.check()
            time.sleep(0.2)
    finally:
        (directory / "ready.json").unlink(missing_ok=True)
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            group.close()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
