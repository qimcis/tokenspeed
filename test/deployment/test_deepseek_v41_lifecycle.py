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
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from tokenspeed.deployment.deepseek_v41.config import Deployment, build_manifest
from tokenspeed.deployment.deepseek_v41.launch import (
    ProcessGroup,
    snapshot_fingerprints,
    validate_ports,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"child did not write {path.name}")
        time.sleep(0.01)


def command(
    code: str, *args: str, role: str = "worker", env: dict | None = None
) -> dict:
    return {"role": role, "argv": [sys.executable, "-c", code, *args], "env": env or {}}


def test_process_failure_is_detected_and_close_is_idempotent(tmp_path):
    group = ProcessGroup(shutdown_grace=0.2)
    group.start(command("raise SystemExit(7)"), tmp_path)
    group.processes[0][1].wait(timeout=5)
    with pytest.raises(RuntimeError, match="worker exited with status 7"):
        group.check()
    group.close()
    group.close()
    assert all(log.closed for log in group.logs)
    with pytest.raises(RuntimeError, match="closed"):
        group.start(command("pass"), tmp_path)


def test_failed_spawn_does_not_hold_log_or_lose_other_children(tmp_path):
    group = ProcessGroup(shutdown_grace=0.2)
    try:
        group.start(command("import time; time.sleep(30)"), tmp_path)
        with pytest.raises(FileNotFoundError):
            group.start(
                {
                    "role": "missing",
                    "argv": [str(tmp_path / "missing-binary")],
                    "env": {},
                },
                tmp_path,
            )
    finally:
        group.close()
    assert group.processes[0][1].returncode is not None
    assert all(log.closed for log in group.logs)


def test_cleanup_reaches_descendant_after_leader_exits(tmp_path):
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    child = """
import signal, sys, time
from pathlib import Path
def stop(signum, frame):
    Path(sys.argv[2]).write_text('terminated')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text('ready')
while True:
    time.sleep(0.1)
"""
    parent = """
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]])
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
"""
    group = ProcessGroup(shutdown_grace=0.2)
    try:
        group.start(command(parent, child, str(ready), str(stopped)), tmp_path)
        wait_for_file(ready)
        group.processes[0][1].wait(timeout=5)
    finally:
        group.close()
    assert stopped.read_text() == "terminated"


def test_cleanup_kills_worker_that_ignores_term(tmp_path):
    ready = tmp_path / "ready"
    code = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text('ready')
while True:
    time.sleep(0.1)
"""
    group = ProcessGroup(shutdown_grace=0.05)
    try:
        group.start(command(code, str(ready)), tmp_path)
        wait_for_file(ready)
    finally:
        group.close()
    assert group.processes[0][1].returncode == -signal.SIGKILL


def test_child_receives_recipe_environment_without_outer_topology(
    tmp_path, monkeypatch
):
    names = (
        "MC_INTRA_NVLINK",
        "MC_INTRANODE_NVLINK",
        "SLURM_STEP_NUM_NODES",
        "SLURM_NODEID",
        "SLURM_STEP_NODELIST",
    )
    for name in names:
        monkeypatch.setenv(name, "wrong")
    result = tmp_path / "environment.json"
    code = """
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)))
"""
    group = ProcessGroup(shutdown_grace=0.2)
    try:
        group.start(
            command(code, str(result), env={"CUDA_VISIBLE_DEVICES": "2,3"}), tmp_path
        )
        group.processes[0][1].wait(timeout=5)
    finally:
        group.close()
    environment = json.loads(result.read_text())
    assert environment["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert not set(names).intersection(environment)


def test_port_preflight_rejects_an_actual_busy_listener():
    config = Deployment.load(ROOT / "recipes/deepseek-v41-b300/epd.toml")
    manifest = build_manifest(config, snapshot="unused")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        manifest["ports"]["gateway"] = port
        with pytest.raises(ValueError, match=f"port {port} is unavailable"):
            validate_ports(manifest)


def test_fingerprints_cover_processor_encoding_and_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding/config.dat").write_bytes(b"encoding")
    (tmp_path / "weights.safetensors").write_bytes(b"large-weights-are-not-hashed")
    before = snapshot_fingerprints(tmp_path)
    (tmp_path / "encoding/config.dat").write_bytes(b"changed")
    after = snapshot_fingerprints(tmp_path)
    assert before.keys() == {"config.json", "tokenizer.json", "encoding/config.dat"}
    assert before["config.json"] == after["config.json"]
    assert before["encoding/config.dat"] != after["encoding/config.dat"]


@pytest.mark.parametrize("grace", [0.0, -1.0, float("nan"), float("inf")])
def test_cleanup_requires_a_finite_bound(grace):
    with pytest.raises(ValueError, match="finite and positive"):
        ProcessGroup(shutdown_grace=grace)
