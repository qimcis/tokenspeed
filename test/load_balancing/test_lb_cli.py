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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_cli_requires_independent_heldout_records(tmp_path):
    # Synthetic algebra fixture only; no serving calibration or speed claim.
    def observation(q, c):
        return {
            "uncached_tokens": q,
            "cached_prefix_tokens": c,
            "elapsed_ms": float(2 * q + 3 * q * c + q * q),
        }

    data = {
        "training": [
            observation(1, 0),
            observation(2, 0),
            observation(1, 2),
            observation(4, 3),
        ],
        "heldout": [observation(3, 1)],
    }
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(data))
    command = [
        sys.executable,
        "-m",
        "tokenspeed.deployment.deepseek_v41",
        "fit-prefill",
        str(path),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    fitted = json.loads(result.stdout)
    assert fitted["heldout_samples"] == 1
    assert fitted["heldout_max_absolute_error_ms"] < 1e-9
    data["heldout"] = []
    path.write_text(json.dumps(data))
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode != 0
    assert "heldout" in result.stderr
