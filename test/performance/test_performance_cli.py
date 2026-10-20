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

import pytest

from tokenspeed.deployment.deepseek_v41.hotset import build_hotset

ROOT = Path(__file__).resolve().parents[2]


def frequencies():
    return {
        "model_revision": "a" * 40,
        "layers": [
            {
                "layer_id": 1,
                "total_rows": 100,
                "counts": [
                    {"row_id": 7, "count": 30},
                    {"row_id": 2, "count": 30},
                    {"row_id": 9, "count": 1},
                ],
            }
        ],
    }


def test_hotset_selects_frequency_then_stable_id_ties():
    result = build_hotset(frequencies(), capacity_rows=1)
    assert result["layers"][0]["global_row_ids"] == [2]
    assert result["model_revision"] == "a" * 40
    assert (
        build_hotset(frequencies(), capacity_rows=0)["layers"][0]["global_row_ids"]
        == []
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(model_revision="main"),
        lambda data: data["layers"][0]["counts"].append({"row_id": 7, "count": 1}),
        lambda data: data["layers"].append(data["layers"][0]),
        lambda data: data["layers"][0]["counts"].append({"row_id": 100, "count": 1}),
    ],
)
def test_invalid_frequency_records_rejected(change):
    data = frequencies()
    change(data)
    with pytest.raises(ValueError):
        build_hotset(data, capacity_rows=2)


def test_hotset_cli_roundtrip(tmp_path):
    path = tmp_path / "frequencies.json"
    path.write_text(json.dumps(frequencies()))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tokenspeed.deployment.deepseek_v41",
            "hotset",
            str(path),
            "--capacity-rows",
            "2",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["layers"][0]["global_row_ids"] == [2, 7]
    from tokenspeed.runtime.models.engram_hot_cache import EngramHotSetManifest

    assert EngramHotSetManifest.from_dict(manifest).to_dict() == manifest
