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

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.config import (
    Deployment,
    build_manifest,
    memory_lower_bound,
)

ROOT = Path(__file__).resolve().parents[2]


def affinity():
    return Deployment.load(ROOT / "recipes/deepseek-v41-b300/performance-affinity.toml")


def test_affinity_recipe_propagates_and_fifo_retains_execution_controls():
    config = affinity()
    candidate = build_manifest(config, snapshot="/snapshot")
    data = config.model_dump()
    data["performance"]["expert_affinity"] = False
    baseline = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    candidate_args = candidate["commands"][0]["argv"]
    baseline_args = baseline["commands"][0]["argv"]
    for args in (candidate_args, baseline_args):
        assert "--enforce-eager" in args
        assert "--disable-overlap-schedule" in args
        assert args[args.index("--max-decode-batch-size") + 1] == "8"
    assert "--expert-affinity" in candidate_args
    assert "--expert-affinity" not in baseline_args
    assert candidate["performance"]["resolved"]["expert_affinity_lookahead"] == 256
    capture_bytes = (
        config.engine.max_num_seqs
        * config.performance.expert_affinity_max_layers
        * 6
        * 8
    )
    assert (
        candidate["memory"]["workers"][0]["expert_route_capture_pinned_host_bytes"]
        == capture_bytes
    )
    assert (
        candidate["memory"]["known_host_bytes"]
        == baseline["memory"]["known_host_bytes"] + capture_bytes
    )


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda d: d["engine"].update(speculation="dspark"), "speculation"),
        (lambda d: d["performance"].update(execution="default"), "eager"),
        (lambda d: d["performance"].update(max_decode_batch_size=17), "max_num_seqs"),
        (lambda d: d["performance"].update(expert_affinity_lookahead=8), "cover"),
        (
            lambda d: d["performance"].update(dspark_candidate_communication="unknown"),
            "literal",
        ),
    ],
)
def test_bad_performance_recipe_rejected_before_launch(edit, match):
    data = affinity().model_dump()
    edit(data)
    with pytest.raises(ValidationError, match=match):
        Deployment.model_validate(data)


def hot_config(tmp_path, layout="host_sharded"):
    data = affinity().model_dump()
    data["engine"].update(speculation="dspark", engram=layout)
    manifest = tmp_path / "hot.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": data["revision"],
                "layers": [
                    {
                        "layer_id": 1,
                        "total_rows": 384006168,
                        "global_row_ids": [0, 384006167],
                    },
                    {"layer_id": 14, "total_rows": 384016682, "global_row_ids": [0]},
                ],
            }
        )
    )
    data["performance"].update(
        expert_affinity=False,
        execution="default",
        dspark_candidate_communication="compact",
        engram_hot_cache_manifest=str(manifest),
        engram_hot_cache_capacity=2,
    )
    return data, manifest


@pytest.mark.parametrize(
    "layout,total_rows", [("host_sharded", 3), ("host_shared", 24)]
)
def test_cache_ledger_counts_rank_copies_metadata_and_staging(
    tmp_path, layout, total_rows
):
    data, _ = hot_config(tmp_path, layout)
    config = Deployment.model_validate(data)
    ledger = memory_lower_bound(config)["workers"][0]
    ranks = ledger["engram_hot_cache_per_rank"]
    assert sum(r["codes_and_scales"] for r in ranks) == total_rows * 264
    assert sum(r["lookup_metadata"] for r in ranks) == total_rows * 8
    assert ledger["engram_hot_host_staging_peak_per_worker"] > 0
    args = build_manifest(config, snapshot="/snapshot")["commands"][0]["argv"]
    assert args[args.index("--dspark-candidate-communication") + 1] == "compact"
    assert args[args.index("--engram-hot-cache-model-revision") + 1] == config.revision


@pytest.mark.parametrize(
    "change,match",
    [
        ({"model_revision": "wrong"}, "revision"),
        (
            {"layers": [{"layer_id": 1, "total_rows": 1, "global_row_ids": [0]}]},
            "layers",
        ),
    ],
)
def test_bad_hot_manifest_rejected_before_commands(tmp_path, change, match):
    data, manifest = hot_config(tmp_path)
    raw = json.loads(manifest.read_text())
    raw.update(change)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match=match):
        Deployment.model_validate(data)


def test_shard_capacity_checked_before_launch(tmp_path):
    data, _ = hot_config(tmp_path, "host_shared")
    data["performance"]["engram_hot_cache_capacity"] = 1
    with pytest.raises(ValidationError, match="capacity"):
        Deployment.model_validate(data)
