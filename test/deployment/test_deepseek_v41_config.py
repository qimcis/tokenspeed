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

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.config import (
    Deployment,
    build_manifest,
    memory_lower_bound,
)
from tokenspeed.deployment.deepseek_v41.launch import (
    validate_hardware,
    validate_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]


def recipe(mode="epd"):
    return Deployment.load(ROOT / "recipes" / "deepseek-v41-b300" / f"{mode}.toml")


@pytest.mark.parametrize(
    "mode,roles",
    [
        ("aggregate", ["aggregate", "gateway"]),
        ("pd", ["prefill", "decode", "gateway"]),
        ("epd", ["encode", "prefill", "decode", "gateway"]),
    ],
)
def test_recipes_build_role_specific_commands(mode, roles):
    config = recipe(mode)
    manifest = build_manifest(
        config, snapshot="/weights/with spaces", python="python-test"
    )
    assert [item["role"] for item in manifest["commands"]] == roles
    assert sum(len(worker.gpus) for worker in config.workers) == 8
    for command in manifest["commands"]:
        assert "/weights/with spaces" in command["argv"]
        assert command["argv"][0] == "python-test"
        assert "--language-model-only" not in command["argv"]
        assert "MC_INTRA_NVLINK" not in command["env"]
        if command["role"] == "encode":
            assert "--speculative-algorithm" not in command["argv"]
            assert "--engram-host-table" not in command["argv"]
        if command["role"] == "decode":
            assert "--disable-prefix-caching" in command["argv"]
        assert "deepseek_v41" in command["argv"]
    assert not manifest["hardware_validated"]


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda d: d.update(revision="main"), "pattern"),
        (lambda d: d.update(gpu_budget=7), "budget"),
        (lambda d: d["workers"][1].update(gpus=[0, 1]), "overlap"),
        (lambda d: d["workers"][0].update(gpus=[0, 0]), "same GPU"),
        (lambda d: d["workers"][0].update(attention_tp=1), "attention TP"),
        (lambda d: d["engine"].update(speculation="tree"), "literal"),
        (lambda d: d["engine"].update(chunked_prefill_size=128), "replay window"),
        (lambda d: d["engine"].update(max_total_tokens=100), "maximum-context"),
        (lambda d: d["engine"].update(max_total_tokens=131073), "aligned"),
        (lambda d: d["engine"].update(kv_parallel_size=2), "Extra inputs"),
        (lambda d: d["transport"].update(receive_slot_mib=1), "receive slot"),
        (lambda d: d["workers"].pop(), "requires exactly"),
    ],
)
def test_invalid_recipes_fail_before_launch(edit, message):
    data = recipe().model_dump()
    edit(data)
    with pytest.raises(ValidationError, match=message):
        Deployment.model_validate(data)


def test_encode_prefill_divisibility():
    data = recipe().model_dump()
    data["workers"][0].update(gpus=[0, 1, 2, 3], attention_tp=4)
    data["workers"][1].update(gpus=[4, 5], attention_tp=2)
    data["workers"][2].update(gpus=[6, 7], attention_tp=2)
    with pytest.raises(ValidationError, match="multiple"):
        Deployment.model_validate(data)


def test_residency_ledger_accounts_for_replicated_kv_and_independent_host_tables():
    data = recipe().model_dump()
    ledger = memory_lower_bound(Deployment.model_validate(data))
    assert ledger["known_host_bytes"] == 2 * 202758032400
    decode = ledger["workers"][2]
    assert decode["components_per_rank"]["global_cache_payload"] == 131072 * 890
    data["engine"]["engram"] = "gpu"
    gpu = memory_lower_bound(Deployment.model_validate(data))
    assert gpu["known_host_bytes"] == 0
    assert (
        gpu["workers"][2]["components_per_rank"]["engram_codes_and_scales"]
        == 202758032400 // 4
    )
    assert "lower_bound" in gpu["status"]


def test_no_speculation_removes_draft_windows_and_flags():
    data = recipe().model_dump()
    data["engine"]["speculation"] = "none"
    config = Deployment.model_validate(data)
    manifest = build_manifest(config, snapshot="/snapshot")
    assert all(
        "--speculative-algorithm" not in item["argv"] for item in manifest["commands"]
    )
    assert manifest["memory"]["workers"][2]["components_per_rank"]["draft_windows"] == 0


def test_hardware_validation_uses_each_rank():
    inventory = [
        {
            "index": i,
            "name": "NVIDIA B300",
            "compute_capability": "10.3",
            "memory_bytes": 268 * 2**30,
        }
        for i in range(8)
    ]
    validate_hardware(recipe(), inventory)
    wrong = copy.deepcopy(inventory)
    wrong[2]["name"] = "NVIDIA GB300"
    with pytest.raises(ValueError, match="not a B300"):
        validate_hardware(recipe(), wrong)
    inventory[3]["memory_bytes"] = 32 * 2**30
    with pytest.raises(ValueError, match="GPU 3"):
        validate_hardware(recipe(), inventory)


def test_snapshot_geometry(tmp_path):
    checkpoint = {
        "model_type": "deepseek_v41",
        "vision_config": {"hidden_size": 1024},
        "quantization_config": {"expert_dtype": "fp4"},
        "text_config": {
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
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [37, 38, 39],
            "engram_num_embeddings": [384006168, 384016682],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    validate_snapshot(recipe(), tmp_path)
    checkpoint["text_config"]["head_dim"] = 576
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="head_dim"):
        validate_snapshot(recipe(), tmp_path)


def test_encoder_cache_and_ring_budgets_are_explicit_and_count_every_rank():
    data = recipe().model_dump()
    data["transport"].update(encode_cache_mib=64, encode_host_cache_mib=128)
    config = Deployment.model_validate(data)
    manifest = build_manifest(config, snapshot="/snapshot")
    encode = manifest["commands"][0]
    assert encode["env"]["TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB"] == "64"
    assert encode["env"]["TOKENSPEED_EPD_ENCODE_EMBED_CACHE_DRAM_MB"] == "128"
    ledger = manifest["memory"]["workers"][0]
    assert ledger["known_device_bytes_per_rank"] == (8 * 32 + 64) * 2**20
    assert ledger["known_host_bytes_per_worker"] == 2 * 128 * 2**20


def test_metrics_outputs_are_role_isolated_and_explicit(tmp_path):
    manifest = build_manifest(recipe(), snapshot="/snapshot", output_directory=tmp_path)
    for command in manifest["commands"][:-1]:
        path = command["argv"][command["argv"].index("--metrics-reporter-config") + 1]
        assert Path(path).parent == tmp_path
        config = manifest["generated_files"][Path(path).name]["jsonl"]
        assert config["output_dir"] == str(tmp_path / "metrics" / command["role"])
        assert config["queue_size"] == 8192
    data = recipe().model_dump()
    data["metrics"]["reporter"] = "none"
    disabled = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    assert disabled["generated_files"] == {}
    assert all("--enable-metrics" not in c["argv"] for c in disabled["commands"])


def test_large_chunk_sets_both_runtime_limits():
    data = recipe().model_dump()
    data["engine"]["chunked_prefill_size"] = 16384
    manifest = build_manifest(Deployment.model_validate(data), snapshot="/snapshot")
    for command in manifest["commands"][:-1]:
        argv = command["argv"]
        assert argv[argv.index("--chunked-prefill-size") + 1] == "16384"
        assert argv[argv.index("--max-prefill-tokens") + 1] == "16384"


def test_metrics_queue_bound_matches_reporter(tmp_path):
    from tokenspeed.runtime.metrics.jsonl import validate_config

    data = recipe().model_dump()
    data["metrics"]["queue_size"] = 1000000
    manifest = build_manifest(
        Deployment.model_validate(data), snapshot="/snapshot", output_directory=tmp_path
    )
    for configuration in manifest["generated_files"].values():
        assert validate_config(configuration["jsonl"])["queue_size"] == 1000000
    data["metrics"]["queue_size"] += 1
    with pytest.raises(ValidationError, match="1000000"):
        Deployment.model_validate(data)
