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

"""CPU-only launch, bounds and lazy GPU-import tests for the draft worker."""

import argparse
import dataclasses
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from tokenspeed.cli.draft_worker import (
    DraftWorkerArgs,
    add_draft_worker_args,
    run_draft_worker_from_args,
)
from tokenspeed.runtime.draft_pool.config import (
    validate_remote_model_pair,
    validate_remote_revision,
    validate_remote_tcp_endpoint,
)


@pytest.fixture
def worker_args() -> DraftWorkerArgs:
    parser = argparse.ArgumentParser()
    add_draft_worker_args(parser)
    args = parser.parse_args(
        [
            "--target-model-path",
            "target/snapshot",
            "--target-revision",
            "a" * 40,
            "--draft-model-path",
            "draft/snapshot",
            "--draft-revision",
            "b" * 40,
            "--listen-address",
            "tcp://127.0.0.1:32000",
            "--max-resident-sessions",
            "32",
            "--max-batch-size",
            "8",
        ]
    )
    return DraftWorkerArgs(**vars(args))


def test_worker_geometry_is_not_user_overridable() -> None:
    parser = argparse.ArgumentParser()
    add_draft_worker_args(parser)
    flags = parser._option_string_actions
    assert "--speculative-num-draft-tokens" not in flags
    assert "--tensor-parallel-size" not in flags
    assert "--quantization" not in flags


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_resident_sessions", 0),
        ("max_batch_size", 0),
        ("staging_slots", 0),
        ("max_queued_jobs", 0),
        ("max_peers", 0),
        ("max_header_bytes", 0),
        ("max_feature_bytes", 0),
        ("max_host_memory_bytes", 0),
        ("lease_ms", 0),
        ("heartbeat_ms", 0),
        ("poll_ms", 0),
        ("linger_ms", -1),
        ("device_id", -1),
    ],
)
def test_worker_rejects_unbounded_or_invalid_resources(
    worker_args, field, value
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(worker_args, **{field: value})


def test_worker_checks_resource_relationships(worker_args) -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        dataclasses.replace(worker_args, max_batch_size=33)
    with pytest.raises(ValueError, match="heartbeat_ms"):
        dataclasses.replace(worker_args, heartbeat_ms=worker_args.lease_ms)
    with pytest.raises(ValueError, match="24 MiB"):
        dataclasses.replace(worker_args, max_feature_bytes=24 * 1024 * 1024 + 1)


@pytest.mark.parametrize("revision", [None, "", "main", "v1", "a" * 39, "g" * 40])
def test_remote_revision_is_immutable(revision) -> None:
    with pytest.raises(ValueError, match="40-character"):
        validate_remote_revision(revision, "--revision")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:1",
        "tcp://localhost",
        "tcp://localhost:0",
        "tcp://localhost:65536",
        "tcp://user:pass@localhost:1",
        "tcp://localhost:1/path",
        "tcp://localhost:1?x=1",
        "tcp://localhost:1#fragment",
        "tcp://bad host:1",
        "tcp://*:1",
        "tcp://0.0.0.0:1",
        "tcp://[::]:1",
        "",
    ],
)
def test_target_endpoint_rejects_invalid_or_bind_only_addresses(endpoint) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        validate_remote_tcp_endpoint(endpoint, allow_wildcard=False)


def test_tcp_ipv6_and_bind_wildcard_contract() -> None:
    validate_remote_tcp_endpoint("tcp://[::1]:32000", allow_wildcard=False)
    validate_remote_tcp_endpoint("tcp://*:32000", allow_wildcard=True)


def test_worker_dispatch_passes_configs_without_constructing_gpu_model(
    worker_args,
) -> None:
    captured = {}

    def run_service(service, model):
        captured.update(service=service, model=model)

    fake_transport = SimpleNamespace(
        DraftPoolServiceConfig=SimpleNamespace, run_draft_worker=run_service
    )
    fake_worker = SimpleNamespace(WorkerModelConfig=SimpleNamespace)
    with mock.patch.dict(
        sys.modules,
        {
            "tokenspeed.runtime.draft_pool.transport": fake_transport,
            "tokenspeed.runtime.draft_pool.worker": fake_worker,
        },
    ):
        run_draft_worker_from_args(
            argparse.Namespace(**dataclasses.asdict(worker_args))
        )
    assert captured["service"].listen_endpoint == worker_args.listen_address
    assert captured["service"].resident_limit == worker_args.max_resident_sessions
    assert (
        captured["service"].max_host_memory_bytes == worker_args.max_host_memory_bytes
    )
    assert captured["model"].target_revision == "a" * 40
    assert captured["model"].draft_revision == "b" * 40
    assert captured["model"].max_batch_size == 8


def test_worker_top_level_dispatch(worker_args, monkeypatch) -> None:
    from tokenspeed.cli import main

    called = []
    monkeypatch.setattr(
        "tokenspeed.cli.draft_worker.run_draft_worker_from_args", called.append
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenspeed",
            "draft-worker",
            "--target-model-path",
            worker_args.target_model_path,
            "--target-revision",
            worker_args.target_revision,
            "--draft-model-path",
            worker_args.draft_model_path,
            "--draft-revision",
            worker_args.draft_revision,
            "--listen-address",
            worker_args.listen_address,
            "--max-resident-sessions",
            "32",
            "--max-batch-size",
            "8",
        ],
    )
    main()
    assert len(called) == 1
    assert called[0].max_resident_sessions == 32


def test_worker_sigterm_reaches_service_cleanup_and_restores_handler(
    worker_args,
) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    retired = []

    def run_service(service, model):
        try:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        finally:
            retired.append(True)

    fake_transport = SimpleNamespace(
        DraftPoolServiceConfig=SimpleNamespace, run_draft_worker=run_service
    )
    fake_worker = SimpleNamespace(WorkerModelConfig=SimpleNamespace)
    with mock.patch.dict(
        sys.modules,
        {
            "tokenspeed.runtime.draft_pool.transport": fake_transport,
            "tokenspeed.runtime.draft_pool.worker": fake_worker,
        },
    ):
        run_draft_worker_from_args(
            argparse.Namespace(**dataclasses.asdict(worker_args))
        )
    assert retired == [True]
    assert signal.getsignal(signal.SIGTERM) is previous


def test_worker_help_never_imports_gpu_runtime() -> None:
    source = """import sys
from tokenspeed.cli import main
sys.argv = ['tokenspeed', 'draft-worker', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
assert 'tokenspeed.runtime.draft_pool.worker' not in sys.modules
assert 'tokenspeed.runtime.utils.server_args' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--max-resident-sessions" in result.stdout


@pytest.fixture
def model_pair():
    target = dict(
        model_type="glm_moe_dsa",
        architectures=["GlmMoeDsaForCausalLM"],
        hidden_size=6144,
        num_hidden_layers=78,
        vocab_size=154880,
        quantization_config={"quant_method": "fp8"},
    )
    draft = dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=6144,
        num_hidden_layers=6,
        num_target_layers=78,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=154880,
        sliding_window=2048,
        layer_types=["sliding_attention"] * 6,
        is_causal=False,
        dtype="bfloat16",
        dflash_config={
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61, 75],
        },
    )
    return target, draft


def test_matching_full_glm_model_pair_accepts_dict_and_hf_style_config(
    model_pair,
) -> None:
    target, draft = model_pair
    validate_remote_model_pair(target, draft)
    validate_remote_model_pair(SimpleNamespace(**target), SimpleNamespace(**draft))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "glm53_flash"),
        ("num_hidden_layers", 64),
        ("vocab_size", 32000),
        ("quantization_config", {"quant_method": "nvfp4"}),
    ],
)
def test_other_target_family_or_precision_is_rejected(model_pair, field, value) -> None:
    target, draft = model_pair
    with pytest.raises(ValueError, match="GLM-5.3"):
        validate_remote_model_pair({**target, field: value}, draft)


def test_draft_native_width_and_capture_order_are_exact(model_pair) -> None:
    target, draft = model_pair
    for changes in (dict(block_size=6), dict(target_layer_ids=[75, 61, 47, 33, 19, 5])):
        changed = {**draft, "dflash_config": {**draft["dflash_config"], **changes}}
        with pytest.raises(ValueError, match="dflash_config"):
            validate_remote_model_pair(target, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_types", ["full_attention"] * 6),
        ("layer_types", ["sliding_attention"] * 5 + ["full_attention"]),
        ("layer_types", ["sliding_attention"] * 5),
        ("is_causal", True),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 64),
        ("head_dim", 192),
    ],
)
def test_worker_cache_and_mask_require_exact_gqa_layout(
    model_pair, field, value
) -> None:
    target, draft = model_pair
    with pytest.raises(ValueError, match=field):
        validate_remote_model_pair(target, {**draft, field: value})


def test_draft_attention_mode_must_be_gqa(model_pair) -> None:
    target, draft = model_pair
    for mode in ("mla", "unknown", None):
        changed = {
            **draft,
            "dflash_config": {**draft["dflash_config"], "attention_mode": mode},
        }
        with pytest.raises(ValueError, match="GQA"):
            validate_remote_model_pair(target, changed)
    draft["dflash_config"]["attention_mode"] = "gqa"
    validate_remote_model_pair(target, draft)


def test_normalized_config_can_use_legacy_torch_dtype(model_pair) -> None:
    target, draft = model_pair
    draft.pop("dtype")
    draft["torch_dtype"] = "bfloat16"
    validate_remote_model_pair(target, draft)
