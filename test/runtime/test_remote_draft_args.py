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

"""Remote launch constraints and preserved local speculative defaults."""

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture
def remote_args():
    with mock.patch.object(ServerArgs, "__post_init__"):
        args = ServerArgs(
            model="zai-org/GLM-5.3",
            revision="a" * 40,
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="incoai/GLM-5.3-DFlash2",
            speculative_draft_model_revision="b" * 40,
            remote_draft_endpoint="tcp://127.0.0.1:32000",
            remote_draft_min_ready=8,
            remote_draft_max_defer_ms=30.0,
            mapping=SimpleNamespace(has_attn_cp=False),
        )
    args.resolve_basic_defaults()
    return args


def test_remote_default_geometry_preserves_native_eight(remote_args) -> None:
    remote_args.resolve_speculative_decoding()
    assert remote_args.speculative_num_steps == 7
    assert remote_args.speculative_num_draft_tokens == 8
    assert remote_args.speculative_verify_tokens == 6
    assert remote_args.draft_model_path_use_base is False


def test_local_defaults_are_unchanged() -> None:
    with mock.patch.object(ServerArgs, "__post_init__"):
        args = ServerArgs(model="target", speculative_algorithm="DFLASH")
    args.resolve_basic_defaults()
    args.resolve_speculative_decoding()
    assert args.remote_draft_endpoint is None
    assert args.remote_draft_min_ready is None
    assert args.remote_draft_max_defer_ms is None
    assert args.speculative_verify_tokens is None
    assert args.speculative_num_steps == 3
    assert args.speculative_num_draft_tokens == 4


def test_local_short_verify_waits_for_checkpoint_native_geometry() -> None:
    with mock.patch.object(ServerArgs, "__post_init__"):
        args = ServerArgs(
            model="target", speculative_algorithm="DFLASH", speculative_verify_tokens=6
        )
    args.resolve_basic_defaults()
    args.resolve_speculative_decoding()
    # Native block is defaulted later from the checkpoint; do not reject six
    # merely because the pre-config default happens to be four.
    assert args.speculative_num_draft_tokens == 4
    assert args.speculative_verify_tokens == 6


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("speculative_algorithm", "EAGLE3", "DFLASH"),
        ("speculative_draft_model_path", None, "separate"),
        ("draft_model_path_use_base", True, "separate"),
        ("speculative_draft_model_quantization", "nvfp4", "BF16"),
        ("speculative_num_draft_tokens", 6, "native draft width 8"),
        ("speculative_verify_tokens", 8, "verification width 6"),
        ("pipeline_parallel_size", 2, "pipeline"),
        ("disaggregation_mode", "decode", "PD/EPD"),
        ("disaggregation_mode", "prefill", "PD/EPD"),
        ("disaggregation_mode", "encode", "PD/EPD"),
        ("dp_sampling", True, "dp-sampling"),
        ("remote_draft_min_ready", None, "min-ready"),
        ("remote_draft_min_ready", 0, "min-ready"),
        ("remote_draft_max_defer_ms", None, "max-defer"),
        ("remote_draft_max_defer_ms", -1, "max-defer"),
        ("remote_draft_max_defer_ms", float("inf"), "max-defer"),
        ("remote_draft_max_defer_ms", float("nan"), "max-defer"),
        ("revision", "main", "--revision"),
        ("speculative_draft_model_revision", None, "draft-model-revision"),
    ],
)
def test_remote_rejects_unsupported_launches(
    remote_args, field, value, message
) -> None:
    setattr(remote_args, field, value)
    with pytest.raises(ValueError, match=message):
        remote_args.validate_remote_draft_options()


def test_remote_allows_immediate_fallback_policy(remote_args) -> None:
    remote_args.remote_draft_max_defer_ms = 0
    remote_args.validate_remote_draft_options()


def test_attention_dp_is_supported_but_context_parallelism_is_not(remote_args) -> None:
    remote_args.mapping = SimpleNamespace(
        has_attn_cp=False, has_attn_dp=True, attn=SimpleNamespace(dp_size=8, tp_size=1)
    )
    remote_args.validate_remote_draft_options()
    remote_args.mapping.has_attn_cp = True
    with pytest.raises(ValueError, match="context parallelism"):
        remote_args.validate_remote_draft_options()


def test_policy_flags_require_remote_endpoint(remote_args) -> None:
    remote_args.remote_draft_endpoint = None
    with pytest.raises(ValueError, match="remote-draft-endpoint"):
        remote_args.validate_remote_draft_options()


def test_parser_keeps_remote_flags_in_engine_contract() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        [
            "--model",
            "zai-org/GLM-5.3",
            "--revision",
            "a" * 40,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "incoai/GLM-5.3-DFlash2",
            "--speculative-draft-model-revision",
            "b" * 40,
            "--remote-draft-endpoint",
            "tcp://127.0.0.1:32000",
            "--remote-draft-min-ready",
            "8",
            "--remote-draft-max-defer-ms",
            "30",
            "--speculative-verify-tokens",
            "6",
        ]
    )
    with mock.patch.object(ServerArgs, "__post_init__"):
        args = ServerArgs.from_cli_args(parsed)
    assert args.remote_draft_min_ready == 8
    assert args.remote_draft_max_defer_ms == 30
    assert args.speculative_verify_tokens == 6
    assert args.speculative_draft_model_revision == "b" * 40


def test_model_pair_validation_is_noop_for_local_serving(remote_args) -> None:
    remote_args.remote_draft_endpoint = None
    remote_args.validate_remote_draft_model_configs(object(), object())
