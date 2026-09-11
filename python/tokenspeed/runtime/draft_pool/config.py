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

"""CPU-only launch validation for the supported remote draft deployment."""

import re
from collections.abc import Mapping
from urllib.parse import urlsplit


def validate_remote_revision(revision: str | None, argument: str) -> None:
    """Require a full immutable checkpoint commit ID for the named argument."""
    if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError(
            f"{argument} requires an immutable 40-character checkpoint commit ID"
        )


def validate_remote_tcp_endpoint(endpoint: str, *, allow_wildcard: bool) -> None:
    """Validate a private-network TCP endpoint; this does not authenticate it.

    Args:
        endpoint: ZMQ TCP endpoint with an explicit hostname and port.
        allow_wildcard: Whether bind-only wildcard hosts are permitted.
    """
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Remote draft endpoint must be tcp://host:port") from exc
    invalid = (
        parsed.scheme != "tcp"
        or not parsed.hostname
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.path or parsed.query or parsed.fragment)
        or any(char.isspace() for char in endpoint)
    )
    if not allow_wildcard and parsed.hostname in ("*", "0.0.0.0", "::"):
        invalid = True
    if invalid:
        raise ValueError(
            "Remote draft endpoint must be tcp://host:port with an explicit valid port"
        )


def _field(config: object, name: str, missing: object) -> object:
    if isinstance(config, Mapping):
        return config.get(name, missing)
    return getattr(config, name, missing)


def validate_remote_model_pair(target_config: object, draft_config: object) -> None:
    """Validate the qualified full-GLM/DFlash2 geometry before allocating weights.

    Args:
        target_config: Full GLM-5.3 FP8 checkpoint configuration.
        draft_config: Matching native-eight BF16 DFlash2 configuration.

    Revisions and the protocol fingerprint bind these structural checks to
    the selected checkpoint bytes. Paths may name local snapshot directories.
    """
    target_fields = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "vocab_size": 154880,
    }
    draft_fields = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 6144,
        "num_hidden_layers": 6,
        "num_target_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 154880,
        "sliding_window": 2048,
        "layer_types": ["sliding_attention"] * 6,
        "is_causal": False,
    }
    for config, expected, label in (
        (target_config, target_fields, "full GLM-5.3 target"),
        (draft_config, draft_fields, "GLM-5.3 DFlash2 draft"),
    ):
        for name, value in expected.items():
            if _field(config, name, None) != value:
                raise ValueError(
                    f"Remote drafting requires the {label}: {name} must be {value!r}"
                )
    quantization = _field(target_config, "quantization_config", None)
    if _field(quantization, "quant_method", None) != "fp8":
        raise ValueError(
            "Remote drafting requires the full GLM-5.3 FP8 target checkpoint"
        )
    draft_quantization = _field(draft_config, "quantization_config", None)
    if draft_quantization:
        raise ValueError("Remote drafting requires unquantized BF16 DFlash2 weights")
    draft_dtype = _field(draft_config, "dtype", None)
    if draft_dtype is None:
        draft_dtype = _field(draft_config, "torch_dtype", None)
    if str(draft_dtype) not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("Remote drafting requires a BF16 DFlash2 checkpoint")
    block = _field(draft_config, "dflash_config", None)
    if _field(block, "attention_mode", "gqa") != "gqa":
        raise ValueError("Remote GLM-5.3 DFlash2 requires GQA draft attention")
    expected_block = {
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61, 75],
    }
    for name, value in expected_block.items():
        if _field(block, name, None) != value:
            raise ValueError(
                f"Remote GLM-5.3 DFlash2 requires dflash_config.{name}={value!r}"
            )
