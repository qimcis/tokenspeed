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

"""Startup keeps native drafting independent of target verification and placement."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tokenspeed.runtime.configs.model_config import _apply_block_spec_widths
from tokenspeed.runtime.execution.factory import create_model_runner
from tokenspeed.runtime.layers.attention.configs.base import (
    resolve_speculative_num_tokens,
)


def test_short_verification_preserves_checkpoint_native_width():
    args = SimpleNamespace(
        speculative_algorithm="DFLASH",
        speculative_num_steps=7,
        speculative_num_draft_tokens=8,
        speculative_verify_tokens=6,
        _speculative_widths_explicit=True,
    )
    checkpoint = SimpleNamespace(dflash_config={"block_size": 8})
    assert _apply_block_spec_widths(args, checkpoint, checkpoint) == 8
    assert (args.speculative_num_steps, args.speculative_num_draft_tokens) == (7, 8)
    assert resolve_speculative_num_tokens(args, is_draft=False) == 6
    assert resolve_speculative_num_tokens(args, is_draft=True) == 8


@pytest.mark.parametrize("verify_width", [0, 1, 9])
def test_verification_must_consume_a_valid_checkpoint_prefix(verify_width):
    args = SimpleNamespace(
        speculative_algorithm="DFLASH",
        speculative_num_steps=7,
        speculative_num_draft_tokens=8,
        speculative_verify_tokens=verify_width,
        _speculative_widths_explicit=True,
    )
    checkpoint = SimpleNamespace(block_size=8)
    with pytest.raises(ValueError, match="speculative-verify-tokens"):
        _apply_block_spec_widths(args, checkpoint, checkpoint)


def test_default_verification_still_uses_the_complete_native_block():
    args = SimpleNamespace(
        speculative_algorithm="DFLASH",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_verify_tokens=None,
        _speculative_widths_explicit=False,
    )
    checkpoint = SimpleNamespace(block_size=8)
    _apply_block_spec_widths(args, checkpoint, checkpoint)
    assert resolve_speculative_num_tokens(args, is_draft=False) == 8
    assert resolve_speculative_num_tokens(args, is_draft=True) == 8


@pytest.mark.parametrize("remote", [False, True])
def test_remote_target_never_constructs_a_full_draft_runner(remote):
    args = SimpleNamespace(
        remote_draft_endpoint="tcp://127.0.0.1:5555" if remote else None,
        speculative_algorithm="DFLASH",
    )
    target_config, draft_config = object(), object()
    target, draft = object(), object()
    with (
        patch(
            "tokenspeed.runtime.execution.factory.ModelRunner",
            side_effect=[target, draft],
        ) as runner,
        patch(
            "tokenspeed.runtime.execution.factory._wire_draft_to_target_model"
        ) as wire,
    ):
        actual_target, actual_draft = create_model_runner(
            args, target_config, draft_config, 0, 0
        )
    assert actual_target is target
    assert actual_draft is (None if remote else draft)
    assert runner.call_count == (1 if remote else 2)
    if remote:
        wire.assert_not_called()
    else:
        wire.assert_called_once_with(args, target, draft)
