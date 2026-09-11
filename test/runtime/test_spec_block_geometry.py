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

"""CPU-only checkpoint/verification geometry checks without GPU imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_source = (
    Path(__file__).resolve().parents[2]
    / "python/tokenspeed/runtime/utils/spec_block_geometry.py"
)
_spec = importlib.util.spec_from_file_location("spec_block_geometry", _source)
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)


@pytest.mark.parametrize("verify_width", (2, 6, 8, None))
def test_native_eight_can_consume_a_shorter_prefix(verify_width: int | None) -> None:
    assert _geometry.resolve_block_widths("DFLASH", 8) == (7, 8)
    assert _geometry.resolve_verify_width(8, verify_width) == (
        8 if verify_width is None else verify_width
    )
    _geometry.validate_block_widths("DFLASH", 8, 7, 8)


@pytest.mark.parametrize(
    "native_width,verify_width", ((8, 1), (8, 9), (1, None), (8, 0))
)
def test_invalid_verifier_capacity_fails_early(
    native_width: int, verify_width: int | None
) -> None:
    with pytest.raises(ValueError, match="speculative-verify-tokens"):
        _geometry.resolve_verify_width(native_width, verify_width)


def test_short_target_prefix_cannot_shrink_native_checkpoint_geometry() -> None:
    config = SimpleNamespace(dflash_config={"block_size": 8})
    native = _geometry.read_checkpoint_block_size(config)
    assert _geometry.resolve_verify_width(native, 6) == 6
    with pytest.raises(ValueError, match="requires --speculative-num-steps 7"):
        _geometry.validate_block_widths("DFLASH", native, 5, 6)


def test_dspark_native_convention_remains_unchanged() -> None:
    assert _geometry.resolve_block_widths("DSPARK", 7) == (7, 8)
    _geometry.validate_block_widths("DSPARK", 7, 7, 8)
