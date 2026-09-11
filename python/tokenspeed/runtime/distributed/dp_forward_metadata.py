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

"""CPU-only decode-width metadata for the existing post-plan DP exchange."""

from collections.abc import Sequence
from typing import Any


def planned_decode_width(forward_op: Any) -> int:
    """Read a cohort's selected decode width without changing its plan.

    Args:
        forward_op: The immutable scheduler batch, or ``None`` on an idle rank.
            Decode rows follow all extend rows in ``input_lengths``.

    Returns:
        Zero when no decode rows run, otherwise their common positive width.
        Older local-serving batch bindings can omit ``decode_input_tokens``;
        their already-planned row lengths provide the same information.
    """
    if forward_op is None:
        return 0
    decode_lengths = forward_op.input_lengths[forward_op.num_extends() :]
    if not decode_lengths:
        return 0
    widths = {int(length) for length in decode_lengths}
    if len(widths) != 1 or min(widths) <= 0:
        raise ValueError("An attention cohort must plan one positive decode width")
    row_width = next(iter(widths))
    selected_width = int(getattr(forward_op, "decode_input_tokens", row_width))
    if selected_width != row_width:
        raise ValueError("Planned decode width disagrees with reserved input rows")
    return selected_width


def common_decode_width(global_decode_input_tokens: Sequence[int]) -> int | None:
    """Find the common non-idle width in gathered physical-rank metadata.

    Args:
        global_decode_input_tokens: One selected width per physical rank. Zero
            means no decode rows; TP peers repeat their cohort's width.

    Returns:
        The common positive width, or ``None`` for mixed widths or no decode
        work. This does not change a cohort's selected width. Callers must
        separately exclude extend/mixed forward modes from decode graphs.
    """
    if any(width < 0 for width in global_decode_input_tokens):
        raise ValueError("Gathered decode widths must be nonnegative")
    widths = {width for width in global_decode_input_tokens if width > 0}
    return next(iter(widths)) if len(widths) == 1 else None
