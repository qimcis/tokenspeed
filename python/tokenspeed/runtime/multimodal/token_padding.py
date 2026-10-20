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

"""Physical multimodal prompt IDs, independent of encoder and GPU imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.multimodal.inputs import MultimodalInputs


def pad_input_tokens(input_ids: list[int], mm_inputs: MultimodalInputs) -> list[int]:
    """Substitute placeholder token IDs with each item's ``pad_value``.

    The gateway produces ``input_ids`` with a single placeholder token
    repeated across every multimodal-token position (e.g. ``<image>``
    repeated 1024 times for a 1024-token image). The prefix cache needs
    each placeholder run to carry a content-derived ID so two different
    images compare unequal. We rewrite each ``offsets`` range to the
    item's pre-computed ``pad_value`` here.
    """
    if not input_ids or not mm_inputs.mm_items:
        return input_ids

    out = None
    for item in mm_inputs.mm_items:
        if item.pad_value is None or not item.offsets:
            continue
        if out is None:
            out = list(input_ids)
        pad_value = int(item.pad_value)
        for offset_start, offset_end in item.offsets:
            out[offset_start : offset_end + 1] = [pad_value] * (
                offset_end - offset_start + 1
            )
    return input_ids if out is None else out
