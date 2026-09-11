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

from collections.abc import Callable

import torch

from tokenspeed.runtime.distributed.comm_manager import MoEInputLayout
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_here,
    is_breakable_capture_active,
)
from tokenspeed.runtime.execution.context import ForwardContext

MoEForward = Callable[
    [torch.Tensor, torch.Tensor | None, ForwardContext, MoEInputLayout], torch.Tensor
]


def _capture_moe_block(
    forward_live: MoEForward,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    ctx: ForwardContext,
    layout: MoEInputLayout,
    dst: torch.Tensor,
) -> torch.Tensor:
    return dst.zero_()


def _run_moe_block_into(
    forward_live: MoEForward,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    ctx: ForwardContext,
    layout: MoEInputLayout,
    dst: torch.Tensor,
) -> torch.Tensor:
    dst.zero_()
    result = forward_live(hidden_states, input_ids, ctx, layout)
    rows = layout.resolve(ctx).output_rows
    if result.shape != (rows, *dst.shape[1:]):
        raise ValueError("MoE output does not match its live return layout")
    dst[:rows].copy_(result)
    return dst


def run_moe_block(
    forward_live: MoEForward,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    ctx: ForwardContext,
    layout: MoEInputLayout,
) -> torch.Tensor:
    """Run one MoE block with exact live math and a stable graph destination.

    ``forward_live`` receives the original physical inputs and current context;
    it alone slices source rows and gathers the result according to ``layout``.
    ``input_ids`` accompanies hash routing, or is ``None``. Returns local or
    replicated output matching the input capacity during graph capture/replay.
    """
    if not is_breakable_capture_active():
        return forward_live(hidden_states, input_ids, ctx, layout)
    dst = torch.empty_like(hidden_states)
    return break_here(
        _run_moe_block_into,
        dst,
        forward_live,
        hidden_states,
        input_ids,
        ctx,
        layout,
        dst,
        capture_stub=_capture_moe_block,
    )
