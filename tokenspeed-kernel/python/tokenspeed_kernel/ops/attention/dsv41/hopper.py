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

"""Measured SM90 dispatch for DeepSeek V4.1 selected attention.

Native sparse BF16 attention accelerates existing prefill workspaces and larger
combined-cache batches. Small SWA-only batches use direct packed-cache reads.
The scalar portable path remains preferable for small combined-cache batches.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv41._hopper_gathered import (
    selected_attention as _gathered_attention,
)
from tokenspeed_kernel.ops.attention.dsv41._hopper_tiled import (
    hopper_tiled_selected_attention,
)
from tokenspeed_kernel.ops.attention.dsv41.flash_mla import is_flash_mla_v41_available
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    selected_attention as _portable_attention,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "attention",
    "dsv41_selected_attention",
    name="hopper_dsv41_selected_attention",
    solution="hopper",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    priority=Priority.PERFORMANT,
)
def selected_attention(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
):
    """Return attention using the measured SM90 TP8 path and existing contract.

    All fourteen arguments and the output obey ``dsv41.selected_attention``.
    Selection uses host-visible tensor geometry only; device lengths and slots
    stay live on graph replay. Packed cache bytes and BF16 dequantization are
    unchanged. Query padding, when needed, belongs to the native gathered path.
    Other local head counts retain the portable implementation until measured.
    """
    arguments = (
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        out,
        query_chunk_size,
        schedule,
        prefill_kv,
        prefill_indices,
    )
    if q.ndim != 3 or q.shape[1:] != (8, 512) or q.shape[0] == 0:
        return _portable_attention(*arguments)
    native_available = is_flash_mla_v41_available()
    if prefill_kv is not None:
        implementation = (
            _gathered_attention if native_available else _portable_attention
        )
        return implementation(*arguments)
    if prefill_indices is not None:
        return _portable_attention(*arguments)
    if global_cache is None:
        # The two measured regimes avoid a global query-head padding allocation.
        tile_k, num_warps = (64, 8) if q.shape[0] <= 64 else (32, 4)
        return hopper_tiled_selected_attention(
            *arguments, pv_mode="bf16", tile_k=tile_k, num_warps=num_warps
        )
    if native_available and q.shape[0] >= 16:
        return _gathered_attention(*arguments)
    return _portable_attention(*arguments)
