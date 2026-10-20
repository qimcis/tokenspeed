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

"""Portable fused Engram hot-row lookup, UVA miss gather and BF16 dequant."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit(do_not_specialize=["W", "S", "N", "H", "START", "END"])
def _engram_cached_embedding_kernel(
    W,
    S,
    I,
    HI,
    HC,
    HS,
    O,
    N,
    H,
    START,
    END,
    D: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, B)
    if row < N:
        index = tl.load(I + row).to(tl.int64)
        local_id = index - START
        local = (index >= START) & (index < END)
        value = tl.full((B,), 0, tl.float32)
        if local:
            # One program owns one requested row: search and branch are uniform.
            lo = tl.full((), 0, tl.int64)
            hi = H.to(tl.int64)
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = tl.load(HI + mid)
                before = candidate < local_id
                lo = tl.where(before, mid + 1, lo)
                hi = tl.where(before, hi, mid)
            found = tl.load(HI + lo, mask=lo < H, other=-1)
            hit = (lo < H) & (found == local_id)
            if hit:
                code = tl.load(HC + lo * D + cols, cols < D, 0)
                exponent = tl.load(HS + lo * (D // 32) + cols // 32, cols < D, 127)
            else:
                host_codes = tl.cast(W, tl.pointer_type(tl.uint8))
                host_scales = tl.cast(S, tl.pointer_type(tl.uint8))
                # Keep explicit miss masks even inside the uniform branch. Host
                # bytes are never fetched on a hit, even after if-conversion.
                code = tl.load(host_codes + local_id * D + cols, (cols < D) & ~hit, 0)
                exponent = tl.load(
                    host_scales + local_id * (D // 32) + cols // 32,
                    (cols < D) & ~hit,
                    127,
                )
            code = code.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            exponent = exponent.to(tl.int32)
            scale = tl.where(
                exponent == 0, 2.0**-127, (exponent << 23).to(tl.float32, bitcast=True)
            )
            scale = tl.where(exponent == 255, float("nan"), scale)
            value = code * scale
        tl.store(O + row * D + cols, value, cols < D)


@register_kernel(
    "embedding",
    "engram_cached_embedding",
    name="triton_engram_cached_embedding",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=[format_signature(weight=dense_tensor_format(torch.uint8))],
    priority=Priority.PORTABLE,
)
def triton_engram_cached_embedding(
    weight,
    scales,
    indices,
    hot_ids,
    hot_codes,
    hot_scales,
    row_start,
    row_end,
    *,
    out,
):
    """Launch into caller-owned output; all metadata except width is runtime."""
    platform = current_platform()
    _engram_cached_embedding_kernel[(indices.numel(),)](
        platform.device_visible_data_ptr(weight),
        platform.device_visible_data_ptr(scales),
        indices,
        hot_ids,
        hot_codes.view(torch.uint8),
        hot_scales,
        out,
        indices.numel(),
        hot_ids.numel(),
        row_start,
        row_end,
        D=weight.shape[1],
        B=triton.next_power_of_2(weight.shape[1]),
        num_warps=4,
    )
    return out
