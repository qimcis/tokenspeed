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

"""Unregistered SM90 candidate: BF16 gather plus native sparse FlashMLA.

Retains the portable page-planar codecs and their BF16 rounding boundary. Only
native calls pad query heads to 64/128 and selected width to 128. Compare the whole
gather/pad/native/copy chain against the portable live-head path before enabling
this candidate: FlashMLA rounds probability/value products differently, so
bitwise parity with the scalar FP32 path is not expected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
    _native_query,
    flash_mla_api,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _cache,
    _integers,
    _output,
    _same_device,
    cache_gather,
)


def _native_tile(q, kv, indices, attn_sink, softmax_scale, out):
    """Run one bounded tile; invalid holes remain masked through the native ABI."""
    q_native, sink_native = _native_query(q, attn_sink)
    width = max(128, (indices.shape[1] + 127) // 128 * 128)
    if indices.shape[1] != width:
        indices = F.pad(indices, (0, width - indices.shape[1]), value=-1)
    result, _, _ = flash_mla_api().flash_mla_sparse_fwd(
        q=q_native,
        kv=kv,
        indices=indices.contiguous().unsqueeze(1),
        sm_scale=float(softmax_scale),
        d_v=512,
        attn_sink=sink_native,
        # SWA has fixed capacity before the global segment. A sum of segment
        # lengths would cut off valid global rows after holes in that capacity.
        topk_length=None,
    )
    # The pinned FlashMLA wheel has no out keyword. Keep only actual heads and
    # release its padded result before allocating the next tile's workspace.
    out.copy_(result[:, : q.shape[1], :])


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
    """Return BF16 selected attention with the portable 14-argument contract.

    Args:
        q: Post-RoPE BF16 [tokens, heads, 512], with 1..128 real heads.
        swa_cache: Page-planar uint8 [pages,64,528], ignored for compact prefill.
        swa_slots: Integer [tokens,width] physical SWA slots, including holes.
        swa_lens: Device-visible integer [tokens] active prefix lengths.
        global_cache: Optional page-planar uint8 [pages,64,288] global cache.
        global_slots: Optional integer [tokens,width] global slots.
        global_lens: Optional device-visible integer [tokens] global lengths.
        attn_sink: FP32 real-head or pre-padded sink logits; counted once.
        softmax_scale: QK logit multiplier.
        out: Contiguous BF16 destination matching q, or None to allocate.
        query_chunk_size: Positive query tile bound for gathered KV scratch.
        schedule: Caller-owned paged schedule, unused by this sparse-prefill ABI.
        prefill_kv: Optional existing BF16 [rows,1,512] compact workspace.
        prefill_indices: Optional matching integer [tokens,width] workspace IDs.

    Returns:
        Contiguous output shaped like q; supplied out is returned in place.
        Invalid selections are omitted, including non-prefix holes. Cache
        selections and lengths are read on-device on every graph replay. The
        caller owns SM90 dispatch; this candidate changes no registry entries.
    """
    if (
        q.ndim != 3
        or q.shape[-1] != 512
        or q.dtype != torch.bfloat16
        or not 1 <= q.shape[1] <= 128
    ):
        raise ValueError("q must be BF16 [tokens,1..128 heads,512]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    if (prefill_kv is None) != (prefill_indices is None):
        raise ValueError("prefill_kv and prefill_indices must be supplied together")
    attn_sink = attn_sink[: q.shape[1]]
    _same_device(
        q,
        (
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            out,
            prefill_kv,
            prefill_indices,
        ),
    )
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must contain FP32 logits for every query head")
    out = _output(out, q.shape, q.dtype, q.device)

    if prefill_kv is not None:
        if prefill_kv.ndim == 2:
            prefill_kv = prefill_kv.unsqueeze(1)
        if (
            prefill_kv.ndim != 3
            or prefill_kv.shape[1:] != (1, 512)
            or prefill_kv.dtype != torch.bfloat16
            or prefill_kv.stride(-1) != 1
        ):
            raise ValueError(
                "prefill_kv must be BF16 [rows,1,512] with contiguous rows"
            )
        if prefill_indices.ndim != 2 or prefill_indices.shape[0] != q.shape[0]:
            raise ValueError("prefill_indices must have one row per query")
        _integers(prefill_indices, prefill_indices.shape, "prefill_indices")
        if prefill_kv.shape[0] > torch.iinfo(torch.int32).max:
            raise ValueError("prefill workspace exceeds native int32 index capacity")
        if prefill_kv.shape[0] == 0 or prefill_indices.shape[1] == 0:
            return out.zero_()
        for start in range(0, q.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, q.shape[0])
            indices = prefill_indices[start:end]
            # Mask before narrowing int64, so out-of-range IDs cannot wrap into
            # valid int32 workspace rows. Do not compact or discard duplicates.
            valid = (indices >= 0) & (indices < prefill_kv.shape[0])
            indices = indices.masked_fill(~valid, -1).to(torch.int32)
            _native_tile(
                q[start:end],
                prefill_kv,
                indices,
                attn_sink,
                softmax_scale,
                out[start:end],
            )
        return out

    segments = [(swa_cache, swa_slots, swa_lens, "swa")]
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            raise ValueError("absent global cache requires absent global metadata")
    else:
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires slots and lengths")
        segments.append((global_cache, global_slots, global_lens, "global"))
    for cache, slots, lens, cache_format in segments:
        _cache(cache, cache_format)
        if slots is None or slots.ndim != 2 or slots.shape[0] != q.shape[0]:
            raise ValueError("slots must have one row per query")
        _integers(slots, slots.shape, "slots")
        if lens is None:
            raise ValueError("each cache requires slots and lengths")
        _integers(lens, (q.shape[0],), "lens")
    width = sum(slots.shape[1] for _, slots, _, _ in segments)
    if width == 0:
        return out.zero_()
    if min(query_chunk_size, q.shape[0]) * width > torch.iinfo(torch.int32).max:
        raise ValueError("gathered tile exceeds native int32 index capacity")
    for start in range(0, q.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, q.shape[0])
        kv_parts, valid_parts = [], []
        for cache, slots, lens, cache_format in segments:
            slots = slots[start:end]
            valid = (
                (torch.arange(slots.shape[1], device=q.device) < lens[start:end, None])
                & (slots >= 0)
                & (slots < cache.shape[0] * 64)
            )
            kv_parts.append(
                cache_gather(cache, slots.masked_fill(~valid, -1), cache_format, None)
            )
            valid_parts.append(valid)
        kv = torch.cat(kv_parts, dim=1).reshape(-1, 1, 512)
        valid = torch.cat(valid_parts, dim=1)
        indices = (
            torch.arange((end - start) * width, dtype=torch.int32, device=q.device)
            .view(end - start, width)
            .masked_fill(~valid, -1)
        )
        _native_tile(
            q[start:end],
            kv,
            indices,
            attn_sink,
            softmax_scale,
            out[start:end],
        )
        del kv, kv_parts, valid_parts, valid, indices
    return out
