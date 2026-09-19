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

"""Unregistered SM90 candidates: tensor-core attention over two packed caches.

The page-planar reader follows the existing V4.1 gather and AMD selected-attention
semantics. Dequantized values round to BF16 before either dot product. The two
explicit PV variants compare BF16 probabilities with FP32 probabilities consumed
by TF32x3; neither promises bitwise equality with scalar FP32 reductions.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _cache,
    _e2m1_decode,
    _integers,
    _output,
    _planar_offset,
    _same_device,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    selected_attention as _portable_selected_attention,
)


@triton.jit
def _load_cache_tile(
    CACHE,
    SLOTS,
    token,
    columns,
    length,
    SLOT_T,
    SLOT_K,
    PAGE_STRIDE,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    CR: tl.constexpr,
    CB: tl.constexpr,
    IS_SWA: tl.constexpr,
):
    slot = tl.load(
        SLOTS + token * SLOT_T + columns * SLOT_K,
        mask=(columns < length) & (columns < WIDTH),
        other=-1,
    ).to(tl.int64)
    valid = (columns < length) & (columns < WIDTH) & (slot >= 0) & (slot < CAPACITY)
    slot = tl.where(valid, slot, 0)
    dims = tl.arange(0, 512)
    if IS_SWA:
        VALUES: tl.constexpr = 512
        GROUP: tl.constexpr = 32
        ROW_BYTES: tl.constexpr = 528
        value_dim = dims
    else:
        VALUES: tl.constexpr = 256
        GROUP: tl.constexpr = 16
        ROW_BYTES: tl.constexpr = 288
        value_dim = dims // 2
    page = slot // 64
    row = slot % 64
    value_byte = row[None, :] * VALUES + value_dim[:, None]
    groups = tl.arange(0, 512 // GROUP)
    # Load each physical scale once per selected row. Broadcasting the float
    # scale afterward also avoids repeated 2-D byte loads for every dimension.
    scale_byte = 64 * VALUES + row[None, :] * (512 // GROUP) + groups[:, None]
    base = CACHE + page[None, :] * PAGE_STRIDE
    encoded = tl.load(
        base + _planar_offset(value_byte, CR, CB, ROW_BYTES),
        valid[None, :],
        other=0,
    )
    encoded_scale = tl.load(
        base + _planar_offset(scale_byte, CR, CB, ROW_BYTES),
        valid[None, :],
        other=0,
    )
    if IS_SWA:
        value = encoded.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale = tl.where(
            encoded_scale == 0,
            2.0**-127,
            (encoded_scale.to(tl.int32) << 23).to(tl.float32, bitcast=True),
        )
    else:
        code = (encoded.to(tl.int32) >> ((dims[:, None] % 2) * 4)) & 15
        value = _e2m1_decode(code)
        scale = encoded_scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale = tl.broadcast_to(
        scale[:, None, :], (512 // GROUP, GROUP, columns.shape[0])
    ).reshape(512, columns.shape[0])
    # This cast is the existing cache_gather -> BF16 workspace boundary.
    decoded = (value * scale).to(tl.bfloat16)
    return tl.where(valid[None, :], decoded, 0.0).to(tl.bfloat16), valid


@triton.jit
def _hopper_tiled_kernel(
    Q,
    SWA,
    SWA_SLOTS,
    SWA_LENGTHS,
    GLOBAL,
    GLOBAL_SLOTS,
    GLOBAL_LENGTHS,
    SINK,
    OUT,
    QT,
    QH,
    QD,
    ST,
    SK,
    SL,
    SP,
    GT,
    GK,
    GL,
    GP,
    SS,
    OT,
    OH,
    SCALE,
    HEADS: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    GLOBAL_WIDTH: tl.constexpr,
    SWA_CAPACITY: tl.constexpr,
    GLOBAL_CAPACITY: tl.constexpr,
    SWA_CR: tl.constexpr,
    SWA_CB: tl.constexpr,
    GLOBAL_CR: tl.constexpr,
    GLOBAL_CB: tl.constexpr,
    HAS_GLOBAL: tl.constexpr,
    TILE_K: tl.constexpr,
    PV_MODE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    dims = tl.arange(0, 512)
    query = tl.load(
        Q + token * QT + heads[:, None] * QH + dims[None, :] * QD,
        mask=(heads < HEADS)[:, None],
        other=0.0,
    )
    swa_length = tl.minimum(
        tl.maximum(tl.load(SWA_LENGTHS + token * SL).to(tl.int64), 0), SWA_WIDTH
    ).to(tl.int32)
    if HAS_GLOBAL:
        global_length = tl.minimum(
            tl.maximum(tl.load(GLOBAL_LENGTHS + token * GL).to(tl.int64), 0),
            GLOBAL_WIDTH,
        ).to(tl.int32)
    else:
        global_length = 0
    swa_tiles = tl.cdiv(swa_length, TILE_K)
    global_tiles = tl.cdiv(global_length, TILE_K)
    sink = tl.load(SINK + heads * SS, heads < HEADS, other=0.0).to(tl.float32)
    infinite_sink = sink == float("inf")
    # A +inf sink owns all probability mass and returns zero. Avoid inf-inf
    # intermediates, and avoid -inf - -inf for an absent sink/empty first tile.
    maximum = tl.where(infinite_sink, 0.0, sink)
    denominator = tl.where(sink > -float("inf"), 1.0, 0.0)
    accumulator = tl.zeros((16, 512), dtype=tl.float32)
    offsets = tl.arange(0, TILE_K)
    for tile in range(swa_tiles + global_tiles):
        if tile < swa_tiles:
            columns = tile * TILE_K + offsets
            values, valid = _load_cache_tile(
                SWA,
                SWA_SLOTS,
                token,
                columns,
                swa_length,
                ST,
                SK,
                SP,
                SWA_CAPACITY,
                SWA_WIDTH,
                SWA_CR,
                SWA_CB,
                True,
            )
        else:
            columns = (tile - swa_tiles) * TILE_K + offsets
            if HAS_GLOBAL:
                values, valid = _load_cache_tile(
                    GLOBAL,
                    GLOBAL_SLOTS,
                    token,
                    columns,
                    global_length,
                    GT,
                    GK,
                    GP,
                    GLOBAL_CAPACITY,
                    GLOBAL_WIDTH,
                    GLOBAL_CR,
                    GLOBAL_CB,
                    False,
                )
            else:
                values = tl.zeros((512, TILE_K), dtype=tl.bfloat16)
                valid = offsets < 0
        logits = tl.dot(query, values, out_dtype=tl.float32) * SCALE
        logits = tl.where(
            valid[None, :] & (heads < HEADS)[:, None], logits, -float("inf")
        )
        tile_maximum = tl.max(logits, axis=1)
        next_maximum = tl.maximum(maximum, tile_maximum)
        safe_maximum = tl.where(next_maximum > -float("inf"), next_maximum, 0.0)
        previous_scale = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(logits - safe_maximum[:, None])
        probabilities = tl.where(valid[None, :], probabilities, 0.0)
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=1)
        accumulator *= previous_scale[:, None]
        if PV_MODE == "bf16":
            accumulator = tl.dot(
                probabilities.to(tl.bfloat16),
                tl.trans(values),
                accumulator,
                out_dtype=tl.float32,
            )
        else:
            accumulator = tl.dot(
                probabilities,
                tl.trans(values.to(tl.float32)),
                accumulator,
                input_precision="tf32x3",
                out_dtype=tl.float32,
            )
        maximum = next_maximum
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    result = accumulator / safe_denominator[:, None]
    result = tl.where(
        (denominator > 0.0)[:, None] & ~infinite_sink[:, None], result, 0.0
    )
    tl.store(
        OUT + token * OT + heads[:, None] * OH + dims[None, :],
        result,
        mask=(heads < HEADS)[:, None],
    )


def hopper_tiled_selected_attention(
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
    pv_mode,
    tile_k,
    num_warps,
):
    """Compute selected attention using M16 live-head tiles and direct cache reads.

    Args:
        q: BF16 [tokens, 8/16/32/64, 512] query; all positive strides are supported.
        swa_cache: Page-planar uint8 [pages, 64, 528] SWA storage.
        swa_slots: Int32/int64 [tokens, width] logical SWA slots, including holes.
        swa_lens: Device-visible int32/int64 [tokens] SWA prefix lengths.
        global_cache: Optional page-planar uint8 [pages, 64, 288] global storage.
        global_slots: Optional int32/int64 [tokens, width] global slots.
        global_lens: Optional device-visible int32/int64 [tokens] global lengths.
        attn_sink: FP32 vector containing at least one sink per live head.
        softmax_scale: Scale for the FP32 QK logits; sink is already a logit.
        out: Optional contiguous BF16 output with exactly q.shape; returned as-is.
        query_chunk_size: Positive portable-path workspace bound. Direct decode
            allocates no per-query KV workspace and launches all rows directly.
        schedule: Opaque native schedule; ignored by direct/portable execution.
        prefill_kv: Optional existing BF16 prefill workspace, delegated unchanged.
        prefill_indices: Optional matching prefill indices for the portable path.
        pv_mode: Explicit "bf16" or "tf32x3" probability/value arithmetic.
        tile_k: Explicit selected-key tile, 32 or 64 for BF16 PV; 32 for TF32x3.
        num_warps: Explicit CTA warp count, 4 or 8.

    Returns:
        BF16 [tokens, live_heads, 512]. This unregistered candidate is SM90-only;
        the caller owns choosing it after correctness and performance validation.
    """
    if (
        pv_mode not in ("bf16", "tf32x3")
        or tile_k not in (32, 64)
        or num_warps not in (4, 8)
    ):
        raise ValueError(
            "Expected pv_mode bf16/tf32x3, tile_k 32/64, and num_warps 4/8"
        )
    if pv_mode == "tf32x3" and tile_k != 32:
        raise ValueError(
            "tf32x3 PV supports tile_k=32 only; K64 exceeds the SM90 shared-memory budget"
        )
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    if (
        q.ndim != 3
        or q.shape[1] not in (8, 16, 32, 64)
        or q.shape[2] != 512
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("q must be BF16 [tokens, 8/16/32/64, 512]")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Hopper tiled attention requires an SM90 CUDA device")
    if prefill_kv is not None:
        return _portable_selected_attention(
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
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
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
        ),
    )
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must be FP32 [heads]")
    _cache(swa_cache, "swa")
    if swa_slots.ndim != 2 or swa_slots.shape[0] != q.shape[0]:
        raise ValueError("swa_slots must have one row per query")
    _integers(swa_slots, swa_slots.shape, "swa_slots")
    _integers(swa_lens, (q.shape[0],), "swa_lens")
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            raise ValueError("absent global cache requires absent global metadata")
    else:
        _cache(global_cache, "global")
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires slots and lengths")
        if global_slots.ndim != 2 or global_slots.shape[0] != q.shape[0]:
            raise ValueError("global_slots must have one row per query")
        _integers(global_slots, global_slots.shape, "global_slots")
        _integers(global_lens, (q.shape[0],), "global_lens")
    result = _output(out, q.shape, q.dtype, q.device)
    if q.shape[0] == 0:
        return result
    _hopper_tiled_kernel[(q.shape[0], triton.cdiv(q.shape[1], 16))](
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        result,
        *q.stride(),
        *swa_slots.stride(),
        swa_lens.stride(0),
        swa_cache.stride(0),
        global_slots.stride(0) if global_slots is not None else 0,
        global_slots.stride(1) if global_slots is not None else 0,
        global_lens.stride(0) if global_lens is not None else 0,
        global_cache.stride(0) if global_cache is not None else 0,
        attn_sink.stride(0),
        result.stride(0),
        result.stride(1),
        softmax_scale,
        HEADS=q.shape[1],
        SWA_WIDTH=swa_slots.shape[1],
        GLOBAL_WIDTH=global_slots.shape[1] if global_slots is not None else 0,
        SWA_CAPACITY=swa_cache.shape[0] * 64,
        GLOBAL_CAPACITY=global_cache.shape[0] * 64 if global_cache is not None else 0,
        SWA_CR=swa_cache.stride(1),
        SWA_CB=swa_cache.stride(2),
        GLOBAL_CR=global_cache.stride(1) if global_cache is not None else 0,
        GLOBAL_CB=global_cache.stride(2) if global_cache is not None else 0,
        HAS_GLOBAL=global_cache is not None,
        TILE_K=tile_k,
        PV_MODE=pv_mode,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return result
