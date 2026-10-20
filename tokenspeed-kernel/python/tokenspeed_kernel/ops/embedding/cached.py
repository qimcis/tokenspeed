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

"""Validated allocation-free interface for immutable Engram hot rows."""

from __future__ import annotations

import torch


def engram_cached_embedding(
    weight: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    hot_ids: torch.Tensor,
    hot_codes: torch.Tensor,
    hot_scales: torch.Tensor,
    row_start: int,
    row_end: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather exact MXFP8 rows from an immutable HBM cache or canonical table.

    Args:
        weight: Contiguous E4M3 or uint8 codes ``[local_capacity, D]``. Host
            storage must already be pinned or registered for GPU access.
        scales: Contiguous uint8 E8M0 bytes ``[local_capacity, D/32]``.
        indices: Contiguous GPU int32/int64 global row IDs, arbitrary shape.
        hot_ids: Contiguous GPU int64 sorted unique LOCAL row offsets ``[H]``.
            The caller validates their values once when building the hot set.
            An all-minus-one array invalidates the cache without moving storage.
        hot_codes: Contiguous GPU E4M3 or uint8 codes ``[H,D]`` copied exactly
            from the canonical table at ``hot_ids``.
        hot_scales: Contiguous GPU uint8 scale bytes ``[H,D/32]``.
        row_start: First global row owned by this shard.
        row_end: Exclusive global end, excluding capacity padding.
        out: Preallocated contiguous GPU BF16 ``[*indices.shape,D]``. Every
            element is written; IDs outside the shard produce positive zero.

    Returns:
        ``out`` itself. No allocation, host transfer, or synchronization occurs.
        D is a fixed model dimension divisible by 32 (normally 256).

    All GPU tensors must share a device. The cache, table and output storage
    must remain alive and unchanged in address throughout captured graph use.
    The caller must quiesce readers before changing canonical/cache bytes;
    weight refresh and hot-set replacement are not forward-path operations.
    """
    code_dtypes = (torch.uint8, torch.float8_e4m3fn)
    if (
        weight.ndim != 2
        or weight.shape[1] == 0
        or weight.shape[1] % 32
        or weight.dtype not in code_dtypes
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Engram weight must be contiguous byte/E4M3 [rows,D], D % 32 == 0"
        )
    width = weight.shape[1]
    if (
        scales.shape != (weight.shape[0], width // 32)
        or scales.dtype != torch.uint8
        or not scales.is_contiguous()
        or scales.device != weight.device
    ):
        raise ValueError("Engram scales must be colocated contiguous uint8 [rows,D/32]")
    if not 0 <= row_start <= row_end <= row_start + weight.shape[0]:
        raise ValueError("Engram shard range exceeds canonical table capacity")
    if hot_ids.ndim != 1 or hot_ids.dtype != torch.int64 or not hot_ids.is_contiguous():
        raise ValueError("Engram hot IDs must be contiguous int64 [H] local offsets")
    if (
        hot_codes.shape != (hot_ids.numel(), width)
        or hot_codes.dtype not in code_dtypes
        or not hot_codes.is_contiguous()
        or hot_scales.shape != (hot_ids.numel(), width // 32)
        or hot_scales.dtype != torch.uint8
        or not hot_scales.is_contiguous()
    ):
        raise ValueError("Engram hot codes/scales must be contiguous [H,D] / [H,D/32]")
    if indices.dtype not in (torch.int32, torch.int64) or not indices.is_contiguous():
        raise ValueError("Engram indices must be contiguous int32/int64")
    if (
        out.shape != (*indices.shape, width)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError("Engram output must be contiguous BF16 [*indices.shape,D]")
    if (
        not indices.is_cuda
        or any(
            t.device != indices.device for t in (hot_ids, hot_codes, hot_scales, out)
        )
        or weight.device.type not in ("cpu", indices.device.type)
        or (weight.device.type != "cpu" and weight.device != indices.device)
    ):
        raise ValueError("Engram indices, hot set and output require one GPU device")
    if indices.numel() == 0:
        return out

    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    kernel = select_kernel(
        "embedding",
        "engram_cached_embedding",
        format_signature(weight=dense_tensor_format(torch.uint8)),
        traits=None,
        override=None,
        solution=None,
    )
    return kernel(
        weight,
        scales,
        indices,
        hot_ids,
        hot_codes,
        hot_scales,
        row_start,
        row_end,
        out=out,
    )
