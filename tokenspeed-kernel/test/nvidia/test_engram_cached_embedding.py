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

"""Opt-in C2 GPU qualification; no GPU validation is claimed by CPU tests.

TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS=1 pytest -q \
    tokenspeed-kernel/test/nvidia/test_engram_cached_embedding.py
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS") != "1":
    pytest.skip(
        "explicit experimental GPU qualification opt-in required",
        allow_module_level=True,
    )

import torch
from tokenspeed_kernel.ops.embedding import engram_cached_embedding
from tokenspeed_kernel.ops.embedding.triton_cached import (
    _engram_cached_embedding_kernel,
)
from utils import assert_no_triton_compile, is_nvidia

pytestmark = pytest.mark.skipif(not is_nvidia(), reason="requires supported NVIDIA GPU")


def _tables(width, hot_rows):
    # At width256 every row contains every FP8 byte, every scale byte appears
    # as a row. This includes both signed zeros, FP8 NaNs and E8M0 0/255.
    codes = (
        torch.arange(width).remainder(256).to(torch.uint8).repeat(256, 1).pin_memory()
    )
    scales = (
        torch.arange(256, dtype=torch.uint8)[:, None]
        .repeat(1, width // 32)
        .pin_memory()
    )
    local = torch.tensor(hot_rows, dtype=torch.int64)
    hot_ids = local.cuda()
    hot_codes = codes[local].cuda()
    hot_scales = scales[local].cuda()
    return codes, scales, hot_ids, hot_codes, hot_scales


def _reference(codes, scales, indices, start, end):
    local = (indices >= start) & (indices < end)
    rows = (indices - start).clamp(0, codes.shape[0] - 1).long()
    c = codes.cuda()[rows].view(torch.float8_e4m3fn).float()
    s = scales.cuda()[rows].view(torch.float8_e8m0fnu).float()
    result = (c.unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2).bfloat16()
    return result.masked_fill(~local.unsqueeze(-1), 0)


def _exact(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    finite = ~expected.isnan()
    # Compare signed zero, subnormals and infinities bit-for-bit as well.
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("width", [32, 96, 256])
@pytest.mark.parametrize("hot_rows", [[], list(range(0, 256, 2)), list(range(256))])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_all_code_and_scale_bytes(width, hot_rows, index_dtype):
    codes, scales, hi, hc, hs = _tables(width, hot_rows)
    ids = torch.tensor(
        [-1, 16, *range(17, 273), 272, 17, 273, 999], device="cuda", dtype=index_dtype
    ).reshape(2, 131)
    out = torch.empty((*ids.shape, width), device="cuda", dtype=torch.bfloat16)
    returned = engram_cached_embedding(codes, scales, ids, hi, hc, hs, 17, 273, out=out)
    assert returned is out
    _exact(out, _reference(codes, scales, ids, 17, 273))


@pytest.mark.parametrize("hot_rows", [[], [0, 2, 17, 255]])
def test_capture_replay_with_changed_hits_misses_and_shard_masks(hot_rows):
    codes, scales, hi, hc, hs = _tables(256, hot_rows)
    ids = torch.tensor([[31, 32, 33], [48, 286, -1]], device="cuda")
    out = torch.empty((*ids.shape, 256), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            engram_cached_embedding(codes, scales, ids, hi, hc, hs, 31, 287, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    ptr = out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        engram_cached_embedding(codes, scales, ids, hi, hc, hs, 31, 287, out=out)
    for replacement in ([[31, 31, 32], [286, 287, -1]], [[33, 48, 31], [32, 286, 999]]):
        ids.copy_(torch.tensor(replacement, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        assert out.data_ptr() == ptr
        _exact(out, _reference(codes, scales, ids, 31, 287))


def test_runtime_counts_do_not_compile_per_batch():
    codes, scales, hi, hc, hs = _tables(256, list(range(256)))

    def run(n, h, start):
        ids = torch.full((n,), start, device="cuda", dtype=torch.int64)
        out = torch.empty((n, 256), device="cuda", dtype=torch.bfloat16)
        engram_cached_embedding(
            codes, scales, ids, hi[:h], hc[:h], hs[:h], start, start + 256, out=out
        )
        return out

    run(16, 16, 16)
    with assert_no_triton_compile(_engram_cached_embedding_kernel):
        for n, h, start in [(1, 0, 0), (3, 1, 7), (17, 17, 31), (32, 256, 16)]:
            run(n, h, start)


def test_stream_ordering_and_independent_concurrent_outputs():
    codes, scales, hi, hc, hs = _tables(256, [0, 17, 255])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream, row in zip(streams, [0, 1], strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ids = torch.full((19,), row, device="cuda", dtype=torch.int64)
            out = torch.empty((19, 256), device="cuda", dtype=torch.bfloat16)
            engram_cached_embedding(codes, scales, ids, hi, hc, hs, 0, 256, out=out)
            outputs.append((ids, out))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for ids, out in outputs:
        _exact(out, _reference(codes, scales, ids, 0, 256))


def test_empty_arbitrary_shape():
    codes, scales, hi, hc, hs = _tables(256, [])
    ids = torch.empty((2, 0, 3), device="cuda", dtype=torch.int64)
    out = torch.empty((2, 0, 3, 256), device="cuda", dtype=torch.bfloat16)
    assert (
        engram_cached_embedding(codes, scales, ids, hi, hc, hs, 0, 256, out=out) is out
    )


def test_invalidated_hot_ids_use_updated_canonical_bytes():
    codes, scales, hi, hc, hs = _tables(256, [0, 17, 255])
    ids = torch.tensor([0, 17, 255, 0, 17], device="cuda", dtype=torch.int64)
    out = torch.empty((5, 256), device="cuda", dtype=torch.bfloat16)
    engram_cached_embedding(codes, scales, ids, hi, hc, hs, 0, 256, out=out)
    torch.cuda.synchronize()
    # Quiesced refresh invalidates the fixed-address keys before new host bytes.
    hi.fill_(-1)
    codes.fill_(56)  # E4M3 +1; old HBM bytes intentionally stay stale.
    scales.fill_(127)
    engram_cached_embedding(codes, scales, ids, hi, hc, hs, 0, 256, out=out)
    _exact(out, torch.ones_like(out))
