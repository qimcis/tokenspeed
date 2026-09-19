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

"""Unregistered direct-reader candidates; independent softmax and graph checks."""

import inspect

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41._hopper_tiled import (
    _load_cache_tile,
    hopper_tiled_selected_attention,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        pytest.skip("requires an SM90 CUDA device")
    return torch.device("cuda:0")


def _cache(rows, fmt, strided):
    row_bytes = 528 if fmt == "swa" else 288
    pages = (rows.shape[0] + 63) // 64
    if strided:
        storage = torch.zeros(
            (pages + 1, 128, row_bytes * 2 + 14), dtype=torch.uint8, device=rows.device
        )
        cache = storage[1:, ::2, 3 : 3 + row_bytes * 2 : 2]
    else:
        cache = torch.zeros(
            (pages, 64, row_bytes), dtype=torch.uint8, device=rows.device
        )
    dsv41.cache_scatter(
        rows, cache, torch.arange(rows.shape[0], device=rows.device), fmt
    )
    return cache


def _inputs(device, heads, global_enabled, strided):
    torch.manual_seed(4119)
    tokens = 6
    q = torch.randn((tokens * 2, heads * 2, 1024), dtype=torch.bfloat16, device=device)[
        ::2, ::2, ::2
    ]
    swa = _cache(torch.randn(128, 512, device=device), "swa", strided)
    slots = torch.randint(0, 128, (tokens * 2, 259), device=device, dtype=torch.int64)[
        ::2, :256:2
    ]
    slots[1, :32] = -1
    slots[2, ::7] = 128  # exact capacity is invalid
    slots[3, ::5] = 10**12  # do not narrow a slot before its capacity check
    slots[4, ::3] = -7
    lens = torch.tensor(
        [128, 0, 128, 0, 37, 0, 0, 0, -3, 0, 200, 0], device=device, dtype=torch.int64
    )[::2]
    sink = torch.linspace(-2, 2, heads * 2, device=device)[::2]
    sink[0], sink[1] = -torch.inf, torch.inf
    global_cache = global_slots = global_lens = None
    if global_enabled:
        global_cache = _cache(torch.randn(192, 512, device=device), "global", strided)
        global_slots = torch.randint(
            0, 192, (tokens * 2, 1027), device=device, dtype=torch.int32
        )[::2, :1024:2]
        global_slots[1, :64] = -1
        global_slots[2, ::11] = 192
        global_slots[4].fill_(-1)
        global_lens = torch.tensor(
            [512, 0, 511, 0, 9, 0, 0, 0, 512, 0, 2**40, 0],
            device=device,
            dtype=torch.int64,
        )[::2]
    return (q, swa, slots, lens, global_cache, global_slots, global_lens, sink)


def _oracle(args):
    q, swa, slots, lens, global_cache, global_slots, global_lens, sink = args
    chunks, masks = [], []
    for cache, locations, lengths, fmt in (
        (swa, slots, lens, "swa"),
        (global_cache, global_slots, global_lens, "global"),
    ):
        if cache is None:
            continue
        valid = (
            (
                torch.arange(locations.shape[1], device=q.device)[None, :]
                < lengths[:, None]
            )
            & (locations >= 0)
            & (locations < cache.shape[0] * 64)
        )
        chunks.append(
            dsv41.cache_gather(cache, locations.masked_fill(~valid, -1), fmt, None)
            .cpu()
            .double()
        )
        masks.append(valid.cpu())
    values = torch.cat(chunks, dim=1)
    valid = torch.cat(masks, dim=1)
    logits = torch.einsum("thd,tkd->thk", q.cpu().double(), values) * (512**-0.5)
    logits.masked_fill_(~valid[:, None, :], -torch.inf)
    sink = sink.cpu().double()
    maximum = torch.maximum(logits.amax(-1), sink[None, :])
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    probabilities = (logits - safe_maximum[:, :, None]).exp()
    denominator = probabilities.sum(-1) + (sink[None, :] - safe_maximum).exp()
    numerator = torch.einsum("thk,tkd->thd", probabilities, values)
    result = (
        numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)[:, :, None]
    )
    result = torch.where(torch.isposinf(sink)[None, :, None], 0.0, result)
    return result.to(torch.bfloat16).to(q.device)


def _run(args, out, pv_mode, tile_k, num_warps):
    return hopper_tiled_selected_attention(
        *args,
        512**-0.5,
        out,
        256,
        object(),
        None,
        None,
        pv_mode,
        tile_k,
        num_warps,
    )


def _assert_close(actual, expected, pv_mode):
    assert torch.isfinite(actual).all()
    # Frozen before GPU evaluation: the existing V4.1 attention tolerance.
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.004)
    relative_l2 = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-20)
    assert relative_l2 < (0.006 if pv_mode == "bf16" else 0.003)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "pv_mode,tile_k,num_warps",
    [
        ("bf16", 32, 4),
        ("bf16", 64, 8),
        ("tf32x3", 32, 4),
        ("tf32x3", 32, 8),
    ],
)
def test_tiled_independent_softmax_and_strided_planar_caches(
    device, heads, pv_mode, tile_k, num_warps
):
    args = _inputs(device, heads, True, True)
    before = (args[1].clone(), args[4].clone())
    expected = _oracle(args)
    out = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    actual = _run(args, out, pv_mode, tile_k, num_warps)
    assert actual is out
    _assert_close(actual, expected, pv_mode)
    torch.testing.assert_close(args[1], before[0], rtol=0, atol=0)
    torch.testing.assert_close(args[4], before[1], rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 1]) == 0  # +inf sink
    assert torch.count_nonzero(actual[3]) == 0  # both segments have zero length


@pytest.mark.parametrize("heads", [32, 64])
def test_tiled_multiple_head_tiles(device, heads):
    args = _inputs(device, heads, True, False)
    expected = _oracle(args)
    actual = _run(args, None, "bf16", 32, 4)
    _assert_close(actual, expected, "bf16")


@pytest.mark.parametrize("pv_mode", ["bf16", "tf32x3"])
def test_tiled_swa_only_graph_lengths_slots_and_holes(device, pv_mode):
    args = list(_inputs(device, 8, False, False))
    out = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(args, out, pv_mode, 32, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run(args, out, pv_mode, 32, 4)
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([0, 1, 17, 63, 127, 128], [128, 37, -1, 200, 9, 0]):
        args[0].normal_()
        args[3].copy_(torch.tensor(lengths, dtype=args[3].dtype, device=device))
        args[2][1, 0] = 127
        args[2][3, :64] = -1
        expected = _oracle(args)
        graph.replay()
        _assert_close(out, expected, pv_mode)


@pytest.mark.parametrize("pv_mode", ["bf16", "tf32x3"])
def test_tiled_combined_graph_replays_live_global_metadata_sink_and_bytes(
    device, pv_mode
):
    args = list(_inputs(device, 8, True, True))
    out = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(args, out, pv_mode, 32, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run(args, out, pv_mode, 32, 4)
    torch.cuda.current_stream().wait_stream(stream)

    original_swa, original_global = args[1].clone(), args[4].clone()
    for iteration, (swa_lengths, global_lengths) in enumerate(
        (
            ([128, 17, 0, 64, 127, 1], [0, 1, 31, 32, 63, 512]),
            ([1, 128, 17, 0, 63, 128], [512, 65, 64, 33, 32, -1]),
        )
    ):
        args[0].normal_()
        args[3].copy_(torch.tensor(swa_lengths, device=device, dtype=args[3].dtype))
        args[6].copy_(torch.tensor(global_lengths, device=device, dtype=args[6].dtype))
        args[2].copy_(
            torch.randint(0, 128, args[2].shape, device=device, dtype=args[2].dtype)
        )
        args[5].copy_(
            torch.randint(0, 192, args[5].shape, device=device, dtype=args[5].dtype)
        )
        args[2][0, ::7] = -1
        args[5][0, ::11] = -1
        args[5][1, 0] = 192  # updated length one, but its sole slot is invalid
        args[5][2, ::5] = -7
        args[5][3, ::3] = 2**30
        args[5][4, :32] = -1  # leading empty tile before later selected keys
        args[7].copy_(
            torch.linspace(-1.5 + iteration, 1.5 + iteration, 8, device=device)
        )
        args[7][iteration * 2] = -torch.inf
        args[7][iteration * 2 + 1] = torch.inf

        # Repack into the same page-planar views, changing both values and
        # scale bytes while the captured graph retains the same pointers.
        for cache, fmt, rows in ((args[1], "swa", 128), (args[4], "global", 192)):
            replacement = torch.randn(rows, 512, device=device) * (iteration + 0.5)
            dsv41.cache_scatter(
                replacement, cache, torch.arange(rows, device=device), fmt
            )
        assert torch.any(args[1] != original_swa)
        assert torch.any(args[4] != original_global)

        expected = _oracle(args)
        graph.replay()
        _assert_close(out, expected, pv_mode)
        assert torch.count_nonzero(out[:, iteration * 2 + 1]) == 0


@triton.jit
def _decode_probe(
    CACHE,
    SLOTS,
    OUT,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    CAP: tl.constexpr,
    IS_SWA: tl.constexpr,
):
    columns = tl.arange(0, 32)
    decoded, valid = _load_cache_tile(
        CACHE,
        SLOTS,
        0,
        columns,
        32,
        32,
        1,
        CP,
        CAP,
        32,
        CR,
        CB,
        IS_SWA,
    )
    dims = tl.arange(0, 512)
    tl.store(OUT + columns[None, :] * 512 + dims[:, None], decoded)


@pytest.mark.parametrize("fmt", ["swa", "global"])
def test_direct_decode_matches_existing_bf16_gather_bits(device, fmt):
    rows = torch.randn(128, 512, device=device)
    cache = _cache(rows, fmt, True)
    row_bytes, value_bytes, groups = (528, 512, 16) if fmt == "swa" else (288, 256, 32)
    # Build a raw page-planar byte oracle, including E8M0 byte zero and signed
    # zero codes. NaN FP8 encodings are excluded; these are valid packed values.
    raw = torch.arange(64 * value_bytes, device=device, dtype=torch.int64)
    if fmt == "swa":
        raw = ((raw % 127) | ((raw // 127 % 2) << 7)).to(torch.uint8)
        scales = torch.tensor([0, 1, 120, 127, 130], dtype=torch.uint8, device=device)
    else:
        raw = (raw % 256).to(torch.uint8)
        scales = torch.tensor([0, 1, 32, 56, 126], dtype=torch.uint8, device=device)
    scale_bytes = scales[torch.arange(64 * groups, device=device) % scales.numel()]
    cache[0].copy_(torch.cat((raw, scale_bytes)).reshape(64, row_bytes))
    slots = torch.arange(32, device=device, dtype=torch.int64)
    slots[1], slots[3], slots[5] = -1, 128, 10**12
    expected = dsv41.cache_gather(cache, slots, fmt, None)
    actual = torch.empty((32, 512), device=device, dtype=torch.bfloat16)
    _decode_probe[(1,)](
        cache,
        slots,
        actual,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        128,
        fmt == "swa",
        num_warps=4,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


@pytest.mark.parametrize("num_warps", [4, 8])
def test_tiled_rejects_tf32x3_k64_before_launch(num_warps):
    with pytest.raises(ValueError, match="tf32x3 PV supports tile_k=32 only"):
        hopper_tiled_selected_attention(*([None] * 14), "tf32x3", 64, num_warps)


def test_tiled_arguments_are_explicit():
    assert all(
        p.default is inspect.Parameter.empty
        for p in inspect.signature(hopper_tiled_selected_attention).parameters.values()
    )
