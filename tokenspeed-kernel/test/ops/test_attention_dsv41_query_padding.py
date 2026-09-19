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

"""Backend padding selection and real-head query parity."""

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import flash_mla
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import SelectedKernel, kernel_override

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.mark.parametrize(
    "platform_name", ["h100_platform", "b200_platform", "b300_platform"]
)
@pytest.mark.parametrize("available", [False, True])
def test_padding_preference_matches_actual_selected_attention(
    monkeypatch, request, platform_name, available
):
    platform = request.getfixturevalue(platform_name)
    monkeypatch.setattr(Platform, "_instance", platform)
    monkeypatch.setattr(flash_mla, "is_flash_mla_v41_available", lambda: available)
    q = torch.empty(1, 8, 512, dtype=torch.bfloat16)
    selected = []

    def record_call(self, *args, **kwargs):
        selected.append(self.name)
        return args[0]

    monkeypatch.setattr(SelectedKernel, "__call__", record_call)
    prefer_padding = dsv41.prefers_padded_query(q)
    result = dsv41.selected_attention(
        q,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        512**-0.5,
        None,
        256,
        object() if available else None,
        None,
        None,
    )
    assert result is q
    native_expected = available and platform.arch_version.major == 10
    assert prefer_padding is native_expected
    assert selected == [
        (
            "flashmla_dsv41_selected_attention"
            if native_expected
            else "triton_dsv41_selected_attention"
        )
    ]


def test_padding_preference_honors_portable_override(monkeypatch, b200_platform):
    monkeypatch.setattr(Platform, "_instance", b200_platform)
    monkeypatch.setattr(flash_mla, "is_flash_mla_v41_available", lambda: True)
    q = torch.empty(1, 8, 512, dtype=torch.bfloat16)
    assert dsv41.prefers_padded_query(q) is True
    with kernel_override(
        "attention", "dsv41_selected_attention", "triton_dsv41_selected_attention"
    ):
        assert dsv41.prefers_padded_query(q) is False
    assert dsv41.prefers_padded_query(q) is True


def _rope_table(device):
    angles = torch.arange(512, device=device, dtype=torch.float32)[:, None]
    angles = angles * torch.linspace(0.001, 0.1, 32, device=device)[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


@requires_cuda
@pytest.mark.parametrize(
    "tokens,heads,strided",
    [
        (0, 8, False),
        (1, 8, False),
        (5, 8, True),
        (6, 8, False),
        (12, 8, True),
        (1, 16, False),
    ],
)
@torch.inference_mode()
def test_owned_query_rope_matches_fused_padding_exactly(tokens, heads, strided):
    torch.manual_seed(412)
    storage = torch.randn(
        tokens * 2, heads * 2, 512, dtype=torch.bfloat16, device="cuda:0"
    )
    original = storage.clone()
    values = storage[::2, ::2] if strided else storage[:tokens, :heads].contiguous()
    source = original[::2, ::2] if strided else original[:tokens, :heads].contiguous()
    positions = torch.arange(tokens, device="cuda:0", dtype=torch.int64) * 7
    if tokens > 1:
        positions[-1] = -1
    before_positions = positions.clone()
    table = _rope_table("cuda:0")
    expected = dsv41.rope_pad_query(source, positions, table, None)
    pointer = values.data_ptr()
    result = dsv41.rope_inplace(values, positions, table, None)
    assert result.shape == (tokens, heads, 512)
    assert result.data_ptr() == pointer
    torch.testing.assert_close(result, expected[:, :heads], rtol=0, atol=0)
    torch.testing.assert_close(result[..., :448], source[..., :448], rtol=0, atol=0)
    torch.testing.assert_close(positions, before_positions, rtol=0, atol=0)
    assert torch.count_nonzero(expected[:, heads:]).item() == 0


def _cache(rows, cache_format, device):
    width = {"swa": 528, "global": 288}[cache_format]
    cache = torch.zeros((rows // 64, 64, width), dtype=torch.uint8, device=device)
    values = torch.randn(rows, 512, dtype=torch.bfloat16, device=device) * 0.3
    dsv41.cache_scatter(values, cache, torch.arange(rows, device=device), cache_format)
    return cache


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("with_global", [False, True])
@torch.inference_mode()
def test_unpadded_portable_attention_matches_padded_heads_and_graph(
    tokens, with_global
):
    torch.manual_seed(413)
    q = torch.randn(tokens, 8, 512, dtype=torch.bfloat16, device="cuda:0") * 0.3
    positions = torch.arange(tokens, dtype=torch.int64, device="cuda:0")
    table = _rope_table("cuda:0")
    sink = torch.linspace(-2, 2, 8, device="cuda:0")
    padded_sink = F.pad(sink, (0, 56), value=-float("inf"))
    swa = _cache(128, "swa", "cuda:0")
    global_cache = _cache(512, "global", "cuda:0") if with_global else None
    swa_slots = torch.arange(128, dtype=torch.int32, device="cuda:0").repeat(tokens, 1)
    global_slots = (
        torch.arange(512, dtype=torch.int32, device="cuda:0").repeat(tokens, 1)
        if with_global
        else None
    )
    swa_lens = torch.full((tokens,), 128, dtype=torch.int32, device="cuda:0")
    global_lens = torch.full_like(swa_lens, 512) if with_global else None
    swa_slots[:, 17] = -1
    if with_global:
        global_slots[:, 23] = -1

    def forward(padded):
        rotated = (
            dsv41.rope_pad_query(q, positions, table, None)
            if padded
            else dsv41.rope_inplace(q.clone(), positions, table, None)
        )
        return dsv41.selected_attention(
            rotated,
            swa,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            padded_sink if padded else sink,
            512**-0.5,
            None,
            256,
            None,
            None,
            None,
        )[:, :8]

    torch.testing.assert_close(forward(False), forward(True), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward(False)
    torch.cuda.current_stream().wait_stream(stream)
    for step, length in enumerate((0, 1, 128)):
        q.normal_(0, 0.3)
        positions.add_(7)
        positions[-1] = -1
        swa_lens.fill_(length)
        swa_slots.copy_(swa_slots.roll(1, dims=1))
        if with_global:
            global_lens.fill_((0, 1, 512)[step])
            global_slots.copy_(global_slots.roll(7, dims=1))
        eager = forward(False)
        padded = forward(True)
        graph.replay()
        assert captured.shape == (tokens, 8, 512)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(captured, padded, rtol=0, atol=0)
        assert torch.isfinite(captured).all().item()
        if length == 0:
            assert torch.count_nonzero(captured).item() == 0
    graph.reset()


class _NativeDecodeSpy:
    def __init__(self):
        self.calls = []

    def flash_mla_with_kvcache(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs["q"]
        values = torch.arange(q.shape[2], dtype=q.dtype, device=q.device)
        result = values[None, None, :, None].expand_as(q).clone()
        return result, None


@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("provide_out", [False, True])
def test_native_adapter_pads_and_trims_query_heads(monkeypatch, heads, provide_out):
    api = _NativeDecodeSpy()
    monkeypatch.setattr(flash_mla, "flash_mla_api", lambda: api)
    q = torch.ones(2, heads, 512, dtype=torch.bfloat16)
    if heads > 8:
        q[:, 8:] = 0
    live_sink = torch.linspace(-2, 2, 8)
    sink = F.pad(live_sink, (0, heads - 8), value=-float("inf"))
    cache = torch.zeros(1, 64, 528, dtype=torch.uint8)
    slots = torch.zeros(2, 64, dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    out = torch.empty_like(q) if provide_out else None
    schedule = object()
    actual = flash_mla.selected_attention(
        q,
        cache,
        slots,
        lengths,
        None,
        None,
        None,
        sink,
        512**-0.5,
        out,
        256,
        schedule,
        None,
        None,
    )
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call["q"].shape == (2, 1, 64, 512)
    assert call["tile_scheduler_metadata"] is schedule
    torch.testing.assert_close(call["q"][:, 0, :8], q[:, :8], rtol=0, atol=0)
    assert torch.count_nonzero(call["q"][:, :, 8:]).item() == 0
    torch.testing.assert_close(call["attn_sink"][:8], live_sink, rtol=0, atol=0)
    assert torch.isneginf(call["attn_sink"][8:]).all().item()
    expected = torch.arange(heads, dtype=q.dtype)[None, :, None].expand_as(q)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if provide_out:
        assert actual is out
    if heads == 64:
        assert call["q"].data_ptr() == q.data_ptr()
        assert call["attn_sink"].data_ptr() == sink.data_ptr()
