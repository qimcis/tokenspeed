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

"""Query/sink geometry at the target and DSpark attention boundaries."""

from test.runtime.test_deepseek_v41_engram import _mapping
from test.runtime.test_deepseek_v41_model import (
    _Backend,
    _config,
    _ctx,
    _initialize,
)

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
    V41RowPlan,
)
from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41Attention
from tokenspeed.runtime.models.deepseek_v41_dspark import _WindowAttention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class _QueryBackend(_Backend):
    def __init__(self, positions, requests, prefer_padding):
        super().__init__(positions, requests)
        self.prefer_padding = prefer_padding
        self.preference_calls = 0
        self.sinks = []
        self.sink_pointers = []

    def prefers_padded_query(self, q):
        assert q.shape[1:] == (8, 512)
        self.preference_calls += 1
        return self.prefer_padding

    def forward_v41(self, q, swa, **kwargs):
        self.sinks.append(kwargs["attn_sink"].clone())
        self.sink_pointers.append(kwargs["attn_sink"].data_ptr())
        return super().forward_v41(q, swa, **kwargs)


def _attention(device):
    config = _config()
    # TP8 local query/output geometry without distributed collectives.
    config.num_attention_heads = 8
    config.o_groups = 1
    config.head_dim = 512
    # Keep the merged SWA projection 128-byte aligned, as in production.
    config.q_lora_rank = 64
    config.qk_rope_head_dim = 64
    config.max_position_embeddings = 512
    with torch.device(device):
        attention = DeepseekV41Attention(
            config,
            _mapping(0, 1, 1),
            0,
            None,
            "layers.0.attn",
            aux_stream=None,
        )
        _initialize(attention)
    with torch.no_grad():
        attention.attn_sink.copy_(torch.linspace(-2, 2, 8, device=device))
    return attention


def test_target_padding_preference_delegates_to_kernel_selection(monkeypatch):
    q = torch.empty(1, 8, 512, dtype=torch.bfloat16)
    backend = DeepseekV41AttentionBackend.__new__(DeepseekV41AttentionBackend)
    received = []

    def prefer(values):
        received.append(values)
        return True

    monkeypatch.setattr(dsv41, "prefers_padded_query", prefer)
    assert backend.prefers_padded_query(q) is True
    assert len(received) == 1 and received[0] is q


def test_dspark_window_never_requests_native_query_padding():
    positions = torch.arange(5)
    cache = torch.zeros(1, 64, 512, dtype=torch.bfloat16)
    history = torch.full((1, 128), -1, dtype=torch.int32)
    backend = _WindowAttention(positions, cache, history, 5)
    q = torch.empty(5, 8, 512, dtype=torch.bfloat16)
    assert backend.prefers_padded_query(q) is False


@requires_cuda
@pytest.mark.parametrize(
    "tokens,prefer_padding,mode",
    [
        (1, False, ForwardMode.DECODE),
        (6, False, ForwardMode.DECODE),
        (1, True, ForwardMode.DECODE),
        (6, True, ForwardMode.DECODE),
        (6, False, ForwardMode.EXTEND),
        (6, True, ForwardMode.EXTEND),
        (6, False, ForwardMode.MIXED),
        (6, True, ForwardMode.MIXED),
    ],
)
@torch.inference_mode()
def test_model_prepares_query_and_sink_for_receiving_backend(
    tokens, prefer_padding, mode
):
    torch.manual_seed(410)
    attention = _attention("cuda:0")
    positions = torch.arange(126, 126 + tokens, device="cuda:0")
    if tokens > 1:
        positions[-1] = -1
    original_positions = positions.clone()
    requests = torch.zeros(tokens, device="cuda:0", dtype=torch.int64)
    backend = _QueryBackend(positions, requests, prefer_padding)
    x = torch.randn(tokens, 128, device="cuda:0", dtype=torch.bfloat16)
    actual = attention(positions, x, _ctx(backend, tokens, mode), backend.rows())

    padded = mode.is_decode() and prefer_padding
    expected_heads = 64 if padded else 8
    expected_sink_heads = 64 if prefer_padding else 8
    q = backend.calls[-1][1]
    sink = backend.sinks[-1]
    assert q.shape == (tokens, expected_heads, 512)
    assert sink.shape == (expected_sink_heads,)
    assert backend.preference_calls == 1
    torch.testing.assert_close(sink[:8], attention.attn_sink, rtol=0, atol=0)
    torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)
    if padded:
        assert torch.count_nonzero(q[:, 8:]).item() == 0
    if prefer_padding:
        assert torch.isneginf(sink[8:]).all().item()
        expected_sink_pointer = attention._padded_attn_sink.data_ptr()
    else:
        assert attention._padded_attn_sink is None
        expected_sink_pointer = attention.attn_sink.data_ptr()
    assert actual.shape == (tokens, 128)
    assert torch.isfinite(actual).all().item()

    attention(positions, x, _ctx(backend, tokens, mode), backend.rows())
    assert backend.preference_calls == 2
    assert backend.sink_pointers == [expected_sink_pointer, expected_sink_pointer]
    torch.testing.assert_close(backend.sinks[1], sink, rtol=0, atol=0)


@requires_cuda
@torch.inference_mode()
def test_dspark_real_query_heads_match_padded_reference_and_graph(monkeypatch):
    torch.manual_seed(411)
    batch, block = 2, 5
    attention = _attention("cuda:0")
    positions = torch.arange(1, 1 + batch * block, device="cuda:0")
    cache = torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda:0")
    history = torch.arange(128, device="cuda:0", dtype=torch.int32).repeat(batch, 1)
    history[0, :123] = -1
    history[1] = -1
    backend = _WindowAttention(positions, cache, history, block)
    rows = V41RowPlan(backend.meta, backend.meta, None)
    ctx = _ctx(backend, batch * block, ForwardMode.DECODE)
    ctx.bs = batch
    x = torch.randn(batch * block, 128, dtype=torch.bfloat16, device="cuda:0")
    original_forward = backend.forward_v41
    expected_heads = 8

    def checked_forward(q, swa, **kwargs):
        assert q.shape == (batch * block, expected_heads, 512)
        assert kwargs["attn_sink"].shape == (expected_heads,)
        return original_forward(q, swa, **kwargs)

    monkeypatch.setattr(backend, "forward_v41", checked_forward)

    def forward():
        return attention(positions, x, ctx, rows)

    actual = forward()
    assert attention._padded_attn_sink is None
    expected_heads = 64
    with monkeypatch.context() as padded_patch:
        padded_patch.setattr(backend, "prefers_padded_query", lambda q: True)
        old_padded = forward()
    torch.testing.assert_close(actual, old_padded, rtol=0.01, atol=0.01)
    expected_heads = 8

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward()
    torch.cuda.current_stream().wait_stream(stream)
    for start in (5, 127, 128):
        positions.copy_(torch.arange(start, start + batch * block, device="cuda:0"))
        history[0].copy_(torch.arange(128, device="cuda:0", dtype=torch.int32))
        history[0, : start % 128] = -1
        x.normal_()
        expected = forward()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        assert torch.isfinite(captured).all().item()
    graph.reset()
