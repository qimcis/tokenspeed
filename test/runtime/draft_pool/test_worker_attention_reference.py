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

"""Dense CPU proof of the worker adapter's native-block visibility contract.

This exercises real ring metadata and adapter kwargs with a dense kernel oracle.
It does not qualify CUDA kernel execution or checkpoint numerics.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.draft_pool.worker_cache import (
    WorkerAttentionBackend,
    WorkerCacheGeometry,
    WorkerKVPool,
)


def _absolute_kv(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    position = positions.float()[:, None, None]
    head = torch.arange(2).float()[None, :, None]
    channel = torch.arange(4).float()[None, None, :]
    return (
        torch.sin(position * 0.013 + head * 0.4 + channel * 0.2).bfloat16(),
        torch.cos(position * 0.017 + head * 0.3 + channel * 0.5).bfloat16(),
    )


def _dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    expanded_k = k.repeat_interleave(2, dim=1).float()
    expanded_v = v.repeat_interleave(2, dim=1).float()
    scores = torch.einsum("hd,thd->ht", q.float(), expanded_k) * 0.5
    return torch.einsum("ht,thd->hd", scores.softmax(dim=-1), expanded_v).to(q.dtype)


@pytest.mark.parametrize("endpoints", ((0, 3), (2047, 2048), (4097, 1000003)))
def test_ring_adapter_matches_full_history_noncausal_window(endpoints, monkeypatch):
    geometry = WorkerCacheGeometry(2, 2048, 8)
    pool = WorkerKVPool(geometry, 1, 2, 4, "cpu")
    backend = WorkerAttentionBackend(geometry, "cpu")
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        qk_head_dim=4,
        sliding_window_size=2047,
        scaling=0.5,
    )
    for slot, endpoint in enumerate(endpoints):
        start = max(0, endpoint - geometry.history_tokens)
        positions = torch.arange(start, endpoint + geometry.native_block_tokens)
        keys, values = _absolute_kv(positions)
        locations = torch.tensor(geometry.page_row(slot, endpoint))
        pool.set_kv_buffer(layer, locations, keys, values, None, None)
    backend.prepare([0, 1], list(endpoints))
    generator = torch.Generator().manual_seed(81)
    q = torch.randn(16, 4, 4, generator=generator, dtype=torch.bfloat16)
    calls = []

    def dense_kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["max_seqlen_q"] == 1
        assert kwargs["window_left"] == 2047
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["solution"] == "triton"
        assert kwargs["return_lse"] is False
        assert kwargs["q_scale"] is kwargs["k_scale"] is kwargs["v_scale"] is None
        output = []
        for row, query in enumerate(kwargs["q"]):
            length = int(kwargs["cache_seqlens"][row])
            # The portable decode contract masks from the block-end length.
            visible_start = max(0, length - kwargs["window_left"] - 1)
            pages = kwargs["page_table"][row, visible_start:length].long()
            output.append(
                _dense_gqa(
                    query, kwargs["k_cache"][pages, 0], kwargs["v_cache"][pages, 0]
                )
            )
        return torch.stack(output)

    module = ModuleType("tokenspeed_kernel.ops.attention.mha")
    module.mha_decode_with_kvcache = dense_kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    actual = backend.forward(q.flatten(1), None, None, layer, pool, "decode", 2, False)
    expected = []
    for index, endpoint in enumerate(endpoints):
        # Full-history local block decode exposes all eight draft keys even to
        # its first query, then applies the window at that shared block end.
        visible_start = max(0, endpoint + 8 - 2048)
        keys, values = _absolute_kv(torch.arange(visible_start, endpoint + 8))
        expected.extend(
            _dense_gqa(query, keys, values) for query in q[index * 8 : (index + 1) * 8]
        )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    assert len(calls) == 1
