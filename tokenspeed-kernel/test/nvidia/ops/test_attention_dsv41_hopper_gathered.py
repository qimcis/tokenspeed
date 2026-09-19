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

"""CPU ABI/masking tests for the isolated Hopper native-gather candidate.

The native function and packed gather are replaced with deterministic CPU
references. These tests verify wrapper semantics, not CUDA numerical accuracy
or graph capture; those require the separate full-chain GPU A/B.
"""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import _hopper_gathered as candidate


@pytest.fixture
def native_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(candidate, "_same_device", lambda q, tensors: None)

    def gather(cache, slots, cache_format, out):
        assert out is None
        offset = 1 if cache_format == "swa" else 100
        rows = (slots + offset).float().masked_fill(slots < 0, 0)
        return rows.unsqueeze(-1).expand(*slots.shape, 512).to(torch.bfloat16)

    def native(*, q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
        # Exact pinned wheel API: deliberately no out keyword or **kwargs.
        assert q.shape[1] in (64, 128)
        assert indices.shape[1] == 1 and indices.shape[2] % 128 == 0
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        assert kv.shape[1:] == (1, 512) and kv.dtype == torch.bfloat16
        assert topk_length is None and d_v == 512
        calls.append((q.clone(), kv.clone(), indices.clone(), attn_sink.clone()))
        ids = indices[:, 0]
        valid = ids >= 0
        selected = kv[ids.clamp_min(0), 0].float()
        logits = torch.einsum("thd,twd->thw", q.float(), selected) * sm_scale
        logits.masked_fill_(~valid[:, None], -torch.inf)
        logits = torch.cat(
            (logits, attn_sink[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
        )
        probs = torch.softmax(logits, dim=-1)[..., :-1].nan_to_num()
        output = torch.einsum("thw,twd->thd", probs, selected).to(torch.bfloat16)
        return output, None, None

    monkeypatch.setattr(candidate, "cache_gather", gather)
    monkeypatch.setattr(
        candidate, "flash_mla_api", lambda: SimpleNamespace(flash_mla_sparse_fwd=native)
    )
    return calls


def _decode_inputs():
    q = torch.zeros((3, 8, 512), dtype=torch.bfloat16)
    swa = torch.empty((1, 64, 528), dtype=torch.uint8)
    global_cache = torch.empty((1, 64, 288), dtype=torch.uint8)
    swa_slots = torch.tensor([[0, 7, 8, 9, 10], [1, 2, 3, 4, 5], [1, -1, 3, 64, 5]])
    global_slots = torch.tensor([[0, 1, -1, 64], [2, 4, 5, 6], [3, 4, 5, 6]])
    swa_lens = torch.tensor([1, 0, 5], dtype=torch.int32)
    global_lens = torch.tensor([4, 1, 0], dtype=torch.int32)
    sink = torch.full((64,), -torch.inf)
    out = torch.empty_like(q)
    return [
        q,
        swa,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        sink,
        512**-0.5,
        out,
        2,
        object(),
        None,
        None,
    ]


def test_decode_preserves_holes_and_rebases_each_chunk(native_calls):
    args = _decode_inputs()
    result = candidate.selected_attention(*args)
    assert result is args[9]
    assert [call[0].shape[0] for call in native_calls] == [2, 1]
    assert native_calls[0][2][0, 0, :9].tolist() == [0, -1, -1, -1, -1, 5, 6, -1, -1]
    assert native_calls[0][2][1, 0, :9].tolist() == [-1, -1, -1, -1, -1, 14, -1, -1, -1]
    assert native_calls[1][2][0, 0, :9].tolist() == [0, -1, 2, -1, 4, -1, -1, -1, -1]
    expected = torch.tensor([202 / 3, 102, 4], dtype=torch.bfloat16)[
        :, None, None
    ].expand_as(result)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.count_nonzero(native_calls[0][0][:, 8:]).item() == 0
    assert torch.isneginf(native_calls[0][3][8:]).all()

    # Re-read changed device metadata on another call, rather than caching a
    # previous prefix length or turning it into a host-side truncation.
    args[3].zero_()
    args[6].zero_()
    candidate.selected_attention(*args)
    assert torch.count_nonzero(result).item() == 0


def test_prefill_workspace_keeps_rows_and_masks_before_int32_cast(native_calls):
    q = torch.zeros((3, 8, 512), dtype=torch.bfloat16)
    kv = (
        torch.tensor([2, 6, 10], dtype=torch.bfloat16)[:, None, None]
        .expand(-1, 1, 512)
        .contiguous()
    )
    indices = torch.tensor(
        [[0, -1, 2**32 + 1, 1], [2, 2, -3, 9], [-1, -1, -1, -1]], dtype=torch.int64
    )
    sink = torch.full((8,), -torch.inf)
    result = candidate.selected_attention(
        q,
        None,
        None,
        None,
        None,
        None,
        None,
        sink,
        512**-0.5,
        None,
        2,
        None,
        kv,
        indices,
    )
    assert [call[0].shape[0] for call in native_calls] == [2, 1]
    assert native_calls[0][2][0, 0, :4].tolist() == [0, -1, -1, 1]
    assert native_calls[0][2][1, 0, :4].tolist() == [2, 2, -1, -1]
    expected = torch.tensor([4, 10, 0], dtype=torch.bfloat16)[:, None, None].expand_as(
        result
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.is_contiguous()
    torch.testing.assert_close(native_calls[1][1], kv, rtol=0, atol=0)


def test_empty_width_and_empty_batch_do_not_call_native(native_calls):
    args = _decode_inputs()
    args[2] = args[2][:, :0]
    args[4:7] = [None, None, None]
    assert candidate.selected_attention(*args) is args[9]
    assert torch.count_nonzero(args[9]).item() == 0
    assert native_calls == []
    for index in (0, 2, 3, 9):
        args[index] = args[index][:0]
    assert candidate.selected_attention(*args).shape == (0, 8, 512)
    assert native_calls == []


def test_prefill_pair_and_output_shape_are_validated(native_calls):
    args = _decode_inputs()
    args[-1] = torch.zeros((3, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="supplied together"):
        candidate.selected_attention(*args)
    args[-1] = None
    args[9] = torch.empty((3, 64, 512), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="specified shape"):
        candidate.selected_attention(*args)
    assert native_calls == []
