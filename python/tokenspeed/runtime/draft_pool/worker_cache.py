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

"""Bounded worker KV geometry and a portable native-model attention adapter.

The target scheduler owns target KV. This private TP1 worker has no target
pages: its resident slots each own a fixed history ring and a disjoint native
draft block. Absolute positions remain unchanged for RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerCacheGeometry:
    """Map resident slots to fixed one-token pages, including a null page."""

    resident_slots: int
    window_tokens: int
    native_block_tokens: int

    def __post_init__(self) -> None:
        if (
            self.resident_slots < 1
            or self.window_tokens < 2
            or self.native_block_tokens < 2
        ):
            raise ValueError(
                "Worker cache dimensions must be positive with a history window."
            )

    @property
    def history_tokens(self) -> int:
        return self.window_tokens - 1

    @property
    def slot_tokens(self) -> int:
        return self.history_tokens + self.native_block_tokens

    @property
    def total_tokens(self) -> int:
        return 1 + self.resident_slots * self.slot_tokens

    def base(self, slot: int) -> int:
        if not 0 <= slot < self.resident_slots:
            raise ValueError(f"Invalid resident slot: {slot}")
        return 1 + slot * self.slot_tokens

    def context_locations(self, slot: int, start: int, endpoint: int) -> list[int]:
        """Physical slots for projected features at absolute [start, endpoint)."""
        if not 0 <= start <= endpoint or endpoint - start > self.history_tokens:
            raise ValueError("Context transfer exceeds the retained history interval.")
        base = self.base(slot)
        return [
            base + position % self.history_tokens for position in range(start, endpoint)
        ]

    def draft_locations(self, slot: int) -> list[int]:
        """Return scratch locations that cannot overwrite confirmed history."""
        start = self.base(slot) + self.history_tokens
        return list(range(start, start + self.native_block_tokens))

    def page_row(self, slot: int, endpoint: int) -> list[int]:
        """Chronological context and the full unverified native block."""
        start = max(0, endpoint - self.history_tokens)
        return self.context_locations(slot, start, endpoint) + self.draft_locations(
            slot
        )

    def kv_bytes(
        self, layers: int, kv_heads: int, head_dim: int, element_bytes: int
    ) -> int:
        """Exact bytes of all preallocated K and V tensors."""
        return self.total_tokens * layers * 2 * kv_heads * head_dim * element_bytes


def validate_worker_attention(
    model, geometry: WorkerCacheGeometry, kv_heads: int, head_dim: int
) -> None:
    """Reject a built layer whose visibility or KV shape exceeds worker storage."""
    if model._uses_mla:
        raise ValueError("The bounded remote worker supports GQA DFlash2 only.")
    for layer in model.layers:
        attention = layer.self_attn
        if attention.attn.sliding_window_size != geometry.history_tokens:
            raise ValueError("Every worker layer must use the retained sliding window.")
        if attention.num_kv_heads != kv_heads or attention.head_dim != head_dim:
            raise ValueError("Built draft attention differs from worker KV geometry.")


class WorkerKVPool:
    """Fixed KV planes satisfying DFlashDraftModel's existing context-write API.

    Construct and call only on the worker execution thread. ``device`` is
    explicit to permit small CPU storage tests without a draft checkpoint.
    """

    def __init__(
        self,
        geometry: WorkerCacheGeometry,
        layers: int,
        kv_heads: int,
        head_dim: int,
        device: str,
    ) -> None:
        import torch

        self.geometry = geometry
        self.dtype = torch.bfloat16
        shape = (geometry.total_tokens, 1, kv_heads, head_dim)
        self.keys = [
            torch.zeros(shape, dtype=self.dtype, device=device) for _ in range(layers)
        ]
        self.values = [
            torch.zeros(shape, dtype=self.dtype, device=device) for _ in range(layers)
        ]

    def get_kv_buffer(self, layer_id: int):
        return self.keys[layer_id], self.values[layer_id]

    def set_kv_buffer(self, layer, cache_locs, k, v, k_scale, v_scale) -> None:
        import torch

        if k_scale not in (None, 1.0) or v_scale not in (None, 1.0):
            raise ValueError("The BF16 worker does not accept quantized KV scales.")
        keys, values = self.get_kv_buffer(layer.layer_id)
        keys[:, 0].index_copy_(0, cache_locs.to(dtype=torch.int64), k)
        values[:, 0].index_copy_(0, cache_locs.to(dtype=torch.int64), v)

    def clear_slot(self, slot: int) -> None:
        start = self.geometry.base(slot)
        stop = start + self.geometry.slot_tokens
        for keys, values in zip(self.keys, self.values, strict=True):
            keys[start:stop].zero_()
            values[start:stop].zero_()


class WorkerAttentionBackend:
    """Run the existing portable GQA kernel on independently batched sessions.

    DFlash uses a bidirectional native block. Expand each of its query rows
    into a separate one-query request with the same block-end visibility,
    exactly like the runtime's ``draft_block_decode`` metadata. Passing N8
    directly to an ordinary decode kernel would incorrectly impose causality.
    """

    supports_layer_sliding_window = True

    def __init__(self, geometry: WorkerCacheGeometry, device: str) -> None:
        self.geometry = geometry
        self.device = device
        self._page_table = None
        self._seq_lens = None
        self._write_locations = None
        self._batch_size = 0
        self._max_k = 0

    def prepare(self, slots: list[int], endpoints: list[int]) -> None:
        """Publish metadata and scratch writes for one immutable native batch."""
        import torch

        if not slots or len(slots) != len(endpoints) or len(set(slots)) != len(slots):
            raise ValueError(
                "A worker batch needs distinct sessions and matching endpoints."
            )
        rows = [
            self.geometry.page_row(slot, end)
            for slot, end in zip(slots, endpoints, strict=True)
        ]
        lengths = [len(row) for row in rows]
        self._max_k = max(lengths)
        rows = [row + [0] * (self._max_k - len(row)) for row in rows]
        width = self.geometry.native_block_tokens
        self._page_table = torch.tensor(
            rows, dtype=torch.int32, device=self.device
        ).repeat_interleave(width, dim=0)
        self._seq_lens = torch.tensor(
            lengths, dtype=torch.int32, device=self.device
        ).repeat_interleave(width)
        self._write_locations = torch.tensor(
            [loc for slot in slots for loc in self.geometry.draft_locations(slot)],
            dtype=torch.int64,
            device=self.device,
        )
        self._batch_size = len(slots)

    def write_locations(self, layer, forward_mode):
        if self._write_locations is None:
            raise RuntimeError("Worker attention batch has not been prepared.")
        return self._write_locations

    def forward(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        forward_mode,
        bs,
        save_kv_cache,
        **kwargs,
    ):
        from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

        if (
            bs != self._batch_size
            or save_kv_cache
            or k is not None
            or v is not None
            or kwargs
        ):
            raise ValueError("Invalid native DFlash worker attention invocation.")
        keys, values = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        return mha_decode_with_kvcache(
            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_cache=keys,
            v_cache=values,
            page_table=self._page_table,
            cache_seqlens=self._seq_lens,
            max_seqlen_k=self._max_k,
            max_seqlen_q=1,
            window_left=layer.sliding_window_size,
            logit_cap=0.0,
            sinks=None,
            return_lse=False,
            softmax_scale=layer.scaling,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="triton",
        )
