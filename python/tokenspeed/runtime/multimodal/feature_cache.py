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

"""LRU byte-bounded cache for encoded multimodal features.

The cache lives inside a single :class:`VisionEmbedder` instance, which itself
is owned by one model executor / tensor-parallel rank.  Keys are
``(modality, content_hash)`` and values are the already-encoded feature tensors
(plus an optional deepstack companion).  Tensors returned by the cache are
read-only from the caller's perspective; they may be shared across requests as
long as the cache holds a reference to keep them alive.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _CacheEntry:
    encoded: torch.Tensor
    encoded_deepstack: torch.Tensor | None
    size: int


class EncodedFeatureCache:
    """Byte-bounded LRU cache for ``MultimodalDataItem.encoded`` tensors.

    Parameters
    ----------
    max_bytes:
        Soft upper bound on the total bytes stored in the cache.  A single
        item whose encoded size exceeds ``max_bytes`` is still cached, but a
        warning is emitted.  ``0`` disables caching entirely.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._cache: OrderedDict[tuple[Any, int], _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0

    @staticmethod
    def _tensor_size(t: torch.Tensor | None) -> int:
        if t is None:
            return 0
        return int(t.element_size() * t.nelement())

    @staticmethod
    def _make_key(modality: Any, content_hash: int | None) -> tuple[Any, int] | None:
        if content_hash is None:
            return None
        return (modality, int(content_hash))

    def get(
        self, modality: Any, content_hash: int | None
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Lookup an encoded feature. Returns ``(encoded, encoded_deepstack)``."""
        if self.max_bytes == 0:
            return None
        key = self._make_key(modality, content_hash)
        if key is None:
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return entry.encoded, entry.encoded_deepstack

    def put(
        self,
        modality: Any,
        content_hash: int | None,
        encoded: torch.Tensor | None,
        encoded_deepstack: torch.Tensor | None = None,
    ) -> bool:
        """Store an encoded feature.  Evicts LRU entries if necessary."""
        if self.max_bytes == 0:
            return False
        if encoded is None:
            return False
        key = self._make_key(modality, content_hash)
        if key is None:
            return False

        new_size = self._tensor_size(encoded) + self._tensor_size(encoded_deepstack)
        if new_size == 0:
            return False

        with self._lock:
            # If the key already exists, remove its old accounting first.
            old_entry = self._cache.pop(key, None)
            if old_entry is not None:
                self._current_bytes -= old_entry.size

            # Make room.  If the new item is larger than the whole budget we
            # still cache it (it is presumably valuable), but warn so operators
            # can adjust the budget.
            if new_size > self.max_bytes:
                logger.warning(
                    "EncodedFeatureCache: item size %s exceeds max_bytes %s; "
                    "consider increasing TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES",
                    new_size,
                    self.max_bytes,
                )

            while (
                self._cache
                and self._current_bytes + new_size > self.max_bytes
                and self._current_bytes > 0
            ):
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.size
                self.evictions += 1

            self._cache[key] = _CacheEntry(
                encoded=encoded,
                encoded_deepstack=encoded_deepstack,
                size=new_size,
            )
            self._current_bytes += new_size
            self.insertions += 1
            self._cache.move_to_end(key)

        logger.debug(
            "EncodedFeatureCache put modality=%s hash=%s size=%d current_bytes=%d "
            "entries=%d",
            modality,
            content_hash,
            new_size,
            self._current_bytes,
            len(self._cache),
        )
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_bytes": self.max_bytes,
                "current_bytes": self._current_bytes,
                "entries": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "insertions": self.insertions,
            }
