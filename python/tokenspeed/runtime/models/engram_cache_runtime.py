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

"""Load-completion boundary for immutable Engram resident rows.

The model loader calls finish_loading only after the shared host-table barrier.
Weight updates already quiesce execution; this class adds no collective or
per-forward synchronization. Failed/partial updates are never marked ready.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotCache,
    EngramHotCachePlan,
)


class EngramCacheRuntime:
    """Track paired canonical table loads and retain captured cache storage."""

    def __init__(self, plan: EngramHotCachePlan, *, device: torch.device | str):
        self.plan = plan
        self.device = torch.device(device)
        self.cache: EngramHotCache | None = None
        self.generation = -1
        self._loaded: set[str] = set()

    def begin_update(self) -> None:
        """Invalidate before table writes, with graph replay already quiesced."""
        self._loaded.clear()
        if self.cache is not None:
            self.cache.invalidate()
            if self.cache.local_row_ids.is_cuda:
                # Failed loaders may return without the runner's success fence.
                # Publish miss-only keys before another stream can replay graphs.
                torch.cuda.current_stream(self.cache.local_row_ids.device).synchronize()

    def begin_table_load(self, field: str) -> None:
        self._validate_field(field)
        if self.cache is not None and self.cache.valid:
            self.begin_update()
        self._loaded.discard(field)

    def complete_table_load(self, field: str) -> None:
        self._validate_field(field)
        self._loaded.add(field)

    @staticmethod
    def _validate_field(field: str) -> None:
        if field not in ("weight", "scale"):
            raise ValueError("Engram table field must be weight or scale")

    def finish_loading(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Publish only after both tensors load and shared-host writers finish."""
        if self._loaded != {"weight", "scale"}:
            raise RuntimeError("Engram cache requires both completed table loads")
        generation = self.generation + 1
        if self.cache is None:
            self.cache = EngramHotCache(
                self.plan, weight, scale, device=self.device, generation=generation
            )
        else:
            self.cache.refresh_from_tables(
                weight,
                scale,
                model_revision=self.plan.model_revision,
                generation=generation,
            )
        self.generation = generation
        self._loaded.clear()

    def require_ready(self) -> EngramHotCache:
        """Eager/capture guard; device keys also disable stale hits on replay."""
        if self.cache is None or not self.cache.valid:
            raise RuntimeError("Engram cache is not ready; load both tables completely")
        return self.cache
