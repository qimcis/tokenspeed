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

"""Patch-aware routing of whole encoder requests to complete workers.

The caller supplies patch counts after image preprocessing. Cache hints describe
one worker incarnation; unknown keys are misses. Image ordering and identities
remain in the request, and the existing EncodeScheduler alone forms ViT batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.load_balancing.types import Lease, WorkerSnapshot, Workload

if TYPE_CHECKING:
    from tokenspeed.runtime.load_balancing.registry import WorkerRegistry
    from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem


@dataclass(frozen=True)
class EncoderItemWork:
    """Post-preprocessing patches for one item, in its original request position."""

    item_index: int
    cache_key: str
    patch_count: int

    def __post_init__(self) -> None:
        if type(self.item_index) is not int or self.item_index < 0:
            raise ValueError("item_index must be a nonnegative integer")
        if type(self.patch_count) is not int or self.patch_count <= 0:
            raise ValueError("patch_count must be a positive integer")
        if not isinstance(self.cache_key, str) or not self.cache_key:
            raise ValueError("cache_key must be a nonempty string")


@dataclass(frozen=True)
class EncoderRequestWork:
    """Immutable ordered work for one request; repeated content stays distinct."""

    request_id: str
    items: tuple[EncoderItemWork, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("items must be a nonempty tuple")
        if any(not isinstance(item, EncoderItemWork) for item in self.items):
            raise ValueError("items must contain EncoderItemWork values")
        if tuple(item.item_index for item in self.items) != tuple(
            range(len(self.items))
        ):
            raise ValueError("items must preserve consecutive original request indices")

    def uncached_patches(self, snapshot: WorkerSnapshot) -> int:
        """Predicted tower work from fresh, incarnation-scoped cache hints."""
        # Repeated misses remain separate: the real batcher queues each item.
        return sum(
            item.patch_count
            for item in self.items
            if item.cache_key not in snapshot.worker.cache_keys
        )

    def workload(self, snapshot: WorkerSnapshot) -> Workload:
        """Reserve a safe upper bound even if hinted cache entries are evicted.

        Cache membership is not a pin. Until the execution path can confirm and
        hold embeddings, use hints for selection but retain every item's patch
        reservation through dispatch. This can overreserve, never underreserve.
        """
        return Workload(
            requests=1,
            encoder_patches=sum(item.patch_count for item in self.items),
        )


class EncoderPolicy:
    """Reserve bounded patch work and choose using predicted uncached work."""

    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def reserve(
        self, request: EncoderRequestWork, *, now: float | None = None
    ) -> Lease:
        return self.registry.reserve(
            request_id=request.request_id,
            role="encoder",
            score=lambda snapshot: (
                snapshot.total.encoder_patches + request.uncached_patches(snapshot)
            ),
            workload=request.workload,
            now=now,
        )


def preprocessed_patch_count(data: dict) -> int | None:
    """Read exact CPU preprocessing grids without a GPU synchronization.

    V4.1 uses ``vit_grid``; Qwen/Kimi-style towers use image/video ``grid_thw``.
    Missing or unsupported geometry is unknown, never guessed from image count
    or merged output-token offsets. SHM pixel payloads need not be attached.
    """
    keys = [
        key for key in ("vit_grid", "image_grid_thw", "video_grid_thw") if key in data
    ]
    if len(keys) != 1:
        return None
    key = keys[0]
    grid = data[key]
    width = 2 if key == "vit_grid" else 3
    if grid.device.type != "cpu" or grid.numel() == 0 or grid.numel() % width:
        return None
    values = grid.reshape(-1).tolist()
    if any(type(value) is not int or value <= 0 for value in values):
        return None
    total = 0
    for start in range(0, len(values), width):
        count = 1
        for value in values[start : start + width]:
            count *= value
        total += count
    return total


def encoder_request_work(
    request_id: str, items: list[MultimodalDataItem]
) -> EncoderRequestWork:
    """Build routing metadata from already preprocessed items without reordering."""
    work = []
    for index, item in enumerate(items):
        patches = preprocessed_patch_count(item.model_specific_data)
        if patches is None or item.hash is None:
            raise ValueError(
                "encoder routing requires CPU patch grids and content hashes"
            )
        work.append(EncoderItemWork(index, str(item.hash), patches))
    return EncoderRequestWork(request_id, tuple(work))
