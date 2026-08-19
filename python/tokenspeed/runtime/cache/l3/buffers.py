# Copyright (c) 2026 LightSeek Foundation

"""Registered cache buffers and bounded pinned-host staging."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tokenspeed.runtime.cache.l3.errors import L3BackendError, L3TransferError
from tokenspeed.runtime.cache.store.base import BaseKVStore
from tokenspeed.runtime.cache.store.errors import KVStoreBackendError
from tokenspeed.runtime.cache.transfer.layout import CacheTransferLayout


@dataclass
class StashSlot:
    buffer: torch.Tensor
    capacity: int
    busy: bool = False


class L3BufferManager:
    """Own Store registration and reusable pinned-host staging buffers."""

    def __init__(
        self,
        layout: CacheTransferLayout,
        stores: Sequence[BaseKVStore],
        max_stash_bytes: int,
    ) -> None:
        self.layout = layout
        self._stores = tuple(dict.fromkeys(stores))
        self._max_stash_bytes = int(max_stash_bytes)
        self._stash_slots: list[StashSlot] = []
        self._stash_total_bytes = 0
        self._registered_ptrs: set[int] = set()
        self._condition = threading.Condition()
        self._registration_lock = threading.RLock()

    @property
    def registration_lock(self) -> threading.RLock:
        return self._registration_lock

    @property
    def can_unregister(self) -> bool:
        return all(store.supports_buffer_unregistration for store in self._stores)

    def register(self, buffer: torch.Tensor) -> None:
        ptr = int(buffer.data_ptr())
        if ptr in self._registered_ptrs:
            return
        size = int(buffer.numel() * buffer.element_size())
        try:
            with self._registration_lock:
                results = [store.register_buffer(ptr, size) for store in self._stores]
        except (
            KVStoreBackendError,
            RuntimeError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise L3BackendError(
                f"Store buffer registration failed ptr={ptr} size={size}: {exc}"
            ) from exc
        failed = [
            result for result in results if result is not None and int(result) != 0
        ]
        if failed:
            raise L3BackendError(
                f"Store rejected buffer registration ptr={ptr} size={size} results={results}"
            )
        self._registered_ptrs.add(ptr)

    def register_device_layout(self) -> None:
        for buffer in self.layout.buffers:
            if not isinstance(buffer, torch.Tensor) or not buffer.is_cuda:
                raise L3TransferError(
                    "cache transfer layout contains a non-CUDA buffer"
                )
            self.register(buffer)

    def acquire_stash(self, nbytes: int) -> tuple[StashSlot, torch.Tensor]:
        with self._condition:
            for slot in sorted(self._stash_slots, key=lambda value: value.capacity):
                if not slot.busy and slot.capacity >= nbytes:
                    slot.busy = True
                    return slot, slot.buffer[:nbytes]
            if self._stash_total_bytes + nbytes > self._max_stash_bytes:
                self._reclaim_idle_stash(nbytes)
            if self._stash_total_bytes + nbytes > self._max_stash_bytes:
                raise L3TransferError(
                    "L3 pinned stash limit exceeded by active transfers: "
                    f"requested={nbytes} retained={self._stash_total_bytes} "
                    f"limit={self._max_stash_bytes}"
                )
            try:
                buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as exc:
                raise L3TransferError(
                    f"failed to allocate {nbytes} bytes of pinned L3 staging memory"
                ) from exc
            slot = StashSlot(buffer=buffer, capacity=nbytes, busy=True)
            self._stash_slots.append(slot)
            self._stash_total_bytes += nbytes
        try:
            self.register(buffer)
        except L3BackendError:
            with self._condition:
                self._stash_slots.remove(slot)
                self._stash_total_bytes -= nbytes
            raise
        return slot, slot.buffer[:nbytes]

    def _reclaim_idle_stash(self, requested: int) -> None:
        if not self.can_unregister:
            return
        idle = sorted(
            (slot for slot in self._stash_slots if not slot.busy),
            key=lambda value: value.capacity,
        )
        for slot in idle:
            ptr = int(slot.buffer.data_ptr())
            try:
                with self._registration_lock:
                    results = [store.unregister_buffer(ptr) for store in self._stores]
            except (KVStoreBackendError, RuntimeError, OSError) as exc:
                raise L3BackendError(
                    f"Store buffer unregistration failed ptr={ptr}: {exc}"
                ) from exc
            failed = [
                result for result in results if result is not None and int(result) != 0
            ]
            if failed:
                raise L3BackendError(
                    f"Store rejected buffer unregistration ptr={ptr} results={results}"
                )
            self._registered_ptrs.discard(ptr)
            self._stash_slots.remove(slot)
            self._stash_total_bytes -= slot.capacity
            if self._stash_total_bytes + requested <= self._max_stash_bytes:
                return

    def release_stash(self, slot: StashSlot) -> None:
        with self._condition:
            if not slot.busy:
                raise L3TransferError("L3 staging slot was released more than once")
            slot.busy = False
            self._condition.notify()

    def transfer_ranges(
        self,
        transfers: Sequence[tuple[int, int, int]],
        field_ids: set[str] | None = None,
        *,
        host_base_offset: int = 0,
    ) -> list[tuple[int, int, int, int]]:
        ranges: list[tuple[int, int, int, int]] = []
        host_offset = host_base_offset
        for group_index, device_block_id, _host_block_id in transfers:
            for field in self.layout.groups[group_index].fields:
                if field_ids is not None and field.field_id not in field_ids:
                    host_offset += field.payload_bytes
                    continue
                ranges.append(
                    (
                        field.device_buffer_index,
                        field.device_block_zero_offset_bytes
                        + device_block_id * field.block_stride_bytes,
                        host_offset,
                        field.payload_bytes,
                    )
                )
                host_offset += field.payload_bytes
        return ranges

    def device_vectors(
        self, transfers: Sequence[tuple[int, int, int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        for group_index, device_block_id, _host_block_id in transfers:
            ptrs: list[int] = []
            sizes: list[int] = []
            for field in self.layout.groups[group_index].fields:
                buffer = self.layout.buffers[field.device_buffer_index]
                offset = (
                    field.device_block_zero_offset_bytes
                    + device_block_id * field.block_stride_bytes
                )
                size = field.payload_bytes
                capacity = int(buffer.numel() * buffer.element_size())
                if offset < 0 or offset + size > capacity:
                    raise L3TransferError(
                        "L3 direct GPU range exceeds cache allocation: "
                        f"offset={offset} size={size} capacity={capacity}"
                    )
                ptrs.append(int(buffer.data_ptr()) + offset)
                sizes.append(size)
            all_ptrs.append(ptrs)
            all_sizes.append(sizes)
        return all_ptrs, all_sizes

    def clear(self, *, unregister: bool = True) -> None:
        with self._condition:
            if any(slot.busy for slot in self._stash_slots):
                raise L3TransferError("cannot clear L3 buffers while staging is active")
            ptrs = tuple(self._registered_ptrs)
        if unregister:
            if not self.can_unregister:
                raise L3BackendError(
                    "Store backend cannot unregister buffers before release"
                )
            with self._registration_lock:
                for ptr in ptrs:
                    try:
                        results = [
                            store.unregister_buffer(ptr) for store in self._stores
                        ]
                    except (KVStoreBackendError, RuntimeError, OSError) as exc:
                        raise L3BackendError(
                            f"Store buffer unregistration failed ptr={ptr}: {exc}"
                        ) from exc
                    failed = [
                        result
                        for result in results
                        if result is not None and int(result) != 0
                    ]
                    if failed:
                        raise L3BackendError(
                            f"Store rejected buffer unregistration ptr={ptr} results={results}"
                        )
        self._stash_slots.clear()
        self._stash_total_bytes = 0
        self._registered_ptrs.clear()
