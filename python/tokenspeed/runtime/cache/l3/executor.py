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

"""Descriptor-driven executor for distributed Store cache transfers."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.store.base import BaseKVStore
from tokenspeed.runtime.cache.transfer.layout import (
    CacheTransferLayout,
    combine_cache_transfer_layouts,
)
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()


def _ordered_unique(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(v) for v in values))


@dataclass
class _StashSlot:
    buffer: torch.Tensor
    capacity: int
    busy: bool = False


class _Ack(NamedTuple):
    finish_event: object
    op_ids: list[int]
    stash_slot: _StashSlot | None


@dataclass
class _PendingLoadSubmission:
    op_ids: tuple[int, ...]
    content_hashes: tuple[str, ...]
    future: Future
    status: str = "pending"
    error: str | None = None


@dataclass
class _PendingStoreSubmission:
    content_hashes: tuple[str, ...]
    future: Future


@dataclass
class _PresenceProbe:
    hashes: tuple[str, ...]
    future: Future


class _StorePriorityGate:
    """Serialize a Store client while allowing queued loads to pass writes."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._waiting_reads = 0

    @contextmanager
    def call(self, *, read: bool):
        with self._condition:
            if read:
                self._waiting_reads += 1
            try:
                while self._active or (not read and self._waiting_reads):
                    self._condition.wait()
                self._active = True
            finally:
                if read:
                    self._waiting_reads -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


class _ReadPriorityState:
    """Let background writers yield while one or more loads are pending."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending_reads = 0

    def begin_read(self) -> None:
        with self._condition:
            self._pending_reads += 1

    def end_read(self) -> None:
        with self._condition:
            self._pending_reads -= 1
            if self._pending_reads < 0:
                raise RuntimeError("L3 read-priority counter underflow")
            if self._pending_reads == 0:
                self._condition.notify_all()

    def wait_for_idle(self) -> None:
        with self._condition:
            while self._pending_reads:
                self._condition.wait()


def _cache_stream_priorities() -> tuple[int | None, int | None]:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None, None
    try:
        return priority_range()
    except (RuntimeError, TypeError):
        return None, None


def _new_cache_stream(priority: int | None = None):
    if priority is None:
        return device_module.Stream()
    try:
        return device_module.Stream(priority=priority)
    except (RuntimeError, TypeError):
        return device_module.Stream()


def _sanitize_namespace_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unknown"


def _build_store_namespace(
    *,
    model_id: str | None,
    model_revision: str | None,
    model_fingerprint: str | None,
    cache_abi_fingerprint: str | None,
    extra_tag: str | None,
) -> str:
    if model_id is None or not str(model_id).strip():
        raise ValueError("L3 Store namespace requires a model identifier")
    model_component = _sanitize_namespace_component(str(model_id))
    revision_component = (
        _sanitize_namespace_component(str(model_revision))
        if model_revision
        else "default"
    )
    abi_component = (
        _sanitize_namespace_component(str(cache_abi_fingerprint))
        if cache_abi_fingerprint
        else "unknown-abi"
    )
    fingerprint_component = (
        _sanitize_namespace_component(model_fingerprint)
        if model_fingerprint
        else "unknown-model"
    )
    tag_component = _sanitize_namespace_component(extra_tag) if extra_tag else "default"
    raw = (
        f"{model_component}@{revision_component}:{fingerprint_component}:"
        f"{abi_component}:{tag_component}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{tag_component}_{_sanitize_namespace_component(model_component)}_{digest}"


def _fingerprint_model_artifacts(model_id: str | None) -> str | None:
    """Cheaply version a local checkpoint without hashing every weight byte.

    Hugging Face snapshots normally expose immutable blob paths. For mutable
    local directories, include file identity plus samples from each weight so
    replacing a checkpoint at the same path changes the L3 namespace.
    """
    if not model_id:
        return None
    root = Path(model_id)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in (
        "config.json",
        "*.safetensors.index.json",
        "*.bin.index.json",
        "*.safetensors",
        "*.bin",
        "*.pt",
    ):
        candidates.extend(root.glob(pattern))
    candidates = sorted(set(candidates), key=lambda path: path.name)
    if not candidates:
        return None
    digest = hashlib.sha256()
    sample_bytes = 1024 * 1024
    for path in candidates:
        try:
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}:".encode("ascii"))
            with path.open("rb") as handle:
                if stat.st_size <= sample_bytes * 2:
                    digest.update(handle.read())
                else:
                    digest.update(handle.read(sample_bytes))
                    handle.seek(-sample_bytes, os.SEEK_END)
                    digest.update(handle.read(sample_bytes))
        except OSError as exc:
            logger.warning("L3 namespace could not fingerprint %s: %s", path, exc)
            return None
    return digest.hexdigest()[:16]


def _fingerprint_cache_layout(layout: Any) -> str:
    parts: list[str] = []
    for group in getattr(layout, "groups", ()):
        fields = getattr(group, "fields", ())
        field_ids = ",".join(sorted(getattr(field, "field_id", "") for field in fields))
        payloads = ",".join(
            str(getattr(field, "payload_bytes", "")) for field in fields
        )
        strides = ",".join(
            str(getattr(field, "block_stride_bytes", "")) for field in fields
        )
        parts.append(
            f"{group.group_id}:{group.cache_blocks_per_lcm_block}:{field_ids}:{payloads}:{strides}"
        )
    raw = "|".join(parts) if parts else "empty-layout"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _tp_aware_store_key(
    content_hash: str,
    group_id: str,
    cache_block_offset: int,
    tp_rank: int | None = None,
    *,
    namespace: str | None = None,
) -> str:
    base = (
        f"{content_hash}_{group_id}"
        if cache_block_offset == 0
        else f"{content_hash}_{group_id}_o{cache_block_offset}"
    )
    if tp_rank is not None:
        base = f"{base}_tp{tp_rank}"
    if namespace:
        base = f"{namespace}:{base}"
    return base


class L3CacheExecutor:
    """Execute group-aware operations against the distributed store."""

    def __init__(
        self,
        store: BaseKVStore,
        device_pool: Any,
        draft_pool: Any | None = None,
        *,
        l2_executor: Any | None = None,
        io_backend: str = "kernel",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
        cache_abi_fingerprint: str | None = None,
        store_namespace: str | None = None,
        max_stash_bytes: int = 4 * 1024**3,
        store_probe_ttl: float = 1.0,
        io_workers: int = 2,
        direct_gpu: str = "auto",
        direct_gpu_chunk_objects: int = 2,
        host_pipeline_chunk_pages: int = 2,
    ) -> None:
        if store is None:
            raise ValueError("L3 requires an initialized Store backend")
        if io_backend not in ("direct", "kernel"):
            raise ValueError(f"unsupported L3 IO backend {io_backend!r}")
        self.store: BaseKVStore | None = store
        self.l2_executor = l2_executor
        self.transfer_backend = "dma" if io_backend == "direct" else "auto"
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._explicit_namespace = (
            store_namespace.strip()
            if isinstance(store_namespace, str) and store_namespace.strip()
            else None
        )
        self._model_id = model_id
        self._model_revision = model_revision
        self._cache_abi_fingerprint = cache_abi_fingerprint
        if max_stash_bytes <= 0:
            raise ValueError("L3 max_stash_bytes must be positive")
        if store_probe_ttl < 0:
            raise ValueError("L3 store_probe_ttl must be non-negative")
        if io_workers <= 0:
            raise ValueError("L3 io_workers must be positive")
        if direct_gpu not in ("auto", "on", "off"):
            raise ValueError("L3 direct_gpu must be one of: auto, on, off")
        if direct_gpu_chunk_objects <= 0:
            raise ValueError("L3 direct_gpu_chunk_objects must be positive")
        if host_pipeline_chunk_pages <= 0:
            raise ValueError("L3 host_pipeline_chunk_pages must be positive")

        target_layout = device_pool.cache_transfer_layout()
        draft_layout = (
            draft_pool.cache_transfer_layout() if draft_pool is not None else None
        )
        scheduler_group_ids = tuple(
            spec.group_id for spec in device_pool.paged_cache_group_specs
        )
        if (
            draft_layout is not None
            and draft_layout.buffers[0] is target_layout.buffers[0]
        ):
            self.layout = target_layout
        else:
            self.layout = combine_cache_transfer_layouts(
                target_layout, draft_layout, group_ids=scheduler_group_ids
            )

        self._stash_slots: list[_StashSlot] = []
        self._registered_ptrs: set[int] = set()
        self._stash_total_bytes = 0
        self._max_stash_bytes = int(max_stash_bytes)
        self._stash_condition = threading.Condition()
        self._store_lock = threading.RLock()
        self._store_gate = _StorePriorityGate()
        self._write_store_gate = _StorePriorityGate()
        self._read_priority = _ReadPriorityState()
        self._write_store = store
        self._load_acks_lock = threading.Lock()
        self._closed = False
        # Reads/probes and writes have separate pools so background eviction
        # writes cannot occupy every worker needed by a TTFT-critical load.
        self._io_pool = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="tokenspeed-l3-io"
        )
        self._write_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenspeed-l3-write"
        )
        self._pending_load_submissions: dict[
            tuple[int, ...], _PendingLoadSubmission
        ] = {}
        self._pending_store_submissions: list[_PendingStoreSubmission] = []
        self._store_index_outcomes: dict[str, bool] = {}
        self._presence_cache: dict[str, tuple[bool, float]] = {}
        self._presence_probe: _PresenceProbe | None = None
        self._store_probe_ttl = float(store_probe_ttl)
        self._direct_gpu_chunk_objects = int(direct_gpu_chunk_objects)
        self._host_pipeline_chunk_pages = int(host_pipeline_chunk_pages)

        pool_layouts = [(device_pool, target_layout)]
        if draft_pool is not None and self.layout is not target_layout:
            pool_layouts.append((draft_pool, draft_layout))
        self._load_trackers: list[tuple[LayerwiseLoadTracker, int]] = []
        for pool, layout in pool_layouts:
            tracker = LayerwiseLoadTracker(len(layout.consumers))
            pool.register_layerwise_load_tracker(tracker)
            self._load_trackers.append((tracker, len(layout.consumers)))

        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)

        self._direct_gpu_io = self._configure_direct_gpu_io(direct_gpu)

        self._load_acks: list[_Ack] = []

        abi = self._cache_abi_fingerprint or _fingerprint_cache_layout(self.layout)
        # Include TP size in ABI so different sharding doesn't collide.
        if self.tp_size is not None and self.tp_size > 1:
            abi = f"{abi}_tp{self.tp_size}"
        self._store_namespace = _build_store_namespace(
            model_id=self._model_id,
            model_revision=self._model_revision,
            model_fingerprint=_fingerprint_model_artifacts(self._model_id),
            cache_abi_fingerprint=abi,
            extra_tag=(
                self._explicit_namespace or getattr(store, "extra_backend_tag", None)
            ),
        )
        # L3 executor owns namespacing via key prefix; disable backend's
        # extra_backend_tag prefix to avoid double-namespacing (e.g.
        # "ns:ns:key"). The executor's namespace already incorporates
        # extra_backend_tag when explicitly provided.
        try:
            for backend in {store, self._write_store}:
                if getattr(backend, "extra_backend_tag", None) is not None:
                    backend.extra_backend_tag = None  # type: ignore[attr-defined]
        except Exception:
            pass

        logger.info(
            "L3 Store: enabled backend=%s groups=%s namespace=%s direct_gpu=%s",
            type(store).__name__,
            scheduler_group_ids,
            self._store_namespace,
            self._direct_gpu_io,
        )

    def _configure_direct_gpu_io(self, mode: str) -> bool:
        if mode == "off":
            return False
        if not bool(getattr(self.store, "supports_device_memory", False)):
            if mode == "on":
                raise RuntimeError("L3 Store does not support direct GPU buffers")
            return False
        try:
            for buffer in self.layout.buffers:
                if not isinstance(buffer, torch.Tensor) or not buffer.is_cuda:
                    raise RuntimeError(
                        "cache transfer layout contains a non-CUDA buffer"
                    )
                if not self._ensure_registered(buffer):
                    raise RuntimeError("Store rejected a CUDA cache buffer")
        except Exception as exc:
            if mode == "on":
                raise RuntimeError("L3 direct GPU registration failed") from exc
            logger.warning(
                "L3 direct GPU I/O unavailable; using pinned-host pipeline: %s", exc
            )
            return False
        return True

    def _ensure_store_gate(self) -> None:
        if not hasattr(self, "_store_gate"):
            self._store_gate = _StorePriorityGate()
        if not hasattr(self, "_write_store_gate"):
            self._write_store_gate = _StorePriorityGate()

    def _store_call(self, *, read: bool):
        self._ensure_store_gate()
        gate = (
            self._write_store_gate
            if not read and self._write_store is not self.store
            else self._store_gate
        )
        return gate.call(read=read)

    def _registered_stores(self) -> tuple[BaseKVStore, ...]:
        stores = [self.store]
        write_store = getattr(self, "_write_store", None)
        if write_store is not None and write_store is not self.store:
            stores.append(write_store)
        return tuple(store for store in stores if store is not None)

    def _ensure_registered(self, buffer: torch.Tensor) -> bool:
        if self.store is None:
            return False
        ptr = int(buffer.data_ptr())
        if ptr in self._registered_ptrs:
            return True
        size = int(buffer.numel() * buffer.element_size())
        try:
            with self._store_lock:
                results = [
                    backend.register_buffer(ptr, size)
                    for backend in self._registered_stores()
                ]
        except Exception as exc:
            logger.warning(
                "L3: register_buffer failed ptr=%s size=%s: %s", ptr, size, exc
            )
            return False
        failed = [
            result for result in results if result is not None and int(result) != 0
        ]
        if failed:
            logger.warning(
                "L3: register_buffer failed ptr=%s size=%s results=%s",
                ptr,
                size,
                results,
            )
            return False
        self._registered_ptrs.add(ptr)
        return True

    def _allocate_stash(self, nbytes: int) -> _StashSlot:
        # Use a regular pinned torch tensor: transfer_cache_ranges validates
        # host_buffer.is_pinned(), which a raw foreign allocation (e.g. from
        # MooncakeHostMemAllocator) can never satisfy, and _ensure_registered
        # already exposes the same memory to the Store backend for zero-copy
        # put/get.
        self._ensure_stash_state()
        with self._stash_condition:
            if self._stash_total_bytes + nbytes > self._max_stash_bytes:
                raise RuntimeError(
                    "L3 pinned stash limit exceeded: "
                    f"requested={nbytes} retained={self._stash_total_bytes} "
                    f"limit={self._max_stash_bytes}"
                )
            buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            slot = _StashSlot(buffer=buffer, capacity=nbytes, busy=True)
            self._stash_slots.append(slot)
            self._stash_total_bytes += nbytes
            if not self._ensure_registered(buffer):
                slot.busy = False
                self._stash_slots.pop()
                self._stash_total_bytes -= nbytes
                raise RuntimeError(
                    f"failed to register L3 host buffer ({nbytes} bytes)"
                )
            return slot

    def _ensure_stash_state(self) -> None:
        # A few focused unit tests instantiate the executor via __new__.
        if not hasattr(self, "_stash_condition"):
            self._stash_condition = threading.Condition()
        if not hasattr(self, "_stash_total_bytes"):
            self._stash_total_bytes = sum(
                int(slot.capacity) for slot in getattr(self, "_stash_slots", ())
            )
        if not hasattr(self, "_max_stash_bytes"):
            self._max_stash_bytes = 1 << 62
        if not hasattr(self, "_store_lock"):
            self._store_lock = threading.RLock()

    def _acquire_stash(self, nbytes: int) -> tuple[_StashSlot, torch.Tensor]:
        self._ensure_stash_state()
        with self._stash_condition:
            for slot in self._stash_slots:
                if not slot.busy and slot.capacity >= nbytes:
                    slot.busy = True
                    return slot, slot.buffer[:nbytes]
        slot = self._allocate_stash(nbytes)
        return slot, slot.buffer[:nbytes]

    def _release_stash(self, slot: _StashSlot) -> None:
        self._ensure_stash_state()
        with self._stash_condition:
            slot.busy = False
            self._stash_condition.notify()

    def _transfer_ranges(
        self,
        transfers: Sequence[tuple[int, int, int]],
        field_ids: set[str] | None = None,
        *,
        host_base_offset: int = 0,
    ) -> list[tuple[int, int, int, int]]:
        ranges: list[tuple[int, int, int, int]] = []
        host_offset = host_base_offset
        for group_index, device_block_id, _host_block_id in transfers:
            group = self.layout.groups[group_index]
            for field in group.fields:
                if field_ids is not None and field.field_id not in field_ids:
                    # The stash packs every field of every transfer contiguously
                    # (see _do_store_put/_do_store_get), so a skipped field must
                    # still advance the cursor: per-layer H2D copies otherwise
                    # read the wrong absolute host offsets beyond the first
                    # layer/transfer.
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

    def _device_transfer_buffers(
        self, transfers: Sequence[tuple[int, int, int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Build key-major scatter/gather vectors over cache fields."""
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
                    raise RuntimeError(
                        "L3 direct GPU range exceeds cache allocation: "
                        f"offset={offset} size={size} capacity={capacity}"
                    )
                ptrs.append(int(buffer.data_ptr()) + offset)
                sizes.append(size)
            all_ptrs.append(ptrs)
            all_sizes.append(sizes)
        return all_ptrs, all_sizes

    @staticmethod
    def _operation_metadata(
        operation: Any,
        transfer_count: int,
        *,
        required: bool,
    ) -> tuple[list[str], list[int]]:
        nested_hashes = getattr(operation, "content_hashes", None) or []
        hashes = [str(value) for values in nested_hashes for value in values]
        nested_offsets = getattr(operation, "cache_block_offsets", None) or []
        offsets = [int(value) for values in nested_offsets for value in values]
        if not offsets:
            offsets = [0] * len(hashes)
        if required and len(hashes) != transfer_count:
            raise RuntimeError(
                "Store load metadata does not cover every group/offset transfer: "
                f"{len(hashes)} hashes for {transfer_count} transfers"
            )
        if offsets and len(offsets) != len(hashes):
            raise RuntimeError(
                "cache block offset metadata length does not match content hashes"
            )
        if len(hashes) < transfer_count:
            hashes.extend([""] * (transfer_count - len(hashes)))
            offsets.extend([0] * (transfer_count - len(offsets)))
        return hashes[:transfer_count], offsets[:transfer_count]

    def submit_plan(
        self, plan: Any, *, cache_zero_event: object | None = None
    ) -> tuple[int, ...]:
        write_op_ids: list[int] = []
        write_transfers: list[tuple[int, int, int]] = []
        write_hashes: list[str] = []
        write_offsets: list[int] = []
        store_op_ids: list[int] = []
        store_transfers: list[tuple[int, int, int]] = []
        store_hashes: list[str] = []
        store_offsets: list[int] = []

        from tokenspeed_scheduler import Cache as _Cache

        for operation in plan.cache:
            if isinstance(operation, _Cache.WriteBackOp):
                before = len(write_transfers)
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    collected_op_ids=write_op_ids,
                    transfers=write_transfers,
                    source_is_device=True,
                )
                count = len(write_transfers) - before
                hashes, offsets = self._operation_metadata(
                    operation, count, required=False
                )
                write_hashes.extend(hashes)
                write_offsets.extend(offsets)
            elif isinstance(operation, _Cache.StoreLoadOp):
                before = len(store_transfers)
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    collected_op_ids=store_op_ids,
                    transfers=store_transfers,
                    source_is_device=False,
                )
                count = len(store_transfers) - before
                hashes, offsets = self._operation_metadata(
                    operation, count, required=True
                )
                store_hashes.extend(hashes)
                store_offsets.extend(offsets)
            elif isinstance(operation, _Cache.LoadBackOp):
                continue
            else:
                raise TypeError(f"unsupported cache op {type(operation).__name__}")

        load_key = tuple(_ordered_unique(store_op_ids))
        if load_key:
            if load_key in self._pending_load_submissions:
                raise RuntimeError(f"duplicate L3 load submission {load_key}")
            self._read_priority.begin_read()
            try:
                future = self._io_pool.submit(
                    self._run_store_get,
                    load_key,
                    tuple(store_transfers),
                    tuple(store_hashes),
                    tuple(store_offsets),
                    cache_zero_event,
                )
            except Exception:
                self._read_priority.end_read()
                raise
            self._pending_load_submissions[load_key] = _PendingLoadSubmission(
                op_ids=load_key,
                content_hashes=tuple(store_hashes),
                future=future,
            )
        if write_op_ids and write_transfers:
            hashes = tuple(hash_value for hash_value in write_hashes if hash_value)
            ready_event = torch.cuda.Event()
            ready_event.record()
            future = self._write_pool.submit(
                self._run_store_put,
                tuple(write_transfers),
                tuple(write_hashes),
                tuple(write_offsets),
                ready_event,
            )
            self._pending_store_submissions.append(
                _PendingStoreSubmission(content_hashes=hashes, future=future)
            )
        return load_key

    def _run_store_get(
        self,
        op_ids: tuple[int, ...],
        transfers: tuple[tuple[int, int, int], ...],
        content_hashes: tuple[str, ...],
        cache_block_offsets: tuple[int, ...],
        cache_zero_event: object | None,
    ) -> int:
        try:
            # Page sanitization is launched asynchronously by the scheduler.
            # Order the H2D Store restore after it without blocking the event loop.
            if cache_zero_event is not None:
                cache_zero_event.synchronize()
            load_index = self._start_loading(
                op_ids,
                transfers,
                content_hashes=content_hashes,
                cache_block_offsets=cache_block_offsets,
            )
            if load_index is None:
                raise RuntimeError("L3 Store load produced no load index")
            for tracker, _ in self._load_trackers:
                tracker.set_consumers(load_index)
            return load_index
        except Exception:
            for tracker, _ in self._load_trackers:
                tracker.set_consumers(-1)
            raise
        finally:
            self._read_priority.end_read()

    def _run_store_put(
        self,
        transfers: tuple[tuple[int, int, int], ...],
        content_hashes: tuple[str, ...],
        cache_block_offsets: tuple[int, ...],
        ready_event: object | None = None,
    ) -> dict[str, bool]:
        requested = tuple(dict.fromkeys(value for value in content_hashes if value))
        try:
            if ready_event is not None:
                ready_event.synchronize()
            completed = set(
                self._do_store_put(
                    transfers,
                    content_hashes,
                    cache_block_offsets,
                )
            )
        except Exception as exc:
            logger.warning("L3 put failed: %s", exc)
            completed = set()
        return {hash_value: hash_value in completed for hash_value in requested}

    @staticmethod
    def _append_transfers(
        operation_ids: Sequence[int],
        group_ids: Sequence[Sequence[int]],
        src_blocks: Sequence[Sequence[int]],
        dst_blocks: Sequence[Sequence[int]],
        *,
        collected_op_ids: list[int],
        transfers: list[tuple[int, int, int]],
        source_is_device: bool,
    ) -> None:
        if not (
            len(operation_ids) == len(group_ids) == len(src_blocks) == len(dst_blocks)
        ):
            raise ValueError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            operation_ids, group_ids, src_blocks, dst_blocks
        ):
            if not (len(groups) == len(sources) == len(destinations)):
                raise ValueError(f"ragged cache operation {op_id}")
            collected_op_ids.append(int(op_id))
            for group, source, destination in zip(groups, sources, destinations):
                device_block_id, host_block_id = (
                    (source, destination) if source_is_device else (destination, source)
                )
                transfers.append((int(group), int(device_block_id), int(host_block_id)))

    def _start_writing(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str] | None = None,
        cache_block_offsets: Sequence[int] | None = None,
    ) -> None:
        if not op_ids or not transfers:
            return
        outcome = self._run_store_put(
            tuple(transfers),
            tuple(content_hashes or ()),
            tuple(cache_block_offsets or ()),
        )
        self._store_index_outcomes.update(outcome)

    def _do_store_put(
        self,
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str] | None = None,
        cache_block_offsets: Sequence[int] | None = None,
    ) -> list[str]:
        if self.store is None:
            raise RuntimeError("L3 Store is closed")
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

        records: list[tuple[int, str, str]] = []
        for index, (group_index, _device_block, _host_block) in enumerate(transfers):
            if (
                content_hashes is None
                or index >= len(content_hashes)
                or not content_hashes[index]
            ):
                continue
            content_hash = content_hashes[index]
            offset = (
                cache_block_offsets[index]
                if cache_block_offsets is not None and index < len(cache_block_offsets)
                else 0
            )
            group_id = self.layout.groups[group_index].group_id
            records.append(
                (
                    index,
                    content_hash,
                    _tp_aware_store_key(
                        content_hash,
                        group_id,
                        offset,
                        tp_rank=self.tp_rank,
                        namespace=self._store_namespace,
                    ),
                )
            )
        if not records:
            return []

        keys = [record[2] for record in records]
        try:
            self._read_priority.wait_for_idle()
            with self._store_call(read=False):
                exists = self._write_store.batch_exists(keys)
            if len(exists) != len(keys):
                raise RuntimeError(
                    f"L3 batch_exists length mismatch {len(exists)} != {len(keys)}"
                )
        except Exception as exc:
            logger.debug("L3 batch_exists failed, attempting all puts: %s", exc)
            exists = [0] * len(keys)

        succeeded = [value == 1 for value in exists]
        missing_record_indices = [
            index for index, value in enumerate(succeeded) if not value
        ]
        if missing_record_indices:
            missing_transfers = [
                transfers[records[index][0]] for index in missing_record_indices
            ]
            missing_keys = [keys[index] for index in missing_record_indices]
            if getattr(self, "_direct_gpu_io", False):
                ptrs, sizes = self._device_transfer_buffers(missing_transfers)
                results = []
                chunk_size = self._direct_gpu_chunk_objects
                for begin in range(0, len(missing_keys), chunk_size):
                    end = min(begin + chunk_size, len(missing_keys))
                    self._read_priority.wait_for_idle()
                    with self._store_call(read=False):
                        results.extend(
                            self._write_store.batch_put_from(
                                missing_keys[begin:end],
                                ptrs[begin:end],
                                sizes[begin:end],
                            )
                        )
                if len(results) != len(missing_keys):
                    raise RuntimeError(
                        "L3 direct batch_put_from length mismatch "
                        f"{len(results)} != {len(missing_keys)}"
                    )
                for record_index, result in zip(missing_record_indices, results):
                    succeeded[record_index] = result == 0
                missing_record_indices = []
        if missing_record_indices:
            missing_transfers = [
                transfers[records[index][0]] for index in missing_record_indices
            ]
            ranges = self._transfer_ranges(missing_transfers)
            total_bytes = sum(byte_count for *_prefix, byte_count in ranges)
            if total_bytes <= 0:
                return []
            slot, stash = self._acquire_stash(total_bytes)
            try:
                start = torch.cuda.Event()
                start.record()
                start.wait(self.write_stream)
                transfer_cache_ranges(
                    "d2h",
                    self.layout.buffers,
                    stash,
                    ranges,
                    self.write_stream,
                    backend=self.transfer_backend,
                )
                self.write_stream.synchronize()
                ptr_base = int(stash.data_ptr())
                ptrs: list[int] = []
                sizes: list[int] = []
                offset = 0
                for group_index, _device_block, _host_block in missing_transfers:
                    nbytes = sum(
                        field.payload_bytes
                        for field in self.layout.groups[group_index].fields
                    )
                    ptrs.append(ptr_base + offset)
                    sizes.append(nbytes)
                    offset += nbytes
                missing_keys = [keys[index] for index in missing_record_indices]
                results = []
                chunk_size = self._host_pipeline_chunk_pages
                for begin in range(0, len(missing_keys), chunk_size):
                    end = min(begin + chunk_size, len(missing_keys))
                    self._read_priority.wait_for_idle()
                    with self._store_call(read=False):
                        results.extend(
                            self._write_store.batch_put_from(
                                missing_keys[begin:end],
                                ptrs[begin:end],
                                sizes[begin:end],
                            )
                        )
                if len(results) != len(missing_keys):
                    raise RuntimeError(
                        "L3 batch_put_from length mismatch "
                        f"{len(results)} != {len(missing_keys)}"
                    )
                for record_index, result in zip(missing_record_indices, results):
                    succeeded[record_index] = result == 0
            finally:
                # Store puts are synchronous, so the host bytes are no longer in use.
                self._release_stash(slot)

        hash_success: dict[str, bool] = {}
        for (_transfer_index, content_hash, _key), success in zip(records, succeeded):
            hash_success[content_hash] = (
                hash_success.get(content_hash, True) and success
            )
        return [
            content_hash for content_hash, success in hash_success.items() if success
        ]

    def _start_loading(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str] | None = None,
        cache_block_offsets: Sequence[int] | None = None,
    ) -> int | None:
        if not op_ids:
            return None
        op_ids = _ordered_unique(op_ids)
        if not transfers:
            raise RuntimeError(f"Store load has no transfers (op_ids={op_ids})")
        try:
            return self._do_store_get(
                op_ids, transfers, content_hashes, cache_block_offsets
            )
        except Exception as exc:
            raise RuntimeError(
                f"L3 Store load failed for op_ids={op_ids}: {exc}"
            ) from exc

    def _do_store_get(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str] | None = None,
        cache_block_offsets: Sequence[int] | None = None,
    ) -> int:
        if self.store is None:
            raise RuntimeError("L3 Store is closed")
        if content_hashes is None or len(content_hashes) != len(transfers):
            raise RuntimeError("Store load is missing content hashes")
        keys: list[str] = []
        for index, (group_index, _device_block, _host_block) in enumerate(transfers):
            content_hash = content_hashes[index]
            if not content_hash:
                raise RuntimeError(f"Store load transfer {index} has no content hash")
            offset = (
                cache_block_offsets[index]
                if cache_block_offsets is not None and index < len(cache_block_offsets)
                else 0
            )
            keys.append(
                _tp_aware_store_key(
                    content_hash,
                    self.layout.groups[group_index].group_id,
                    offset,
                    tp_rank=self.tp_rank,
                    namespace=self._store_namespace,
                )
            )

        sizes = [
            sum(field.payload_bytes for field in self.layout.groups[group].fields)
            for group, _device_block, _host_block in transfers
        ]
        total_bytes = sum(sizes)
        if total_bytes <= 0:
            raise RuntimeError("Store load requested zero bytes")
        if getattr(self, "_direct_gpu_io", False):
            return self._do_direct_store_get(op_ids, transfers, keys, sizes)
        return self._do_pipelined_host_store_get(
            op_ids, transfers, keys, sizes, total_bytes
        )

    @staticmethod
    def _validate_store_reads(
        keys: Sequence[str], results: Sequence[int], sizes: Sequence[int]
    ) -> None:
        if len(results) != len(keys):
            raise RuntimeError(
                f"L3 batch_get_into length mismatch {len(results)} != {len(keys)}"
            )
        short_reads = [
            (key, result, requested)
            for key, result, requested in zip(keys, results, sizes)
            if result is None or int(result) != requested
        ]
        if short_reads:
            details = ", ".join(
                f"{key}: got {result}, expected {requested}"
                for key, result, requested in short_reads
            )
            raise RuntimeError(f"incomplete L3 Store read ({details})")

    def _begin_load_tracking(self):
        load_index = None
        tracked = []
        consumer_offset = 0
        for tracker, consumer_count in self._load_trackers:
            current_load_index = tracker.begin_load()
            if load_index is None:
                load_index = current_load_index
            elif current_load_index != load_index:
                raise RuntimeError("target and draft Store-load trackers diverged")
            load_events = tracker.event_sets[current_load_index]
            load_events.start_event.record()
            load_events.start_event.wait(self.load_stream)
            tracked.append((load_events, consumer_offset, consumer_count))
            consumer_offset += consumer_count
        if load_index is None or not tracked:
            raise RuntimeError("cache transfer layout has no layer consumers")
        return load_index, tracked

    def _do_direct_store_get(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        keys: Sequence[str],
        sizes: Sequence[int],
    ) -> int:
        if self.store is None:
            raise RuntimeError("L3 Store is closed")
        ptrs, field_sizes = self._device_transfer_buffers(transfers)
        started = time.perf_counter()
        chunk_size = self._direct_gpu_chunk_objects
        results: list[int] = []
        for begin in range(0, len(keys), chunk_size):
            end = min(begin + chunk_size, len(keys))
            chunk_keys = list(keys[begin:end])
            chunk_sizes = list(sizes[begin:end])
            with self._store_call(read=True):
                chunk_results = self.store.batch_get_into(
                    chunk_keys,
                    ptrs[begin:end],
                    field_sizes[begin:end],
                )
            # Validate each completed chunk before issuing another one. A short
            # read raises through the single load-submission future, so the
            # scheduler observes one failed load and requeues/recomputes every
            # admitted request even if earlier chunks populated some pages.
            self._validate_store_reads(chunk_keys, chunk_results, chunk_sizes)
            results.extend(chunk_results)
        store_ms = (time.perf_counter() - started) * 1000
        # Keep one aggregate outcome for the scheduler-facing StoreLoadOp.
        self._validate_store_reads(keys, results, sizes)

        load_index, tracked = self._begin_load_tracking()
        finish = None
        for load_events, _consumer_offset, consumer_count in tracked:
            for layer_index in range(consumer_count):
                finish = torch.cuda.Event()
                finish.record(self.load_stream)
                load_events.layer_done_events[layer_index] = finish
        if finish is None:
            raise RuntimeError("cache transfer layout has no layer consumers")
        with self._load_acks_lock:
            self._load_acks.append(_Ack(finish, list(op_ids), None))
        logger.info(
            "L3 load direct_gpu objects=%s bytes=%s chunks=%s store_ms=%.3f",
            len(keys),
            sum(sizes),
            (len(keys) + chunk_size - 1) // chunk_size,
            store_ms,
        )
        return load_index

    def _do_pipelined_host_store_get(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        keys: Sequence[str],
        sizes: Sequence[int],
        total_bytes: int,
    ) -> int:
        if self.store is None:
            raise RuntimeError("L3 Store is closed")
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

        slot, stash = self._acquire_stash(total_bytes)
        retained_for_async_copy = False
        try:
            ptr_base = int(stash.data_ptr())
            byte_offsets: list[int] = []
            offset = 0
            for nbytes in sizes:
                byte_offsets.append(offset)
                offset += nbytes
            load_index, tracked = self._begin_load_tracking()
            finish = None
            store_seconds = 0.0
            enqueue_seconds = 0.0
            chunk_size = self._host_pipeline_chunk_pages
            for begin in range(0, len(keys), chunk_size):
                end = min(begin + chunk_size, len(keys))
                chunk_keys = list(keys[begin:end])
                chunk_sizes = list(sizes[begin:end])
                chunk_ptrs = [ptr_base + value for value in byte_offsets[begin:end]]
                started = time.perf_counter()
                with self._store_call(read=True):
                    results = self.store.batch_get_into(
                        chunk_keys, chunk_ptrs, chunk_sizes
                    )
                store_seconds += time.perf_counter() - started
                self._validate_store_reads(chunk_keys, results, chunk_sizes)

                started = time.perf_counter()
                chunk_transfers = transfers[begin:end]
                for load_events, consumer_offset, consumer_count in tracked:
                    for layer_index in range(consumer_count):
                        consumer = self.layout.consumers[consumer_offset + layer_index]
                        transfer_cache_ranges(
                            "h2d",
                            self.layout.buffers,
                            stash,
                            self._transfer_ranges(
                                chunk_transfers,
                                set(consumer),
                                host_base_offset=byte_offsets[begin],
                            ),
                            self.load_stream,
                            backend=self.transfer_backend,
                        )
                        finish = torch.cuda.Event()
                        finish.record(self.load_stream)
                        load_events.layer_done_events[layer_index] = finish
                enqueue_seconds += time.perf_counter() - started
            if finish is None:
                raise RuntimeError("cache transfer layout has no layer consumers")
            with self._load_acks_lock:
                self._load_acks.append(_Ack(finish, list(op_ids), slot))
            retained_for_async_copy = True
            logger.info(
                "L3 load host_pipeline objects=%s bytes=%s chunks=%s "
                "store_ms=%.3f enqueue_ms=%.3f",
                len(keys),
                total_bytes,
                (len(keys) + chunk_size - 1) // chunk_size,
                store_seconds * 1000,
                enqueue_seconds * 1000,
            )
            return load_index
        finally:
            if not retained_for_async_copy:
                self._release_stash(slot)

    def poll_results(self) -> list:
        self._poll_store_submissions()
        results: list = []
        with self._load_acks_lock:
            self._load_acks[:] = self._drain_loads(self._load_acks, results)
        return results

    def _poll_store_submissions(self) -> None:
        pending: list[_PendingStoreSubmission] = []
        now = time.monotonic()
        for submission in self._pending_store_submissions:
            if not submission.future.done():
                pending.append(submission)
                continue
            try:
                outcome = submission.future.result()
            except Exception as exc:
                logger.warning("L3 background put failed: %s", exc)
                outcome = {
                    hash_value: False for hash_value in submission.content_hashes
                }
            self._store_index_outcomes.update(outcome)
            for hash_value, present in outcome.items():
                self._presence_cache[hash_value] = (
                    bool(present),
                    now + self._store_probe_ttl,
                )
        self._pending_store_submissions = pending

    def load_submission_status(
        self, op_ids: Sequence[int]
    ) -> tuple[str, list[str], str | None]:
        key = tuple(_ordered_unique(op_ids))
        submission = self._pending_load_submissions.get(key)
        if submission is None:
            return "failed", [], f"unknown L3 load submission {key}"
        if submission.status == "pending" and submission.future.done():
            try:
                submission.future.result()
                submission.status = "succeeded"
                self.record_presence(submission.content_hashes, present=True)
            except Exception as exc:
                submission.status = "failed"
                submission.error = str(exc)
        failed_hashes = (
            list(dict.fromkeys(submission.content_hashes))
            if submission.status == "failed"
            else []
        )
        return submission.status, failed_hashes, submission.error

    def acknowledge_load_submission(self, op_ids: Sequence[int]) -> None:
        self._pending_load_submissions.pop(tuple(_ordered_unique(op_ids)), None)

    def abort_load_submission(self, op_ids: Sequence[int]) -> None:
        key = tuple(_ordered_unique(op_ids))
        submission = self._pending_load_submissions.pop(key, None)
        if submission is not None:
            if submission.future.cancel():
                self._read_priority.end_read()
            else:
                try:
                    submission.future.result()
                except Exception:
                    pass
        try:
            self.load_stream.synchronize()
        except Exception:
            pass
        with self._load_acks_lock:
            retained: list[_Ack] = []
            for ack in self._load_acks:
                if any(op_id in key for op_id in ack.op_ids):
                    if ack.stash_slot is not None:
                        self._release_stash(ack.stash_slot)
                else:
                    retained.append(ack)
            self._load_acks[:] = retained
        for tracker, _ in self._load_trackers:
            tracker.reset()

    def probe_store_presence(
        self, content_hashes: Sequence[str]
    ) -> tuple[str, dict[str, bool], str | None]:
        hashes = tuple(dict.fromkeys(value for value in content_hashes if value))
        if not hashes:
            return "ready", {}, None
        now = time.monotonic()
        probe_error: str | None = None
        if self._presence_probe is not None:
            if not self._presence_probe.future.done():
                known = {
                    hash_value: self._presence_cache[hash_value][0]
                    for hash_value in hashes
                    if hash_value in self._presence_cache
                    and self._presence_cache[hash_value][0]
                }
                if len(known) == len(hashes):
                    return "ready", known, None
                return "pending", {}, None
            probe = self._presence_probe
            self._presence_probe = None
            try:
                outcome = probe.future.result()
                error = None
            except Exception as exc:
                logger.warning("L3 Store existence probe failed: %s", exc)
                outcome = {hash_value: False for hash_value in probe.hashes}
                error = str(exc)
            for hash_value, present in outcome.items():
                self._presence_cache[hash_value] = (
                    bool(present),
                    now + self._store_probe_ttl,
                )
            probe_error = error
        # Positive entries are optimistic after their TTL. A failed direct get
        # is now safe and invalidates/requeues the admission, so paying a
        # blocking existence RPC before every known-key hit only adds TTFT.
        # Unknown and previously-missing entries are still probed, preserving
        # cross-process discovery and eventual discovery after a new put.
        missing = [
            hash_value
            for hash_value in hashes
            if hash_value not in self._presence_cache
            or (
                not self._presence_cache[hash_value][0]
                and self._presence_cache[hash_value][1] <= now
            )
        ]
        if missing:
            future = self._io_pool.submit(self._do_probe_store_presence, tuple(missing))
            self._presence_probe = _PresenceProbe(tuple(missing), future)
            return "pending", {}, None
        return (
            "ready",
            {hash_value: self._presence_cache[hash_value][0] for hash_value in hashes},
            probe_error,
        )

    def _do_probe_store_presence(
        self, content_hashes: tuple[str, ...]
    ) -> dict[str, bool]:
        if self.store is None:
            raise RuntimeError("L3 Store is closed")
        keys: list[str] = []
        owners: list[str] = []
        for content_hash in content_hashes:
            for group in self.layout.groups:
                for offset in range(group.cache_blocks_per_lcm_block):
                    keys.append(
                        _tp_aware_store_key(
                            content_hash,
                            group.group_id,
                            offset,
                            tp_rank=self.tp_rank,
                            namespace=self._store_namespace,
                        )
                    )
                    owners.append(content_hash)
        with self._store_call(read=True):
            exists = self.store.batch_exists(keys)
        if len(exists) != len(keys):
            raise RuntimeError(
                f"L3 batch_exists length mismatch {len(exists)} != {len(keys)}"
            )
        result = {hash_value: True for hash_value in content_hashes}
        for owner, present in zip(owners, exists):
            result[owner] = result[owner] and int(present) == 1
        return result

    def invalidate_presence(self, content_hashes: Sequence[str]) -> None:
        for hash_value in content_hashes:
            self._presence_cache.pop(hash_value, None)

    def record_presence(self, content_hashes: Sequence[str], *, present: bool) -> None:
        expiry = time.monotonic() + self._store_probe_ttl
        for hash_value in content_hashes:
            if hash_value:
                self._presence_cache[str(hash_value)] = (bool(present), expiry)

    def _drain_loads(self, queue: list[_Ack], results: list) -> list[_Ack]:
        pending: list[_Ack] = []
        for ack in queue:
            if ack.finish_event.query():
                results.extend(self._load_done(op_id) for op_id in ack.op_ids)
                if ack.stash_slot is not None:
                    self._release_stash(ack.stash_slot)
            else:
                pending.append(ack)
        return pending

    def peek_store_index_outcomes(self) -> dict[str, bool]:
        self._poll_store_submissions()
        return dict(self._store_index_outcomes)

    def acknowledge_store_index_outcomes(self, content_hashes: Sequence[str]) -> None:
        for hash_value in content_hashes:
            self._store_index_outcomes.pop(hash_value, None)

    def record_store_index_outcomes(self, outcomes: dict[str, bool]) -> None:
        self._store_index_outcomes.update(
            {str(value): bool(present) for value, present in outcomes.items() if value}
        )

    @staticmethod
    def _load_done(op_id: int):
        from tokenspeed_scheduler import Cache as _Cache

        event = _Cache.StoreLoadDoneEvent()
        event.op_id = op_id
        return event

    def _synchronize(self) -> None:
        try:
            self.write_stream.synchronize()
        except Exception:
            pass
        try:
            self.load_stream.synchronize()
        except Exception:
            pass
        with self._load_acks_lock:
            for ack in self._load_acks:
                if ack.stash_slot is not None:
                    self._release_stash(ack.stash_slot)
            self._load_acks.clear()

    def _wait_for_io(self) -> None:
        futures = [item.future for item in self._pending_load_submissions.values()]
        futures.extend(item.future for item in self._pending_store_submissions)
        if self._presence_probe is not None:
            futures.append(self._presence_probe.future)
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
        self._poll_store_submissions()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._wait_for_io()
        self._io_pool.shutdown(wait=True, cancel_futures=False)
        self._write_pool.shutdown(wait=True, cancel_futures=False)
        self._synchronize()
        store, self.store = self.store, None
        write_store, self._write_store = getattr(self, "_write_store", None), None
        for backend in dict.fromkeys((write_store, store)):
            if backend is None:
                continue
            try:
                with self._store_lock:
                    backend.close()
            except Exception as exc:
                logger.warning("L3 Store close failed: %s", exc)
        self._stash_slots.clear()
        self._stash_total_bytes = 0
        self._registered_ptrs.clear()
        self._pending_load_submissions.clear()
        self._pending_store_submissions.clear()
        self._store_index_outcomes.clear()
        self._presence_cache.clear()
        self._presence_probe = None
        self._closed = True

    def reset(self) -> None:
        self._wait_for_io()
        self._synchronize()
        self._pending_load_submissions.clear()
        self._pending_store_submissions.clear()
        self._store_index_outcomes.clear()
        self._presence_cache.clear()
        self._presence_probe = None
        for tracker, _ in self._load_trackers:
            tracker.reset()
