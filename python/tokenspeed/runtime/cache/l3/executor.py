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
import re
from collections.abc import Sequence
from dataclasses import dataclass
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
    stash_slot: _StashSlot


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
    cache_abi_fingerprint: str | None,
    extra_tag: str | None,
) -> str:
    if extra_tag is not None and extra_tag.strip():
        return _sanitize_namespace_component(extra_tag.strip())
    if model_id is None or not str(model_id).strip():
        raise ValueError(
            "L3 Store namespace requires a model identifier: set "
            '--kvstore-storage-backend-extra-config \'{"extra_backend_tag": "<ns>"}\' '
            "or ensure the model path/revision is available"
        )
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
    raw = f"{model_component}@{revision_component}:{abi_component}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{_sanitize_namespace_component(model_component)}_{digest}"


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
        self._closed = False

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

        self._load_acks: list[_Ack] = []
        self._ready_store_hashes: list[str] = []

        if self._explicit_namespace is not None:
            self._store_namespace = _sanitize_namespace_component(
                self._explicit_namespace
            )
        else:
            abi = self._cache_abi_fingerprint or _fingerprint_cache_layout(self.layout)
            # Include TP size in ABI so different sharding doesn't collide.
            if self.tp_size is not None and self.tp_size > 1:
                abi = f"{abi}_tp{self.tp_size}"
            self._store_namespace = _build_store_namespace(
                model_id=self._model_id,
                model_revision=self._model_revision,
                cache_abi_fingerprint=abi,
                extra_tag=getattr(store, "extra_backend_tag", None),
            )
        # L3 executor owns namespacing via key prefix; disable backend's
        # extra_backend_tag prefix to avoid double-namespacing (e.g.
        # "ns:ns:key"). The executor's namespace already incorporates
        # extra_backend_tag when explicitly provided.
        try:
            if getattr(store, "extra_backend_tag", None) is not None:
                store.extra_backend_tag = None  # type: ignore[attr-defined]
        except Exception:
            pass

        logger.info(
            "L3 Store: enabled backend=%s groups=%s namespace=%s",
            type(store).__name__,
            scheduler_group_ids,
            self._store_namespace,
        )

    def _ensure_registered(self, buffer: torch.Tensor) -> bool:
        if self.store is None:
            return False
        ptr = int(buffer.data_ptr())
        if ptr in self._registered_ptrs:
            return True
        size = int(buffer.numel() * buffer.element_size())
        try:
            result = self.store.register_buffer(ptr, size)
        except Exception as exc:
            logger.warning(
                "L3: register_buffer failed ptr=%s size=%s: %s", ptr, size, exc
            )
            return False
        if result is not None and int(result) != 0:
            logger.warning(
                "L3: register_buffer failed ptr=%s size=%s ret=%s",
                ptr,
                size,
                result,
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
        buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        slot = _StashSlot(buffer=buffer, capacity=nbytes, busy=True)
        self._stash_slots.append(slot)
        if not self._ensure_registered(buffer):
            slot.busy = False
            raise RuntimeError(f"failed to register L3 host buffer ({nbytes} bytes)")
        return slot

    def _acquire_stash(self, nbytes: int) -> tuple[_StashSlot, torch.Tensor]:
        for slot in self._stash_slots:
            if not slot.busy and slot.capacity >= nbytes:
                slot.busy = True
                return slot, slot.buffer[:nbytes]
        slot = self._allocate_stash(nbytes)
        return slot, slot.buffer[:nbytes]

    @staticmethod
    def _release_stash(slot: _StashSlot) -> None:
        slot.busy = False

    def _transfer_ranges(
        self,
        transfers: Sequence[tuple[int, int, int]],
        field_ids: set[str] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        ranges: list[tuple[int, int, int, int]] = []
        host_offset = 0
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

    def submit_plan(self, plan: Any) -> None:
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

        load_error: Exception | None = None
        load_index: int | None = None
        try:
            load_index = self._start_loading(
                store_op_ids,
                store_transfers,
                content_hashes=store_hashes or None,
                cache_block_offsets=store_offsets or None,
            )
        except Exception as exc:
            load_error = exc
        for tracker, _ in self._load_trackers:
            tracker.set_consumers(load_index if load_index is not None else -1)
        self._start_writing(
            write_op_ids,
            write_transfers,
            content_hashes=write_hashes or None,
            cache_block_offsets=write_offsets or None,
        )
        if load_error is not None:
            raise load_error

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
        try:
            completed_hashes = self._do_store_put(
                transfers, content_hashes, cache_block_offsets
            )
        except Exception as exc:
            logger.warning(
                "L3 put failed (op_ids=%s): %s", _ordered_unique(op_ids), exc
            )
            return
        self._ready_store_hashes.extend(completed_hashes)

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
            exists = self.store.batch_exists(keys)
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
                results = self.store.batch_put_from(missing_keys, ptrs, sizes)
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
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

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
        slot, stash = self._acquire_stash(total_bytes)
        retained_for_async_copy = False
        try:
            ptr_base = int(stash.data_ptr())
            ptrs: list[int] = []
            offset = 0
            for nbytes in sizes:
                ptrs.append(ptr_base + offset)
                offset += nbytes
            results = self.store.batch_get_into(keys, ptrs, sizes)
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

            load_index = None
            consumer_offset = 0
            finish = None
            for tracker, consumer_count in self._load_trackers:
                current_load_index = tracker.begin_load()
                if load_index is None:
                    load_index = current_load_index
                elif current_load_index != load_index:
                    raise RuntimeError("target and draft Store-load trackers diverged")
                load_events = tracker.event_sets[current_load_index]
                load_events.start_event.record()
                load_events.start_event.wait(self.load_stream)
                for layer_index in range(consumer_count):
                    consumer = self.layout.consumers[consumer_offset + layer_index]
                    transfer_cache_ranges(
                        "h2d",
                        self.layout.buffers,
                        stash,
                        self._transfer_ranges(transfers, set(consumer)),
                        self.load_stream,
                        backend=self.transfer_backend,
                    )
                    finish = torch.cuda.Event()
                    finish.record(self.load_stream)
                    load_events.layer_done_events[layer_index] = finish
                consumer_offset += consumer_count
            if load_index is None or finish is None:
                raise RuntimeError("cache transfer layout has no layer consumers")
            self._load_acks.append(_Ack(finish, list(op_ids), slot))
            retained_for_async_copy = True
            return load_index
        finally:
            if not retained_for_async_copy:
                self._release_stash(slot)

    def poll_results(self) -> list:
        results: list = []
        self._load_acks[:] = self._drain_loads(self._load_acks, results)
        return results

    def _drain_loads(self, queue: list[_Ack], results: list) -> list[_Ack]:
        pending: list[_Ack] = []
        for ack in queue:
            if ack.finish_event.query():
                results.extend(self._load_done(op_id) for op_id in ack.op_ids)
                self._release_stash(ack.stash_slot)
            else:
                pending.append(ack)
        return pending

    def take_store_index_updates(self) -> list[str]:
        hashes = list(dict.fromkeys(self._ready_store_hashes))
        self._ready_store_hashes.clear()
        return hashes

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
        for ack in self._load_acks:
            self._release_stash(ack.stash_slot)
        self._load_acks.clear()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._synchronize()
        store, self.store = self.store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                logger.warning("L3 Store close failed: %s", exc)
        self._stash_slots.clear()
        self._registered_ptrs.clear()
        self._closed = True

    def reset(self) -> None:
        self._synchronize()
        self._ready_store_hashes.clear()
        for tracker, _ in self._load_trackers:
            tracker.reset()
