# Copyright (c) 2026 LightSeek Foundation

"""Store calls and cache-memory transfers for the L3 tier."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.l3.buffers import L3BufferManager, StashSlot
from tokenspeed.runtime.cache.l3.errors import L3BackendError, L3TransferError
from tokenspeed.runtime.cache.l3.namespace import store_key
from tokenspeed.runtime.cache.store.base import BaseKVStore
from tokenspeed.runtime.cache.store.errors import KVStoreBackendError
from tokenspeed.runtime.cache.transfer.layout import CacheTransferLayout
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()


@dataclass(frozen=True)
class TransferChunk:
    begin: int
    end: int
    keys: list[str]
    ptrs: list[Any]
    sizes: list[Any]


class _PriorityGate:
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


class StoreBatcher:
    """One bounded batching policy for direct and staged Store I/O."""

    def __init__(self, read_store: BaseKVStore, write_store: BaseKVStore) -> None:
        self.read_store = read_store
        self.write_store = write_store
        self._read_gate = _PriorityGate()
        self._write_gate = (
            self._read_gate if read_store is write_store else _PriorityGate()
        )
        self._read_condition = threading.Condition()
        self._pending_reads = 0

    def begin_read(self) -> None:
        with self._read_condition:
            self._pending_reads += 1

    def end_read(self) -> None:
        with self._read_condition:
            self._pending_reads -= 1
            if self._pending_reads < 0:
                raise L3TransferError("L3 read-priority counter underflow")
            if self._pending_reads == 0:
                self._read_condition.notify_all()

    def _wait_for_reads(self) -> None:
        with self._read_condition:
            while self._pending_reads:
                self._read_condition.wait()

    @staticmethod
    def chunks(
        keys: Sequence[str],
        ptrs: Sequence[Any],
        sizes: Sequence[Any],
        limit: int,
    ) -> Iterator[TransferChunk]:
        if not (len(keys) == len(ptrs) == len(sizes)):
            raise L3TransferError(
                f"Store vectors are ragged: keys={len(keys)} ptrs={len(ptrs)} sizes={len(sizes)}"
            )
        for begin in range(0, len(keys), limit):
            end = min(begin + limit, len(keys))
            yield TransferChunk(
                begin,
                end,
                list(keys[begin:end]),
                list(ptrs[begin:end]),
                list(sizes[begin:end]),
            )

    def get_chunk(self, chunk: TransferChunk) -> list[int]:
        try:
            with self._read_gate.call(read=True):
                return self.read_store.batch_get_into(
                    chunk.keys, chunk.ptrs, chunk.sizes
                )
        except (
            KVStoreBackendError,
            RuntimeError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise L3BackendError(f"Store read failed: {exc}") from exc

    def get_all(
        self,
        keys: Sequence[str],
        ptrs: Sequence[Any],
        sizes: Sequence[Any],
        limit: int,
    ) -> tuple[list[int], int]:
        results: list[int] = []
        chunks = 0
        for chunk in self.chunks(keys, ptrs, sizes, limit):
            results.extend(self.get_chunk(chunk))
            chunks += 1
        return results, chunks

    def put_all(
        self,
        keys: Sequence[str],
        ptrs: Sequence[Any],
        sizes: Sequence[Any],
        limit: int,
    ) -> list[int]:
        results: list[int] = []
        for chunk in self.chunks(keys, ptrs, sizes, limit):
            self._wait_for_reads()
            try:
                with self._write_gate.call(read=False):
                    results.extend(
                        self.write_store.batch_put_from(
                            chunk.keys, chunk.ptrs, chunk.sizes
                        )
                    )
            except (
                KVStoreBackendError,
                RuntimeError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                raise L3BackendError(f"Store write failed: {exc}") from exc
        return results

    def exists(self, keys: Sequence[str], *, write: bool = False) -> list[int]:
        if write:
            self._wait_for_reads()
        store = self.write_store if write else self.read_store
        gate = self._write_gate if write else self._read_gate
        try:
            with gate.call(read=not write):
                return store.batch_exists(list(keys))
        except (
            KVStoreBackendError,
            RuntimeError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise L3BackendError(f"Store existence check failed: {exc}") from exc


@dataclass
class _LoadAck:
    finish_event: object
    op_ids: tuple[int, ...]
    stash_slot: StashSlot | None


def _stream_priorities() -> tuple[int | None, int | None]:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None, None
    try:
        return priority_range()
    except (RuntimeError, TypeError):
        return None, None


def _new_stream(priority: int | None = None):
    try:
        if priority is None:
            return device_module.Stream()
        try:
            return device_module.Stream(priority=priority)
        except TypeError:
            return device_module.Stream()
    except RuntimeError as exc:
        raise L3TransferError("failed to create an L3 CUDA stream") from exc


class L3StoreIO:
    """Perform direct or staged Store transfers and track CUDA completion."""

    def __init__(
        self,
        *,
        store: BaseKVStore,
        write_store: BaseKVStore,
        layout: CacheTransferLayout,
        buffers: L3BufferManager,
        pool_layouts: Sequence[tuple[Any, CacheTransferLayout]],
        transfer_backend: str,
        tp_rank: int | None,
        namespace: str,
        direct_gpu: str,
        direct_chunk_objects: int,
        host_chunk_objects: int,
    ) -> None:
        self.store = store
        self.write_store = write_store
        self.layout = layout
        self.buffers = buffers
        self.batcher = StoreBatcher(store, write_store)
        self.transfer_backend = transfer_backend
        self.tp_rank = tp_rank
        self.namespace = namespace
        self.direct_chunk_objects = direct_chunk_objects
        self.host_chunk_objects = host_chunk_objects
        self.load_trackers: list[tuple[LayerwiseLoadTracker, int]] = []
        for pool, pool_layout in pool_layouts:
            try:
                tracker = LayerwiseLoadTracker(len(pool_layout.consumers))
                pool.register_layerwise_load_tracker(tracker)
            except RuntimeError as exc:
                raise L3TransferError(
                    "failed to register L3 layerwise load tracking"
                ) from exc
            self.load_trackers.append((tracker, len(pool_layout.consumers)))
        write_priority, load_priority = _stream_priorities()
        self.write_stream = _new_stream(write_priority)
        self.load_stream = _new_stream(load_priority)
        self._acks: list[_LoadAck] = []
        self._acks_lock = threading.Lock()
        self.direct_gpu = self._configure_direct_gpu(direct_gpu)

    def _configure_direct_gpu(self, mode: str) -> bool:
        if mode == "off":
            return False
        if not self.store.supports_device_memory:
            if mode == "on":
                raise L3BackendError("L3 Store does not support direct GPU buffers")
            return False
        try:
            self.buffers.register_device_layout()
        except (L3BackendError, L3TransferError) as exc:
            if mode == "on":
                raise
            logger.warning(
                "L3 direct GPU I/O unavailable; using pinned-host pipeline: %s", exc
            )
            return False
        return True

    def _key(self, content_hash: str, group_index: int, offset: int) -> str:
        return store_key(
            content_hash,
            self.layout.groups[group_index].group_id,
            offset,
            tp_rank=self.tp_rank,
            namespace=self.namespace,
        )

    @staticmethod
    def _validate_reads(
        keys: Sequence[str], results: Sequence[int], sizes: Sequence[int]
    ) -> None:
        if len(results) != len(keys):
            raise L3BackendError(
                f"L3 batch_get_into length mismatch {len(results)} != {len(keys)}"
            )
        short = [
            (key, result, requested)
            for key, result, requested in zip(keys, results, sizes)
            if result is None or int(result) != requested
        ]
        if short:
            details = ", ".join(
                f"{key}: got {result}, expected {requested}"
                for key, result, requested in short
            )
            raise L3BackendError(f"incomplete L3 Store read ({details})")

    def put(
        self,
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str],
        offsets: Sequence[int],
    ) -> list[str]:
        records = [
            (index, content_hash, self._key(content_hash, group, offsets[index]))
            for index, (group, _device, _host) in enumerate(transfers)
            if index < len(content_hashes) and (content_hash := content_hashes[index])
        ]
        if not records:
            return []
        keys = [record[2] for record in records]
        try:
            exists = self.batcher.exists(keys, write=True)
            if len(exists) != len(keys):
                raise L3BackendError(
                    f"L3 batch_exists length mismatch {len(exists)} != {len(keys)}"
                )
        except L3BackendError as exc:
            logger.debug("L3 batch_exists failed, attempting all puts: %s", exc)
            exists = [0] * len(keys)
        succeeded = [value == 1 for value in exists]
        missing = [index for index, present in enumerate(succeeded) if not present]
        if missing and self.direct_gpu:
            selected = [transfers[records[index][0]] for index in missing]
            ptrs, sizes = self.buffers.device_vectors(selected)
            results = self.batcher.put_all(
                [keys[index] for index in missing],
                ptrs,
                sizes,
                self.direct_chunk_objects,
            )
            if len(results) != len(missing):
                raise L3BackendError(
                    f"L3 direct batch_put_from length mismatch {len(results)} != {len(missing)}"
                )
            for record_index, result in zip(missing, results):
                succeeded[record_index] = result == 0
            missing = []
        if missing:
            self._put_staged(transfers, records, keys, succeeded, missing)
        by_hash: dict[str, bool] = {}
        for (_index, content_hash, _key), success in zip(records, succeeded):
            by_hash[content_hash] = by_hash.get(content_hash, True) and success
        return [value for value, success in by_hash.items() if success]

    def _put_staged(self, transfers, records, keys, succeeded, missing) -> None:
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

        selected = [transfers[records[index][0]] for index in missing]
        ranges = self.buffers.transfer_ranges(selected)
        total_bytes = sum(byte_count for *_prefix, byte_count in ranges)
        if total_bytes <= 0:
            return
        slot, stash = self.buffers.acquire_stash(total_bytes)
        try:
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
            except RuntimeError as exc:
                raise L3TransferError("L3 device-to-host staging failed") from exc
            ptr_base = int(stash.data_ptr())
            ptrs: list[int] = []
            sizes: list[int] = []
            offset = 0
            for group_index, _device, _host in selected:
                nbytes = sum(
                    field.payload_bytes
                    for field in self.layout.groups[group_index].fields
                )
                ptrs.append(ptr_base + offset)
                sizes.append(nbytes)
                offset += nbytes
            results = self.batcher.put_all(
                [keys[index] for index in missing],
                ptrs,
                sizes,
                self.host_chunk_objects,
            )
            if len(results) != len(missing):
                raise L3BackendError(
                    f"L3 batch_put_from length mismatch {len(results)} != {len(missing)}"
                )
            for record_index, result in zip(missing, results):
                succeeded[record_index] = result == 0
        finally:
            self.buffers.release_stash(slot)

    def load(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        content_hashes: Sequence[str],
        offsets: Sequence[int],
    ) -> int:
        if not transfers or len(content_hashes) != len(transfers):
            raise L3TransferError("Store load is missing transfers or content hashes")
        keys: list[str] = []
        sizes: list[int] = []
        for index, (group, _device, _host) in enumerate(transfers):
            content_hash = content_hashes[index]
            if not content_hash:
                raise L3TransferError(
                    f"Store load transfer {index} has no content hash"
                )
            keys.append(self._key(content_hash, group, offsets[index]))
            sizes.append(
                sum(field.payload_bytes for field in self.layout.groups[group].fields)
            )
        if self.direct_gpu:
            return self._load_direct(op_ids, transfers, keys, sizes)
        return self._load_staged(op_ids, transfers, keys, sizes)

    def _begin_load(self):
        load_index = None
        tracked = []
        consumer_offset = 0
        for tracker, consumer_count in self.load_trackers:
            current = tracker.begin_load()
            if load_index is None:
                load_index = current
            elif current != load_index:
                raise L3TransferError("target and draft Store-load trackers diverged")
            events = tracker.event_sets[current]
            try:
                events.start_event.record()
                events.start_event.wait(self.load_stream)
            except RuntimeError as exc:
                raise L3TransferError("failed to begin L3 cache restore") from exc
            tracked.append((events, consumer_offset, consumer_count))
            consumer_offset += consumer_count
        if load_index is None or not tracked:
            raise L3TransferError("cache transfer layout has no layer consumers")
        return load_index, tracked

    def _load_direct(self, op_ids, transfers, keys, sizes) -> int:
        ptrs, field_sizes = self.buffers.device_vectors(transfers)
        started = time.perf_counter()
        results: list[int] = []
        chunks = 0
        for chunk in self.batcher.chunks(
            keys, ptrs, field_sizes, self.direct_chunk_objects
        ):
            chunk_results = self.batcher.get_chunk(chunk)
            self._validate_reads(
                chunk.keys, chunk_results, sizes[chunk.begin : chunk.end]
            )
            results.extend(chunk_results)
            chunks += 1
        self._validate_reads(keys, results, sizes)
        load_index, tracked = self._begin_load()
        finish = self._record_layer_events(tracked)
        self._append_ack(finish, op_ids, None)
        logger.info(
            "L3 load direct_gpu objects=%s bytes=%s chunks=%s store_ms=%.3f",
            len(keys),
            sum(sizes),
            chunks,
            (time.perf_counter() - started) * 1000,
        )
        return load_index

    def _load_staged(self, op_ids, transfers, keys, sizes) -> int:
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

        total_bytes = sum(sizes)
        slot, stash = self.buffers.acquire_stash(total_bytes)
        retained = False
        try:
            ptr_base = int(stash.data_ptr())
            byte_offsets: list[int] = []
            offset = 0
            for nbytes in sizes:
                byte_offsets.append(offset)
                offset += nbytes
            load_index, tracked = self._begin_load()
            finish = None
            store_seconds = 0.0
            enqueue_seconds = 0.0
            chunks = 0
            ptrs = [ptr_base + value for value in byte_offsets]
            for chunk in self.batcher.chunks(
                keys, ptrs, sizes, self.host_chunk_objects
            ):
                started = time.perf_counter()
                results = self.batcher.get_chunk(chunk)
                store_seconds += time.perf_counter() - started
                self._validate_reads(chunk.keys, results, chunk.sizes)
                started = time.perf_counter()
                for events, consumer_offset, consumer_count in tracked:
                    for layer_index in range(consumer_count):
                        consumer = self.layout.consumers[consumer_offset + layer_index]
                        try:
                            transfer_cache_ranges(
                                "h2d",
                                self.layout.buffers,
                                stash,
                                self.buffers.transfer_ranges(
                                    transfers[chunk.begin : chunk.end],
                                    set(consumer),
                                    host_base_offset=byte_offsets[chunk.begin],
                                ),
                                self.load_stream,
                                backend=self.transfer_backend,
                            )
                            finish = torch.cuda.Event()
                            finish.record(self.load_stream)
                        except RuntimeError as exc:
                            raise L3TransferError(
                                "L3 host-to-device restore failed"
                            ) from exc
                        events.layer_done_events[layer_index] = finish
                enqueue_seconds += time.perf_counter() - started
                chunks += 1
            if finish is None:
                raise L3TransferError("cache transfer layout has no layer consumers")
            self._append_ack(finish, op_ids, slot)
            retained = True
            logger.info(
                "L3 load host_pipeline objects=%s bytes=%s chunks=%s "
                "store_ms=%.3f enqueue_ms=%.3f",
                len(keys),
                total_bytes,
                chunks,
                store_seconds * 1000,
                enqueue_seconds * 1000,
            )
            return load_index
        finally:
            if not retained:
                self.buffers.release_stash(slot)

    def _record_layer_events(self, tracked) -> object:
        finish = None
        try:
            for events, _consumer_offset, consumer_count in tracked:
                for layer_index in range(consumer_count):
                    finish = torch.cuda.Event()
                    finish.record(self.load_stream)
                    events.layer_done_events[layer_index] = finish
        except RuntimeError as exc:
            raise L3TransferError("failed to record L3 load completion") from exc
        if finish is None:
            raise L3TransferError("cache transfer layout has no layer consumers")
        return finish

    def _append_ack(
        self, finish: object, op_ids: Sequence[int], slot: StashSlot | None
    ) -> None:
        with self._acks_lock:
            self._acks.append(_LoadAck(finish, tuple(op_ids), slot))

    def poll_results(self) -> list:
        from tokenspeed_scheduler import Cache

        results = []
        pending = []
        with self._acks_lock:
            for ack in self._acks:
                try:
                    finished = ack.finish_event.query()
                except RuntimeError as exc:
                    raise L3TransferError("failed to query L3 load completion") from exc
                if finished:
                    for op_id in ack.op_ids:
                        event = Cache.StoreLoadDoneEvent()
                        event.op_id = op_id
                        results.append(event)
                    if ack.stash_slot is not None:
                        self.buffers.release_stash(ack.stash_slot)
                else:
                    pending.append(ack)
            self._acks = pending
        return results

    def probe(self, content_hashes: Sequence[str]) -> dict[str, bool]:
        keys: list[str] = []
        owners: list[str] = []
        for content_hash in content_hashes:
            for group in self.layout.groups:
                for offset in range(group.cache_blocks_per_lcm_block):
                    keys.append(
                        store_key(
                            content_hash,
                            group.group_id,
                            offset,
                            tp_rank=self.tp_rank,
                            namespace=self.namespace,
                        )
                    )
                    owners.append(content_hash)
        exists = self.batcher.exists(keys)
        if len(exists) != len(keys):
            raise L3BackendError(
                f"L3 batch_exists length mismatch {len(exists)} != {len(keys)}"
            )
        outcome = {value: True for value in content_hashes}
        for owner, present in zip(owners, exists):
            outcome[owner] = outcome[owner] and int(present) == 1
        return outcome

    def abort(self, op_ids: Sequence[int]) -> None:
        key = set(op_ids)
        try:
            self.load_stream.synchronize()
        except RuntimeError as exc:
            raise L3TransferError("failed to synchronize aborted L3 load") from exc
        with self._acks_lock:
            retained = []
            for ack in self._acks:
                if key.intersection(ack.op_ids):
                    if ack.stash_slot is not None:
                        self.buffers.release_stash(ack.stash_slot)
                else:
                    retained.append(ack)
            self._acks = retained
        for tracker, _ in self.load_trackers:
            tracker.reset()

    def synchronize(self) -> None:
        try:
            self.write_stream.synchronize()
            self.load_stream.synchronize()
        except RuntimeError as exc:
            raise L3TransferError("failed to synchronize L3 CUDA streams") from exc
        with self._acks_lock:
            for ack in self._acks:
                if ack.stash_slot is not None:
                    self.buffers.release_stash(ack.stash_slot)
            self._acks.clear()

    def reset(self) -> None:
        self.synchronize()
        for tracker, _ in self.load_trackers:
            tracker.reset()
