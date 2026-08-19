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
StoreObjectKey = tuple[int, str, int]


@dataclass(frozen=True)
class TransferChunk:
    begin: int
    end: int
    keys: list[str]
    ptrs: list[Any]
    sizes: list[Any]


@dataclass(frozen=True)
class TransferLimits:
    max_objects: int
    max_bytes: int
    max_fragments: int


@dataclass
class WriteSnapshot:
    slot: StashSlot
    buffer: torch.Tensor
    ready_event: object
    transfers: tuple[tuple[int, int, int], ...]
    ptrs: tuple[int, ...]
    sizes: tuple[int, ...]
    nbytes: int


class _PriorityGate:
    def __init__(self, *, max_read_burst: int = 8) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._waiting_reads = 0
        self._waiting_writes = 0
        self._consecutive_reads = 0
        self._max_read_burst = max_read_burst

    @contextmanager
    def call(self, *, read: bool):
        with self._condition:
            if read:
                self._waiting_reads += 1
            else:
                self._waiting_writes += 1
            try:
                while (
                    self._active
                    or (
                        read
                        and self._waiting_writes
                        and self._consecutive_reads >= self._max_read_burst
                    )
                    or (
                        not read
                        and self._waiting_reads
                        and self._consecutive_reads < self._max_read_burst
                    )
                ):
                    self._condition.wait()
                self._active = True
                if read:
                    self._consecutive_reads += 1
                else:
                    self._consecutive_reads = 0
            finally:
                if read:
                    self._waiting_reads -= 1
                else:
                    self._waiting_writes -= 1
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

    @staticmethod
    def chunks(
        keys: Sequence[str],
        ptrs: Sequence[Any],
        sizes: Sequence[Any],
        limits: TransferLimits,
    ) -> Iterator[TransferChunk]:
        if not (len(keys) == len(ptrs) == len(sizes)):
            raise L3TransferError(
                f"Store vectors are ragged: keys={len(keys)} ptrs={len(ptrs)} sizes={len(sizes)}"
            )
        begin = 0
        while begin < len(keys):
            end = begin
            chunk_bytes = 0
            chunk_fragments = 0
            while end < len(keys):
                item_sizes = sizes[end]
                if isinstance(item_sizes, (list, tuple)):
                    item_bytes = sum(int(value) for value in item_sizes)
                    item_fragments = len(item_sizes)
                else:
                    item_bytes = int(item_sizes)
                    item_fragments = 1
                if item_bytes <= 0 or item_fragments <= 0:
                    raise L3TransferError(
                        f"Store object {end} has an empty transfer payload"
                    )
                if (
                    item_bytes > limits.max_bytes
                    or item_fragments > limits.max_fragments
                ):
                    raise L3TransferError(
                        "Store object exceeds one-call transfer limits: "
                        f"object={end} bytes={item_bytes}/{limits.max_bytes} "
                        f"fragments={item_fragments}/{limits.max_fragments}"
                    )
                exceeds = end > begin and (
                    end - begin >= limits.max_objects
                    or chunk_bytes + item_bytes > limits.max_bytes
                    or chunk_fragments + item_fragments > limits.max_fragments
                )
                if exceeds:
                    break
                chunk_bytes += item_bytes
                chunk_fragments += item_fragments
                end += 1
                if end - begin >= limits.max_objects:
                    break
            yield TransferChunk(
                begin,
                end,
                list(keys[begin:end]),
                list(ptrs[begin:end]),
                list(sizes[begin:end]),
            )
            begin = end

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
        limits: TransferLimits,
    ) -> tuple[list[int], int]:
        results: list[int] = []
        chunks = 0
        for chunk in self.chunks(keys, ptrs, sizes, limits):
            results.extend(self.get_chunk(chunk))
            chunks += 1
        return results, chunks

    def put_all(
        self,
        keys: Sequence[str],
        ptrs: Sequence[Any],
        sizes: Sequence[Any],
        limits: TransferLimits,
    ) -> list[int]:
        results: list[int] = []
        for chunk in self.chunks(keys, ptrs, sizes, limits):
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
        transfer_chunk_bytes: int,
        transfer_chunk_fragments: int,
        cache_blocks_per_hash: Sequence[int],
    ) -> None:
        self.store = store
        self.write_store = write_store
        self.layout = layout
        self.buffers = buffers
        self.batcher = StoreBatcher(store, write_store)
        self.transfer_backend = transfer_backend
        self.tp_rank = tp_rank
        self.namespace = namespace
        self.direct_limits = TransferLimits(
            direct_chunk_objects,
            transfer_chunk_bytes,
            transfer_chunk_fragments,
        )
        self.host_limits = TransferLimits(
            host_chunk_objects,
            transfer_chunk_bytes,
            transfer_chunk_fragments,
        )
        self.cache_blocks_per_hash = tuple(
            int(value) for value in cache_blocks_per_hash
        )
        if len(self.cache_blocks_per_hash) != len(self.layout.groups) or any(
            value <= 0 for value in self.cache_blocks_per_hash
        ):
            raise L3TransferError("invalid per-group L3 cache blocks per hash")
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

    def snapshot_write(
        self,
        transfers: Sequence[tuple[int, int, int]],
    ) -> WriteSnapshot:
        """Enqueue a bounded D2H copy before scheduler source pages are released."""
        from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges

        stable_transfers = tuple(transfers)
        object_sizes = [
            sum(field.payload_bytes for field in self.layout.groups[group_index].fields)
            for group_index, _device, _host in stable_transfers
        ]
        # Validate singleton object limits before allocating or copying.
        list(
            self.batcher.chunks(
                [""] * len(stable_transfers),
                [0] * len(stable_transfers),
                object_sizes,
                self.host_limits,
            )
        )
        ranges = self.buffers.transfer_ranges(stable_transfers)
        total_bytes = sum(byte_count for *_prefix, byte_count in ranges)
        if total_bytes <= 0:
            raise L3TransferError("L3 write snapshot has no payload")
        slot, stash = self.buffers.acquire_stash(total_bytes)
        retained = False
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
            ready = torch.cuda.Event()
            ready.record(self.write_stream)
            ptr_base = int(stash.data_ptr())
            ptrs: list[int] = []
            sizes: list[int] = []
            offset = 0
            for group_index, _device, _host in stable_transfers:
                nbytes = sum(
                    field.payload_bytes
                    for field in self.layout.groups[group_index].fields
                )
                ptrs.append(ptr_base + offset)
                sizes.append(nbytes)
                offset += nbytes
            retained = True
            return WriteSnapshot(
                slot=slot,
                buffer=stash,
                ready_event=ready,
                transfers=stable_transfers,
                ptrs=tuple(ptrs),
                sizes=tuple(sizes),
                nbytes=total_bytes,
            )
        except RuntimeError as exc:
            try:
                self.write_stream.synchronize()
            except RuntimeError as sync_exc:
                raise L3TransferError(
                    "L3 write stream failed while abandoning a snapshot"
                ) from sync_exc
            raise L3TransferError("L3 device-to-host snapshot failed") from exc
        finally:
            if not retained:
                self.buffers.release_stash(slot)

    def write_nbytes(self, transfers: Sequence[tuple[int, int, int]]) -> int:
        return sum(
            field.payload_bytes
            for group_index, _device, _host in transfers
            for field in self.layout.groups[group_index].fields
        )

    def release_write_snapshot(self, snapshot: WriteSnapshot) -> None:
        self.buffers.release_stash(snapshot.slot)

    def discard_write_snapshot(self, snapshot: WriteSnapshot) -> None:
        try:
            snapshot.ready_event.synchronize()
        except RuntimeError as exc:
            raise L3TransferError("failed to discard L3 write snapshot") from exc
        self.release_write_snapshot(snapshot)

    def put_snapshot(
        self,
        snapshot: WriteSnapshot,
        content_hashes: Sequence[str],
        offsets: Sequence[int],
    ) -> dict[StoreObjectKey, bool]:
        transfers = snapshot.transfers
        records = [
            (index, content_hash, self._key(content_hash, group, offsets[index]))
            for index, (group, _device, _host) in enumerate(transfers)
            if index < len(content_hashes) and (content_hash := content_hashes[index])
        ]
        if not records:
            return {}
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
        if missing:
            selected_indices = [records[index][0] for index in missing]
            results = self.batcher.put_all(
                [keys[index] for index in missing],
                [snapshot.ptrs[index] for index in selected_indices],
                [snapshot.sizes[index] for index in selected_indices],
                self.host_limits,
            )
            if len(results) != len(missing):
                raise L3BackendError(
                    f"L3 batch_put_from length mismatch {len(results)} != {len(missing)}"
                )
            for record_index, result in zip(missing, results):
                succeeded[record_index] = result == 0
        return {
            (
                int(transfers[index][0]),
                content_hash,
                int(offsets[index]),
            ): bool(success)
            for (index, content_hash, _key), success in zip(records, succeeded)
        }

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
        for chunk in self.batcher.chunks(keys, ptrs, field_sizes, self.direct_limits):
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
        load_index, tracked = self._begin_load()
        finish = None
        store_seconds = 0.0
        enqueue_seconds = 0.0
        chunks = 0
        chunk_plan = self.batcher.chunks(
            keys,
            [0] * len(keys),
            sizes,
            self.host_limits,
        )
        for planned in chunk_plan:
            chunk_sizes = [int(value) for value in planned.sizes]
            chunk_bytes = sum(chunk_sizes)
            slot, stash = self.buffers.acquire_stash(chunk_bytes)
            copy_enqueued = False
            copy_complete = False
            try:
                ptr_base = int(stash.data_ptr())
                byte_offsets: list[int] = []
                offset = 0
                for nbytes in chunk_sizes:
                    byte_offsets.append(offset)
                    offset += nbytes
                chunk = TransferChunk(
                    begin=planned.begin,
                    end=planned.end,
                    keys=planned.keys,
                    ptrs=[ptr_base + value for value in byte_offsets],
                    sizes=chunk_sizes,
                )
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
                                ),
                                self.load_stream,
                                backend=self.transfer_backend,
                            )
                            copy_enqueued = True
                            finish = torch.cuda.Event()
                            finish.record(self.load_stream)
                        except RuntimeError as exc:
                            raise L3TransferError(
                                "L3 host-to-device restore failed"
                            ) from exc
                        events.layer_done_events[layer_index] = finish
                enqueue_seconds += time.perf_counter() - started
                if finish is None:
                    raise L3TransferError(
                        "cache transfer layout has no layer consumers"
                    )
                try:
                    finish.synchronize()
                except RuntimeError as exc:
                    raise L3TransferError(
                        "L3 host-to-device chunk completion failed"
                    ) from exc
                copy_complete = True
                chunks += 1
            finally:
                if copy_enqueued and not copy_complete:
                    try:
                        self.load_stream.synchronize()
                    except RuntimeError as exc:
                        raise L3TransferError(
                            "failed to drain L3 load stream before releasing staging"
                        ) from exc
                self.buffers.release_stash(slot)
        if finish is None:
            raise L3TransferError("cache transfer layout has no layer consumers")
        self._append_ack(finish, op_ids, None)
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
        objects = self.probe_objects(content_hashes)
        outcome = {value: True for value in content_hashes}
        for (_group, content_hash, _offset), present in objects.items():
            outcome[content_hash] = outcome[content_hash] and present
        return outcome

    def probe_objects(
        self, content_hashes: Sequence[str]
    ) -> dict[StoreObjectKey, bool]:
        keys: list[str] = []
        owners: list[StoreObjectKey] = []
        for content_hash in content_hashes:
            for group_index, (group, blocks_per_hash) in enumerate(
                zip(self.layout.groups, self.cache_blocks_per_hash)
            ):
                for offset in range(blocks_per_hash):
                    keys.append(
                        store_key(
                            content_hash,
                            group.group_id,
                            offset,
                            tp_rank=self.tp_rank,
                            namespace=self.namespace,
                        )
                    )
                    owners.append((group_index, content_hash, offset))
        exists = self.batcher.exists(keys)
        if len(exists) != len(keys):
            raise L3BackendError(
                f"L3 batch_exists length mismatch {len(exists)} != {len(keys)}"
            )
        return {owner: int(present) == 1 for owner, present in zip(owners, exists)}

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
