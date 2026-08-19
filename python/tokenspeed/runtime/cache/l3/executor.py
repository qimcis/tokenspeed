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

"""Plan-level façade for the distributed L3 cache tier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tokenspeed.runtime.cache.l3.buffers import L3BufferManager
from tokenspeed.runtime.cache.l3.errors import (
    L3BackendError,
    L3Error,
    L3ShutdownError,
    L3SubmissionError,
    L3TransferError,
)
from tokenspeed.runtime.cache.l3.namespace import (
    build_store_namespace,
    fingerprint_cache_layout,
    fingerprint_model_artifacts,
    store_key,
)
from tokenspeed.runtime.cache.l3.store_io import L3StoreIO, StoreObjectKey
from tokenspeed.runtime.cache.l3.submissions import L3SubmissionTracker
from tokenspeed.runtime.cache.store.base import BaseKVStore
from tokenspeed.runtime.cache.store.errors import KVStoreShutdownError
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Preserve the helper API used by engine integration and external backends.
_build_store_namespace = build_store_namespace
_fingerprint_model_artifacts = fingerprint_model_artifacts
_fingerprint_cache_layout = fingerprint_cache_layout
_tp_aware_store_key = store_key


def _ordered_unique(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class L3CacheExecutor:
    """Translate scheduler cache plans into asynchronous L3 operations."""

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
        self._validate_config(
            io_backend=io_backend,
            max_stash_bytes=max_stash_bytes,
            store_probe_ttl=store_probe_ttl,
            io_workers=io_workers,
            direct_gpu=direct_gpu,
            direct_gpu_chunk_objects=direct_gpu_chunk_objects,
            host_pipeline_chunk_pages=host_pipeline_chunk_pages,
        )
        self.store: BaseKVStore | None = store
        self._write_store: BaseKVStore | None = store
        self.l2_executor = l2_executor
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        target_layout = device_pool.cache_transfer_layout()
        draft_layout = (
            draft_pool.cache_transfer_layout() if draft_pool is not None else None
        )
        scheduler_group_ids = tuple(
            spec.group_id for spec in device_pool.paged_cache_group_specs
        )
        cache_blocks_per_hash = self._cache_blocks_per_hash(
            device_pool, target_layout, scheduler_group_ids
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

        explicit_namespace = (
            store_namespace.strip()
            if isinstance(store_namespace, str) and store_namespace.strip()
            else None
        )
        abi = cache_abi_fingerprint or fingerprint_cache_layout(self.layout)
        if tp_size is not None and tp_size > 1:
            abi = f"{abi}_tp{tp_size}"
        self._store_namespace = build_store_namespace(
            model_id=model_id,
            model_revision=model_revision,
            model_fingerprint=fingerprint_model_artifacts(model_id),
            cache_abi_fingerprint=abi,
            extra_tag=explicit_namespace or store.extra_backend_tag,
        )
        store.extra_backend_tag = None

        self._buffers = L3BufferManager(self.layout, (store,), max_stash_bytes)
        pool_layouts = [(device_pool, target_layout)]
        if draft_pool is not None and self.layout is not target_layout:
            pool_layouts.append((draft_pool, draft_layout))
        self._io = L3StoreIO(
            store=store,
            write_store=store,
            layout=self.layout,
            buffers=self._buffers,
            pool_layouts=pool_layouts,
            transfer_backend="dma" if io_backend == "direct" else "auto",
            tp_rank=tp_rank,
            namespace=self._store_namespace,
            direct_gpu=direct_gpu,
            direct_chunk_objects=direct_gpu_chunk_objects,
            host_chunk_objects=host_pipeline_chunk_pages,
            cache_blocks_per_hash=cache_blocks_per_hash,
        )
        self._submissions = L3SubmissionTracker(
            io_workers=io_workers, presence_ttl=store_probe_ttl
        )
        self._closed = False
        logger.info(
            "L3 Store: enabled backend=%s groups=%s namespace=%s direct_gpu=%s",
            type(store).__name__,
            scheduler_group_ids,
            self._store_namespace,
            self._io.direct_gpu,
        )

    @staticmethod
    def _validate_config(**config) -> None:
        if config["io_backend"] not in ("direct", "kernel"):
            raise ValueError(f"unsupported L3 IO backend {config['io_backend']!r}")
        if config["max_stash_bytes"] <= 0:
            raise ValueError("L3 max_stash_bytes must be positive")
        if config["store_probe_ttl"] < 0:
            raise ValueError("L3 store_probe_ttl must be non-negative")
        if config["io_workers"] <= 0:
            raise ValueError("L3 io_workers must be positive")
        if config["direct_gpu"] not in ("auto", "on", "off"):
            raise ValueError("L3 direct_gpu must be one of: auto, on, off")
        if config["direct_gpu_chunk_objects"] <= 0:
            raise ValueError("L3 direct_gpu_chunk_objects must be positive")
        if config["host_pipeline_chunk_pages"] <= 0:
            raise ValueError("L3 host_pipeline_chunk_pages must be positive")

    @staticmethod
    def _cache_blocks_per_hash(
        device_pool, layout, group_ids: Sequence[str]
    ) -> tuple[int, ...]:
        contract = getattr(device_pool, "runtime_contract", None)
        if contract is None:
            return tuple(group.cache_blocks_per_lcm_block for group in layout.groups)
        specs = {spec.group_id: spec for spec in contract.group_specs}
        counts = []
        for group_id in group_ids:
            spec = specs[group_id]
            cache_block_tokens = int(spec.rows_per_page) * int(spec.entry_stride_tokens)
            if contract.block_size % cache_block_tokens:
                raise ValueError(
                    f"L3 cache group {group_id!r} block size does not divide "
                    "the scheduler hash span"
                )
            counts.append(contract.block_size // cache_block_tokens)
        return tuple(counts)

    @staticmethod
    def _metadata(operation: Any, transfer_count: int, *, required: bool):
        hashes = [str(value) for values in operation.content_hashes for value in values]
        offsets = [
            int(value) for values in operation.cache_block_offsets for value in values
        ]
        if not offsets:
            offsets = [0] * len(hashes)
        if required and len(hashes) != transfer_count:
            raise L3SubmissionError(
                "Store load metadata does not cover every transfer: "
                f"{len(hashes)} hashes for {transfer_count} transfers"
            )
        if len(offsets) != len(hashes):
            raise L3SubmissionError(
                "cache block offset metadata length does not match content hashes"
            )
        if len(hashes) < transfer_count:
            hashes.extend([""] * (transfer_count - len(hashes)))
            offsets.extend([0] * (transfer_count - len(offsets)))
        return hashes[:transfer_count], offsets[:transfer_count]

    @staticmethod
    def _append_transfers(
        operation_ids,
        group_ids,
        src_pages,
        dst_pages,
        *,
        op_ids: list[int],
        transfers: list[tuple[int, int, int]],
        source_is_device: bool,
    ) -> None:
        if not (
            len(operation_ids) == len(group_ids) == len(src_pages) == len(dst_pages)
        ):
            raise L3SubmissionError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            operation_ids, group_ids, src_pages, dst_pages
        ):
            if not (len(groups) == len(sources) == len(destinations)):
                raise L3SubmissionError(f"ragged cache operation {op_id}")
            op_ids.append(int(op_id))
            for group, source, destination in zip(groups, sources, destinations):
                device, host = (
                    (source, destination) if source_is_device else (destination, source)
                )
                transfers.append((int(group), int(device), int(host)))

    def submit_plan(
        self, plan: Any, *, cache_zero_event: object | None = None
    ) -> tuple[int, ...]:
        from tokenspeed_scheduler import Cache

        write_ids: list[int] = []
        write_transfers: list[tuple[int, int, int]] = []
        write_hashes: list[str] = []
        write_offsets: list[int] = []
        load_ids: list[int] = []
        load_transfers: list[tuple[int, int, int]] = []
        load_hashes: list[str] = []
        load_offsets: list[int] = []
        for operation in plan.cache:
            if isinstance(operation, Cache.WriteBackOp):
                before = len(write_transfers)
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    op_ids=write_ids,
                    transfers=write_transfers,
                    source_is_device=True,
                )
                hashes, offsets = self._metadata(
                    operation, len(write_transfers) - before, required=False
                )
                write_hashes.extend(hashes)
                write_offsets.extend(offsets)
            elif isinstance(operation, Cache.StoreLoadOp):
                before = len(load_transfers)
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    op_ids=load_ids,
                    transfers=load_transfers,
                    source_is_device=False,
                )
                hashes, offsets = self._metadata(
                    operation, len(load_transfers) - before, required=True
                )
                load_hashes.extend(hashes)
                load_offsets.extend(offsets)
            elif not isinstance(operation, Cache.LoadBackOp):
                raise L3SubmissionError(
                    f"unsupported cache op {type(operation).__name__}"
                )

        load_key = _ordered_unique(load_ids)
        if load_key:
            self._io.batcher.begin_read()
            try:
                self._submissions.submit_load(
                    load_key,
                    tuple(load_hashes),
                    lambda: self._run_load(
                        load_key,
                        tuple(load_transfers),
                        tuple(load_hashes),
                        tuple(load_offsets),
                        cache_zero_event,
                    ),
                )
            except L3SubmissionError:
                self._io.batcher.end_read()
                raise
        if write_ids and write_transfers:
            try:
                ready_event = torch.cuda.Event()
                ready_event.record()
            except RuntimeError as exc:
                raise L3TransferError("failed to record L3 write readiness") from exc
            objects = tuple(
                (int(transfer[0]), value, int(offset))
                for transfer, value, offset in zip(
                    write_transfers, write_hashes, write_offsets
                )
                if value
            )
            self._submissions.submit_write(
                objects,
                lambda: self._run_put(
                    tuple(write_transfers),
                    tuple(write_hashes),
                    tuple(write_offsets),
                    ready_event,
                ),
            )
        return load_key

    def _run_load(self, op_ids, transfers, hashes, offsets, zero_event) -> int:
        try:
            if zero_event is not None:
                try:
                    zero_event.synchronize()
                except RuntimeError as exc:
                    raise L3TransferError(
                        "failed to order L3 load after cache-page zeroing"
                    ) from exc
            load_index = self._io.load(op_ids, transfers, hashes, offsets)
            try:
                for tracker, _ in self._io.load_trackers:
                    tracker.set_consumers(load_index)
            except RuntimeError as exc:
                raise L3TransferError(
                    "failed to publish L3 load to cache consumers"
                ) from exc
            return load_index
        except L3Error:
            try:
                for tracker, _ in self._io.load_trackers:
                    tracker.set_consumers(-1)
            except RuntimeError as exc:
                raise L3TransferError("failed to cancel L3 cache consumers") from exc
            raise
        finally:
            self._io.batcher.end_read()

    def _run_put(
        self, transfers, hashes, offsets, ready_event
    ) -> dict[StoreObjectKey, bool]:
        requested = tuple(
            dict.fromkeys(
                (int(transfer[0]), value, int(offset))
                for transfer, value, offset in zip(transfers, hashes, offsets)
                if value
            )
        )
        try:
            try:
                ready_event.synchronize()
            except RuntimeError as exc:
                raise L3TransferError("failed to wait for L3 write readiness") from exc
            present = self._io.put(transfers, hashes, offsets)
        except (L3BackendError, L3TransferError) as exc:
            logger.warning("L3 put failed: %s", exc)
            present = {}
        return {value: bool(present.get(value, False)) for value in requested}

    def poll_results(self) -> list:
        self._submissions.poll_writes()
        return self._io.poll_results()

    def load_submission_status(self, op_ids: Sequence[int]):
        return self._submissions.load_status(_ordered_unique(op_ids))

    def acknowledge_load_submission(self, op_ids: Sequence[int]) -> None:
        self._submissions.acknowledge_load(_ordered_unique(op_ids))

    def abort_load_submission(self, op_ids: Sequence[int]) -> None:
        key = _ordered_unique(op_ids)
        if self._submissions.abort_load(key):
            self._io.batcher.end_read()
        self._io.abort(key)

    def probe_store_presence(self, content_hashes: Sequence[str]):
        return self._submissions.probe_presence(content_hashes, self._io.probe)

    def probe_store_objects(self, content_hashes: Sequence[str]):
        return self._submissions.probe_objects(content_hashes, self._io.probe_objects)

    def invalidate_presence(self, content_hashes: Sequence[str]) -> None:
        self._submissions.invalidate_presence(content_hashes)

    def record_presence(self, content_hashes: Sequence[str], *, present: bool) -> None:
        self._submissions.record_presence(content_hashes, present=present)

    def peek_store_index_outcomes(self) -> dict[StoreObjectKey, bool]:
        return self._submissions.outcomes()

    def acknowledge_store_index_outcomes(
        self, objects: Sequence[StoreObjectKey]
    ) -> None:
        self._submissions.acknowledge_outcomes(objects)

    def record_store_index_outcomes(self, outcomes: dict[StoreObjectKey, bool]) -> None:
        self._submissions.record_outcomes(outcomes)

    def shutdown(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        try:
            self._submissions.shutdown()
        except L3ShutdownError as exc:
            errors.append(f"submission pools: {exc}")
        try:
            self._io.synchronize()
        except L3TransferError as exc:
            errors.append(str(exc))
        stores = tuple(
            dict.fromkeys(
                store for store in (self._write_store, self.store) if store is not None
            )
        )
        self.store = None
        self._write_store = None
        for store in stores:
            try:
                with self._buffers.registration_lock:
                    store.close()
            except (KVStoreShutdownError, RuntimeError, OSError) as exc:
                errors.append(f"{type(store).__name__}: {exc}")
        self._buffers.clear()
        self._submissions.clear()
        self._closed = True
        if errors:
            raise L3ShutdownError("; ".join(errors))

    def reset(self) -> None:
        self._submissions.wait()
        self._io.reset()
        self._submissions.clear()
