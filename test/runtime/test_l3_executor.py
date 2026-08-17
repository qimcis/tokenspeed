"""Distributed L3 Store cache executor regression tests."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _try_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except (ImportError, ModuleNotFoundError) as exc:
        raise unittest.SkipTest(f"needs runtime dependencies: {exc}") from exc


def _two_layer_executor(executor_module):
    """One cache group whose fields span two layer consumers, two transfers."""
    from tokenspeed.runtime.cache.transfer.layout import (
        CacheField,
        CacheGroupLayout,
        CacheTransferLayout,
    )

    def field(field_id: str, payload: int) -> CacheField:
        return CacheField(
            field_id=field_id,
            device_buffer_index=0,
            device_block_zero_offset_bytes=0,
            block_stride_bytes=1024,
            payload_bytes=payload,
        )

    fields = (
        field("layer.0.k", 100),
        field("layer.0.v", 100),
        field("layer.1.k", 100),
        field("layer.1.v", 100),
    )
    layout = CacheTransferLayout(
        num_lcm_blocks=16,
        groups=(
            CacheGroupLayout(group_id="full", cache_blocks_per_lcm_block=1, fields=fields),
        ),
        buffers=(object(),),
        consumers=(("layer.0.k", "layer.0.v"), ("layer.1.k", "layer.1.v")),
    )
    executor = executor_module.L3CacheExecutor.__new__(executor_module.L3CacheExecutor)
    executor.layout = layout
    return executor


class L3TransferRangeTest(unittest.TestCase):
    """Store-load host offsets must stay absolute within the packed stash.

    The stash produced by _do_store_get packs EVERY field of every transfer
    contiguously. Per-layer H2D copies filter by that layer's fields, so the
    filtered `_transfer_ranges` must still use the absolute offsets from the
    full packing (a regression test for corrupted multi-layer store loads).
    """

    def setUp(self):
        self.executor_module = _try_import("tokenspeed.runtime.cache.l3.executor")

    def test_filtered_ranges_keep_absolute_stash_offsets(self):
        executor = _two_layer_executor(self.executor_module)
        transfers = [(0, 5, -1), (0, 6, -1)]
        group = executor.layout.groups[0]

        # Reference packing: per transfer, all fields in group order.
        per_transfer_bytes = sum(field.payload_bytes for field in group.fields)
        expected_offsets: dict[str, list[int]] = {}
        cursor = 0
        for _transfer in transfers:
            for field in group.fields:
                expected_offsets.setdefault(field.field_id, []).append(cursor)
                cursor += field.payload_bytes
        self.assertEqual(cursor, per_transfer_bytes * len(transfers))

        for consumer in (
            {"layer.0.k", "layer.0.v"},
            {"layer.1.k", "layer.1.v"},
        ):
            with self.subTest(consumer=consumer):
                ranges = executor._transfer_ranges(transfers, consumer)
                got: dict[str, list[int]] = {}
                for _buffer, _device_block, host_offset, num_bytes in ranges:
                    field_id = _field_id_at(executor, host_offset, per_transfer_bytes)
                    got.setdefault(field_id, []).append(host_offset)
                    self.assertEqual(
                        num_bytes,
                        next(
                            f.payload_bytes for f in group.fields if f.field_id == field_id
                        ),
                    )
                for field_id in consumer:
                    self.assertEqual(
                        got[field_id],
                        expected_offsets[field_id],
                        msg=f"field {field_id} must read the packed-stash offsets",
                    )

    def test_unfiltered_ranges_are_fully_contiguous(self):
        executor = _two_layer_executor(self.executor_module)
        transfers = [(0, 5, -1), (0, 6, -1)]
        ranges = executor._transfer_ranges(transfers)
        self.assertEqual(
            [host_offset for _, _, host_offset, _ in ranges],
            [0, 100, 200, 300, 400, 500, 600, 700],
        )


def _field_id_at(executor, host_offset: int, per_transfer_bytes: int) -> str:
    offset = host_offset % per_transfer_bytes
    cursor = 0
    for field in executor.layout.groups[0].fields:
        if cursor == offset:
            return field.field_id
        cursor += field.payload_bytes
    raise AssertionError(f"no field at packed stash offset {host_offset}")


class L3StashTest(unittest.TestCase):
    """Stash staging buffers must satisfy the pinned-host transfer contract."""

    def setUp(self):
        self.executor_module = _try_import("tokenspeed.runtime.cache.l3.executor")

    def test_stash_is_pinned_registered_and_reused(self):
        import torch

        store = Mock()
        store.register_buffer.return_value = 0
        executor = self.executor_module.L3CacheExecutor.__new__(
            self.executor_module.L3CacheExecutor
        )
        executor.store = store
        executor._stash_slots = []
        executor._registered_ptrs = set()

        slot = executor._allocate_stash(4096)

        self.assertIs(slot.busy, True)
        buffer = slot.buffer
        self.assertEqual(buffer.dtype, torch.uint8)
        self.assertTrue(buffer.is_contiguous())
        self.assertTrue(buffer.is_pinned())
        store.register_buffer.assert_called_once_with(
            int(buffer.data_ptr()), int(buffer.numel() * buffer.element_size())
        )
        # First-fit reuse must not allocate or register again.
        reused, _ = executor._acquire_stash(2048)
        self.assertIs(reused, slot)

    def test_stash_registration_failure_raises_and_releases(self):
        store = Mock()
        store.register_buffer.return_value = -1
        executor = self.executor_module.L3CacheExecutor.__new__(
            self.executor_module.L3CacheExecutor
        )
        executor.store = store
        executor._stash_slots = []
        executor._registered_ptrs = set()

        with self.assertRaises(RuntimeError):
            executor._allocate_stash(4096)
        self.assertFalse(executor._stash_slots[0].busy)


class PoolMultiTrackerTest(unittest.TestCase):
    """Pools must keep every cache tier's load tracker (L2 and L3)."""

    def setUp(self):
        _try_import("tokenspeed.runtime.layers.attention.kv_cache.base")
        from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool

        self.CachePool = CachePool

    def _new_pool(self) -> object:
        pool = self.CachePool.__new__(self.CachePool)
        pool.layerwise_load_tracker = None
        pool._layerwise_load_trackers = []
        return pool

    def test_pool_waits_on_all_registered_trackers(self):
        pool = self._new_pool()
        first = Mock()
        second = Mock()
        pool.register_layerwise_load_tracker(first)
        pool.register_layerwise_load_tracker(second)

        pool.wait_for_layerwise_load(3)

        first.wait_for_layer.assert_called_once_with(3)
        second.wait_for_layer.assert_called_once_with(3)

    def test_pool_keeps_first_tracker_for_direct_readers(self):
        pool = self._new_pool()
        first = Mock()
        second = Mock()
        pool.register_layerwise_load_tracker(first)
        pool.register_layerwise_load_tracker(second)

        self.assertIs(pool.layerwise_load_tracker, first)
        self.assertEqual(pool._layerwise_load_trackers, [first, second])

    def test_pool_direct_assignment_still_waits(self):
        pool = self._new_pool()
        tracker = Mock()
        pool.layerwise_load_tracker = tracker

        pool.wait_for_layerwise_load(3)

        tracker.wait_for_layer.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()