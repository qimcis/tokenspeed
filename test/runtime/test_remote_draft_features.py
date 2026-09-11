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

"""CPU checks of feature frontiers, graph metadata and asynchronous ownership."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.draft_pool.features import (
    AsyncFeatureExporter,
    FeatureExportDescriptor,
    RemoteFeatureCapture,
    committed_feature_interval,
)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [(0, (0, 0)), (2047, (0, 2047)), (2048, (1, 2048)), (262143, (260096, 262143))],
)
def test_anchor_never_enters_committed_feature_window(endpoint, expected):
    assert committed_feature_interval(endpoint, 2047) == expected


class _Projection:
    hidden_size = 2
    context_in_features = 4
    context_dtype = torch.bfloat16

    def project_target_hidden(self, hidden):
        # Nontrivial tap mixing catches pruning/reordering before projection.
        return hidden[:, :2] + 2 * hidden[:, 2:]


class _Cache:
    hidden_size = 2
    block_granularity = 4
    group_id = "features"

    def __init__(self):
        self.writes = []
        self.waits = 0

    def wait_ready(self):
        self.waits += 1

    def write(self, **kwargs):
        self.writes.append({name: value.clone() for name, value in kwargs.items()})


def _capture_fixture(max_bs, max_tokens):
    buffers = SimpleNamespace(
        max_bs=max_bs,
        device="cpu",
        positions_buf=torch.zeros(max_tokens, dtype=torch.int64),
        input_lengths_buf=torch.zeros(max_bs, dtype=torch.int32),
    )
    cache = _Cache()
    capture = RemoteFeatureCapture(_Projection(), cache, buffers, 32)
    return capture, buffers, cache


def test_chunked_prefill_and_mixed_rows_preserve_every_input_position():
    capture, buffers, cache = _capture_fixture(3, 12)
    buffers.input_lengths_buf[:] = torch.tensor([3, 1, 6])
    buffers.positions_buf[:10] = torch.tensor([9, 10, 11, 50, 70, 71, 72, 73, 74, 75])
    ctx = SimpleNamespace(bs=3, input_num_tokens=10, num_extends=2)
    table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    capture.prepare_batch(ctx, {"features": table}, 3)
    hidden = torch.arange(40, dtype=torch.bfloat16).view(10, 4)
    capture.capture(ctx, SimpleNamespace(hidden_states=hidden))
    written = cache.writes[0]
    assert written["positions"].tolist() == [9, 10, 11, 50, 70, 71, 72, 73, 74, 75]
    assert written["request_indices"].tolist() == [0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
    torch.testing.assert_close(
        written["projected_features"], hidden[:, :2] + 2 * hidden[:, 2:]
    )
    assert written["is_valid_token"].all()


def test_graph_padding_stays_null_after_larger_batch_and_width_change():
    capture, _, cache = _capture_fixture(4, 24)
    pointers = (capture.block_table.data_ptr(), capture.valid_requests.data_ptr())
    wide = SimpleNamespace(bs=4, input_num_tokens=24, num_extends=0)
    capture.prepare_batch(wide, {"features": torch.ones((4, 8), dtype=torch.int32)}, 4)
    narrow = SimpleNamespace(bs=4, input_num_tokens=4, num_extends=0)
    capture.prepare_batch(
        narrow, {"features": torch.tensor([[7, 8]], dtype=torch.int32)}, 1
    )
    capture.capture(
        narrow, SimpleNamespace(hidden_states=torch.ones((4, 4), dtype=torch.bfloat16))
    )
    assert (
        capture.block_table.data_ptr(),
        capture.valid_requests.data_ptr(),
    ) == pointers
    assert capture.block_table[0].tolist() == [7, 8, 0, 0, 0, 0, 0, 0]
    assert not capture.block_table[1:].any()
    assert cache.writes[-1]["request_indices"].tolist() == [0, 1, 2, 3]
    assert cache.writes[-1]["is_valid_token"].tolist() == [True, False, False, False]
    capture.prepare_batch(wide, {}, 0)
    capture.capture(
        wide, SimpleNamespace(hidden_states=torch.ones((24, 4), dtype=torch.bfloat16))
    )
    assert not cache.writes[-1]["is_valid_token"].any()


def test_missing_taps_fail_before_writing_cache():
    capture, _, cache = _capture_fixture(1, 6)
    ctx = SimpleNamespace(bs=1, input_num_tokens=6, num_extends=0)
    with pytest.raises(RuntimeError, match="complete ordered"):
        capture.capture(ctx, SimpleNamespace(hidden_states=torch.zeros((6, 2))))
    assert not cache.writes


def test_restore_generation_is_fenced_when_only_graph_metadata_is_refreshed():
    capture, _, cache = _capture_fixture(1, 6)
    ctx = SimpleNamespace(bs=1, input_num_tokens=6, num_extends=0)
    tables = {"features": torch.ones((1, 4), dtype=torch.int32)}
    capture.prepare_batch(ctx, tables, 1)
    # A graph replay never re-enters capture(); a new restore must still fence.
    capture.prepare_batch(ctx, tables, 1)
    assert cache.waits == 2


class _Lease:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1
        assert self.releases == 1


class _Event:
    def __init__(self):
        self.ready = False
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.ready


class _Stream:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class _ManualExporter(AsyncFeatureExporter):
    def __init__(self, max_pending, max_pending_bytes):
        self.stream = _Stream()
        super().__init__(2, 4, max_pending, max_pending_bytes, None, self.stream)
        self.events = []

    def _copy_to_host(self, pending, prerequisite_stream):
        assert prerequisite_stream == "target-execution"
        event = _Event()
        self.events.append(event)
        return pending.source.clone(), event


def _descriptor(ticket, session, start, end):
    return FeatureExportDescriptor(ticket, session, start, end, 123)


def test_cancellation_does_not_retire_source_or_make_staging_reusable_early():
    exporter = _ManualExporter(1, 16)
    old = _Lease()
    fresh = _Lease()
    rows = torch.arange(8, dtype=torch.bfloat16).view(4, 2)
    assert exporter.submit(_descriptor(1, "old", 7, 11), rows, "target-execution", old)
    exporter.cancel("old")
    assert exporter.poll() == []
    assert old.releases == 0
    assert exporter.pending_bytes == 16
    assert not exporter.submit(
        _descriptor(2, "new", 7, 11), rows, "target-execution", fresh
    )
    assert fresh.releases == 0
    assert exporter.stream.synchronizations == 0
    exporter.events[0].ready = True
    assert exporter.poll() == []
    assert old.releases == 1
    assert exporter.pending_count == 0
    assert exporter.submit(
        _descriptor(2, "new", 7, 11), rows, "target-execution", fresh
    )
    # An old session cancellation cannot invalidate a slot's new incarnation.
    exporter.cancel("old")
    exporter.events[1].ready = True
    ready = exporter.poll()
    assert len(ready) == 1 and ready[0].descriptor.session_id == "new"
    assert ready[0].features.device.type == "cpu"
    torch.testing.assert_close(ready[0].features, rows)
    assert fresh.releases == 1


def test_completed_copy_passes_unrelated_pending_copy_without_waiting():
    exporter = _ManualExporter(2, 32)
    leases = [_Lease(), _Lease()]
    rows = torch.ones((4, 2), dtype=torch.bfloat16)
    for i in range(2):
        assert exporter.submit(
            _descriptor(i, str(i), 0, 4), rows, "target-execution", leases[i]
        )
    exporter.events[1].ready = True
    assert [ready.descriptor.ticket_id for ready in exporter.poll()] == [1]
    assert [lease.releases for lease in leases] == [0, 1]
    assert exporter.pending_bytes == 16
    assert exporter.stream.synchronizations == 0
    exporter.shutdown()
    assert leases[0].releases == 1
    assert exporter.stream.synchronizations == 1
    assert exporter.pending_bytes == 0
    assert exporter.poll() == []
    rejected = _Lease()
    assert not exporter.submit(
        _descriptor(3, "3", 0, 4), rows, "target-execution", rejected
    )
    assert rejected.releases == 0


def test_byte_limit_bounds_ragged_exports_separately_from_count():
    exporter = _ManualExporter(4, 16)
    rows = torch.ones((3, 2), dtype=torch.bfloat16)
    assert exporter.submit(
        _descriptor(1, "a", 5, 8), rows, "target-execution", _Lease()
    )
    assert not exporter.submit(
        _descriptor(2, "b", 5, 8), rows, "target-execution", _Lease()
    )
    assert exporter.pending_count == 1
    assert exporter.pending_bytes == 12
    assert len(exporter.events) == 1


def test_export_rejects_rows_before_retained_window():
    exporter = _ManualExporter(1, 16)
    with pytest.raises(ValueError, match="confirmed window"):
        exporter.submit(
            _descriptor(1, "a", 4, 9),
            torch.zeros((5, 2), dtype=torch.bfloat16),
            "target-execution",
            _Lease(),
        )
    assert exporter.pending_count == 0


def test_failed_copy_retires_before_source_lease_release(monkeypatch):
    exporter = _ManualExporter(1, 16)
    lease = _Lease()

    def fail_copy(pending, prerequisite_stream):
        pending.host = pending.source.clone()
        raise RuntimeError("copy enqueue failed")

    monkeypatch.setattr(exporter, "_copy_to_host", fail_copy)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        exporter.submit(
            _descriptor(1, "a", 0, 4),
            torch.zeros((4, 2), dtype=torch.bfloat16),
            "target-execution",
            lease,
        )
    assert exporter.stream.synchronizations == 1
    assert lease.releases == 1
    assert exporter.pending_count == 0


def test_cancel_during_submission_retains_owner_until_published_event(monkeypatch):
    exporter = _ManualExporter(1, 16)
    lease = _Lease()
    entered = threading.Event()
    finish = threading.Event()
    event = _Event()

    def deferred_copy(pending, prerequisite_stream):
        pending.host = pending.source.clone()
        entered.set()
        assert finish.wait(5)
        return pending.host, event

    monkeypatch.setattr(exporter, "_copy_to_host", deferred_copy)
    result = []

    def submit():
        result.append(
            exporter.submit(
                _descriptor(1, "old", 0, 4),
                torch.zeros((4, 2), dtype=torch.bfloat16),
                "target-execution",
                lease,
            )
        )

    producer = threading.Thread(target=submit)
    producer.start()
    try:
        assert entered.wait(5)
        exporter.cancel("old")
        assert exporter.poll() == []
        assert lease.releases == 0
        assert not exporter.has_capacity(4)
    finally:
        finish.set()
        producer.join(5)
    assert not producer.is_alive()
    assert result == [True]
    event.ready = True
    assert exporter.poll() == []
    assert lease.releases == 1


def test_failed_retirement_keeps_source_pinned_and_stops_new_exports(monkeypatch):
    exporter = _ManualExporter(1, 16)
    lease = _Lease()

    def failed_copy(pending, prerequisite_stream):
        pending.host = pending.source.clone()
        raise RuntimeError("copy failed")

    def failed_retirement():
        raise RuntimeError("cannot prove copy retirement")

    monkeypatch.setattr(exporter, "_copy_to_host", failed_copy)
    monkeypatch.setattr(exporter.copy_stream, "synchronize", failed_retirement)
    with pytest.raises(RuntimeError, match="cannot prove"):
        exporter.submit(
            _descriptor(1, "old", 0, 4),
            torch.zeros((4, 2), dtype=torch.bfloat16),
            "target-execution",
            lease,
        )
    assert lease.releases == 0
    assert exporter.pending_bytes == 16
    assert exporter.owns_ticket(1)
    assert not exporter.has_capacity(0)
    assert exporter.poll() == []
    monkeypatch.setattr(exporter.copy_stream, "synchronize", lambda: None)
    exporter.shutdown()
    assert lease.releases == 1
    assert not exporter.owns_ticket(1)
