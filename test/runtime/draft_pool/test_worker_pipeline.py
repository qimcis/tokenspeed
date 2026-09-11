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

"""CPU orchestration checks; fake CUDA events do not qualify device overlap."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.draft_pool.worker import (
    DFlash2WorkerEngine,
    WorkerDraftJob,
    WorkerFeatureUpdate,
    WorkerSessionTable,
    worker_pipeline_buffer_bytes,
)
from tokenspeed.runtime.draft_pool.worker_cache import WorkerCacheGeometry


class _Event:
    def __init__(self, cuda):
        self.cuda = cuda
        self.ready = False
        self.recorded_stream = None
        cuda.events.append(self)

    def record(self, stream):
        self.recorded_stream = stream
        self.ready = False
        self.cuda.operations.append(("record", self, stream))

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True
        self.cuda.synchronizations.append(self)


class _Stream:
    def __init__(self, cuda):
        self.cuda = cuda
        cuda.streams.append(self)

    def wait_event(self, event):
        self.cuda.operations.append(("wait_event", self, event))

    def wait_stream(self, stream):
        self.cuda.operations.append(("wait_stream", self, stream))

    def synchronize(self):
        self.cuda.synchronizations.append(self)


@pytest.fixture
def pipeline(monkeypatch):
    cuda = SimpleNamespace(
        events=[], streams=[], operations=[], synchronizations=[], allocations=[]
    )
    cuda.current = _Stream(cuda)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: _Stream(cuda))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: _Event(cuda))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: cuda.current)

    @contextmanager
    def stream_context(stream):
        previous = cuda.current
        cuda.current = stream
        try:
            yield
        finally:
            cuda.current = previous

    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    original_empty = torch.empty

    def empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        tensor = original_empty(*args, **kwargs)
        cuda.allocations.append((tensor, pinned))
        return tensor

    monkeypatch.setattr(torch, "empty", empty)
    engine = object.__new__(DFlash2WorkerEngine)
    engine._thread_id = threading.get_ident()
    engine._closed = False
    engine.config = SimpleNamespace(device_id=0, max_batch_size=3)
    engine.contract = SimpleNamespace(feature_width=2, vocab_size=64)
    engine.geometry = WorkerCacheGeometry(4, 8, 8)
    engine.sessions = WorkerSessionTable(4, 7)
    engine.device = "cpu"
    engine.max_position = 1048576
    engine.pool = SimpleNamespace(clear_slot=lambda slot: None)
    writes = []

    def write_context_kv(features, positions, locations, pool):
        cuda.operations.append(("write_context", cuda.current))
        writes.append((features.clone(), positions.clone(), locations.clone()))

    engine.model = SimpleNamespace(write_context_kv=write_context_kv)
    engine._init_pipeline()
    return engine, cuda, writes


def _features(start, endpoint):
    positions = torch.arange(start, endpoint, dtype=torch.float32)
    return torch.stack((positions, -positions), dim=1).bfloat16()


def _update(session_id, start, endpoint, is_snapshot):
    return WorkerFeatureUpdate(
        session_id, start, endpoint, _features(start, endpoint), is_snapshot
    )


def _complete(cuda):
    for event in cuda.events:
        event.ready = True


def test_packed_context_preserves_absolute_positions_and_session_rings(pipeline):
    engine, cuda, writes = pipeline
    a = engine.sessions.open("delta")
    a.confirmed_endpoint = 5
    b = engine.sessions.open("snapshot")
    engine.sessions.open("empty")
    updates = (
        _update("delta", 5, 10, False),
        _update("snapshot", 100, 107, True),
        _update("empty", 0, 0, True),
    )
    ticket = engine.stage_batch(updates)
    assert writes == []
    assert a.confirmed_endpoint == 5
    assert b.confirmed_endpoint is None
    engine.launch_batch(ticket, [])
    assert len(writes) == 1
    features, positions, locations = writes[0]
    torch.testing.assert_close(
        features, torch.cat([update.features for update in updates]), rtol=0, atol=0
    )
    assert positions.tolist() == list(range(5, 10)) + list(range(100, 107))
    assert locations.tolist() == engine.geometry.context_locations(
        a.slot, 5, 10
    ) + engine.geometry.context_locations(b.slot, 100, 107)
    assert engine.poll_batch(ticket) is None
    assert a.confirmed_endpoint == 5
    assert b.confirmed_endpoint is None
    assert cuda.synchronizations == []
    assert any(operation[0] == "wait_event" for operation in cuda.operations)
    assert ("write_context", engine._compute_stream) in cuda.operations
    slot = engine._staging[engine._batches[ticket].slot]
    assert slot.upload_ready.recorded_stream is engine._upload_stream
    assert slot.completed.recorded_stream is engine._compute_stream
    assert engine._upload_stream is not engine._compute_stream
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ("delta", "snapshot", "empty")
    assert completion.results == ()
    assert completion.install_error is completion.draft_error is None
    assert a.confirmed_endpoint == 10
    assert b.confirmed_endpoint == 107
    assert engine.sessions.get("empty").confirmed_endpoint == 0


def test_staging_owns_input_snapshot_before_caller_reuses_payload(pipeline):
    engine, cuda, writes = pipeline
    engine.sessions.open("a")
    update = _update("a", 0, 3, True)
    expected = update.features.clone()
    ticket = engine.stage_batch([update])
    update.features.fill_(99)
    engine.launch_batch(ticket, [])
    torch.testing.assert_close(writes[0][0], expected, rtol=0, atol=0)
    _complete(cuda)
    assert engine.poll_batch(ticket).installed == ("a",)


def test_second_batch_upload_does_not_mutate_active_compute_state(pipeline):
    engine, cuda, writes = pipeline
    for session_id in ("a", "b", "c"):
        engine.sessions.open(session_id)
    forwarded = []

    def forward_tokens(jobs):
        forwarded.extend(jobs)
        return torch.arange(8, dtype=torch.int32).view(1, 8)

    engine._forward_tokens = forward_tokens
    jobs = [WorkerDraftJob("a", 3, 0)]
    first = engine.stage_batch([_update("a", 0, 3, True)])
    engine.launch_batch(first, jobs)
    second = engine.stage_batch([_update("b", 0, 2, True)])
    assert len(writes) == 1
    assert forwarded == jobs
    assert engine.poll_batch(first) is None
    with pytest.raises(RuntimeError):
        engine.launch_batch(second, [])
    with pytest.raises(RuntimeError):
        engine.stage_batch([_update("c", 0, 1, True)])
    assert cuda.synchronizations == []
    _complete(cuda)
    assert engine.poll_batch(first).installed == ("a",)
    engine.launch_batch(second, [])
    assert len(writes) == 2
    # A third transfer can use the first slot only after its completion retired.
    third = engine.stage_batch([_update("c", 0, 1, True)])
    assert third not in (first, second)
    _complete(cuda)
    assert engine.poll_batch(second).installed == ("b",)
    engine.launch_batch(third, [])
    _complete(cuda)
    assert engine.poll_batch(third).installed == ("c",)
    assert len(writes) == 3


def test_pending_tickets_prevent_session_release_and_duplicate_updates(pipeline):
    engine, cuda, _ = pipeline
    engine.sessions.open("a")
    ticket = engine.stage_batch([_update("a", 0, 2, True)])
    with pytest.raises(RuntimeError):
        engine.close_session("a")
    with pytest.raises((RuntimeError, ValueError)):
        engine.stage_batch([_update("a", 0, 2, True)])
    engine.launch_batch(ticket, [])
    with pytest.raises(RuntimeError):
        engine.close_session("a")
    _complete(cuda)
    engine.poll_batch(ticket)
    engine.close_session("a")
    with pytest.raises(ValueError, match="missing"):
        engine.sessions.get("a")


def test_invalid_later_update_is_rejected_before_any_batch_is_staged(pipeline):
    engine, _, writes = pipeline
    engine.sessions.open("valid")
    engine.sessions.open("invalid").confirmed_endpoint = 5
    with pytest.raises(ValueError, match="installed endpoint"):
        engine.stage_batch(
            [_update("valid", 0, 3, True), _update("invalid", 4, 6, False)]
        )
    assert writes == []
    assert engine.sessions.get("valid").confirmed_endpoint is None
    assert engine._batches == {}
    assert len(engine._free_staging) == 2


@pytest.mark.parametrize("invalid", ("dtype", "width", "range", "duplicate"))
def test_invalid_batch_schema_is_rejected_before_staging(pipeline, invalid):
    engine, _, writes = pipeline
    engine.sessions.open("a")
    update = _update("a", 0, 2, True)
    if invalid == "dtype":
        update = WorkerFeatureUpdate("a", 0, 2, update.features.float(), True)
    elif invalid == "width":
        update = WorkerFeatureUpdate("a", 0, 2, update.features[:, :1], True)
    elif invalid == "range":
        endpoint = engine.max_position - engine.geometry.native_block_tokens + 1
        update = _update("a", endpoint - engine.geometry.history_tokens, endpoint, True)
    updates = [update, update] if invalid == "duplicate" else [update]
    with pytest.raises(ValueError):
        engine.stage_batch(updates)
    assert writes == []
    assert engine._batches == {}


def test_launch_revalidates_endpoint_before_writing_staged_context(pipeline):
    engine, cuda, writes = pipeline
    session = engine.sessions.open("a")
    session.confirmed_endpoint = 5
    ticket = engine.stage_batch([_update("a", 5, 7, False)])
    session.confirmed_endpoint = 6
    engine.launch_batch(ticket, [])
    assert writes == []
    assert engine.poll_batch(ticket) is None
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ()
    assert completion.results == ()
    assert completion.install_error is not None


def test_empty_snapshot_has_ack_without_zero_row_context_kernels(pipeline):
    engine, cuda, writes = pipeline
    engine.sessions.open("a")
    ticket = engine.stage_batch([_update("a", 0, 0, True)])
    engine.launch_batch(ticket, [])
    assert writes == []
    assert engine.poll_batch(ticket) is None
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ("a",)
    assert completion.install_error is None
    assert engine.sessions.get("a").confirmed_endpoint == 0


def test_completion_preserves_native_proposal_job_order(pipeline):
    engine, cuda, _ = pipeline
    engine.sessions.open("a")
    engine.sessions.open("b")
    updates = [_update("a", 0, 2, True), _update("b", 0, 3, True)]
    jobs = [WorkerDraftJob("b", 3, 23), WorkerDraftJob("a", 2, 13)]
    forwarded = []

    def forward_tokens(forward_jobs):
        forwarded.extend(forward_jobs)
        return torch.tensor(
            [[job.anchor_token + i for i in range(8)] for job in forward_jobs],
            dtype=torch.int32,
        )

    engine._forward_tokens = forward_tokens
    ticket = engine.stage_batch(updates)
    engine.launch_batch(ticket, jobs)
    assert forwarded == jobs
    assert engine.poll_batch(ticket) is None
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ("a", "b")
    assert [result.session_id for result in completion.results] == ["b", "a"]
    assert [result.confirmed_endpoint for result in completion.results] == [3, 2]
    assert [result.anchor_token for result in completion.results] == [23, 13]
    assert [result.candidate_ids for result in completion.results] == [
        tuple(range(24, 31)),
        tuple(range(14, 21)),
    ]


def test_draft_failure_retains_successful_install_only_after_completion(pipeline):
    engine, cuda, _ = pipeline
    session = engine.sessions.open("a")

    def fail_draft(jobs):
        raise RuntimeError("selector failure after context write")

    engine._forward_tokens = fail_draft
    ticket = engine.stage_batch([_update("a", 0, 3, True)])
    engine.launch_batch(ticket, [WorkerDraftJob("a", 3, 7)])
    assert session.confirmed_endpoint is None
    assert engine.poll_batch(ticket) is None
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ("a",)
    assert completion.results == ()
    assert completion.install_error is None
    assert "selector failure" in completion.draft_error
    assert session.confirmed_endpoint == 3


def test_synchronous_install_wrapper_uses_same_packed_writer(pipeline):
    engine, cuda, writes = pipeline
    engine.sessions.open("a")
    features = _features(0, 3)
    engine.install_features("a", 0, 3, features, True)
    assert len(writes) == 1
    torch.testing.assert_close(writes[0][0], features, rtol=0, atol=0)
    assert engine.sessions.get("a").confirmed_endpoint == 3
    assert engine._batches == {}
    assert len(cuda.synchronizations) == 1


def test_failed_upload_retires_issued_copies_before_slot_reuse(pipeline, monkeypatch):
    engine, cuda, writes = pipeline
    engine.sessions.open("a")
    original_copy = torch.Tensor.copy_
    uploads = []

    def copy(destination, source, **kwargs):
        if kwargs.get("non_blocking", False):
            uploads.append(destination)
            if len(uploads) == 2:
                raise RuntimeError("second upload failed")
        return original_copy(destination, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError, match="second upload failed"):
        engine.stage_batch([_update("a", 0, 3, True)])
    assert len(uploads) == 2
    assert cuda.synchronizations == [engine._upload_stream]
    assert engine._batches == {}
    assert len(engine._free_staging) == 2
    assert engine.sessions.get("a").confirmed_endpoint is None
    assert writes == []
    monkeypatch.setattr(torch.Tensor, "copy_", original_copy)
    ticket = engine.stage_batch([_update("a", 0, 3, True)])
    engine.launch_batch(ticket, [])
    _complete(cuda)
    assert engine.poll_batch(ticket).installed == ("a",)


def test_context_write_failure_holds_slot_and_session_until_completion(pipeline):
    engine, cuda, _ = pipeline
    session = engine.sessions.open("a")
    session.confirmed_endpoint = 5

    def failed_write(features, positions, locations, pool):
        raise RuntimeError("partial context write")

    engine.model.write_context_kv = failed_write
    ticket = engine.stage_batch([_update("a", 5, 7, False)])
    engine.launch_batch(ticket, [])
    assert engine.poll_batch(ticket) is None
    assert len(engine._free_staging) == 1
    with pytest.raises(RuntimeError):
        engine.close_session("a")
    assert session.confirmed_endpoint == 5
    assert cuda.synchronizations == []
    _complete(cuda)
    completion = engine.poll_batch(ticket)
    assert completion.installed == ()
    assert completion.results == ()
    assert "partial context write" in completion.install_error
    assert len(engine._free_staging) == 2
    # The service invalidates the potentially overwritten ring after retirement.
    engine.close_session("a")


def test_staging_storage_is_preallocated_pinned_and_reused_after_retirement(pipeline):
    engine, cuda, _ = pipeline
    engine.sessions.open("a")
    initial_allocations = len(cuda.allocations)
    for slot in engine._staging:
        for tensor in (
            slot.host_features,
            slot.host_positions,
            slot.host_locations,
            slot.host_tokens,
        ):
            assert any(
                allocated is tensor and pinned for allocated, pinned in cuda.allocations
            )
    pointers = [
        (slot.host_features.data_ptr(), slot.device_features.data_ptr())
        for slot in engine._staging
    ]
    tickets = []
    for endpoint in (2, 4, 6):
        start = max(0, endpoint - 2)
        ticket = engine.stage_batch([_update("a", start, endpoint, endpoint == 2)])
        tickets.append(ticket)
        engine.launch_batch(ticket, [])
        _complete(cuda)
        engine.poll_batch(ticket)
    assert len(set(tickets)) == 3
    assert len(cuda.allocations) == initial_allocations
    assert pointers == [
        (slot.host_features.data_ptr(), slot.device_features.data_ptr())
        for slot in engine._staging
    ]


def test_pipeline_budget_matches_actual_host_and_device_allocations(pipeline):
    engine, cuda, _ = pipeline
    expected_host, expected_device = worker_pipeline_buffer_bytes(
        engine.config.max_batch_size,
        engine.geometry.history_tokens,
        engine.contract.feature_width,
        engine.geometry.native_block_tokens,
    )
    actual_host = sum(
        tensor.numel() * tensor.element_size()
        for tensor, pinned in cuda.allocations
        if pinned
    )
    actual_device = sum(
        tensor.numel() * tensor.element_size()
        for tensor, pinned in cuda.allocations
        if not pinned
    )
    assert expected_host == actual_host
    assert expected_device == actual_device
    # The advertised eight-request/full-window case includes metadata and IDs,
    # rather than charging only the two BF16 feature planes.
    assert worker_pipeline_buffer_bytes(8, 2047, 6144, 8) == (
        402981056,
        402980608,
    )
