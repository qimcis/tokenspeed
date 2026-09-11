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

"""Real TCP bounds/lifecycle tests with a thread-affine CPU draft engine."""

import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import zmq

from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    Busy,
    Close,
    DraftMessageCodec,
    DraftProtocolContract,
    Failure,
    Hello,
    MissingState,
    Open,
    Opened,
    Proposal,
    Ready,
    Update,
)
from tokenspeed.runtime.draft_pool.transport import (
    DraftPoolService,
    DraftPoolServiceConfig,
    TargetDraftClient,
    TargetDraftClientConfig,
)


def _contract():
    return DraftProtocolContract(
        protocol_version=1,
        target_model="test/target",
        target_revision="a" * 40,
        draft_model="test/draft",
        draft_revision="b" * 40,
        projection_fingerprint="c" * 64,
        feature_schema="dflash2.projected.bf16.v1",
        feature_dtype="bfloat16",
        feature_width=4,
        vocab_size=128,
        window_tokens=16,
        native_block_tokens=8,
        target_verify_tokens=6,
    )


def _config():
    return DraftPoolServiceConfig(
        listen_endpoint="tcp://127.0.0.1:*",
        resident_limit=3,
        staging_limit=2,
        max_queued_jobs=2,
        max_batch_size=2,
        max_peers=2,
        max_header_bytes=4096,
        max_feature_bytes=4096,
        max_host_memory_bytes=1024 * 1024,
        lease_ms=5000,
        heartbeat_ms=30,
        poll_ms=2,
        linger_ms=0,
    )


class _Engine:
    def __init__(self, control):
        self.contract = _contract()
        self.control = control
        self.thread_id = threading.get_ident()
        self.sessions = {}
        self.control.engine = self

    def _check(self):
        assert threading.get_ident() == self.thread_id

    def open_session(self, session_id):
        self._check()
        assert session_id not in self.sessions
        self.sessions[session_id] = None

    def install_features(
        self, session_id, feature_start, confirmed_endpoint, features, is_snapshot
    ):
        self._check()
        assert features.device.type == "cpu"
        if self.control.fail_install:
            raise RuntimeError("injected install failure")
        previous = self.sessions[session_id]
        if is_snapshot:
            assert previous is None
            assert feature_start == 0
        else:
            assert previous[0] == feature_start
        self.sessions[session_id] = (confirmed_endpoint, features.clone())

    def draft_batch(self, jobs):
        self._check()
        self.control.entered.set()
        assert self.control.gate.wait(3), "test did not release fake forward"
        if self.control.fail_proposal:
            self.control.fail_proposal = False
            raise RuntimeError("injected proposal failure")
        return [
            SimpleNamespace(
                session_id=job.session_id,
                confirmed_endpoint=job.confirmed_endpoint,
                anchor_token=job.anchor_token,
                candidate_ids=(1, 2, 3, 4, 5, 6, 7),
            )
            for job in jobs
        ]

    def close_session(self, session_id):
        self._check()
        self.sessions.pop(session_id, None)

    def close(self):
        self._check()
        self.sessions.clear()
        self.control.closed.set()


class _Inbox:
    def __init__(self, client):
        self.client = client
        self.messages = []

    def wait(self, kind, session_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.messages.extend(item.message for item in self.client.poll())
            for index, message in enumerate(self.messages):
                if isinstance(message, kind) and (
                    session_id is None
                    or getattr(message, "session_id", None) == session_id
                ):
                    return self.messages.pop(index)
            time.sleep(0.002)
        raise AssertionError(f"Missing {kind.__name__}/{session_id}: {self.messages}")


@contextmanager
def _running(config):
    control = SimpleNamespace(
        gate=threading.Event(),
        entered=threading.Event(),
        closed=threading.Event(),
        fail_install=False,
        fail_proposal=False,
        engine=None,
    )
    control.gate.set()
    service = DraftPoolService(config=config, engine_factory=lambda: _Engine(control))
    errors = []

    def run():
        try:
            service.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert service.started.wait(3), errors
    client = TargetDraftClient(
        config=TargetDraftClientConfig(
            endpoint=service.bound_endpoint,
            identity=b"test-cohort",
            max_pending_messages=8,
            max_pending_bytes=8192,
            max_reply_messages=32,
            max_header_bytes=4096,
            max_feature_bytes=4096,
            heartbeat_ms=30,
            reconnect_ms=20,
            poll_ms=2,
            linger_ms=0,
        ),
        contract=_contract(),
    )
    inbox = _Inbox(client)
    try:
        inbox.wait(Ready, None)
        yield service, client, inbox, control
    finally:
        control.gate.set()
        client.close()
        service.close()
        thread.join(3)
        assert not thread.is_alive()
        assert not errors
        assert control.closed.is_set()


def _open(client, inbox, session_id):
    assert client.submit(
        Open(
            session_id=session_id, confirmed_endpoint=2, anchor_token=9, history_start=0
        ),
        None,
    )
    return inbox.wait(Opened, session_id)


def _snapshot(session_id):
    return Update(
        session_id=session_id,
        confirmed_endpoint=2,
        anchor_token=9,
        feature_start=0,
        is_snapshot=True,
    )


def test_open_reserves_global_staging_before_allocating_snapshot():
    config = replace(_config(), staging_limit=1, max_batch_size=1)
    with _running(config) as (service, client, inbox, _):
        _open(client, inbox, "first")
        assert service.staging_slots == 1
        assert client.submit(
            Open(
                session_id="second",
                confirmed_endpoint=2,
                anchor_token=9,
                history_start=0,
            ),
            None,
        )
        assert "staging" in inbox.wait(Busy, "second").reason
        assert service.resident_sessions == 1
        assert client.submit(Close(session_id="first"), None)
        deadline = time.monotonic() + 2
        while service.resident_sessions and time.monotonic() < deadline:
            time.sleep(0.002)
        _open(client, inbox, "second")
        assert service.staging_slots == 1


def test_unadmitted_update_is_rejected_before_tensor_decode(monkeypatch):
    with _running(_config()) as (service, client, inbox, _):
        decoded_updates = []
        original = service._codec.decode

        def decode(frames):
            decoded_updates.append(frames)
            return original(frames)

        monkeypatch.setattr(service._codec, "decode", decode)
        assert client.submit(
            _snapshot("unopened"), torch.ones((2, 4), dtype=torch.bfloat16)
        )
        inbox.wait(MissingState, "unopened")
        assert decoded_updates == []
        assert service.staging_slots == service.resident_sessions == 0


def test_single_outstanding_transfer_and_owned_source_copy():
    with _running(_config()) as (service, client, inbox, control):
        _open(client, inbox, "one")
        control.gate.clear()
        features = torch.full((2, 4), 3, dtype=torch.bfloat16)
        assert client.submit(_snapshot("one"), features)
        features.fill_(99)  # Successful submit transferred ownership to copied frames.
        assert control.entered.wait(3)
        assert client.submit(_snapshot("one"), features)
        inbox.wait(Busy, "one")
        assert service.staging_slots == 1
        control.gate.set()
        assert inbox.wait(Ack, "one").confirmed_endpoint == 2
        assert inbox.wait(Proposal, "one").candidate_ids == (1, 2, 3, 4, 5, 6, 7)
        torch.testing.assert_close(
            control.engine.sessions["one"][1],
            torch.full((2, 4), 3, dtype=torch.bfloat16),
        )


def test_context_ack_survives_proposal_failure_and_next_delta():
    with _running(_config()) as (_, client, inbox, control):
        _open(client, inbox, "one")
        control.fail_proposal = True
        assert client.submit(_snapshot("one"), torch.ones((2, 4), dtype=torch.bfloat16))
        assert inbox.wait(Ack, "one").confirmed_endpoint == 2
        assert inbox.wait(Failure, "one").code == "proposal_failed"
        # A stale/failed proposal does not erase healthy installed worker context.
        assert client.submit(
            Update(
                session_id="one",
                confirmed_endpoint=3,
                anchor_token=10,
                feature_start=2,
                is_snapshot=False,
            ),
            torch.ones((1, 4), dtype=torch.bfloat16),
        )
        assert inbox.wait(Ack, "one").confirmed_endpoint == 3
        assert inbox.wait(Proposal, "one").anchor_token == 10


def test_install_failure_releases_session_and_reports_failure():
    with _running(_config()) as (service, client, inbox, control):
        _open(client, inbox, "one")
        control.fail_install = True
        assert client.submit(_snapshot("one"), torch.ones((2, 4), dtype=torch.bfloat16))
        assert inbox.wait(Failure, "one").code == "context_install_failed"
        deadline = time.monotonic() + 2
        while service.resident_sessions and time.monotonic() < deadline:
            time.sleep(0.002)
        assert service.resident_sessions == service.staging_slots == 0


def test_unsent_snapshot_reservation_expires_without_worker_allocation():
    with _running(replace(_config(), lease_ms=100)) as (
        service,
        client,
        inbox,
        control,
    ):
        _open(client, inbox, "unused")
        deadline = time.monotonic() + 2
        while service.resident_sessions and time.monotonic() < deadline:
            time.sleep(0.002)
        assert service.resident_sessions == service.staging_slots == 0
        assert control.engine.sessions == {}


def test_host_memory_bound_and_bad_capacity_fail_at_configuration():
    config = _config()
    assert config.max_host_bytes > config.staging_limit * config.max_feature_bytes
    with pytest.raises(ValueError, match="host bound"):
        replace(config, max_host_memory_bytes=config.max_host_bytes - 1)
    with pytest.raises(ValueError, match="full worker batch"):
        replace(config, staging_limit=1)
    with pytest.raises(ValueError, match="resident_limit"):
        replace(config, staging_limit=config.resident_limit + 1)


def test_submit_rejects_non_cpu_features_before_copy():
    with _running(_config()) as (_, client, inbox, _):
        _open(client, inbox, "one")
        with pytest.raises(ValueError, match="non-CPU"):
            client.submit(
                _snapshot("one"),
                torch.empty((2, 4), device="meta", dtype=torch.bfloat16),
            )
        assert client.pending_bytes == 0


def test_worker_queue_bound_counts_waiting_jobs_separately_from_active_batch():
    config = replace(_config(), staging_limit=3, max_queued_jobs=1, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "queued", "waiting"):
            _open(client, inbox, session_id)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("queued"), features)
        assert client.submit(_snapshot("waiting"), features)
        assert "queue" in inbox.wait(Busy, "waiting").reason
        assert service.queued_jobs == 1
        assert service.staging_slots == 3
        control.gate.set()
        inbox.wait(Proposal, "running")
        inbox.wait(Proposal, "queued")
        assert client.submit(_snapshot("waiting"), features)
        inbox.wait(Proposal, "waiting")


def test_client_byte_budget_rejects_before_queueing():
    with _running(_config()) as (_, client, _, _):
        limited = TargetDraftClient(
            config=replace(client.config, identity=b"limited", max_pending_bytes=1),
            contract=_contract(),
        )
        try:
            _Inbox(limited).wait(Ready, None)
            assert not limited.submit(
                Open(
                    session_id="one",
                    confirmed_endpoint=2,
                    anchor_token=9,
                    history_start=0,
                ),
                None,
            )
            assert limited.pending_bytes == 0
        finally:
            limited.close()


def test_extra_multipart_frames_are_drained_without_stopping_service():
    with _running(_config()) as (service, client, inbox, _):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(service.bound_endpoint)
        codec = DraftMessageCodec(
            contract=_contract(), max_header_bytes=4096, max_feature_bytes=4096
        )
        try:
            socket.send_multipart(
                codec.encode(Hello(contract=_contract()), None) + [b"extra"] * 20,
                copy=True,
            )
            assert socket.poll(2000, zmq.POLLIN)
            result = codec.decode(socket.recv_multipart(copy=True)).message
            assert isinstance(result, Failure)
            assert result.code == "invalid_message"
            _open(client, inbox, "still_healthy")
            assert service.resident_sessions == 1
        finally:
            socket.close(linger=0)
            context.term()


def test_engine_shutdown_failure_still_releases_network_resources():
    control = SimpleNamespace(engine=None, closed=threading.Event())

    class ClosingFailureEngine(_Engine):
        def close(self):
            super().close()
            raise RuntimeError("injected close failure")

    service = DraftPoolService(
        config=_config(), engine_factory=lambda: ClosingFailureEngine(control)
    )
    errors = []

    def run():
        try:
            service.run()
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert service.started.wait(3)
    endpoint = service.bound_endpoint
    service.close()
    thread.join(3)
    assert not thread.is_alive()
    assert control.closed.is_set()
    assert len(errors) == 1 and "close failure" in str(errors[0])
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    try:
        socket.bind(endpoint)
    finally:
        socket.close(linger=0)
        context.term()
