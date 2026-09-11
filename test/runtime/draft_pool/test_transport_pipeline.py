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

"""Real TCP scheduling tests for independently staged worker batches."""

import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from test.runtime.test_draft_pool_transport import (
    _config,
    _contract,
    _Inbox,
    _open,
    _running,
    _snapshot,
)

import pytest
import torch
import zmq

from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    Busy,
    Close,
    DraftMessageCodec,
    Hello,
    Open,
    Proposal,
    Ready,
)
from tokenspeed.runtime.draft_pool.transport import TargetDraftClient


def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate(), "Timed out waiting for pipeline state"


@contextmanager
def _quiet_client():
    """Handshake against a real socket with no periodic application traffic."""
    from tokenspeed.runtime.draft_pool.transport import TargetDraftClientConfig

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.setsockopt(zmq.RCVHWM, 1)
    router.bind("tcp://127.0.0.1:*")
    codec = DraftMessageCodec(
        contract=_contract(), max_header_bytes=4096, max_feature_bytes=4096
    )
    client = TargetDraftClient(
        config=TargetDraftClientConfig(
            endpoint=router.getsockopt_string(zmq.LAST_ENDPOINT),
            identity=b"quiet-client",
            max_pending_messages=64,
            max_pending_bytes=262144,
            max_reply_messages=32,
            max_header_bytes=4096,
            max_feature_bytes=4096,
            heartbeat_ms=5000,
            reconnect_ms=20,
            poll_ms=2000,
            linger_ms=0,
        ),
        contract=_contract(),
    )
    try:
        assert router.poll(3000, zmq.POLLIN)
        identity, *frames = router.recv_multipart()
        assert isinstance(codec.decode(frames).message, Hello)
        router.send_multipart(
            [
                identity,
                *codec.encode(
                    Ready(
                        contract=_contract(),
                        resident_limit=64,
                        staging_limit=64,
                        max_batch_size=1,
                        lease_ms=10000,
                    ),
                    None,
                ),
            ]
        )
        _Inbox(client).wait(Ready, None)
        # Let the I/O thread settle into an otherwise two-second poll.
        time.sleep(0.03)
        yield client, router, codec
    finally:
        client.close()
        router.close(linger=0)
        context.term()


def test_client_submit_wakes_quiet_long_poll():
    with _quiet_client() as (client, router, codec):
        assert client.submit(
            Open(
                session_id="wake",
                confirmed_endpoint=2,
                anchor_token=9,
                history_start=0,
            ),
            None,
        )
        assert router.poll(800, zmq.POLLIN), "Queued send waited for poll_ms"
        _, *frames = router.recv_multipart()
        assert codec.decode(frames).message.session_id == "wake"


def test_client_drains_backpressured_queue_without_incoming_replies():
    with _quiet_client() as (client, router, codec):
        for index in range(24):
            assert client.submit(
                Open(
                    session_id=f"queued-{index}",
                    confirmed_endpoint=2,
                    anchor_token=9,
                    history_start=0,
                ),
                None,
            )
        received = []
        deadline = time.monotonic() + 0.8
        while len(received) < 24 and time.monotonic() < deadline:
            if router.poll(10, zmq.POLLIN):
                _, *frames = router.recv_multipart()
                received.append(codec.decode(frames).message.session_id)
        assert received == [f"queued-{index}" for index in range(24)]
        _wait_until(lambda: client.pending_bytes == 0, 0.5)


def test_client_close_wakes_quiet_long_poll():
    with _quiet_client() as (client, _, _):
        started = time.monotonic()
        client.close()
        assert time.monotonic() - started < 0.8
        assert not client._thread.is_alive()


def test_next_batch_is_staged_while_current_forward_owns_kv():
    config = replace(_config(), staging_limit=2, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        _open(client, inbox, "running")
        _open(client, inbox, "next")
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        assert control.launched == [("running",)]
        assert control.completed == []
        assert control.engine.sessions["next"] is None
        assert service.staging_slots == 2
        # An upload never makes a still-running session eligible for another update.
        assert client.submit(_snapshot("running"), features)
        assert "outstanding" in inbox.wait(Busy, "running").reason
        assert client.submit(
            Open(
                session_id="overflow",
                confirmed_endpoint=2,
                anchor_token=9,
                history_start=0,
            ),
            None,
        )
        assert "staging" in inbox.wait(Busy, "overflow").reason
        control.gate.set()
        for session_id in ("running", "next"):
            inbox.wait(Ack, session_id)
            inbox.wait(Proposal, session_id)
        assert control.launched == [("running",), ("next",)]
        assert control.completed == control.launched
        assert service.staging_slots == 0


def test_waiting_contexts_form_one_batch_after_both_pipeline_slots_fill():
    config = replace(_config(), resident_limit=4, staging_limit=4, max_queued_jobs=2)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "next", "third", "fourth"):
            _open(client, inbox, session_id)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        for session_id in ("third", "fourth"):
            assert client.submit(_snapshot(session_id), features)
        _wait_until(lambda: service.queued_jobs == 2, 1)
        assert len(control.stages) == 2
        control.gate.set()
        for session_id in ("running", "next", "third", "fourth"):
            inbox.wait(Proposal, session_id)
        assert ("third", "fourth") in control.stages
        assert ("third", "fourth") in control.launched


@pytest.mark.parametrize("session_id", ["running", "next"])
def test_close_retains_inflight_stage_until_completion_and_suppresses_replies(
    session_id,
):
    config = replace(_config(), staging_limit=2, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        for name in ("running", "next"):
            _open(client, inbox, name)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        assert client.submit(Close(session_id=session_id), None)
        _wait_until(lambda: service._sessions[session_id].closing, 1)
        assert service.resident_sessions == 2
        assert service.staging_slots == 2
        control.gate.set()
        survivor = "next" if session_id == "running" else "running"
        inbox.wait(Proposal, survivor)
        _wait_until(lambda: session_id not in control.engine.sessions, 1)
        _wait_until(lambda: service.resident_sessions == 1, 1)
        inbox.messages.extend(item.message for item in client.poll())
        assert not any(
            isinstance(message, (Ack, Proposal)) and message.session_id == session_id
            for message in inbox.messages
        )
        assert service.staging_slots == 0


def test_close_queued_context_releases_reservation_without_upload():
    config = replace(_config(), staging_limit=3, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "next", "queued"):
            _open(client, inbox, session_id)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        assert client.submit(_snapshot("queued"), features)
        _wait_until(lambda: service.queued_jobs == 1, 1)
        assert client.submit(Close(session_id="queued"), None)
        _wait_until(lambda: service.resident_sessions == 2, 1)
        assert service.queued_jobs == 0
        assert service.staging_slots == 2
        assert ("queued",) not in control.stages
        control.gate.set()
        inbox.wait(Proposal, "running")
        inbox.wait(Proposal, "next")


def test_worker_completion_does_not_wait_for_service_poll_timeout():
    config = replace(_config(), poll_ms=2000, heartbeat_ms=5000, lease_ms=10000)
    with _running(config) as (_, client, inbox, control):
        _open(client, inbox, "completion")
        control.gate.clear()
        assert client.submit(
            _snapshot("completion"), torch.ones((2, 4), dtype=torch.bfloat16)
        )
        assert control.entered.wait(3)
        time.sleep(0.03)
        started = time.monotonic()
        control.gate.set()
        inbox.wait(Proposal, "completion")
        assert time.monotonic() - started < 0.8


def test_service_shutdown_retires_active_and_staged_ownership():
    config = replace(_config(), staging_limit=2, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "next"):
            _open(client, inbox, session_id)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        service.close()
        time.sleep(0.03)
        assert not control.closed.is_set()
        control.gate.set()
        assert control.closed.wait(1)
        assert control.engine.sessions == {}
        assert control.engine.staged == {}


def test_expiry_retains_current_and_next_stages_until_device_completion():
    config = replace(_config(), staging_limit=2, max_batch_size=1, lease_ms=100)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "next"):
            _open(client, inbox, session_id)
        control.gate.clear()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        _wait_until(
            lambda: all(state.closing for state in service._sessions.values()), 1
        )
        assert service.resident_sessions == service.staging_slots == 2
        control.gate.set()
        _wait_until(lambda: service.resident_sessions == 0, 1)
        assert service.staging_slots == 0
        inbox.messages.extend(item.message for item in client.poll())
        assert not any(
            isinstance(message, (Ack, Proposal)) for message in inbox.messages
        )


def test_completed_closed_session_releases_while_next_forward_is_running():
    config = replace(_config(), staging_limit=3, max_batch_size=1)
    with _running(config) as (service, client, inbox, control):
        for session_id in ("running", "next", "queued"):
            _open(client, inbox, session_id)
        control.gate.clear()
        control.completion_gates["next"] = threading.Event()
        features = torch.ones((2, 4), dtype=torch.bfloat16)
        assert client.submit(_snapshot("running"), features)
        assert control.entered.wait(3)
        assert client.submit(_snapshot("next"), features)
        _wait_until(lambda: ("next",) in control.stages, 1)
        assert client.submit(_snapshot("queued"), features)
        _wait_until(lambda: service.queued_jobs == 1, 1)
        assert client.submit(Close(session_id="running"), None)
        _wait_until(lambda: service._sessions["running"].closing, 1)
        control.gate.set()
        _wait_until(lambda: ("next",) in control.launched, 1)
        _wait_until(lambda: "running" not in control.engine.sessions, 1)
        assert ("next",) not in control.completed
        assert service.resident_sessions == 2
        control.completion_gates["next"].set()
        inbox.wait(Proposal, "next")
        inbox.wait(Proposal, "queued")
