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

"""Confirmed-prefix control: admission, stale ACKs, cancellation and batching."""

import time
from types import SimpleNamespace

import torch

from tokenspeed.runtime.draft_pool.controller import RemoteDraftController
from tokenspeed.runtime.draft_pool.features import (
    CompletedFeatureExport,
    FeatureExportDescriptor,
)
from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    DecodedMessage,
    Opened,
    Proposal,
    Ready,
    Update,
)


class Client:
    def __init__(self):
        self.incoming = []
        self.sent = []
        self.accepting = True

    def submit(self, message, features):
        if not self.accepting:
            return False
        self.sent.append((message, features))
        return True

    def poll(self):
        result, self.incoming = self.incoming, []
        return result

    def close(self):
        pass


class Device:
    def __init__(self):
        self.completed = []
        self.released = []
        self.cancelled = []
        self.queued = []

    def poll_remote_draft_exports(self):
        result = self.completed, self.released
        self.completed, self.released = [], []
        return result

    def cancel_remote_draft_exports(self, session_id):
        self.cancelled.append(session_id)

    def queue_remote_draft_exports(self, snapshots):
        self.queued.extend(snapshots)

    def close_remote_draft_exports(self):
        pass


class Scheduler:
    def __init__(self, descriptors):
        self.descriptors = descriptors
        self.snapshots = []

    def remote_draft_requests(self):
        return self.descriptors

    def remote_draft_snapshots(self):
        result, self.snapshots = self.snapshots, []
        return result


def descriptor(request_id, endpoint, anchor):
    return SimpleNamespace(
        request_id=request_id,
        endpoint=endpoint,
        anchor_id=anchor,
        session_id="",
        status="unavailable",
        admission_allowed=True,
        results_in_flight=0,
    )


def setup():
    client, device = Client(), Device()
    contract = SimpleNamespace(window_tokens=2048, target_verify_tokens=6)
    controller = RemoteDraftController(
        enabled=True,
        device=device,
        client=client,
        contract=contract,
        max_sessions=4,
        max_exports=2,
        max_defer_ms=100,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        leader_rank=0,
    )
    controller._ready = True
    return controller, client, device


def begin(controller, descriptors):
    commands = []
    controller._drive({d.request_id: d for d in descriptors}, commands, 1000)
    for d in descriptors:
        session = controller._sessions.get(d.request_id)
        if session is not None:
            d.session_id = session.session_id
            d.status = "pending"
    return commands


def test_open_reserves_before_snapshot_export_and_keeps_other_requests_independent():
    controller, client, device = setup()
    a, b = descriptor("a", 100, 7), descriptor("b", 200, 9)
    commands = begin(controller, [a, b])
    assert [kind for kind, _ in commands] == [
        "RemoteDraftPending",
        "RemoteDraftPending",
    ]
    assert not device.queued
    client.incoming = [
        DecodedMessage(
            Opened(session_id=a.session_id, confirmed_endpoint=100, anchor_token=7),
            None,
        )
    ]
    commands = []
    controller._receive(commands, {"a": a, "b": b}, 1001)
    controller._drive({"a": a, "b": b}, commands, 1001)
    assert [(kind, data["request_id"]) for kind, data in commands] == [
        ("RemoteDraftExport", "a")
    ]
    assert controller._sessions["b"].opened is False


def test_stale_ack_advances_context_but_stale_proposal_cannot_enter_scheduler():
    controller, client, _ = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    session = controller._sessions["a"]
    session.opened = True
    session.transfer_endpoint = 100
    session.transfer_anchor = 7
    session.export_requested = True
    d.endpoint, d.anchor_id, d.status = 101, 8, "unavailable"
    client.incoming = [
        DecodedMessage(
            Ack(session_id=session.session_id, confirmed_endpoint=100, anchor_token=7),
            None,
        ),
        DecodedMessage(
            Proposal(
                session_id=session.session_id,
                confirmed_endpoint=100,
                anchor_token=7,
                candidate_ids=(1, 2, 3, 4, 5, 6, 7),
            ),
            None,
        ),
    ]
    commands = []
    controller._receive(commands, {"a": d}, 1002)
    assert commands == []
    assert session.ack_endpoint == 100
    controller._drive({"a": d}, commands, 1002)
    assert [kind for kind, _ in commands] == ["RemoteDraftPending", "RemoteDraftExport"]
    assert commands[-1][1]["start"] == 100


def test_ack_does_not_request_same_prefix_again_while_proposal_is_in_transit():
    controller, client, _ = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    session = controller._sessions["a"]
    session.opened = True
    session.transfer_endpoint, session.transfer_anchor = 100, 7
    client.incoming = [
        DecodedMessage(
            Ack(session_id=session.session_id, confirmed_endpoint=100, anchor_token=7),
            None,
        )
    ]
    commands = []
    controller._receive(commands, {"a": d}, 1002)
    controller._drive({"a": d}, commands, 1002)
    assert commands == []


def test_ready_reply_consumes_exactly_five_candidates_with_explicit_anchor():
    controller, client, _ = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    client.incoming = [
        DecodedMessage(
            Proposal(
                session_id=d.session_id,
                confirmed_endpoint=100,
                anchor_token=7,
                candidate_ids=(10, 11, 12, 13, 14, 15, 16),
            ),
            None,
        )
    ]
    commands = []
    controller._receive(commands, {"a": d}, 1001)
    event = controller._scheduler_events(commands)[0]
    assert event.anchor_id == 7
    assert event.candidate_ids == [10, 11, 12, 13, 14]


def test_cancelled_export_still_releases_pin_and_never_sends_update():
    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    sid = d.session_id
    controller.after_commit(set())
    device.completed = [
        CompletedFeatureExport(
            FeatureExportDescriptor(
                ticket_id=1, session_id=sid, start=98, end=100, anchor_token=7
            ),
            torch.zeros(2, 4, dtype=torch.bfloat16),
        )
    ]
    device.released = [1]
    commands = []
    controller._poll_exports(commands)
    assert commands == [("ReleaseRemoteDraftSnapshot", {"ticket_id": 1})]
    assert not any(isinstance(message, Update) for message, _ in client.sent)


def test_export_transport_backpressure_retains_bounded_cpu_rows_until_copy():
    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    session = controller._sessions["a"]
    session.opened, session.export_requested = True, True
    rows = torch.ones(2, 4, dtype=torch.bfloat16)
    device.completed = [
        CompletedFeatureExport(
            FeatureExportDescriptor(
                ticket_id=1, session_id=d.session_id, start=98, end=100, anchor_token=7
            ),
            rows,
        )
    ]
    device.released = [1]
    client.accepting = False
    controller._poll_exports([])
    assert len(controller._completed_exports) == 1
    assert session.transfer_endpoint is None
    client.accepting = True
    controller._poll_exports([])
    assert not controller._completed_exports
    assert session.transfer_endpoint == 100
    assert client.sent[-1][1] is rows


def test_inflight_prefix_is_not_reopened_and_finished_prefix_is_closed():
    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    d.results_in_flight = 1
    count = len(client.sent)
    controller._sessions["a"].last_progress_ms = time.monotonic_ns() // 1_000_000
    events = controller.poll_ready_events(Scheduler([d]), {"a"}, False)
    assert len(events) == 1  # mirrored clock only
    assert len(client.sent) == count
    controller.after_commit(set())
    assert d.session_id in device.cancelled
    assert not controller._sessions


def test_rejected_snapshot_pin_cannot_leave_export_requested_forever():
    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    session = controller._sessions["a"]
    session.opened = True
    controller._request_export(session, d, [])
    assert session.export_requested
    controller.queue_exports(Scheduler([d]))  # scheduler rejected this prefix
    assert not session.export_requested
    assert "a" not in controller._sessions
    assert d.session_id in device.cancelled
    assert controller._deferred_commands[0][0] == "RemoteDraftUnavailable"


def test_open_waiting_on_old_prefix_resets_after_fallback():
    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    old_session = d.session_id
    d.endpoint, d.anchor_id, d.status = 101, 8, "unavailable"
    controller._drive({"a": d}, [], 1002)
    assert old_session in device.cancelled
    assert controller._sessions["a"].session_id != old_session


def test_proposal_failure_preserves_acknowledged_context():
    from tokenspeed.runtime.draft_pool.protocol import Failure

    controller, client, device = setup()
    d = descriptor("a", 100, 7)
    begin(controller, [d])
    session = controller._sessions["a"]
    session.opened = True
    session.transfer_endpoint, session.transfer_anchor = 100, 7
    client.incoming = [
        DecodedMessage(
            Ack(session_id=session.session_id, confirmed_endpoint=100, anchor_token=7),
            None,
        ),
        DecodedMessage(
            Failure(
                session_id=session.session_id, code="proposal_failed", detail="injected"
            ),
            None,
        ),
    ]
    commands = []
    controller._receive(commands, {"a": d}, 1002)
    assert session.ack_endpoint == 100
    assert controller._sessions["a"] is session
    assert not device.cancelled
    assert [kind for kind, _ in commands] == ["RemoteDraftUnavailable"]


def test_operational_snapshot_uses_scheduler_readiness():
    controller, client, _ = setup()
    client.connected = True
    a, b = descriptor("a", 100, 7), descriptor("b", 200, 9)
    begin(controller, [a, b])
    controller._sessions["a"].opened = True
    b.status = "ready"
    controller._outstanding_exports.add(4)
    assert controller.get_stats(Scheduler([a, b])) == {
        "connected": 1,
        "resident_sessions": 1,
        "pending_requests": 1,
        "ready_requests": 1,
        "outstanding_exports": 1,
    }
