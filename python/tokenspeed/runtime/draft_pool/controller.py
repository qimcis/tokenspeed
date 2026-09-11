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

"""Confirmed-prefix remote proposal control, without a second scheduler.

Only the attention-cohort leader owns a client. It broadcasts plain scheduler
commands; every mirrored rank applies the same commands at the existing head
advance. GPU exports are independently polled through DeviceHandle.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    Busy,
    Close,
    Failure,
    MissingState,
    Open,
    Opened,
    Proposal,
    Ready,
    Update,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
    PROJECTED_FEATURE_EXPORT_SLOTS as REMOTE_DRAFT_EXPORT_SLOTS,
)

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    request_id: str
    session_id: str
    endpoint: int
    anchor: int
    opened: bool
    ack_endpoint: int | None
    transfer_endpoint: int | None
    transfer_anchor: int | None
    export_requested: bool
    last_progress_ms: float


class RemoteDraftController:
    """Translate bounded worker events into mirrored scheduler feedback.

    ``device`` is the narrow DeviceHandle, never an executor or cache pool.
    ``client`` is present only on the cohort leader. All decisions about which
    requests may wait or execute remain in the existing C++ scheduler.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: Any,
        client: Any,
        contract: Any,
        max_sessions: int,
        max_exports: int,
        max_defer_ms: float,
        attn_tp_rank: int,
        attn_tp_size: int,
        attn_tp_cpu_group: Any,
        leader_rank: int,
    ) -> None:
        self._enabled = enabled
        self._device = device
        self._client = client
        self._contract = contract
        self._max_sessions = max_sessions
        self._max_exports = max_exports
        self._max_defer_ms = max_defer_ms
        self._attn_tp_rank = attn_tp_rank
        self._attn_tp_size = attn_tp_size
        self._group = attn_tp_cpu_group
        self._leader_rank = leader_rank
        self._sessions: dict[str, _Session] = {}
        self._by_session: dict[str, _Session] = {}
        self._retry_after: dict[str, float] = {}
        self._ready = False
        self._resident_limit = max_sessions
        self._completed_exports: deque = deque()
        self._outstanding_exports: set[int] = set()
        self._admission_cursor = 0
        self._lease_ms = max(1000, max_defer_ms * 2)
        self._pin_requests: dict[tuple[str, int], _Session] = {}
        self._deferred_commands: list = []

    @staticmethod
    def _command(kind: str, **values: Any) -> tuple[str, dict[str, Any]]:
        return kind, values

    @staticmethod
    def _scheduler_events(commands: list) -> list:
        from tokenspeed_scheduler import ForwardEvent

        events = []
        for kind, values in commands:
            event = getattr(ForwardEvent, kind)()
            for name, value in values.items():
                setattr(event, name, value)
            events.append(event)
        return events

    def _unavailable(self, session: _Session) -> tuple:
        return self._command(
            "RemoteDraftUnavailable",
            request_id=session.request_id,
            session_id=session.session_id,
            endpoint=session.endpoint,
            anchor_id=session.anchor,
        )

    def _drop(self, session: _Session, commands: list, retry_at: float) -> None:
        commands.append(self._unavailable(session))
        self._client.submit(Close(session_id=session.session_id), None)
        self._device.cancel_remote_draft_exports(session.session_id)
        self._sessions.pop(session.request_id, None)
        self._by_session.pop(session.session_id, None)
        self._retry_after[session.request_id] = retry_at

    def _reset(self, commands: list, now_ms: float) -> None:
        for session in list(self._sessions.values()):
            self._drop(session, commands, now_ms)

    def _receive(self, commands: list, descriptors: dict, now_ms: float) -> None:
        for decoded in self._client.poll():
            message = decoded.message
            if isinstance(message, Ready):
                if message.contract != self._contract:
                    self._ready = False
                    self._reset(commands, now_ms)
                    logger.error(
                        "Remote draft worker has an incompatible model contract"
                    )
                    continue
                # A fresh handshake cannot inherit GPU sessions from an old
                # connection, even when the model contract is unchanged.
                self._reset(commands, now_ms)
                self._ready = True
                self._resident_limit = min(self._max_sessions, message.resident_limit)
                self._lease_ms = message.lease_ms
                continue
            if isinstance(message, Failure) and message.session_id is None:
                self._ready = False
                self._reset(commands, now_ms)
                logger.warning("Remote draft worker unavailable: %s", message.code)
                continue
            session = self._by_session.get(message.session_id)
            if session is None:
                continue
            if isinstance(message, Busy):
                retry_at = now_ms + max(message.retry_after_ms, 1)
                if session.opened:
                    commands.append(self._unavailable(session))
                    session.transfer_endpoint = None
                    session.transfer_anchor = None
                    session.export_requested = False
                    self._retry_after[session.request_id] = retry_at
                else:
                    self._drop(session, commands, retry_at)
                continue
            if isinstance(message, Failure) and message.code == "proposal_failed":
                # Installed context was ACKed separately and remains useful.
                commands.append(self._unavailable(session))
                session.export_requested = False
                self._retry_after[session.request_id] = now_ms + max(
                    self._max_defer_ms, 1
                )
                continue
            if isinstance(message, (MissingState, Failure)):
                self._drop(session, commands, now_ms + max(self._max_defer_ms, 1))
                continue
            descriptor = descriptors.get(session.request_id)
            if isinstance(message, Opened):
                if (
                    descriptor is None
                    or descriptor.results_in_flight
                    or message.confirmed_endpoint != descriptor.endpoint
                    or message.anchor_token != descriptor.anchor_id
                ):
                    self._drop(session, commands, now_ms)
                    continue
                session.opened = True
                session.last_progress_ms = now_ms
            elif isinstance(message, Ack):
                # ACK validity is about the submitted transfer, not the current
                # target prefix. Healthy fallback can make its proposal stale
                # while this ACK still advances useful remote context.
                if (
                    message.confirmed_endpoint == session.transfer_endpoint
                    and message.anchor_token == session.transfer_anchor
                ):
                    session.ack_endpoint = message.confirmed_endpoint
                    session.last_progress_ms = now_ms
                    session.transfer_endpoint = None
                    session.transfer_anchor = None
                    session.export_requested = False
            elif isinstance(message, Proposal):
                if (
                    descriptor is not None
                    and not descriptor.results_in_flight
                    and message.confirmed_endpoint == descriptor.endpoint
                    and message.anchor_token == descriptor.anchor_id
                    and descriptor.session_id == session.session_id
                    and descriptor.status == "pending"
                ):
                    commands.append(
                        self._command(
                            "RemoteDraftReady",
                            request_id=session.request_id,
                            session_id=session.session_id,
                            endpoint=message.confirmed_endpoint,
                            anchor_id=message.anchor_token,
                            candidate_ids=list(
                                message.candidate_ids[
                                    : self._contract.target_verify_tokens - 1
                                ]
                            ),
                        )
                    )

    def _poll_exports(self, commands: list) -> None:
        completed_exports, released_tickets = self._device.poll_remote_draft_exports()
        for ticket in released_tickets:
            self._outstanding_exports.discard(ticket)
            commands.append(
                self._command("ReleaseRemoteDraftSnapshot", ticket_id=ticket)
            )
        for completed in completed_exports:
            if completed.descriptor.session_id in self._by_session:
                self._completed_exports.append(completed)
        while self._completed_exports:
            completed = self._completed_exports[0]
            descriptor = completed.descriptor
            session = self._by_session.get(descriptor.session_id)
            if session is None:
                self._completed_exports.popleft()
                continue
            message = Update(
                session_id=descriptor.session_id,
                confirmed_endpoint=descriptor.end,
                anchor_token=descriptor.anchor_token,
                feature_start=descriptor.start,
                is_snapshot=session.ack_endpoint is None,
            )
            if not self._client.submit(message, completed.features):
                break
            # submit copied the CPU bytes into its bounded outbox. No worker
            # ACK or socket timing can retain an exporter staging slot.
            self._completed_exports.popleft()
            session.transfer_endpoint = descriptor.end
            session.transfer_anchor = descriptor.anchor_token

    def _request_export(
        self, session: _Session, descriptor: Any, commands: list
    ) -> None:
        if (
            session.export_requested
            or session.transfer_endpoint is not None
            or session.ack_endpoint == descriptor.endpoint
        ):
            return
        queued = max(
            sum(s.export_requested for s in self._sessions.values()),
            len(self._outstanding_exports) + len(self._completed_exports),
        )
        if queued >= self._max_exports:
            return
        start = (
            max(0, descriptor.endpoint - self._contract.window_tokens + 1)
            if session.ack_endpoint is None
            else session.ack_endpoint
        )
        commands.append(
            self._command(
                "RemoteDraftExport",
                request_id=session.request_id,
                session_id=session.session_id,
                endpoint=descriptor.endpoint,
                anchor_id=descriptor.anchor_id,
                start=start,
            )
        )
        session.export_requested = True
        self._pin_requests[(session.session_id, descriptor.endpoint)] = session

    def _drive(self, descriptors: dict, commands: list, now_ms: float) -> None:
        if not self._ready:
            return
        candidates = list(descriptors.values())
        if candidates:
            pivot = self._admission_cursor % len(candidates)
            candidates = candidates[pivot:] + candidates[:pivot]
        for descriptor in candidates:
            if descriptor.results_in_flight:
                continue
            if now_ms < self._retry_after.get(descriptor.request_id, 0):
                continue
            session = self._sessions.get(descriptor.request_id)
            if session is not None and descriptor.session_id not in (
                "",
                session.session_id,
            ):
                self._drop(session, commands, now_ms)
                session = None
            if (
                session is not None
                and not session.opened
                and (descriptor.endpoint, descriptor.anchor_id)
                != (session.endpoint, session.anchor)
            ):
                self._drop(session, commands, now_ms)
                session = None
            if session is not None and descriptor.status == "pending":
                if session.opened:
                    self._request_export(session, descriptor, commands)
                continue
            if descriptor.status != "unavailable" or not descriptor.admission_allowed:
                continue
            if now_ms < self._retry_after.get(descriptor.request_id, 0):
                continue
            if session is not None:
                if session.transfer_endpoint is not None or session.export_requested:
                    continue
                if not session.opened:
                    continue
                if session.ack_endpoint is None:
                    # OPEN reserved storage, but a busy snapshot installed no
                    # context yet; retry it against an unchanged OPEN prefix.
                    if (descriptor.endpoint, descriptor.anchor_id) != (
                        session.endpoint,
                        session.anchor,
                    ):
                        self._drop(session, commands, now_ms)
                        session = None
                if session is not None and session.ack_endpoint == descriptor.endpoint:
                    # A failed/expired proposal falls back before another
                    # confirmed interval exists. Retain its installed context.
                    continue
                if (
                    session is not None
                    and session.ack_endpoint is not None
                    and not max(
                        0, descriptor.endpoint - self._contract.window_tokens + 1
                    )
                    <= session.ack_endpoint
                    < descriptor.endpoint
                ):
                    self._drop(session, commands, now_ms)
                    session = None
            if session is None:
                if len(self._sessions) >= self._resident_limit:
                    continue
                session = _Session(
                    request_id=descriptor.request_id,
                    session_id=uuid.uuid4().hex,
                    endpoint=descriptor.endpoint,
                    anchor=descriptor.anchor_id,
                    opened=False,
                    ack_endpoint=None,
                    transfer_endpoint=None,
                    transfer_anchor=None,
                    export_requested=False,
                    last_progress_ms=now_ms,
                )
                if not self._client.submit(
                    Open(
                        session_id=session.session_id,
                        confirmed_endpoint=session.endpoint,
                        anchor_token=session.anchor,
                        history_start=max(
                            0, session.endpoint - self._contract.window_tokens + 1
                        ),
                    ),
                    None,
                ):
                    continue
                self._sessions[session.request_id] = session
                self._by_session[session.session_id] = session
                self._admission_cursor += 1
            session.endpoint = descriptor.endpoint
            session.anchor = descriptor.anchor_id
            commands.append(
                self._command(
                    "RemoteDraftPending",
                    request_id=session.request_id,
                    session_id=session.session_id,
                    endpoint=session.endpoint,
                    anchor_id=session.anchor,
                    now_ms=int(now_ms),
                )
            )
            if session.opened:
                self._request_export(session, descriptor, commands)

    def poll_ready_events(
        self, scheduler: Any, live_request_ids: set[str], paused: bool
    ) -> list:
        """Return identically ordered head feedback; never advance the scheduler."""
        if not self._enabled:
            return []
        commands = []
        if self._attn_tp_rank == 0:
            now_ms = time.monotonic_ns() // 1_000_000
            descriptors = {d.request_id: d for d in scheduler.remote_draft_requests()}
            commands.append(self._command("RemoteDraftTick", now_ms=now_ms))
            commands.extend(self._deferred_commands)
            self._deferred_commands.clear()
            self._poll_exports(commands)
            for session in list(self._sessions.values()):
                if (
                    session.request_id not in live_request_ids
                    or session.request_id not in descriptors
                    or now_ms - session.last_progress_ms >= self._lease_ms
                ):
                    self._drop(session, commands, now_ms)
            self._receive(commands, descriptors, now_ms)
            if not paused:
                self._drive(descriptors, commands, now_ms)
            for request_id in list(self._retry_after):
                if request_id not in live_request_ids:
                    self._retry_after.pop(request_id)
        if self._attn_tp_size > 1:
            import torch.distributed as dist

            payload = [commands]
            dist.broadcast_object_list(
                payload, src=self._leader_rank, group=self._group
            )
            commands = payload[0]
        return self._scheduler_events(commands)

    def queue_exports(self, scheduler: Any) -> None:
        """Drain the C++ pin descriptors after head feedback and before planning."""
        if not self._enabled:
            return
        snapshots = scheduler.remote_draft_snapshots()
        if self._attn_tp_rank == 0:
            accepted = {(s.session_id, s.endpoint) for s in snapshots}
            for key, session in tuple(self._pin_requests.items()):
                if key not in accepted and session.session_id in self._by_session:
                    # The authoritative scheduler rejected the pin (for
                    # example a reset superseded this prefix). Do not strand
                    # a session behind an export which will never complete.
                    session.export_requested = False
                    self._drop(
                        session,
                        self._deferred_commands,
                        time.monotonic_ns() // 1_000_000 + self._max_defer_ms,
                    )
            self._pin_requests.clear()
            self._outstanding_exports.update(
                snapshot.ticket_id for snapshot in snapshots
            )
            self._device.queue_remote_draft_exports(snapshots)

    def after_commit(self, live_request_ids: set[str]) -> None:
        """Close terminal worker state only after all tail feedback was applied."""
        if not self._enabled or self._attn_tp_rank != 0:
            return
        for session in list(self._sessions.values()):
            if session.request_id not in live_request_ids:
                self._client.submit(Close(session_id=session.session_id), None)
                self._device.cancel_remote_draft_exports(session.session_id)
                self._sessions.pop(session.request_id, None)
                self._by_session.pop(session.session_id, None)
                self._retry_after.pop(session.request_id, None)

    def get_stats(self, scheduler: Any) -> dict[str, int] | None:
        """Return leader-local operational gauges from authoritative state."""
        if not self._enabled or self._attn_tp_rank != 0:
            return None
        requests = scheduler.remote_draft_requests()
        return {
            "connected": int(self._ready and self._client.connected),
            "resident_sessions": sum(
                session.opened for session in self._sessions.values()
            ),
            "pending_requests": sum(
                request.status == "pending" for request in requests
            ),
            "ready_requests": sum(request.status == "ready" for request in requests),
            "outstanding_exports": len(self._outstanding_exports),
        }

    def close(self) -> None:
        """Stop admission and retire GPU exports before releasing host storage."""
        if not self._enabled:
            return
        if self._attn_tp_rank == 0:
            for session in self._sessions.values():
                self._client.submit(Close(session_id=session.session_id), None)
            self._client.close()
            self._sessions.clear()
            self._by_session.clear()
            self._completed_exports.clear()
        self._device.close_remote_draft_exports()


def build_remote_draft_controller(
    *,
    server_args: Any,
    model_config: Any,
    draft_model_config: Any,
    device: Any,
    max_sessions: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_tp_cpu_group: Any,
    leader_rank: int,
) -> RemoteDraftController:
    """Construct a private client only on a remote-enabled cohort leader."""
    endpoint = server_args.remote_draft_endpoint
    contract = None
    client = None
    if endpoint is not None:
        from tokenspeed.runtime.draft_pool.protocol import (
            MAX_FEATURE_BYTES,
            MAX_HEADER_BYTES,
            build_contract,
        )
        from tokenspeed.runtime.draft_pool.transport import (
            TargetDraftClient,
            TargetDraftClientConfig,
        )

        target_revision = (
            getattr(model_config.hf_config, "_commit_hash", None)
            or server_args.revision
        )
        draft_revision = (
            getattr(draft_model_config.hf_config, "_commit_hash", None)
            or server_args.speculative_draft_model_revision
        )
        if not target_revision or not draft_revision:
            raise ValueError(
                "Remote drafting requires resolved or explicitly pinned target and draft revisions"
            )
        contract = build_contract(
            target_model=server_args.model,
            target_revision=target_revision,
            draft_model=server_args.speculative_draft_model_path,
            draft_revision=draft_revision,
            target_hf_config=model_config.hf_config.to_dict(),
            draft_hf_config=draft_model_config.hf_config.to_dict(),
            target_verify_tokens=server_args.speculative_verify_tokens,
        )
        if attn_tp_rank == 0:
            message_limit = max(8, max_sessions * 4)
            client = TargetDraftClient(
                TargetDraftClientConfig(
                    endpoint=endpoint,
                    identity=uuid.uuid4().hex.encode("ascii"),
                    max_pending_messages=message_limit,
                    max_pending_bytes=REMOTE_DRAFT_EXPORT_SLOTS * MAX_FEATURE_BYTES
                    + message_limit * MAX_HEADER_BYTES,
                    max_reply_messages=message_limit,
                    max_header_bytes=MAX_HEADER_BYTES,
                    max_feature_bytes=MAX_FEATURE_BYTES,
                    heartbeat_ms=1000,
                    reconnect_ms=100,
                    poll_ms=10,
                    linger_ms=0,
                ),
                contract,
            )
    return RemoteDraftController(
        enabled=endpoint is not None,
        device=device,
        client=client,
        contract=contract,
        max_sessions=max_sessions,
        max_exports=REMOTE_DRAFT_EXPORT_SLOTS,
        max_defer_ms=float(server_args.remote_draft_max_defer_ms or 0),
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_cpu_group=attn_tp_cpu_group,
        leader_rank=leader_rank,
    )
