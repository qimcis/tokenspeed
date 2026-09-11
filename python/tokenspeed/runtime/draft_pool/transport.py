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

"""Bounded private draft transport; sockets and model execution never share a thread.

The target controller owns prefix eligibility and coalescing. This module owns
copied wire buffers, connection recovery, and worker admission. A context ACK
is independent of proposal success: timeout/fallback must not lose the installed
worker frontier. All engine methods, including construction and destruction,
run on the worker's single data-plane thread.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import zmq
from zmq.utils.monitor import recv_monitor_message

from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    Busy,
    Close,
    DecodedMessage,
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
from tokenspeed.runtime.draft_pool.worker import WorkerDraftJob as DraftPoolJob


@dataclass(frozen=True)
class TargetDraftClientConfig:
    """Explicit per-cohort wire limits and socket timings, in bytes/ms."""

    endpoint: str
    identity: bytes
    max_pending_messages: int
    max_pending_bytes: int
    max_reply_messages: int
    max_header_bytes: int
    max_feature_bytes: int
    heartbeat_ms: int
    reconnect_ms: int
    poll_ms: int
    linger_ms: int

    def __post_init__(self) -> None:
        if not self.endpoint or not 1 <= len(self.identity) <= 255:
            raise ValueError(
                "A private endpoint and a 1..255-byte identity are required"
            )
        _positive_limits(self)


def _positive_limits(config: Any) -> None:
    for name, value in vars(config).items():
        if isinstance(value, int) and (
            value < 0 or (value == 0 and name != "linger_ms")
        ):
            raise ValueError(f"{name} must be positive (linger_ms may be zero)")


def _recv_bounded(
    socket: zmq.Socket, max_frames: int, max_total_bytes: int
) -> tuple[list[bytes], bool]:
    """Drain one multipart message without accumulating unbounded extra frames."""
    frames = []
    total_bytes = 0
    count = 0
    valid = True
    while True:
        frame = socket.recv(copy=True)
        count += 1
        total_bytes += len(frame)
        if count > max_frames or total_bytes > max_total_bytes:
            valid = False
        elif valid:
            frames.append(frame)
        if not socket.getsockopt(zmq.RCVMORE):
            return frames, valid


class TargetDraftClient:
    """Nonblocking target-facing DEALER with one CPU I/O thread.

    ``submit`` copies CPU frames before returning True; the caller may then
    release/reuse export staging. False retains nothing. ``poll`` drains typed
    messages, including ``Failure(code='connection_reset')`` and a new ``Ready``
    after reconnection. Either event requires fresh controller session IDs.
    The caller must not mutate staging concurrently with ``submit``.
    """

    def __init__(
        self, config: TargetDraftClientConfig, contract: DraftProtocolContract
    ) -> None:
        self.config = config
        self.contract = contract
        self._codec = DraftMessageCodec(
            contract=contract,
            max_header_bytes=config.max_header_bytes,
            max_feature_bytes=config.max_feature_bytes,
        )
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._outbound: deque[tuple[Any, list[bytes]]] = deque()
        self._inbound: deque[DecodedMessage] = deque()
        self._session_ids: set[str] = set()
        self._pending_bytes = 0
        self._connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="draft-pool-client", daemon=True
        )
        self._thread.start()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def pending_bytes(self) -> int:
        """Bytes owned by queued copied frames, excluding bounded socket buffers."""
        with self._lock:
            return self._pending_bytes

    def submit(self, message: Any, features: Any) -> bool:
        """Queue a CPU-only protocol message; admission failure retains no payload."""
        if isinstance(message, Hello):
            raise ValueError("The transport owns HELLO handshakes")
        with self._encode_lock:
            feature_bytes = 0
            if features is not None:
                if getattr(getattr(features, "device", None), "type", None) != "cpu":
                    raise ValueError("non-CPU feature tensors are forbidden")
                feature_bytes = features.numel() * features.element_size()
            with self._lock:
                if (
                    self._stop.is_set()
                    or not self._connected
                    or len(self._outbound) >= self.config.max_pending_messages
                    or (
                        features is not None
                        and self._pending_bytes
                        + feature_bytes
                        + self.config.max_header_bytes
                        > self.config.max_pending_bytes
                    )
                ):
                    return False
            # The private codec validates device/dtype and bounds before copying.
            frames = [bytes(frame) for frame in self._codec.encode(message, features)]
            frame_bytes = sum(map(len, frames))
            with self._lock:
                if (
                    self._stop.is_set()
                    or not self._connected
                    or len(self._outbound) >= self.config.max_pending_messages
                    or self._pending_bytes + frame_bytes > self.config.max_pending_bytes
                ):
                    return False
                self._outbound.append((message, frames))
                self._pending_bytes += frame_bytes
                return True

    def poll(self) -> list[DecodedMessage]:
        """Drain wire replies in receive order, preserving ACK before proposal."""
        with self._lock:
            result = list(self._inbound)
            self._inbound.clear()
            return result

    def close(self) -> None:
        """Stop admissions and retire copied sends with bounded socket linger."""
        self._stop.set()
        self._thread.join((self.config.linger_ms + 4 * self.config.poll_ms) / 1000 + 1)
        if self._thread.is_alive():
            raise RuntimeError("Draft client I/O thread did not stop")

    def _reset(self, detail: str) -> None:
        with self._lock:
            was_connected = self._connected
            self._connected = False
            self._outbound.clear()
            self._pending_bytes = 0
            # A reset invalidates every prior session, including queued replies.
            if was_connected:
                self._inbound.clear()
                self._session_ids.clear()
                self._inbound.append(
                    DecodedMessage(
                        message=Failure(
                            session_id=None, code="connection_reset", detail=detail
                        ),
                        features=None,
                    )
                )

    def _receive(self, decoded: DecodedMessage) -> bool:
        with self._lock:
            if isinstance(decoded.message, Ready) and self._connected:
                # Retransmitted HELLOs may already be queued before first READY.
                return True
            if len(self._inbound) >= self.config.max_reply_messages:
                return False
            self._inbound.append(decoded)
            if isinstance(decoded.message, Ready):
                self._connected = True
            elif isinstance(decoded.message, Opened):
                self._session_ids.add(decoded.message.session_id)
            return True

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        monitor = None
        try:
            socket.setsockopt(zmq.IDENTITY, self.config.identity)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.LINGER, self.config.linger_ms)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.RCVHWM, self.config.max_reply_messages)
            socket.setsockopt(zmq.MAXMSGSIZE, self.config.max_header_bytes)
            socket.setsockopt(zmq.RECONNECT_IVL, self.config.reconnect_ms)
            socket.setsockopt(zmq.HEARTBEAT_IVL, self.config.heartbeat_ms)
            socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3 * self.config.heartbeat_ms)
            monitor = socket.get_monitor_socket(
                events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED
            )
            socket.connect(self.config.endpoint)
            decoder = DraftMessageCodec(
                contract=self.contract,
                max_header_bytes=self.config.max_header_bytes,
                max_feature_bytes=self.config.max_feature_bytes,
            )
            hello = decoder.encode(Hello(contract=self.contract), None)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            poller.register(monitor, zmq.POLLIN)
            last_hello = float("-inf")
            while not self._stop.is_set():
                events = dict(poller.poll(self.config.poll_ms))
                if monitor in events:
                    event = recv_monitor_message(monitor)
                    if event["event"] == zmq.EVENT_DISCONNECTED:
                        self._reset(
                            "Worker connection lost; reseed with fresh sessions"
                        )
                    elif event["event"] == zmq.EVENT_CONNECTED:
                        last_hello = float("-inf")
                if not self.connected:
                    now = time.monotonic()
                    if (now - last_hello) * 1000 >= self.config.reconnect_ms:
                        try:
                            socket.send_multipart(hello, flags=zmq.DONTWAIT, copy=True)
                            last_hello = now
                        except zmq.Again:
                            pass
                if socket in events:
                    try:
                        frames, valid = _recv_bounded(
                            socket, 1, self.config.max_header_bytes
                        )
                        if not valid:
                            raise ValueError("Worker reply exceeds the frame bound")
                        decoded = decoder.decode(frames)
                        if not isinstance(
                            decoded.message,
                            (Ready, Opened, Ack, Proposal, Busy, MissingState, Failure),
                        ):
                            raise ValueError(
                                "Unexpected target-to-worker message in reply"
                            )
                        if (
                            isinstance(decoded.message, Failure)
                            and decoded.message.code == "handshake_required"
                        ):
                            self._reset(
                                "Worker handshake expired; reseed with fresh sessions"
                            )
                            continue
                        if not self._receive(decoded):
                            self._reset(
                                "Reply queue overflow; reseed with fresh sessions"
                            )
                            # Flush already-buffered replies from the old session set.
                            while socket.poll(0, zmq.POLLIN):
                                _recv_bounded(socket, 1, self.config.max_header_bytes)
                    except (ValueError, TypeError) as exc:
                        self._reset(f"Invalid worker reply: {str(exc)[:256]}")
                if self.connected:
                    with self._lock:
                        entry = self._outbound[0] if self._outbound else None
                    if entry is not None:
                        message, frames = entry
                        try:
                            socket.send_multipart(frames, flags=zmq.DONTWAIT, copy=True)
                        except zmq.Again:
                            continue
                        with self._lock:
                            if self._outbound and self._outbound[0] is entry:
                                self._outbound.popleft()
                                self._pending_bytes -= sum(map(len, frames))
                                if isinstance(message, Close):
                                    self._session_ids.discard(message.session_id)
        except Exception as exc:
            self._reset(f"Draft transport stopped: {str(exc)[:256]}")
        finally:
            with self._lock:
                sessions = tuple(self._session_ids)
            for session_id in sessions:
                try:
                    with self._encode_lock:
                        frames = self._codec.encode(Close(session_id=session_id), None)
                    socket.send_multipart(
                        frames,
                        flags=zmq.DONTWAIT,
                        copy=True,
                    )
                except zmq.ZMQError:
                    break
            self._reset("Draft client closed")
            if monitor is not None:
                monitor.close(linger=0)
            socket.close(linger=self.config.linger_ms)
            context.term()


@dataclass(frozen=True)
class DraftPoolServiceConfig:
    """Worker admission limits and private ROUTER socket configuration.

    Resident, staging and batch limits are global across all cohort connections;
    READY does not grant each peer a separate copy of these capacities.
    """

    listen_endpoint: str
    resident_limit: int
    staging_limit: int
    max_queued_jobs: int
    max_batch_size: int
    max_peers: int
    max_header_bytes: int
    max_feature_bytes: int
    max_host_memory_bytes: int
    lease_ms: int
    heartbeat_ms: int
    poll_ms: int
    linger_ms: int

    def __post_init__(self) -> None:
        _positive_limits(self)
        if not self.listen_endpoint:
            raise ValueError("A private listen endpoint is required")
        if self.max_batch_size > self.staging_limit:
            raise ValueError("staging_limit must cover a full worker batch")
        if self.staging_limit > self.resident_limit:
            raise ValueError("staging_limit cannot exceed resident_limit")
        if self.max_host_bytes > self.max_host_memory_bytes:
            raise ValueError(
                f"Draft host bound {self.max_host_bytes} exceeds max_host_memory_bytes "
                f"{self.max_host_memory_bytes}; reduce staging/peers or raise the budget"
            )

    @property
    def max_host_bytes(self) -> int:
        """Conservative application/socket budget for admitted private peers.

        Three payload copies cover decoded staging, receive, and codec scratch.
        ROUTER HWM=1 adds one incoming payload per configured private peer.
        The private network must enforce the configured peer connection count;
        unauthenticated public exposure is deliberately unsupported.
        """
        return (
            3 * self.staging_limit + self.max_peers + 1
        ) * self.max_feature_bytes + (
            4 * self.resident_limit + 3 * self.max_peers
        ) * self.max_header_bytes


class DraftEngine(Protocol):
    contract: DraftProtocolContract

    def open_session(self, session_id: str) -> None: ...

    def install_features(
        self,
        session_id: str,
        feature_start: int,
        confirmed_endpoint: int,
        features: Any,
        is_snapshot: bool,
    ) -> None: ...

    def draft_batch(self, jobs: Sequence[DraftPoolJob]) -> Sequence[Any]: ...

    def close_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...


@dataclass
class _Session:
    identity: bytes
    opening: Open
    endpoint: int
    anchor_token: int
    last_activity: float
    initial: bool
    pending: bool
    staged: bool
    engine_opened: bool
    closing: bool


@dataclass(frozen=True)
class _BatchResult:
    installed: tuple[Ack, ...]
    proposals: tuple[Proposal, ...]
    failures: tuple[Failure, ...]
    invalid_sessions: tuple[str, ...]


class DraftPoolService:
    """ROUTER orchestration with bounded sessions, staging, and ready jobs.

    Construct on a control thread; ``run`` owns all socket operations. The
    factory and every engine method run on a separate single execution thread.
    ``close`` only requests shutdown and is safe from another thread. ``run``
    retires outstanding engine work before destroying storage and returning.
    """

    def __init__(
        self,
        config: DraftPoolServiceConfig,
        engine_factory: Callable[[], DraftEngine],
    ) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="draft-pool-device"
        )
        self._engine_future = self._executor.submit(engine_factory)
        self._stop = threading.Event()
        self._sessions: dict[str, _Session] = {}
        self._peers: dict[bytes, float] = {}
        self._queued: OrderedDict[str, DecodedMessage] = OrderedDict()
        self._outbox: OrderedDict[tuple[Any, ...], tuple[bytes, Any]] = OrderedDict()
        self._staged = 0
        self._running_ids: tuple[str, ...] = ()
        self._closing_ids: tuple[str, ...] = ()
        self._inflight: Future[Any] | None = None
        self.bound_endpoint: str | None = None
        self.started = threading.Event()

    @property
    def resident_sessions(self) -> int:
        return len(self._sessions)

    @property
    def staging_slots(self) -> int:
        return self._staged

    @property
    def queued_jobs(self) -> int:
        return len(self._queued)

    def close(self) -> None:
        """Stop new work; the run thread safely retires device work and leases."""
        self._stop.set()

    def run(self) -> None:
        """Serve until close or an engine/transport failure; propagate failures."""
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        engine = None
        try:
            engine = self._engine_future.result()
            self._codec = DraftMessageCodec(
                contract=engine.contract,
                max_header_bytes=self.config.max_header_bytes,
                max_feature_bytes=self.config.max_feature_bytes,
            )
            socket.setsockopt(zmq.LINGER, self.config.linger_ms)
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(
                zmq.MAXMSGSIZE,
                max(self.config.max_header_bytes, self.config.max_feature_bytes),
            )
            socket.setsockopt(zmq.HEARTBEAT_IVL, self.config.heartbeat_ms)
            socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3 * self.config.heartbeat_ms)
            socket.bind(self.config.listen_endpoint)
            self.bound_endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self.started.set()
            while not self._stop.is_set():
                if socket.poll(self.config.poll_ms, zmq.POLLIN):
                    frames, valid = _recv_bounded(
                        socket,
                        3,
                        255
                        + self.config.max_header_bytes
                        + self.config.max_feature_bytes,
                    )
                    identity = frames[0]
                    try:
                        if not valid:
                            raise ValueError(
                                "Message exceeds the multipart frame bound"
                            )
                        if len(frames) < 2:
                            raise ValueError("Missing protocol header")
                        header = self._codec.decode_header(frame=frames[1])
                        if self._preflight(identity, header):
                            self._handle(
                                identity, self._codec.decode(frames[1:]), engine
                            )
                    except (ValueError, TypeError) as exc:
                        self._reply(
                            identity,
                            Failure(
                                session_id=None,
                                code="invalid_message",
                                detail=str(exc)[:256],
                            ),
                        )
                self._complete()
                self._expire()
                self._dispatch(engine)
                self._flush(socket)
        finally:
            self._stop.set()
            self._queued.clear()
            self._outbox.clear()
            # Never release worker memory while an install/forward still owns it.
            try:
                if engine is None:
                    # A termination signal may have interrupted model loading.
                    # Let construction retire on its owning thread before close.
                    try:
                        engine = self._engine_future.result()
                    except Exception:
                        pass
                if engine is not None:
                    self._executor.submit(engine.close).result()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._sessions.clear()
                self._staged = 0
                socket.close(linger=self.config.linger_ms)
                context.term()

    def _reply(self, identity: bytes, message: Any) -> None:
        session_id = getattr(message, "session_id", None)
        # Unadmitted identities cannot consume one queue slot per invented ID.
        key = (
            identity,
            type(message),
            session_id if session_id in self._sessions else None,
        )
        capacity = 4 * self.config.resident_limit + 2 * self.config.max_peers
        if (
            key not in self._outbox
            and len(self._outbox) >= capacity
            and isinstance(message, (Ack, Opened, Ready))
        ):
            for old_key, (_, old_message) in tuple(self._outbox.items()):
                if not isinstance(old_message, (Ack, Opened, Ready)):
                    del self._outbox[old_key]
                    break
        if key in self._outbox or len(self._outbox) < capacity:
            self._outbox[key] = (identity, message)
        # Nonessential overload/error replies can be dropped at the hard bound.
        # At most four lifecycle replies per admitted session are retained.

    def _flush(self, socket: zmq.Socket) -> None:
        for key, (identity, message) in tuple(self._outbox.items()):
            try:
                socket.send_multipart(
                    [identity, *self._codec.encode(message, None)],
                    flags=zmq.DONTWAIT,
                    copy=True,
                )
                del self._outbox[key]
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                if exc.errno != zmq.EHOSTUNREACH:
                    raise
                del self._outbox[key]

    def _busy(self, identity: bytes, session_id: str, reason: str) -> None:
        self._reply(
            identity,
            Busy(
                session_id=session_id,
                reason=reason,
                retry_after_ms=self.config.poll_ms,
            ),
        )

    def _preflight(self, identity: bytes, message: Any) -> bool:
        """Refuse unadmitted feature payloads before allocating tensor staging."""
        if not isinstance(message, Update):
            return True
        session = self._sessions.get(message.session_id)
        if (
            identity not in self._peers
            or session is None
            or session.identity != identity
            or session.closing
        ):
            self._reply(
                identity,
                MissingState(
                    session_id=message.session_id,
                    reason="OPEN admission is required before a feature transfer",
                ),
            )
            return False
        if session.pending:
            self._busy(
                identity, message.session_id, "One context transfer is outstanding"
            )
            return False
        if len(self._queued) >= self.config.max_queued_jobs:
            self._busy(identity, message.session_id, "Worker job queue is full")
            return False
        if not session.staged and self._staged >= self.config.staging_limit:
            self._busy(identity, message.session_id, "Context staging limit reached")
            return False
        return True

    def _handle(
        self, identity: bytes, decoded: DecodedMessage, engine: DraftEngine
    ) -> None:
        message = decoded.message
        now = time.monotonic()
        if isinstance(message, Hello):
            if message.contract != engine.contract:
                self._reply(
                    identity,
                    Failure(
                        session_id=None,
                        code="incompatible_contract",
                        detail="Target and worker contracts differ",
                    ),
                )
                return
            if (
                identity not in self._peers
                and len(self._peers) >= self.config.max_peers
            ):
                self._reply(
                    identity,
                    Failure(
                        session_id=None,
                        code="busy",
                        detail="Private peer admission limit reached",
                    ),
                )
                return
            self._peers[identity] = now
            self._reply(
                identity,
                Ready(
                    contract=engine.contract,
                    resident_limit=self.config.resident_limit,
                    staging_limit=self.config.staging_limit,
                    max_batch_size=self.config.max_batch_size,
                    lease_ms=self.config.lease_ms,
                ),
            )
            return
        if identity not in self._peers:
            self._reply(
                identity,
                Failure(
                    session_id=None,
                    code="handshake_required",
                    detail="Send HELLO first",
                ),
            )
            return
        self._peers[identity] = now
        if not isinstance(message, (Open, Update, Close)):
            raise ValueError("Expected OPEN, UPDATE, or CLOSE from target")
        session_id = message.session_id
        session = self._sessions.get(session_id)
        if isinstance(message, Open):
            if session is not None:
                if (
                    session.identity == identity
                    and session.opening == message
                    and session.initial
                ):
                    session.last_activity = now
                    self._reply(
                        identity,
                        Opened(
                            session_id=session_id,
                            confirmed_endpoint=message.confirmed_endpoint,
                            anchor_token=message.anchor_token,
                        ),
                    )
                else:
                    self._reply(
                        identity,
                        MissingState(
                            session_id=session_id, reason="Use a fresh session ID"
                        ),
                    )
                return
            if len(self._sessions) >= self.config.resident_limit:
                self._busy(identity, session_id, "Resident session limit reached")
                return
            if self._staged >= self.config.staging_limit:
                self._busy(identity, session_id, "Snapshot staging limit reached")
                return
            self._sessions[session_id] = _Session(
                identity=identity,
                opening=message,
                endpoint=message.history_start,
                anchor_token=message.anchor_token,
                last_activity=now,
                initial=True,
                pending=False,
                staged=True,
                engine_opened=False,
                closing=False,
            )
            self._staged += 1
            self._reply(
                identity,
                Opened(
                    session_id=session_id,
                    confirmed_endpoint=message.confirmed_endpoint,
                    anchor_token=message.anchor_token,
                ),
            )
            return
        if session is None or session.identity != identity or session.closing:
            if not isinstance(message, Close):
                self._reply(
                    identity,
                    MissingState(
                        session_id=session_id, reason="Session is absent or expired"
                    ),
                )
            return
        session.last_activity = now
        if isinstance(message, Close):
            self._close_session(session_id)
            return
        if session.pending:
            self._busy(
                identity, session_id, "One context transfer is already outstanding"
            )
            return
        if session.initial:
            valid = (
                message.is_snapshot
                and message.feature_start == session.opening.history_start
                and message.confirmed_endpoint == session.opening.confirmed_endpoint
                and message.anchor_token == session.opening.anchor_token
            )
        else:
            valid = (
                not message.is_snapshot
                and message.feature_start == session.endpoint
                and message.confirmed_endpoint >= session.endpoint
                and (
                    message.confirmed_endpoint != session.endpoint
                    or message.anchor_token == session.anchor_token
                )
            )
        if not valid:
            self._reply(
                identity,
                MissingState(
                    session_id=session_id, reason="Context frontier mismatch; reseed"
                ),
            )
            self._close_session(session_id)
            return
        if len(self._queued) >= self.config.max_queued_jobs:
            self._busy(identity, session_id, "Worker job queue is full")
            return
        if not session.staged:
            if self._staged >= self.config.staging_limit:
                self._busy(identity, session_id, "Context staging limit reached")
                return
            session.staged = True
            self._staged += 1
        session.pending = True
        self._queued[session_id] = decoded

    def _close_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        session.closing = True
        if session_id in self._queued:
            del self._queued[session_id]
            session.pending = False
        if session_id not in self._running_ids and session.staged:
            session.staged = False
            self._staged -= 1
        if not session.engine_opened and session_id not in self._running_ids:
            del self._sessions[session_id]
        for key in tuple(self._outbox):
            if key[2] == session_id and key[1] in (Opened, Ack, Proposal):
                del self._outbox[key]

    def _expire(self) -> None:
        deadline = time.monotonic() - self.config.lease_ms / 1000
        for session_id, session in tuple(self._sessions.items()):
            if not session.closing and session.last_activity < deadline:
                self._close_session(session_id)
        for identity, last_activity in tuple(self._peers.items()):
            if last_activity < deadline:
                del self._peers[identity]

    def _dispatch(self, engine: DraftEngine) -> None:
        if self._inflight is not None:
            return
        closing = tuple(
            session_id for session_id, state in self._sessions.items() if state.closing
        )
        if closing:
            self._closing_ids = closing
            self._inflight = self._executor.submit(self._release, engine, closing)
            return
        selected = tuple(self._queued)[: self.config.max_batch_size]
        if not selected:
            return
        work = []
        for session_id in selected:
            session = self._sessions[session_id]
            work.append((self._queued.pop(session_id), not session.engine_opened))
            session.engine_opened = True
        self._running_ids = selected
        self._inflight = self._executor.submit(self._execute, engine, tuple(work))

    @staticmethod
    def _release(engine: DraftEngine, sessions: tuple[str, ...]) -> None:
        for session_id in sessions:
            engine.close_session(session_id)

    @staticmethod
    def _execute(
        engine: DraftEngine, work: tuple[tuple[DecodedMessage, bool], ...]
    ) -> _BatchResult:
        installed, proposals, failures, invalid, jobs = [], [], [], [], []
        for decoded, needs_open in work:
            update = decoded.message
            try:
                if needs_open:
                    engine.open_session(update.session_id)
                engine.install_features(
                    session_id=update.session_id,
                    feature_start=update.feature_start,
                    confirmed_endpoint=update.confirmed_endpoint,
                    features=decoded.features,
                    is_snapshot=update.is_snapshot,
                )
                installed.append(
                    Ack(
                        session_id=update.session_id,
                        confirmed_endpoint=update.confirmed_endpoint,
                        anchor_token=update.anchor_token,
                    )
                )
                jobs.append(
                    DraftPoolJob(
                        session_id=update.session_id,
                        confirmed_endpoint=update.confirmed_endpoint,
                        anchor_token=update.anchor_token,
                    )
                )
            except Exception as exc:
                invalid.append(update.session_id)
                failures.append(
                    Failure(
                        session_id=update.session_id,
                        code="context_install_failed",
                        detail=str(exc)[:256],
                    )
                )
        if jobs:
            try:
                results = engine.draft_batch(jobs)
                expected = {
                    (job.session_id, job.confirmed_endpoint, job.anchor_token)
                    for job in jobs
                }
                actual = [
                    (result.session_id, result.confirmed_endpoint, result.anchor_token)
                    for result in results
                ]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("Worker returned mismatched proposal identities")
                if any(
                    len(result.candidate_ids) != engine.contract.native_block_tokens - 1
                    or any(
                        type(token) is not int
                        or not 0 <= token < engine.contract.vocab_size
                        for token in result.candidate_ids
                    )
                    for result in results
                ):
                    raise ValueError("Worker returned invalid native proposal tokens")
                proposals.extend(
                    Proposal(
                        session_id=result.session_id,
                        confirmed_endpoint=result.confirmed_endpoint,
                        anchor_token=result.anchor_token,
                        candidate_ids=tuple(result.candidate_ids),
                    )
                    for result in results
                )
            except Exception as exc:
                failures.extend(
                    Failure(
                        session_id=job.session_id,
                        code="proposal_failed",
                        detail=str(exc)[:256],
                    )
                    for job in jobs
                )
        return _BatchResult(
            tuple(installed), tuple(proposals), tuple(failures), tuple(invalid)
        )

    def _complete(self) -> None:
        if self._inflight is None or not self._inflight.done():
            return
        result = self._inflight.result()
        self._inflight = None
        if self._closing_ids:
            for session_id in self._closing_ids:
                del self._sessions[session_id]
            self._closing_ids = ()
            return
        for session_id in self._running_ids:
            session = self._sessions[session_id]
            session.pending = False
            if session.staged:
                session.staged = False
                self._staged -= 1
        self._running_ids = ()
        for ack in result.installed:
            session = self._sessions[ack.session_id]
            session.endpoint = ack.confirmed_endpoint
            session.anchor_token = ack.anchor_token
            session.initial = False
            if not session.closing:
                self._reply(session.identity, ack)
        for message in (*result.proposals, *result.failures):
            session = self._sessions[message.session_id]
            if not session.closing:
                self._reply(session.identity, message)
        for session_id in result.invalid_sessions:
            self._close_session(session_id)


def run_draft_worker(
    service_config: DraftPoolServiceConfig, worker_model_config: Any
) -> None:
    """Construct the standalone engine on its data-plane thread and serve.

    ``service_config`` sets bounded private transport resources.
    ``worker_model_config`` is the worker's explicit ``WorkerModelConfig``;
    its heavy model dependencies are imported only in the worker command.
    """
    from tokenspeed.runtime.draft_pool.worker import DFlash2WorkerEngine

    service = DraftPoolService(
        config=service_config,
        engine_factory=lambda: DFlash2WorkerEngine(worker_model_config),
    )
    try:
        service.run()
    finally:
        service.close()
