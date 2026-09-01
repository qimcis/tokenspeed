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

"""Nonblocking shadow hooks for an external speculation control plane.

This module deliberately has no candidate data plane. It exports bounded seal
telemetry and accepts exact-prefix decode-order hints; native autoregressive or
local speculative execution remains the only execution path.
"""

from __future__ import annotations

import hashlib
import ipaddress
import itertools
import json
import queue
import struct
import threading
import time
import urllib.parse
import urllib.request
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

PROTOCOL_VERSION = 1
_MAX_REQUESTS_PER_SEAL = 16_384
_MAX_ID_BYTES = 512
_SAMPLING_CONTRACT_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "min_new_tokens",
    "ignore_eos",
    "stop_token_ids",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
    "logit_bias",
    "seed",
    "custom_params",
)


class RemoteSpecFallbackReason(str, Enum):
    """One primary reason for every shadow opportunity that falls back."""

    NO_PROPOSAL = "no_proposal"
    STATE_UPDATE_PENDING = "state_update_pending"
    DRAFT_QUEUED = "draft_queued"
    DRAFT_IN_FLIGHT = "draft_in_flight"
    LATE_AFTER_SEAL = "late_after_seal"
    WRONG_IDENTITY = "wrong_identity"
    WRONG_EPOCH = "wrong_epoch"
    WRONG_OFFSET = "wrong_offset"
    WRONG_PREFIX = "wrong_prefix"
    WRONG_ANCHOR = "wrong_anchor"
    WRONG_REVISION = "wrong_revision"
    WRONG_WIDTH = "wrong_width"
    MALFORMED_TOKEN = "malformed_token"
    LEASED_ELSEWHERE = "leased_elsewhere"
    ALREADY_CONSUMED = "already_consumed"
    EVICTED = "evicted"
    DEPTH_UNAVAILABLE = "depth_unavailable"
    POLICY_REJECTED = "policy_rejected"
    TARGET_STEP_UNSETTLED = "target_step_unsettled"
    REMOTE_DISABLED = "remote_disabled"
    COORDINATOR_UNAVAILABLE = "coordinator_unavailable"
    ENGINE_REJECTED_HINT = "engine_rejected_hint"
    UNSUPPORTED_SAMPLING = "unsupported_sampling"


@dataclass(frozen=True)
class PrefixStamp:
    request_id: str
    request_incarnation: str
    token_count: int
    digest: str
    anchor_token_id: int | None
    sampling_contract_digest: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_incarnation": self.request_incarnation,
            "token_count": self.token_count,
            "digest": self.digest,
            "anchor_token_id": self.anchor_token_id,
            "sampling_contract_digest": self.sampling_contract_digest,
        }


@dataclass(frozen=True)
class ReadyHint:
    prefix: PrefixStamp
    candidate_id: str
    ready: bool
    priority: int
    draft_depth: int
    target_revision: str
    draft_revision: str
    fallback_reason: RemoteSpecFallbackReason | None = None


@dataclass(frozen=True)
class RemoteSpecDirective:
    generation: int
    created_at_ns: int
    received_at_ns: int
    engine_id: str
    engine_incarnation: str
    requests: tuple[ReadyHint, ...]


@dataclass(frozen=True)
class SemanticChecks:
    identity: bool | None = None
    epoch: bool | None = None
    count: bool | None = None
    digest: bool | None = None
    anchor: bool | None = None
    revision: bool | None = None
    width: bool | None = None
    token_bounds: bool | None = None
    sampling: bool | None = None
    lease: bool | None = None
    consume_once: bool | None = None

    def to_wire(self) -> dict[str, bool | None]:
        return {
            "identity": self.identity,
            "epoch": self.epoch,
            "count": self.count,
            "digest": self.digest,
            "anchor": self.anchor,
            "revision": self.revision,
            "width": self.width,
            "token_bounds": self.token_bounds,
            "sampling": self.sampling,
            "lease": self.lease,
            "consume_once": self.consume_once,
        }


@dataclass(frozen=True)
class RemoteSpecRequestRow:
    prefix: PrefixStamp
    request_pool_index: int | None
    runnable: bool
    native_position: int | None
    selected_position: int | None
    ready_before_ordering: bool
    ready_after_ordering: bool
    candidate_id: str | None
    draft_depth: int | None
    target_revision: str | None
    draft_revision: str | None
    semantic_checks: SemanticChecks
    fallback_reason: RemoteSpecFallbackReason | None
    committed_tokens: int = 0
    committed_digest: str | None = None
    raw_target_tokens: tuple[int, ...] = ()
    committed_target_tokens: tuple[int, ...] = ()
    raw_accepted_draft_tokens: int | None = None
    correction_token: int | None = None
    bonus_token: int | None = None
    tombstoned: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            **self.prefix.to_wire(),
            "request_pool_index": self.request_pool_index,
            "runnable": self.runnable,
            "native_position": self.native_position,
            "selected_position": self.selected_position,
            "ready_before_ordering": self.ready_before_ordering,
            "ready_after_ordering": self.ready_after_ordering,
            "candidate_id": self.candidate_id,
            "draft_depth": self.draft_depth,
            "target_revision": self.target_revision,
            "draft_revision": self.draft_revision,
            "semantic_checks": self.semantic_checks.to_wire(),
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason is not None else None
            ),
            "committed_tokens": self.committed_tokens,
            "committed_digest": self.committed_digest,
            "raw_target_tokens": list(self.raw_target_tokens),
            "committed_target_tokens": list(self.committed_target_tokens),
            "raw_accepted_draft_tokens": self.raw_accepted_draft_tokens,
            "correction_token": self.correction_token,
            "bonus_token": self.bonus_token,
            "tombstoned": self.tombstoned,
        }


@dataclass(frozen=True)
class RemoteSpecSealTelemetry:
    engine_id: str
    engine_incarnation: str
    target_revision: str
    native_speculative_algorithm: str | None
    max_depth: int
    seal_id: str
    generation: int
    sealed_at_ns: int
    committed_at_ns: int | None
    scheduler_stats: Mapping[str, int]
    directive_generation: int | None
    directive_created_at_ns: int | None
    directive_received_at_ns: int | None
    batch_signature: Mapping[str, Any]
    preferred_decode_ids: tuple[str, ...]
    native_decode_ids: tuple[str, ...]
    selected_decode_ids: tuple[str, ...]
    rows: tuple[RemoteSpecRequestRow, ...]

    @property
    def fallback_opportunities(self) -> int:
        return sum(row.selected_position is not None for row in self.rows)

    @property
    def fallback_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            if row.selected_position is None:
                continue
            if row.fallback_reason is None:
                raise ValueError("selected shadow row has no fallback reason")
            key = row.fallback_reason.value
            counts[key] = counts.get(key, 0) + 1
        if sum(counts.values()) != self.fallback_opportunities:
            raise ValueError("fallback counts do not cover every opportunity")
        return counts

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "seal_telemetry",
            "engine_id": self.engine_id,
            "engine_incarnation": self.engine_incarnation,
            "target_revision": self.target_revision,
            "native_speculative_algorithm": self.native_speculative_algorithm,
            "max_depth": self.max_depth,
            "seal_id": self.seal_id,
            "generation": self.generation,
            "sealed_at_ns": self.sealed_at_ns,
            "committed_at_ns": self.committed_at_ns,
            "scheduler_stats": dict(self.scheduler_stats),
            "directive_generation": self.directive_generation,
            "directive_created_at_ns": self.directive_created_at_ns,
            "directive_received_at_ns": self.directive_received_at_ns,
            "batch_signature": dict(self.batch_signature),
            "preferred_decode_ids": list(self.preferred_decode_ids),
            "native_decode_ids": list(self.native_decode_ids),
            "selected_decode_ids": list(self.selected_decode_ids),
            "fallback_opportunities": self.fallback_opportunities,
            "fallback_counts": self.fallback_counts,
            "ready_before_ordering": sum(
                row.ready_before_ordering for row in self.rows
            ),
            "ready_after_ordering": sum(row.ready_after_ordering for row in self.rows),
            "rows": [row.to_wire() for row in self.rows],
        }


@dataclass(frozen=True)
class RemoteSpecBinding:
    telemetry: RemoteSpecSealTelemetry


@dataclass
class _TrackedRequest:
    state_ref: Any
    incarnation: str
    prompt_tokens: tuple[int, ...]
    output_count: int
    hasher: Any


@dataclass(frozen=True)
class _SealContext:
    seal_id: str
    generation: int
    sealed_at_ns: int
    scheduler_stats: Mapping[str, int]
    directive_generation: int | None
    directive_created_at_ns: int | None
    directive_received_at_ns: int | None
    prefixes: Mapping[str, PrefixStamp]
    native_decode_ids: tuple[str, ...]
    ready: Mapping[str, ReadyHint]
    checks: Mapping[str, SemanticChecks]
    rejected: Mapping[str, RemoteSpecFallbackReason]
    preferred_decode_ids: tuple[str, ...]
    coordinator_unavailable: bool


@dataclass(frozen=True)
class _ClientSnapshot:
    directive: RemoteSpecDirective | None
    last_error: str | None


class _BoundedJsonClient:
    """A latest-value HTTP mailbox whose worker never blocks the event loop."""

    def __init__(
        self,
        endpoint: str,
        *,
        capacity: int,
        timeout_secs: float,
        max_message_bytes: int,
        transport: Callable[[bytes], bytes] | None = None,
    ) -> None:
        _validate_local_endpoint(endpoint)
        if capacity <= 0:
            raise ValueError("remote speculation mailbox capacity must be positive")
        if timeout_secs <= 0:
            raise ValueError("remote speculation timeout must be positive")
        if max_message_bytes <= 0:
            raise ValueError("remote speculation message limit must be positive")
        self._endpoint = endpoint
        self._timeout_secs = timeout_secs
        self._close_timeout_secs = min(
            max(timeout_secs * (capacity + 1) + 0.1, 0.1), 5.0
        )
        self._max_message_bytes = max_message_bytes
        self._transport = transport or self._post
        self._mailbox: queue.Queue[Any | None] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._offer_lock = threading.Lock()
        self._dropped_seals_total = 0
        self._closed = False
        self._snapshot = _ClientSnapshot(None, None)
        self._thread = threading.Thread(
            target=self._run, name="remote-spec-shadow", daemon=True
        )
        self._thread.start()

    def offer(self, telemetry: RemoteSpecSealTelemetry) -> None:
        with self._offer_lock:
            if self._closed:
                return
            try:
                self._mailbox.put_nowait(telemetry)
                return
            except queue.Full:
                try:
                    dropped = self._mailbox.get_nowait()
                except queue.Empty:
                    dropped = None
                if dropped is not None:
                    self._dropped_seals_total += 1
            try:
                self._mailbox.put_nowait(telemetry)
            except queue.Full:
                self._dropped_seals_total += 1
                self._set_error("telemetry mailbox remained full after eviction")

    def _encode_telemetry(self, telemetry: RemoteSpecSealTelemetry) -> bytes:
        with self._offer_lock:
            dropped_seals_total = self._dropped_seals_total
        wire = telemetry.to_wire()
        wire["transport_dropped_seals_total"] = dropped_seals_total
        return json.dumps(wire, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def snapshot(self) -> _ClientSnapshot:
        with self._lock:
            return self._snapshot

    def close(self) -> None:
        with self._offer_lock:
            if self._closed:
                should_signal = False
            else:
                self._closed = True
                should_signal = True
        if should_signal:
            try:
                self._mailbox.put(
                    None,
                    timeout=min(max(self._timeout_secs * 2, 0.1), 1.0),
                )
            except queue.Full as exc:
                raise RuntimeError(
                    "remote speculation background client did not accept shutdown"
                ) from exc
        self._thread.join(timeout=self._close_timeout_secs)
        if self._thread.is_alive():
            raise RuntimeError("remote speculation background client did not stop")

    def _mark_dropped(self) -> None:
        with self._offer_lock:
            self._dropped_seals_total += 1

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = _ClientSnapshot(self._snapshot.directive, message)

    def _run(self) -> None:
        while True:
            telemetry = self._mailbox.get()
            if telemetry is None:
                return
            try:
                payload = self._encode_telemetry(telemetry)
            except Exception as exc:
                self._mark_dropped()
                self._set_error(f"{type(exc).__name__}: telemetry encoding failed")
                continue
            if len(payload) > self._max_message_bytes:
                self._mark_dropped()
                self._set_error("outbound telemetry exceeds configured byte limit")
                continue
            try:
                raw = self._transport(payload)
                if len(raw) > self._max_message_bytes:
                    raise ValueError(
                        "coordinator response exceeds configured byte limit"
                    )
                directive = _parse_directive(json.loads(raw.decode("utf-8")))
            except Exception as exc:  # the shadow path is failure-independent
                self._set_error(f"{type(exc).__name__}: {exc}")
                continue
            with self._lock:
                current = self._snapshot.directive
                if current is None or directive.generation > current.generation:
                    self._snapshot = _ClientSnapshot(directive, None)

    def _post(self, payload: bytes) -> bytes:
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_secs) as response:
            content_length = response.headers.get("Content-Length")
            if (
                content_length is not None
                and int(content_length) > self._max_message_bytes
            ):
                raise ValueError("coordinator response exceeds configured byte limit")
            raw = response.read(self._max_message_bytes + 1)
        return raw


def _validate_local_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote speculation endpoint must be an HTTP(S) URL")
    host = parsed.hostname.lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "remote speculation shadow endpoint must be loopback-local"
            ) from exc
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("remote speculation endpoint contains unsupported URL fields")


def _bounded_string(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_ID_BYTES
    ):
        raise ValueError(f"{field_name} must be a nonempty bounded string")
    return value


def _digest_string(value: Any, field_name: str) -> str:
    value = _bounded_string(value, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _parse_directive(payload: Any) -> RemoteSpecDirective:
    if (
        not isinstance(payload, dict)
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError("unsupported remote speculation directive")
    requests = payload.get("requests", [])
    if not isinstance(requests, list) or len(requests) > _MAX_REQUESTS_PER_SEAL:
        raise ValueError("directive requests must be a bounded list")
    hints = []
    request_ids: set[str] = set()
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("directive request entries must be objects")
        anchor = item.get("anchor_token_id")
        if anchor is not None and (
            isinstance(anchor, bool) or not isinstance(anchor, int)
        ):
            raise ValueError("anchor_token_id must be an integer or null")
        if not isinstance(item.get("ready"), bool):
            raise ValueError("ready must be a boolean")
        reason = item.get("fallback_reason")
        request_id = _bounded_string(item.get("request_id"), "request_id")
        if request_id in request_ids:
            raise ValueError("directive contains duplicate request hints")
        request_ids.add(request_id)
        hints.append(
            ReadyHint(
                prefix=PrefixStamp(
                    request_id=request_id,
                    request_incarnation=_bounded_string(
                        item.get("request_incarnation"), "request_incarnation"
                    ),
                    token_count=_nonnegative_int(
                        item.get("token_count"), "token_count"
                    ),
                    digest=_digest_string(item.get("digest"), "digest"),
                    anchor_token_id=anchor,
                    sampling_contract_digest=_digest_string(
                        item.get("sampling_contract_digest"),
                        "sampling_contract_digest",
                    ),
                ),
                candidate_id=_bounded_string(item.get("candidate_id"), "candidate_id"),
                ready=item["ready"],
                priority=_strict_int(item.get("priority"), "priority"),
                draft_depth=_nonnegative_int(item.get("draft_depth"), "draft_depth"),
                target_revision=_bounded_string(
                    item.get("target_revision", "unknown"), "target_revision"
                ),
                draft_revision=_bounded_string(
                    item.get("draft_revision", "unknown"), "draft_revision"
                ),
                fallback_reason=(
                    RemoteSpecFallbackReason(reason) if reason is not None else None
                ),
            )
        )
    return RemoteSpecDirective(
        generation=_nonnegative_int(payload.get("generation"), "generation"),
        created_at_ns=_nonnegative_int(payload.get("created_at_ns"), "created_at_ns"),
        received_at_ns=time.monotonic_ns(),
        engine_id=_bounded_string(payload.get("engine_id"), "engine_id"),
        engine_incarnation=_bounded_string(
            payload.get("engine_incarnation"), "engine_incarnation"
        ),
        requests=tuple(hints),
    )


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


class RemoteSpecHooks:
    """Control-plane-only hooks for shadow readiness and soft decode ordering."""

    def __init__(
        self,
        *,
        mode: str = "off",
        endpoint: str | None = None,
        engine_id: str = "tokenspeed",
        mailbox_capacity: int = 8,
        timeout_secs: float = 0.1,
        max_message_bytes: int = 1 << 20,
        max_hint_age_ms: int = 1_000,
        target_revision: str = "unknown",
        native_speculative_algorithm: str | None = None,
        max_depth: int = 8,
        local_spec_width: int = 0,
        attn_tp_rank: int = 0,
        attn_tp_size: int = 1,
        attn_tp_cpu_group: Any = None,
        attn_tp_src_global_rank: int = 0,
        client: Any = None,
    ) -> None:
        if mode not in {"off", "shadow"}:
            raise ValueError("remote speculation mode must be 'off' or 'shadow'")
        if max_hint_age_ms <= 0:
            raise ValueError("remote speculation hint age must be positive")
        if max_depth <= 0:
            raise ValueError("remote speculation max depth must be positive")
        self.enabled = mode == "shadow"
        self._engine_id = engine_id
        self._engine_incarnation = uuid.uuid4().hex
        self._max_hint_age_ns = max_hint_age_ms * 1_000_000
        self._target_revision = target_revision
        self._native_speculative_algorithm = native_speculative_algorithm
        self._max_depth = max_depth
        self._local_spec_width = max(local_spec_width, 0)
        self._attn_tp_rank = attn_tp_rank
        self._attn_tp_size = attn_tp_size
        self._attn_tp_cpu_group = attn_tp_cpu_group
        self._attn_tp_src_global_rank = attn_tp_src_global_rank
        self._generation = 0
        self._tracked: dict[str, _TrackedRequest] = {}
        self._request_epochs: dict[str, int] = {}
        self._open: _SealContext | None = None
        self._last_hook_error: str | None = None
        if not self.enabled:
            self._client = None
        elif client is not None:
            self._client = client if attn_tp_rank == 0 else None
        else:
            if endpoint is None:
                raise ValueError("shadow remote speculation requires an endpoint")
            self._client = (
                _BoundedJsonClient(
                    endpoint,
                    capacity=mailbox_capacity,
                    timeout_secs=timeout_secs,
                    max_message_bytes=max_message_bytes,
                )
                if attn_tp_rank == 0
                else None
            )

    def before_plan(
        self,
        rid_to_state: Mapping[str, Any],
        scheduler_stats: Mapping[str, int] | Callable[[], Mapping[str, int]],
        now_ns: int | None = None,
        unsettled_request_ids: Iterable[str] = (),
    ) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        preferred_ids: tuple[str, ...] = ()
        if self._attn_tp_rank == 0:
            try:
                preferred_ids = self._before_plan_root(
                    rid_to_state,
                    scheduler_stats,
                    now_ns,
                    unsettled_request_ids,
                )
            except Exception as exc:
                self._open = None
                self._record_hook_error("before_plan", exc)
        try:
            broadcast_ids = self._broadcast_preferences(preferred_ids)
        except Exception as exc:
            self._record_hook_error("preference_broadcast", exc)
            broadcast_ids = ()
        if (
            self._attn_tp_rank == 0
            and self._open is not None
            and self._open.preferred_decode_ids != broadcast_ids
        ):
            self._open = replace(
                self._open,
                preferred_decode_ids=broadcast_ids,
            )
        return broadcast_ids

    def _before_plan_root(
        self,
        rid_to_state: Mapping[str, Any],
        scheduler_stats: Mapping[str, int] | Callable[[], Mapping[str, int]],
        now_ns: int | None = None,
        unsettled_request_ids: Iterable[str] = (),
    ) -> tuple[str, ...]:
        if callable(scheduler_stats):
            scheduler_stats = scheduler_stats()
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        client_snapshot = self._safe_client_snapshot()
        self._generation += 1
        prefixes: dict[str, PrefixStamp] = {}
        native_ids = []
        live = set(rid_to_state)
        for request_id in tuple(self._tracked):
            if request_id not in live:
                del self._tracked[request_id]
        for request_id, state in itertools.islice(
            rid_to_state.items(), _MAX_REQUESTS_PER_SEAL
        ):
            stamp = self._prefix_stamp(request_id, state)
            prefixes[request_id] = stamp
            if (
                bool(getattr(state, "prefill_finished", False))
                and getattr(state, "finished_reason", None) is None
                and not bool(getattr(state, "to_abort", False))
            ):
                native_ids.append(request_id)

        unsettled = set(unsettled_request_ids)
        observed_directive = client_snapshot.directive
        directive = observed_directive
        expired_request_ids: set[str] = set()
        if (
            directive is not None
            and now_ns - directive.received_at_ns > self._max_hint_age_ns
        ):
            expired_request_ids = {
                hint.prefix.request_id for hint in directive.requests
            }
            directive = None
        ready: dict[str, ReadyHint] = {}
        checks: dict[str, SemanticChecks] = {}
        rejected: dict[str, RemoteSpecFallbackReason] = {
            request_id: RemoteSpecFallbackReason.TARGET_STEP_UNSETTLED
            for request_id in unsettled
            if request_id in prefixes
        }
        rejected.update(
            {
                request_id: RemoteSpecFallbackReason.LATE_AFTER_SEAL
                for request_id in expired_request_ids
                if request_id in prefixes and request_id not in unsettled
            }
        )
        if directive is not None:
            for hint in directive.requests:
                actual = prefixes.get(hint.prefix.request_id)
                hint_checks = _semantic_checks(
                    actual,
                    hint,
                    target_revision=self._target_revision,
                    max_depth=self._max_depth,
                )
                checks[hint.prefix.request_id] = hint_checks
                reason = _semantic_mismatch_reason(hint_checks)
                if directive.engine_id != self._engine_id:
                    rejected[hint.prefix.request_id] = (
                        RemoteSpecFallbackReason.WRONG_IDENTITY
                    )
                elif directive.engine_incarnation != self._engine_incarnation:
                    rejected[hint.prefix.request_id] = (
                        RemoteSpecFallbackReason.WRONG_EPOCH
                    )
                elif hint.prefix.request_id in unsettled:
                    rejected[hint.prefix.request_id] = (
                        RemoteSpecFallbackReason.TARGET_STEP_UNSETTLED
                    )
                elif reason is not None:
                    rejected[hint.prefix.request_id] = reason
                elif hint.fallback_reason is not None:
                    rejected[hint.prefix.request_id] = hint.fallback_reason
                elif hint.ready:
                    ready[hint.prefix.request_id] = hint
                else:
                    rejected[hint.prefix.request_id] = (
                        RemoteSpecFallbackReason.DRAFT_IN_FLIGHT
                    )
        preferred_ids = tuple(
            hint.prefix.request_id
            for hint in sorted(ready.values(), key=lambda item: item.priority)
            if hint.prefix.request_id in native_ids
        )
        self._open = _SealContext(
            seal_id=uuid.uuid4().hex,
            generation=self._generation,
            sealed_at_ns=now_ns,
            scheduler_stats={key: int(value) for key, value in scheduler_stats.items()},
            directive_generation=(
                observed_directive.generation
                if observed_directive is not None
                else None
            ),
            directive_created_at_ns=(
                observed_directive.created_at_ns
                if observed_directive is not None
                else None
            ),
            directive_received_at_ns=(
                observed_directive.received_at_ns
                if observed_directive is not None
                else None
            ),
            prefixes=prefixes,
            native_decode_ids=tuple(native_ids),
            ready=ready,
            checks=checks,
            rejected=rejected,
            preferred_decode_ids=preferred_ids,
            coordinator_unavailable=(
                client_snapshot.last_error is not None
                or self._last_hook_error is not None
            ),
        )
        return preferred_ids

    def bind_plan(
        self, forward_op: Any, sampling_params_list: Sequence[Any] = ()
    ) -> RemoteSpecBinding | None:
        if not self.enabled or self._attn_tp_rank != 0:
            return None
        try:
            return self._bind_plan_root(forward_op, sampling_params_list)
        except Exception as exc:
            self._open = None
            self._record_hook_error("bind_plan", exc)
            return None

    def _bind_plan_root(
        self, forward_op: Any, sampling_params_list: Sequence[Any] = ()
    ) -> RemoteSpecBinding | None:
        del sampling_params_list
        if self._open is None:
            return None
        context, self._open = self._open, None
        selected = (
            tuple(forward_op.request_ids[forward_op.num_extends() :])
            if forward_op is not None
            else ()
        )
        native_position = {
            request_id: index
            for index, request_id in enumerate(context.native_decode_ids)
        }
        selected_position = {
            request_id: index for index, request_id in enumerate(selected)
        }
        pool_index_by_request = (
            {
                request_id: int(pool_index)
                for request_id, pool_index in zip(
                    forward_op.request_ids,
                    getattr(forward_op, "request_pool_indices", ()),
                )
            }
            if forward_op is not None
            else {}
        )
        batch_signature = (
            {
                "request_ids": list(forward_op.request_ids),
                "request_pool_indices": list(
                    getattr(forward_op, "request_pool_indices", ())
                ),
                "input_lengths": list(getattr(forward_op, "input_lengths", ())),
                "extend_prefix_lens": list(
                    getattr(forward_op, "extend_prefix_lens", ())
                ),
                "num_extends": int(forward_op.num_extends()),
            }
            if forward_op is not None
            else {
                "request_ids": [],
                "request_pool_indices": [],
                "input_lengths": [],
                "extend_prefix_lens": [],
                "num_extends": 0,
            }
        )
        rows = []
        for request_id, prefix in context.prefixes.items():
            is_selected = request_id in selected_position
            is_runnable = request_id in native_position
            hint = context.ready.get(request_id)
            reason = None
            if is_selected:
                if hint is not None:
                    reason = RemoteSpecFallbackReason.REMOTE_DISABLED
                elif request_id in context.rejected:
                    reason = context.rejected[request_id]
                elif context.coordinator_unavailable:
                    reason = RemoteSpecFallbackReason.COORDINATOR_UNAVAILABLE
                else:
                    reason = RemoteSpecFallbackReason.NO_PROPOSAL
            rows.append(
                RemoteSpecRequestRow(
                    prefix=prefix,
                    request_pool_index=pool_index_by_request.get(request_id),
                    runnable=is_runnable,
                    native_position=native_position.get(request_id),
                    selected_position=selected_position.get(request_id),
                    ready_before_ordering=hint is not None and is_runnable,
                    ready_after_ordering=hint is not None and is_selected,
                    candidate_id=hint.candidate_id if hint is not None else None,
                    draft_depth=hint.draft_depth if hint is not None else None,
                    target_revision=(
                        hint.target_revision if hint is not None else None
                    ),
                    draft_revision=(hint.draft_revision if hint is not None else None),
                    semantic_checks=context.checks.get(request_id, SemanticChecks()),
                    fallback_reason=reason,
                )
            )
        return RemoteSpecBinding(
            RemoteSpecSealTelemetry(
                engine_id=self._engine_id,
                engine_incarnation=self._engine_incarnation,
                target_revision=self._target_revision,
                native_speculative_algorithm=self._native_speculative_algorithm,
                max_depth=self._max_depth,
                seal_id=context.seal_id,
                generation=context.generation,
                sealed_at_ns=context.sealed_at_ns,
                committed_at_ns=None,
                scheduler_stats=context.scheduler_stats,
                directive_generation=context.directive_generation,
                directive_created_at_ns=context.directive_created_at_ns,
                directive_received_at_ns=context.directive_received_at_ns,
                batch_signature=batch_signature,
                preferred_decode_ids=context.preferred_decode_ids,
                native_decode_ids=context.native_decode_ids,
                selected_decode_ids=selected,
                rows=tuple(rows),
            )
        )

    def observe_commit(
        self,
        forward_op: Any,
        results: Any,
        request_changes: Sequence[Any],
        binding: RemoteSpecBinding | None,
        rid_to_state: Mapping[str, Any] | None = None,
        now_ns: int | None = None,
    ) -> None:
        if not self.enabled or self._attn_tp_rank != 0 or binding is None:
            return
        try:
            self._observe_commit_root(
                forward_op,
                results,
                request_changes,
                binding,
                rid_to_state,
                now_ns,
            )
        except Exception as exc:
            self._record_hook_error("observe_commit", exc)

    def _observe_commit_root(
        self,
        forward_op: Any,
        results: Any,
        request_changes: Sequence[Any],
        binding: RemoteSpecBinding,
        rid_to_state: Mapping[str, Any] | None,
        now_ns: int | None,
    ) -> None:
        tokens: dict[str, list[int]] = {}
        for event in request_changes:
            request_id = getattr(event, "request_id", None)
            event_tokens = getattr(event, "tokens", None)
            if isinstance(request_id, str) and event_tokens is not None:
                tokens.setdefault(request_id, []).extend(
                    int(token) for token in event_tokens
                )
        raw_results = self._raw_results_by_request(forward_op, results)
        rows = []
        for row in binding.telemetry.rows:
            committed = tokens.get(row.prefix.request_id, [])
            raw = raw_results.get(row.prefix.request_id, ())
            accepted = None
            correction = None
            bonus = None
            if self._local_spec_width > 0 and row.selected_position is not None:
                accepted = max(0, len(raw) - 1)
                if raw:
                    if len(raw) >= self._local_spec_width:
                        bonus = raw[-1]
                    else:
                        correction = raw[-1]
            rows.append(
                replace(
                    row,
                    committed_tokens=len(committed),
                    committed_digest=(_digest_tokens(committed) if committed else None),
                    raw_target_tokens=raw,
                    committed_target_tokens=tuple(committed),
                    raw_accepted_draft_tokens=accepted,
                    correction_token=correction,
                    bonus_token=bonus,
                    tombstoned=(
                        rid_to_state is not None
                        and row.prefix.request_id not in rid_to_state
                    ),
                )
            )
        telemetry = replace(
            binding.telemetry,
            committed_at_ns=time.monotonic_ns() if now_ns is None else now_ns,
            rows=tuple(rows),
        )
        self._offer(telemetry)

    def _raw_results_by_request(
        self, forward_op: Any, results: Any
    ) -> dict[str, tuple[int, ...]]:
        if forward_op is None or results is None:
            return {}
        lengths = results.output_lengths.tolist()
        flat_tokens = results.output_tokens.tolist()
        num_extends = int(forward_op.num_extends())
        offset = 0
        output: dict[str, tuple[int, ...]] = {}
        for index, request_id in enumerate(forward_op.request_ids):
            length = int(lengths[index])
            output[request_id] = tuple(
                int(token) for token in flat_tokens[offset : offset + length]
            )
            if self._local_spec_width > 0 and index >= num_extends:
                offset += self._local_spec_width
            else:
                offset += length
        return output

    def observe_unlaunched(self, binding: RemoteSpecBinding | None) -> None:
        if not self.enabled or self._attn_tp_rank != 0 or binding is None:
            return
        try:
            rows = tuple(
                replace(
                    row,
                    fallback_reason=(
                        RemoteSpecFallbackReason.ENGINE_REJECTED_HINT
                        if row.selected_position is not None
                        else row.fallback_reason
                    ),
                )
                for row in binding.telemetry.rows
            )
            self._offer(replace(binding.telemetry, rows=rows))
        except Exception as exc:
            self._record_hook_error("observe_unlaunched", exc)

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception as exc:
            self._record_hook_error("close", exc)

    def _offer(self, telemetry: RemoteSpecSealTelemetry) -> None:
        if self._attn_tp_rank == 0 and self._client is not None:
            self._client.offer(telemetry)

    def _safe_client_snapshot(self) -> _ClientSnapshot:
        if self._client is None:
            return _ClientSnapshot(None, None)
        try:
            snapshot = self._client.snapshot()
            directive = getattr(snapshot, "directive")
            last_error = getattr(snapshot, "last_error")
            if directive is not None and not isinstance(directive, RemoteSpecDirective):
                raise TypeError("client returned an invalid directive object")
            if last_error is not None and not isinstance(last_error, str):
                raise TypeError("client returned an invalid error value")
            return _ClientSnapshot(directive, last_error)
        except Exception as exc:
            self._record_hook_error("client_snapshot", exc)
            return _ClientSnapshot(None, f"{type(exc).__name__}: {exc}")

    def _broadcast_preferences(self, preferred_ids: tuple[str, ...]) -> tuple[str, ...]:
        if self._attn_tp_size <= 1:
            return preferred_ids
        import torch.distributed as dist

        payload = [
            (
                (self._engine_incarnation, preferred_ids)
                if self._attn_tp_rank == 0
                else None
            )
        ]
        dist.broadcast_object_list(
            payload,
            src=self._attn_tp_src_global_rank,
            group=self._attn_tp_cpu_group,
        )
        engine_incarnation, preferred_ids = payload[0]
        if not isinstance(engine_incarnation, str) or not isinstance(
            preferred_ids, tuple
        ):
            return ()
        if not all(isinstance(request_id, str) for request_id in preferred_ids):
            return ()
        self._engine_incarnation = engine_incarnation
        return preferred_ids

    def _record_hook_error(self, stage: str, exc: Exception) -> None:
        try:
            detail = str(exc)
        except Exception:
            detail = "unprintable exception"
        self._last_hook_error = f"{stage}: {type(exc).__name__}: {detail}"

    def _prefix_stamp(self, request_id: str, state: Any) -> PrefixStamp:
        prompt = tuple(int(token) for token in getattr(state, "prompt_input_ids", ()))
        output = [int(token) for token in getattr(state, "output_ids", ())]
        tracked = self._tracked.get(request_id)
        if (
            tracked is None
            or tracked.state_ref() is not state
            or tracked.prompt_tokens != prompt
            or len(output) < tracked.output_count
        ):
            hasher = hashlib.sha256()
            _update_token_digest(hasher, prompt)
            epoch = self._request_epochs.get(request_id, 0) + 1
            self._request_epochs[request_id] = epoch
            incarnation = hashlib.sha256(
                f"{self._engine_id}\0{self._engine_incarnation}\0{request_id}\0{epoch}".encode(
                    "utf-8"
                )
            ).hexdigest()
            tracked = _TrackedRequest(
                weakref.ref(state), incarnation, prompt, 0, hasher
            )
            self._tracked[request_id] = tracked
        _update_token_digest(tracked.hasher, output[tracked.output_count :])
        tracked.output_count = len(output)
        all_count = len(prompt) + len(output)
        anchor = output[-1] if output else (prompt[-1] if prompt else None)
        return PrefixStamp(
            request_id=request_id,
            request_incarnation=tracked.incarnation,
            token_count=all_count,
            digest=tracked.hasher.hexdigest(),
            anchor_token_id=anchor,
            sampling_contract_digest=_sampling_contract_digest(
                getattr(state, "sampling_params", None)
            ),
        )


def _semantic_checks(
    actual: PrefixStamp | None,
    hint: ReadyHint,
    *,
    target_revision: str,
    max_depth: int,
) -> SemanticChecks:
    proposed = hint.prefix
    identity = actual is not None and actual.request_id == proposed.request_id
    epoch = (
        actual is not None
        and actual.request_incarnation == proposed.request_incarnation
    )
    count = actual is not None and actual.token_count == proposed.token_count
    digest = actual is not None and actual.digest == proposed.digest
    anchor = actual is not None and actual.anchor_token_id == proposed.anchor_token_id
    sampling = (
        actual is not None
        and actual.sampling_contract_digest == proposed.sampling_contract_digest
    )
    return SemanticChecks(
        identity=identity,
        epoch=epoch,
        count=count,
        digest=digest,
        anchor=anchor,
        revision=hint.target_revision == target_revision,
        width=0 < hint.draft_depth <= max_depth,
        token_bounds=None,
        sampling=sampling,
        lease=None,
        consume_once=None,
    )


def _semantic_mismatch_reason(
    checks: SemanticChecks,
) -> RemoteSpecFallbackReason | None:
    if not checks.identity:
        return RemoteSpecFallbackReason.WRONG_IDENTITY
    if not checks.epoch:
        return RemoteSpecFallbackReason.WRONG_EPOCH
    if not checks.count:
        return RemoteSpecFallbackReason.WRONG_OFFSET
    if not checks.digest:
        return RemoteSpecFallbackReason.WRONG_PREFIX
    if not checks.anchor:
        return RemoteSpecFallbackReason.WRONG_ANCHOR
    if not checks.revision:
        return RemoteSpecFallbackReason.WRONG_REVISION
    if not checks.width:
        return RemoteSpecFallbackReason.WRONG_WIDTH
    if not checks.sampling:
        return RemoteSpecFallbackReason.UNSUPPORTED_SAMPLING
    return None


def _update_token_digest(hasher: Any, tokens: Sequence[int]) -> None:
    for token in tokens:
        hasher.update(struct.pack("<i", int(token)))


def _digest_tokens(tokens: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    _update_token_digest(hasher, tokens)
    return hasher.hexdigest()


def _sampling_contract_digest(sampling_params: Any) -> str:
    contract = {
        field_name: _canonical_contract_value(
            getattr(sampling_params, field_name, None)
        )
        for field_name in _SAMPLING_CONTRACT_FIELDS
    }
    encoded = json.dumps(
        contract,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return {"non_finite_float": repr(value)}
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_contract_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, separators=(",", ":"), sort_keys=True),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_contract_value(item) for item in value]
    return {"unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}"}
