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

import json
import threading
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine.remote_spec import (
    PrefixStamp,
    ReadyHint,
    RemoteSpecDirective,
    RemoteSpecFallbackReason,
    RemoteSpecHooks,
    _BoundedJsonClient,
    _parse_directive,
)


@dataclass
class _State:
    prompt_input_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    prefill_finished: bool = True
    finished_reason: object | None = None
    to_abort: bool = False


@dataclass
class _Forward:
    request_ids: list[str]
    extends: int = 0

    def num_extends(self) -> int:
        return self.extends


@dataclass
class _Tensor:
    values: list[int]

    def tolist(self) -> list[int]:
        return self.values


@dataclass
class _Results:
    output_lengths: _Tensor
    output_tokens: _Tensor


@dataclass
class _Event:
    request_id: str
    tokens: list[int]


class _Client:
    def __init__(self) -> None:
        self.directive = None
        self.last_error = None
        self.offered = []
        self.closed = False

    def snapshot(self):
        return type(
            "Snapshot",
            (),
            {"directive": self.directive, "last_error": self.last_error},
        )()

    def offer(self, telemetry) -> None:
        self.offered.append(telemetry)

    def close(self) -> None:
        self.closed = True


class _ExplodingClient(_Client):
    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage

    def snapshot(self):
        if self.stage == "snapshot":
            raise RuntimeError("snapshot exploded")
        return super().snapshot()

    def offer(self, telemetry) -> None:
        if self.stage == "offer":
            raise RuntimeError("offer exploded")
        super().offer(telemetry)

    def close(self) -> None:
        if self.stage == "close":
            raise RuntimeError("close exploded")
        super().close()


class _Telemetry:
    def to_wire(self) -> dict:
        return {"protocol_version": 1, "generation": 1}


def _hint(prefix: PrefixStamp, *, priority: int = 0) -> ReadyHint:
    return ReadyHint(
        prefix=prefix,
        candidate_id="candidate-1",
        ready=True,
        priority=priority,
        draft_depth=2,
        target_revision="unknown",
        draft_revision="draft-revision",
    )


def _directive(
    binding,
    *hints: ReadyHint,
    generation: int = 1,
    created_at_ns: int = 1,
    received_at_ns: int = 1,
) -> RemoteSpecDirective:
    telemetry = binding.telemetry
    return RemoteSpecDirective(
        generation=generation,
        created_at_ns=created_at_ns,
        received_at_ns=received_at_ns,
        engine_id=telemetry.engine_id,
        engine_incarnation=telemetry.engine_incarnation,
        requests=tuple(hints),
    )


def _directive_payload(
    *,
    generation: int = 1,
    created_at_ns: int = 1,
    engine_id: str = "engine",
    engine_incarnation: str = "incarnation",
    requests: list[dict] | None = None,
) -> dict:
    return {
        "protocol_version": 1,
        "generation": generation,
        "created_at_ns": created_at_ns,
        "engine_id": engine_id,
        "engine_incarnation": engine_incarnation,
        "requests": requests or [],
    }


def test_fallback_reason_vocabulary_is_exhaustive_and_stable() -> None:
    assert [reason.name for reason in RemoteSpecFallbackReason] == [
        "NO_PROPOSAL",
        "STATE_UPDATE_PENDING",
        "DRAFT_QUEUED",
        "DRAFT_IN_FLIGHT",
        "LATE_AFTER_SEAL",
        "WRONG_IDENTITY",
        "WRONG_EPOCH",
        "WRONG_OFFSET",
        "WRONG_PREFIX",
        "WRONG_ANCHOR",
        "WRONG_REVISION",
        "WRONG_WIDTH",
        "MALFORMED_TOKEN",
        "LEASED_ELSEWHERE",
        "ALREADY_CONSUMED",
        "EVICTED",
        "DEPTH_UNAVAILABLE",
        "POLICY_REJECTED",
        "TARGET_STEP_UNSETTLED",
        "REMOTE_DISABLED",
        "COORDINATOR_UNAVAILABLE",
        "ENGINE_REJECTED_HINT",
        "UNSUPPORTED_SAMPLING",
    ]


def test_disabled_hooks_are_a_true_noop() -> None:
    hooks = RemoteSpecHooks(mode="off", endpoint="http://not-local.invalid")
    assert hooks.before_plan({}, {}, now_ns=1) == ()
    assert hooks.bind_plan(None) is None
    hooks.observe_commit(None, None, [], None)
    hooks.close()


def test_shadow_endpoint_must_be_loopback_local() -> None:
    with pytest.raises(ValueError, match="loopback-local"):
        RemoteSpecHooks(mode="shadow", endpoint="http://example.com:9000")


def test_background_transport_failure_is_contained_off_the_caller() -> None:
    attempted = threading.Event()

    def fail_transport(payload: bytes) -> bytes:
        assert payload
        attempted.set()
        raise OSError("offline")

    client = _BoundedJsonClient(
        "http://127.0.0.1:9000",
        capacity=1,
        timeout_secs=0.1,
        max_message_bytes=1_024,
        transport=fail_transport,
    )
    started = time.monotonic()
    client.offer(_Telemetry())
    assert time.monotonic() - started < 0.05
    assert attempted.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while client.snapshot().last_error is None and time.monotonic() < deadline:
        time.sleep(0.001)
    error = client.snapshot().last_error
    assert error is not None
    assert "offline" in error
    client.close()


def test_telemetry_encoding_occurs_off_the_event_loop() -> None:
    encoding_started = threading.Event()
    release_encoding = threading.Event()

    class SlowTelemetry:
        def to_wire(self) -> dict:
            encoding_started.set()
            assert release_encoding.wait(timeout=1.0)
            return {"protocol_version": 1, "generation": 1}

    client = _BoundedJsonClient(
        "http://127.0.0.1:9000",
        capacity=1,
        timeout_secs=0.1,
        max_message_bytes=1_024,
        transport=lambda payload: json.dumps(_directive_payload()).encode(),
    )
    started = time.monotonic()
    client.offer(SlowTelemetry())
    assert time.monotonic() - started < 0.05
    assert encoding_started.wait(timeout=1.0)
    release_encoding.set()
    client.close()


def test_equal_generation_cannot_replace_an_existing_directive() -> None:
    calls = 0
    completed = threading.Event()

    def transport(payload: bytes) -> bytes:
        nonlocal calls
        assert payload
        calls += 1
        if calls == 2:
            completed.set()
        return json.dumps(
            _directive_payload(generation=7, created_at_ns=calls)
        ).encode()

    client = _BoundedJsonClient(
        "http://127.0.0.1:9000",
        capacity=2,
        timeout_secs=0.1,
        max_message_bytes=1_024,
        transport=transport,
    )
    client.offer(_Telemetry())
    deadline = time.monotonic() + 1.0
    while client.snapshot().directive is None and time.monotonic() < deadline:
        time.sleep(0.001)
    first = client.snapshot().directive
    assert first is not None
    assert first.created_at_ns == 1

    client.offer(_Telemetry())
    assert completed.wait(timeout=1.0)
    client.close()
    assert client.snapshot().directive is first


def test_mailbox_overflow_marks_the_trace_gap_without_blocking() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    delivered = []

    def transport(payload: bytes) -> bytes:
        delivered.append(json.loads(payload))
        if len(delivered) == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        return json.dumps(_directive_payload(generation=len(delivered))).encode()

    client = _BoundedJsonClient(
        "http://127.0.0.1:9000",
        capacity=1,
        timeout_secs=0.1,
        max_message_bytes=1_024,
        transport=transport,
    )
    client.offer(_Telemetry())
    assert first_started.wait(timeout=1.0)
    client.offer(_Telemetry())
    started = time.monotonic()
    client.offer(_Telemetry())
    assert time.monotonic() - started < 0.05
    release_first.set()
    deadline = time.monotonic() + 1.0
    while len(delivered) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(delivered) == 2
    assert delivered[0]["transport_dropped_seals_total"] == 0
    assert delivered[1]["transport_dropped_seals_total"] == 1
    client.close()


def test_directive_parser_rejects_duplicate_request_hints() -> None:
    request = {
        "request_id": "r",
        "request_incarnation": "request-incarnation",
        "token_count": 1,
        "digest": "0" * 64,
        "anchor_token_id": 1,
        "sampling_contract_digest": "1" * 64,
        "candidate_id": "candidate",
        "ready": True,
        "priority": 0,
        "draft_depth": 1,
        "target_revision": "target",
        "draft_revision": "draft",
    }
    payload = _directive_payload(requests=[request, dict(request)])
    with pytest.raises(ValueError, match="duplicate request hints"):
        _parse_directive(payload)


@pytest.mark.parametrize(
    ("field_name", "fallback_reason"),
    [
        ("engine_id", RemoteSpecFallbackReason.WRONG_IDENTITY),
        ("engine_incarnation", RemoteSpecFallbackReason.WRONG_EPOCH),
    ],
)
def test_every_directive_path_requires_exact_engine_incarnation_fencing(
    field_name: str, fallback_reason: RemoteSpecFallbackReason
) -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1])
    hooks.before_plan({"r": state}, {}, now_ns=1)
    initial = hooks.bind_plan(_Forward(["r"]))
    assert initial is not None
    prefix = initial.telemetry.rows[0].prefix
    directive = _directive(initial, _hint(prefix), received_at_ns=2)
    client.directive = replace(directive, **{field_name: "wrong"})

    assert hooks.before_plan({"r": state}, {}, now_ns=3) == ()
    binding = hooks.bind_plan(_Forward(["r"]))
    assert binding is not None
    assert binding.telemetry.rows[0].fallback_reason is fallback_reason


def test_before_plan_snapshot_failure_is_fail_open() -> None:
    hooks = RemoteSpecHooks(mode="shadow", client=_ExplodingClient("snapshot"))
    state = _State([1])
    assert hooks.before_plan({"r": state}, {}, now_ns=1) == ()
    binding = hooks.bind_plan(_Forward(["r"]))
    assert binding is not None
    assert (
        binding.telemetry.rows[0].fallback_reason
        is RemoteSpecFallbackReason.COORDINATOR_UNAVAILABLE
    )


def test_before_plan_prefix_failure_is_fail_open() -> None:
    hooks = RemoteSpecHooks(mode="shadow", client=_Client())
    assert hooks.before_plan({"r": object()}, {}, now_ns=1) == ()
    assert hooks.bind_plan(_Forward(["r"])) is None


def test_preference_broadcast_failure_is_fail_open_and_clears_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1])
    hooks.before_plan({"r": state}, {}, now_ns=1)
    initial = hooks.bind_plan(_Forward(["r"]))
    assert initial is not None
    prefix = initial.telemetry.rows[0].prefix
    client.directive = _directive(initial, _hint(prefix), received_at_ns=2)

    def fail_broadcast(preferred_ids):
        del preferred_ids
        raise RuntimeError("broadcast exploded")

    monkeypatch.setattr(hooks, "_broadcast_preferences", fail_broadcast)
    assert hooks.before_plan({"r": state}, {}, now_ns=3) == ()
    binding = hooks.bind_plan(_Forward(["r"]))
    assert binding is not None
    assert binding.telemetry.preferred_decode_ids == ()


def test_bind_plan_failure_is_fail_open() -> None:
    class BrokenForward:
        @property
        def request_ids(self):
            raise RuntimeError("forward exploded")

    hooks = RemoteSpecHooks(mode="shadow", client=_Client())
    hooks.before_plan({"r": _State([1])}, {}, now_ns=1)
    assert hooks.bind_plan(BrokenForward()) is None


def test_observe_commit_failure_is_fail_open() -> None:
    class BrokenTensor:
        def tolist(self):
            raise RuntimeError("results exploded")

    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    forward = _Forward(["r"])
    hooks.before_plan({"r": _State([1])}, {}, now_ns=1)
    binding = hooks.bind_plan(forward)
    hooks.observe_commit(
        forward,
        _Results(BrokenTensor(), BrokenTensor()),
        [],
        binding,
    )
    assert client.offered == []


def test_observe_unlaunched_and_close_fail_open() -> None:
    offer_client = _ExplodingClient("offer")
    hooks = RemoteSpecHooks(mode="shadow", client=offer_client)
    hooks.before_plan({"r": _State([1])}, {}, now_ns=1)
    hooks.observe_unlaunched(hooks.bind_plan(_Forward(["r"])))

    close_hooks = RemoteSpecHooks(mode="shadow", client=_ExplodingClient("close"))
    close_hooks.close()


def test_exact_prefix_hint_is_advisory_and_every_selected_row_falls_back() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1, 2], [3])

    assert hooks.before_plan({"r": state}, {}, now_ns=10) == ()
    first = hooks.bind_plan(_Forward(["r"]))
    assert first is not None
    prefix = first.telemetry.rows[0].prefix
    client.directive = _directive(
        first,
        _hint(prefix),
        generation=1,
        created_at_ns=10,
        received_at_ns=11,
    )

    assert hooks.before_plan({"r": state}, {}, now_ns=12) == ("r",)
    binding = hooks.bind_plan(_Forward(["r"]))
    assert binding is not None
    row = binding.telemetry.rows[0]
    assert row.ready_before_ordering
    assert row.ready_after_ordering
    assert row.fallback_reason is RemoteSpecFallbackReason.REMOTE_DISABLED
    assert binding.telemetry.fallback_counts == {"remote_disabled": 1}

    hooks.observe_commit(_Forward(["r"]), None, [], binding, now_ns=13)
    assert client.offered[-1].fallback_opportunities == 1


def test_stale_and_unsettled_prefixes_cannot_reorder_decode() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1, 2], [3])
    hooks.before_plan({"r": state}, {}, now_ns=10)
    initial = hooks.bind_plan(_Forward(["r"]))
    prefix = initial.telemetry.rows[0].prefix
    client.directive = _directive(
        initial,
        _hint(prefix),
        generation=1,
        created_at_ns=10,
        received_at_ns=11,
    )

    state.output_ids.append(4)
    assert hooks.before_plan({"r": state}, {}, now_ns=12) == ()
    stale = hooks.bind_plan(_Forward(["r"]))
    assert (
        stale.telemetry.rows[0].fallback_reason is RemoteSpecFallbackReason.WRONG_OFFSET
    )

    current_prefix = stale.telemetry.rows[0].prefix
    client.directive = _directive(
        stale,
        _hint(current_prefix),
        generation=2,
        created_at_ns=12,
        received_at_ns=13,
    )
    assert (
        hooks.before_plan({"r": state}, {}, now_ns=14, unsettled_request_ids=("r",))
        == ()
    )
    unsettled = hooks.bind_plan(_Forward(["r"]))
    assert (
        unsettled.telemetry.rows[0].fallback_reason
        is RemoteSpecFallbackReason.TARGET_STEP_UNSETTLED
    )


def test_unlaunched_selected_plan_is_accounted_as_engine_rejected() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    hooks.before_plan({"r": _State([1])}, {}, now_ns=1)
    binding = hooks.bind_plan(_Forward(["r"]))
    hooks.observe_unlaunched(binding)
    assert client.offered[-1].fallback_counts == {"engine_rejected_hint": 1}


def test_revision_and_width_are_part_of_shadow_readiness() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(
        mode="shadow",
        client=client,
        target_revision="target-r1",
        max_depth=2,
    )
    state = _State([1])
    hooks.before_plan({"r": state}, {}, now_ns=1)
    initial = hooks.bind_plan(_Forward(["r"]))
    prefix = initial.telemetry.rows[0].prefix
    invalid = replace(_hint(prefix), target_revision="target-r0", draft_depth=3)
    client.directive = _directive(initial, invalid, received_at_ns=2)
    assert hooks.before_plan({"r": state}, {}, now_ns=3) == ()
    row = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0]
    assert row.fallback_reason is RemoteSpecFallbackReason.WRONG_REVISION
    assert row.semantic_checks.revision is False
    assert row.semantic_checks.width is False


def test_sampling_contract_change_rejects_an_otherwise_exact_hint() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1])
    state.sampling_params = SimpleNamespace(temperature=1.0)
    hooks.before_plan({"r": state}, {}, now_ns=1)
    initial = hooks.bind_plan(_Forward(["r"]))
    prefix = initial.telemetry.rows[0].prefix
    client.directive = _directive(initial, _hint(prefix), received_at_ns=2)

    state.sampling_params.temperature = 0.5
    assert hooks.before_plan({"r": state}, {}, now_ns=3) == ()
    row = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0]
    assert row.fallback_reason is RemoteSpecFallbackReason.UNSUPPORTED_SAMPLING
    assert row.semantic_checks.sampling is False


def test_commit_telemetry_keeps_raw_and_committed_local_outcomes() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(
        mode="shadow",
        client=client,
        local_spec_width=3,
        native_speculative_algorithm="mtp",
    )
    state = _State([1])
    forward = _Forward(["r"])
    hooks.before_plan({"r": state}, {}, now_ns=1)
    binding = hooks.bind_plan(forward)

    hooks.observe_commit(
        forward,
        _Results(_Tensor([3]), _Tensor([10, 11, 12])),
        [_Event("r", [10, 11, 12])],
        binding,
        {},
        now_ns=2,
    )
    row = client.offered[-1].rows[0]
    assert client.offered[-1].native_speculative_algorithm == "mtp"
    assert row.raw_target_tokens == (10, 11, 12)
    assert row.committed_target_tokens == (10, 11, 12)
    assert row.raw_accepted_draft_tokens == 2
    assert row.bonus_token == 12
    assert row.correction_token is None
    assert row.tombstoned


def test_request_id_reuse_gets_a_new_incarnation() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    hooks.before_plan({"r": _State([1])}, {}, now_ns=1)
    first = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0].prefix
    hooks.before_plan({"r": _State([1])}, {}, now_ns=2)
    second = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0].prefix
    assert first.request_incarnation != second.request_incarnation


def test_same_state_prompt_mutation_reseeds_digest_and_incarnation() -> None:
    client = _Client()
    hooks = RemoteSpecHooks(mode="shadow", client=client)
    state = _State([1, 2])
    hooks.before_plan({"r": state}, {}, now_ns=1)
    first = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0].prefix

    state.prompt_input_ids[0] = 9
    hooks.before_plan({"r": state}, {}, now_ns=2)
    second = hooks.bind_plan(_Forward(["r"])).telemetry.rows[0].prefix
    assert second.request_incarnation != first.request_incarnation
    assert second.digest != first.digest
