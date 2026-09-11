# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Real TCP pool acceptance with deterministic CPU inference, not GPU qualification."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.draft_pool.protocol import (
    Ack,
    Close,
    Failure,
    Open,
    Opened,
    Proposal,
    Update,
    build_contract,
)
from tokenspeed.runtime.draft_pool.transport import (
    DraftPoolService,
    DraftPoolServiceConfig,
    TargetDraftClient,
    TargetDraftClientConfig,
)


def await_condition(condition, label: str) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {label}")
        threading.Event().wait(0.002)


class DeterministicEngine:
    """Only inference is replaced; production codec, I/O and worker lifecycle run."""

    def __init__(self, contract, block_first: bool) -> None:
        self.contract = contract
        self.owner = threading.get_ident()
        self.sessions = {}
        self.batches = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.block_first = block_first

    def open_session(self, session_id: str) -> None:
        assert threading.get_ident() == self.owner
        self.sessions[session_id] = torch.empty((0, 4), dtype=torch.bfloat16)

    def install_features(
        self, session_id, feature_start, confirmed_endpoint, features, is_snapshot
    ) -> None:
        assert threading.get_ident() == self.owner
        assert features.device.type == "cpu"
        assert features.dtype == torch.bfloat16
        assert features.shape[0] == confirmed_endpoint - feature_start
        if is_snapshot:
            assert feature_start == 0
            self.sessions[session_id] = features.clone()
        else:
            assert len(self.sessions[session_id]) == feature_start
            self.sessions[session_id] = torch.cat((self.sessions[session_id], features))

    def draft_batch(self, jobs):
        assert threading.get_ident() == self.owner
        self.batches.append(tuple(job.session_id for job in jobs))
        if self.block_first:
            self.block_first = False
            self.started.set()
            assert self.release.wait(5), "test did not release worker inference"
        results = []
        for job in jobs:
            value = int(self.sessions[job.session_id].float().sum()) + job.anchor_token
            results.append(
                SimpleNamespace(
                    session_id=job.session_id,
                    confirmed_endpoint=job.confirmed_endpoint,
                    anchor_token=job.anchor_token,
                    candidate_ids=tuple((value + index) % 128 for index in range(7)),
                )
            )
        return results

    def close_session(self, session_id: str) -> None:
        assert threading.get_ident() == self.owner
        del self.sessions[session_id]

    def close(self) -> None:
        assert threading.get_ident() == self.owner
        self.sessions.clear()
        self.closed.set()


@dataclass
class RunningService:
    service: DraftPoolService
    thread: threading.Thread
    engines: list
    errors: list

    def stop(self) -> None:
        self.engines[0].release.set()
        self.service.close()
        self.thread.join(5)
        assert not self.thread.is_alive()
        assert not self.errors
        assert self.engines[0].closed.is_set()


@pytest.fixture
def pool():
    contract = build_contract(
        target_model="example-target",
        target_revision="a" * 40,
        draft_model="example-draft",
        draft_revision="b" * 40,
        target_hf_config={"hidden_size": 4, "vocab_size": 128, "num_hidden_layers": 8},
        draft_hf_config={
            "hidden_size": 4,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6,
            "sliding_window": 8,
            "dflash_config": {"block_size": 8, "target_layer_ids": [1, 4, 7]},
        },
        target_verify_tokens=6,
    )
    running = []
    clients = []

    def start(endpoint: str, block_first: bool):
        engines, errors = [], []

        def factory():
            engine = DeterministicEngine(contract, block_first)
            engines.append(engine)
            return engine

        service = DraftPoolService(
            config=DraftPoolServiceConfig(
                listen_endpoint=endpoint,
                resident_limit=4,
                staging_limit=4,
                max_queued_jobs=4,
                max_batch_size=2,
                max_peers=2,
                max_header_bytes=4096,
                max_feature_bytes=1024,
                max_host_memory_bytes=1 << 20,
                lease_ms=30000,
                heartbeat_ms=100,
                poll_ms=2,
                linger_ms=0,
            ),
            engine_factory=factory,
        )

        def serve():
            try:
                service.run()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        entry = RunningService(service, thread, engines, errors)
        running.append(entry)
        thread.start()
        await_condition(lambda: service.started.is_set() or errors, "worker listen")
        assert not errors
        return entry

    def connect(endpoint: str, identity: bytes):
        client = TargetDraftClient(
            config=TargetDraftClientConfig(
                endpoint=endpoint,
                identity=identity,
                max_pending_messages=16,
                max_pending_bytes=65536,
                max_reply_messages=32,
                max_header_bytes=4096,
                max_feature_bytes=1024,
                heartbeat_ms=100,
                reconnect_ms=20,
                poll_ms=2,
                linger_ms=0,
            ),
            contract=contract,
        )
        clients.append(client)
        await_condition(lambda: client.connected, "target handshake")
        return Inbox(client)

    yield start, connect
    for entry in running:
        entry.engines[0].release.set()
    for client in clients:
        client.close()
    for entry in running:
        if entry.thread.is_alive():
            entry.stop()


class Inbox:
    def __init__(self, client) -> None:
        self.client = client
        self.messages = []

    def receive(self, predicate):
        def available():
            self.messages.extend(item.message for item in self.client.poll())
            return any(predicate(message) for message in self.messages)

        await_condition(available, "expected worker response")
        index = next(i for i, message in enumerate(self.messages) if predicate(message))
        return self.messages.pop(index)

    def open(self, session_id: str, endpoint: int, anchor: int) -> None:
        assert self.client.submit(
            Open(
                session_id=session_id,
                confirmed_endpoint=endpoint,
                anchor_token=anchor,
                history_start=0,
            ),
            None,
        )
        self.receive(
            lambda message: isinstance(message, Opened)
            and message.session_id == session_id
        )

    def update(
        self,
        session_id: str,
        endpoint: int,
        anchor: int,
        start: int,
        snapshot: bool,
        features,
    ) -> None:
        assert self.client.submit(
            Update(
                session_id=session_id,
                confirmed_endpoint=endpoint,
                anchor_token=anchor,
                feature_start=start,
                is_snapshot=snapshot,
            ),
            features,
        )

    def proposal(self, session_id: str, endpoint: int):
        ack = self.receive(
            lambda message: isinstance(message, Ack)
            and message.session_id == session_id
            and message.confirmed_endpoint == endpoint
        )
        proposal = self.receive(
            lambda message: isinstance(message, Proposal)
            and message.session_id == session_id
            and message.confirmed_endpoint == endpoint
        )
        assert ack.anchor_token == proposal.anchor_token
        return proposal


def test_pool_batches_independent_cohorts_and_retires_cancelled_work(pool):
    start, connect = pool
    running = start("tcp://127.0.0.1:*", True)
    first = connect(running.service.bound_endpoint, b"cohort-a")
    second = connect(running.service.bound_endpoint, b"cohort-b")
    snapshot = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    expected = snapshot.clone()
    first.open("cancelled", 4, 10)
    first.update("cancelled", 4, 10, 0, True, snapshot)
    assert running.engines[0].started.wait(5)
    first.open("other-a", 4, 11)
    second.open("other-b", 4, 12)
    first.update("other-a", 4, 11, 0, True, snapshot)
    second.update("other-b", 4, 12, 0, True, snapshot)
    snapshot.zero_()  # Successful submit owns a copy before staging can be reused.
    await_condition(
        lambda: running.service.queued_jobs == 2, "independent queued requests"
    )
    assert first.client.submit(Close(session_id="cancelled"), None)
    # One ordered message behind CLOSE proves the service processed cancellation.
    first.open("after-cancel", 4, 13)
    running.engines[0].release.set()
    a = first.proposal("other-a", 4)
    b = second.proposal("other-b", 4)
    assert a.candidate_ids[0] == (int(expected.float().sum()) + 11) % 128
    assert b.candidate_ids[0] == (int(expected.float().sum()) + 12) % 128
    assert any(
        set(batch) == {"other-a", "other-b"} for batch in running.engines[0].batches
    )
    await_condition(
        lambda: "cancelled" not in running.engines[0].sessions,
        "cancelled storage retirement",
    )
    first.messages.extend(item.message for item in first.client.poll())
    assert not any(
        isinstance(message, Proposal) and message.session_id == "cancelled"
        for message in first.messages
    )


def test_worker_restart_reseeds_equivalent_history_on_existing_connection(pool):
    start, connect = pool
    running = start("tcp://127.0.0.1:*", False)
    endpoint = running.service.bound_endpoint
    inbox = connect(endpoint, b"cohort")
    history = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    inbox.open("before-restart", 4, 10)
    inbox.update("before-restart", 4, 10, 0, True, history[:4].contiguous())
    inbox.proposal("before-restart", 4)
    inbox.update("before-restart", 6, 20, 4, False, history[4:].contiguous())
    incremental = inbox.proposal("before-restart", 6)
    running.stop()
    inbox.receive(
        lambda message: isinstance(message, Failure)
        and message.code == "connection_reset"
    )
    replacement = start(endpoint, False)
    await_condition(lambda: inbox.client.connected, "worker reconnect handshake")
    inbox.open("fresh-session", 6, 20)
    inbox.update("fresh-session", 6, 20, 0, True, history)
    reseeded = inbox.proposal("fresh-session", 6)
    assert incremental.candidate_ids == reseeded.candidate_ids
    assert set(replacement.engines[0].sessions) == {"fresh-session"}
