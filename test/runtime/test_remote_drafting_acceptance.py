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
from dataclasses import dataclass, replace

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
from tokenspeed.runtime.draft_pool.worker import (
    WorkerBatchCompletion,
    WorkerDraftResult,
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
        self.stages = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.block_first = block_first
        self._next_ticket = 0
        self._staged = {}
        self._active = None
        self._blocked = None
        self._jobs = ()

    def open_session(self, session_id: str) -> None:
        assert threading.get_ident() == self.owner
        assert session_id not in self.sessions
        self.sessions[session_id] = torch.empty((0, 4), dtype=torch.bfloat16)

    def validate_features(self, update) -> None:
        assert threading.get_ident() == self.owner
        assert update.features.device.type == "cpu"
        assert update.features.dtype == torch.bfloat16
        assert (
            update.features.shape[0] == update.confirmed_endpoint - update.feature_start
        )
        if update.is_snapshot:
            assert update.feature_start == 0
            assert len(self.sessions[update.session_id]) == 0
        else:
            assert len(self.sessions[update.session_id]) == update.feature_start

    def stage_batch(self, updates) -> int:
        assert threading.get_ident() == self.owner
        assert len(self._staged) < 2
        self._next_ticket += 1
        self._staged[self._next_ticket] = tuple(
            replace(update, features=update.features.clone()) for update in updates
        )
        self.stages.append(tuple(update.session_id for update in updates))
        return self._next_ticket

    def launch_batch(self, ticket, jobs) -> None:
        assert threading.get_ident() == self.owner
        assert self._active is None
        for update in self._staged[ticket]:
            self.validate_features(update)
            if update.is_snapshot:
                self.sessions[update.session_id] = update.features
            else:
                self.sessions[update.session_id] = torch.cat(
                    (self.sessions[update.session_id], update.features)
                )
        self._active = ticket
        self._jobs = tuple(jobs)
        self.batches.append(tuple(job.session_id for job in jobs))
        if self.block_first:
            self.block_first = False
            self._blocked = ticket
            self.started.set()

    def poll_batch(self, ticket):
        assert threading.get_ident() == self.owner
        assert self._active == ticket
        if ticket == self._blocked and not self.release.is_set():
            return None
        results = []
        for job in self._jobs:
            value = int(self.sessions[job.session_id].float().sum()) + job.anchor_token
            results.append(
                WorkerDraftResult(
                    session_id=job.session_id,
                    confirmed_endpoint=job.confirmed_endpoint,
                    anchor_token=job.anchor_token,
                    candidate_ids=tuple((value + index) % 128 for index in range(7)),
                )
            )
        installed = tuple(update.session_id for update in self._staged.pop(ticket))
        self._active = None
        self._jobs = ()
        return WorkerBatchCompletion(
            installed=installed,
            results=tuple(results),
            install_error=None,
            draft_error=None,
        )

    def close_session(self, session_id: str) -> None:
        assert threading.get_ident() == self.owner
        assert all(
            update.session_id != session_id
            for batch in self._staged.values()
            for update in batch
        )
        del self.sessions[session_id]

    def close(self) -> None:
        assert threading.get_ident() == self.owner
        if self._active is not None and self._active == self._blocked:
            assert self.release.wait(5), "test did not retire worker inference"
        self._staged.clear()
        self._active = None
        self._jobs = ()
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
    first.open("uploaded", 4, 13)
    first.update("uploaded", 4, 13, 0, True, snapshot)
    await_condition(
        lambda: ("uploaded",) in running.engines[0].stages,
        "next batch upload while first inference runs",
    )
    first.open("other-a", 4, 11)
    second.open("other-b", 4, 12)
    first.update("other-a", 4, 11, 0, True, snapshot)
    second.update("other-b", 4, 12, 0, True, snapshot)
    snapshot.zero_()  # Successful submit owns a copy before staging can be reused.
    await_condition(
        lambda: running.service.queued_jobs == 2, "independent queued requests"
    )
    assert first.client.submit(Close(session_id="cancelled"), None)
    # Repeating an outstanding OPEN is idempotent and forms an ordered barrier
    # behind CLOSE without consuming another resident/staging reservation.
    first.open("other-a", 4, 11)
    running.engines[0].release.set()
    uploaded = first.proposal("uploaded", 4)
    assert uploaded.candidate_ids[0] == (int(expected.float().sum()) + 13) % 128
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
