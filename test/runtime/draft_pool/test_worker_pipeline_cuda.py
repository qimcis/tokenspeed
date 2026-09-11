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

"""Opt-in checkpoint qualification for batched context and two-slot uploads.

Use the existing TOKENSPEED_TEST_DRAFT_WORKER_CUDA=1 gate and the four
TOKENSPEED_TEST_{TARGET,DRAFT}_{CHECKPOINT,REVISION} variables. This loads one
model with four resident sessions, batch size two and a 1 GiB pinned-memory
budget. CPU collection skips these checks; it does not qualify GPU execution.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pytest
import torch

from tokenspeed.runtime.draft_pool import worker

pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_DRAFT_WORKER_CUDA") != "1",
    reason="Requires explicit CUDA allocation and the actual pinned checkpoint pair",
)


@pytest.fixture(scope="module")
def pipeline_engine():
    config = worker.WorkerModelConfig(
        target_model_path=os.environ["TOKENSPEED_TEST_TARGET_CHECKPOINT"],
        target_revision=os.environ["TOKENSPEED_TEST_TARGET_REVISION"],
        draft_model_path=os.environ["TOKENSPEED_TEST_DRAFT_CHECKPOINT"],
        draft_revision=os.environ["TOKENSPEED_TEST_DRAFT_REVISION"],
        device_id=0,
        max_resident_sessions=4,
        max_batch_size=2,
    )
    engine = worker.DFlash2WorkerEngine(config, pipeline_host_budget_bytes=1024**3)
    try:
        yield engine
    finally:
        engine.close()


@dataclass(frozen=True)
class _FeatureBank:
    start: int
    initial_endpoint: int
    values: torch.Tensor

    def interval(self, start: int, endpoint: int) -> torch.Tensor:
        assert self.start <= start <= endpoint <= self.start + self.values.shape[0]
        return self.values[start - self.start : endpoint - self.start].contiguous()


def _banks(engine) -> tuple[_FeatureBank, ...]:
    history = engine.geometry.history_tokens
    banks = []
    # Three histories cross the physical ring boundary; the fourth also covers
    # short context mixed into a batch with a full snapshot.
    for index, endpoint in enumerate(
        (2 * history - 1, 3 * history - 1, 4 * history - 1, 3)
    ):
        start = max(0, endpoint - history)
        generator = torch.Generator(device="cpu").manual_seed(123 + index)
        values = torch.randn(
            endpoint - start + 12,
            engine.contract.feature_width,
            generator=generator,
            dtype=torch.bfloat16,
        )
        banks.append(_FeatureBank(start, endpoint, values))
    return tuple(banks)


def _wait(engine, ticket: int):
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        completion = engine.poll_batch(ticket)
        if completion is not None:
            assert completion.install_error is None, completion.install_error
            assert completion.draft_error is None, completion.draft_error
            return completion
        time.sleep(0.001)
    pytest.fail(f"Worker pipeline ticket {ticket} did not retire within 120 seconds")


def _update(name: str, bank: _FeatureBank, start: int, endpoint: int, snapshot: bool):
    return worker.WorkerFeatureUpdate(
        session_id=name,
        feature_start=start,
        confirmed_endpoint=endpoint,
        features=bank.interval(start, endpoint),
        is_snapshot=snapshot,
    )


def _assert_completion(engine, completion, updates, jobs) -> None:
    assert completion.installed == tuple(update.session_id for update in updates)
    assert len(completion.results) == len(jobs)
    for result, job in zip(completion.results, jobs, strict=True):
        assert (result.session_id, result.confirmed_endpoint, result.anchor_token) == (
            job.session_id,
            job.confirmed_endpoint,
            job.anchor_token,
        )
        assert len(result.candidate_ids) == engine.geometry.native_block_tokens - 1 == 7
        assert all(
            0 <= token < engine.contract.vocab_size for token in result.candidate_ids
        )
        assert (
            engine.sessions.get(job.session_id).confirmed_endpoint
            == job.confirmed_endpoint
        )


def _cache_snapshot(engine, names: tuple[str, ...]):
    snapshots = []
    for name in names:
        session = engine.sessions.get(name)
        endpoint = session.confirmed_endpoint
        assert endpoint is not None
        start = max(0, endpoint - engine.geometry.history_tokens)
        locations = engine.geometry.context_locations(session.slot, start, endpoint)
        snapshots.append(
            tuple(
                plane[locations].cpu().clone()
                for layer in range(len(engine.model.layers))
                for plane in engine.pool.get_kv_buffer(layer)
            )
        )
    return snapshots


def _assert_caches_close(actual, expected) -> None:
    for actual_session, expected_session in zip(actual, expected, strict=True):
        for actual_plane, expected_plane in zip(
            actual_session, expected_session, strict=True
        ):
            # Packing different context row counts can change BF16 GEMM rounding.
            torch.testing.assert_close(
                actual_plane, expected_plane, rtol=0.02, atol=0.02
            )


def test_batched_mixed_deltas_match_individual_cold_context(pipeline_engine):
    engine = pipeline_engine
    banks = _banks(engine)[:2]
    names = ("pipeline-cold-a", "pipeline-cold-b")
    final_endpoints = tuple(
        bank.initial_endpoint + delta for bank, delta in zip(banks, (1, 6), strict=True)
    )
    for name in names:
        engine.open_session(name)
    # Establish reference KV with one cold, independently installed snapshot
    # per request, then replay exactly those features through batched deltas.
    for name, bank, endpoint in zip(names, banks, final_endpoints, strict=True):
        start = endpoint - engine.geometry.history_tokens
        engine.install_features(
            name, start, endpoint, bank.interval(start, endpoint), True
        )
    expected = _cache_snapshot(engine, names)
    for name in names:
        engine.close_session(name)
        engine.open_session(name)
    snapshots = tuple(
        _update(name, bank, bank.start, bank.initial_endpoint, True)
        for name, bank in zip(names, banks, strict=True)
    )
    ticket = engine.stage_batch(snapshots)
    engine.launch_batch(ticket, [])
    completion = _wait(engine, ticket)
    assert completion.installed == names
    assert completion.results == ()
    deltas = tuple(
        _update(name, bank, bank.initial_endpoint, endpoint, False)
        for name, bank, endpoint in zip(names, banks, final_endpoints, strict=True)
    )
    jobs = tuple(
        worker.WorkerDraftJob(name, endpoint, 42 + index)
        for index, (name, endpoint) in enumerate(
            zip(names, final_endpoints, strict=True)
        )
    )
    ticket = engine.stage_batch(deltas)
    engine.launch_batch(ticket, jobs)
    _assert_completion(engine, _wait(engine, ticket), deltas, jobs)
    _assert_caches_close(_cache_snapshot(engine, names), expected)
    for name in names:
        engine.close_session(name)


def _run_trace(engine, banks, pipeline: bool):
    names = tuple(f"pipeline-trace-{index}" for index in range(4))
    for name in names:
        engine.open_session(name)
    hidden_states = []

    def capture_native_hidden(module, arguments, output):
        # Keep a stream-ordered GPU clone. Copying to CPU here would serialize
        # the upload/compute overlap this test is intended to exercise.
        hidden_states.append(output.hidden_states.detach().clone())

    hook = engine.model.register_forward_hook(capture_native_hidden)
    try:
        endpoints = [bank.initial_endpoint for bank in banks]
        for phase, advances in enumerate(((0, 0, 0, 0), (1, 6, 5, 2), (6, 1, 1, 3))):
            batches = []
            for first in (0, 2):
                updates = []
                jobs = []
                for index in range(first, first + 2):
                    bank = banks[index]
                    start = bank.start if phase == 0 else endpoints[index]
                    endpoints[index] += advances[index]
                    updates.append(
                        _update(names[index], bank, start, endpoints[index], phase == 0)
                    )
                    jobs.append(
                        worker.WorkerDraftJob(
                            names[index], endpoints[index], 51 + phase + index
                        )
                    )
                batches.append((tuple(updates), tuple(jobs)))
            if pipeline:
                first_updates, first_jobs = batches[0]
                second_updates, second_jobs = batches[1]
                first_ticket = engine.stage_batch(first_updates)
                engine.launch_batch(first_ticket, first_jobs)
                second_ticket = engine.stage_batch(second_updates)
                assert first_ticket != second_ticket
                # The second upload owns a different staging slot while the first
                # compute/result copy still owns its ticket. Model compute remains
                # sequential; launch the next batch only after retiring the first.
                _assert_completion(
                    engine, _wait(engine, first_ticket), first_updates, first_jobs
                )
                engine.launch_batch(second_ticket, second_jobs)
                _assert_completion(
                    engine, _wait(engine, second_ticket), second_updates, second_jobs
                )
            else:
                for updates, jobs in batches:
                    ticket = engine.stage_batch(updates)
                    engine.launch_batch(ticket, jobs)
                    _assert_completion(engine, _wait(engine, ticket), updates, jobs)
    finally:
        hook.remove()
    assert len(hidden_states) == 6
    expected_shape = (
        2 * engine.geometry.native_block_tokens,
        engine.contract.feature_width,
    )
    assert all(tuple(hidden.shape) == expected_shape for hidden in hidden_states)
    snapshots = _cache_snapshot(engine, names)
    hidden_cpu = [hidden.cpu() for hidden in hidden_states]
    for name in names:
        engine.close_session(name)
    return snapshots, hidden_cpu


def test_two_upload_slots_preserve_native_forward_and_context(pipeline_engine):
    engine = pipeline_engine
    banks = _banks(engine)
    serial_cache, serial_hidden = _run_trace(engine, banks, pipeline=False)
    pipeline_cache, pipeline_hidden = _run_trace(engine, banks, pipeline=True)
    _assert_caches_close(pipeline_cache, serial_cache)
    for actual, expected in zip(pipeline_hidden, serial_hidden, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    # Candidate IDs are validated above, but are not required to be bitwise
    # equal: small BF16 differences at a top-k/selector boundary may change a
    # valid proposal. Target verification remains authoritative.
