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

"""Adversarial contracts exercised through the compiled native scheduler."""

import pytest

native = pytest.importorskip("tokenspeed_scheduler")


def scheduler(
    *,
    cap=32,
    replay=False,
    state=False,
    budget=1.0,
    linear=1.0,
    context=0.0,
    quadratic=0.0,
    mixed=False,
):
    cfg = native.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = cap
    cfg.max_batch_size = 4
    cfg.num_device_pages = 256
    cfg.num_snapshot_pages = 1
    cfg.enable_mixed_prefill_decode = mixed
    cfg.disable_l2_cache = True
    cfg.cache_groups = [native.CacheGroupConfig("full", 2, 256, native.AttnKind.Full)]
    if replay:
        cfg.cache_groups += [
            native.CacheGroupConfig(
                "swa",
                2,
                256,
                native.AttnKind.SlidingWindow,
                sliding_window_tokens=4,
                replayable=True,
            )
        ]
    if state:
        cfg.cache_groups += [
            native.CacheGroupConfig("state", 0, 256, native.AttnKind.State)
        ]
    result = native.Scheduler(cfg)
    result.configure_prefill_work_budget(budget, linear, context, quadratic)
    return result


def submit(s, request_id, tokens):
    spec = native.RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    spec.max_new_tokens = 4
    s.submit_requests([spec])


def complete(s, request_id, tokens=()):
    event = native.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = list(tokens)
    batch = native.ExecutionEvent()
    batch.add_event(event)
    s.advance(batch)


def finish(s, request_id):
    event = native.ForwardEvent.Finish()
    event.request_id = request_id
    batch = native.ExecutionEvent()
    batch.add_event(event)
    s.advance(batch)


def prefill(s, request_id, tokens):
    submit(s, request_id, tokens)
    chunks = []
    for _ in range(len(tokens) + 2):
        plan = s.next_execution_plan()
        assert plan.forward, "work quota stalled a feasible prompt"
        batch = plan.forward[0]
        assert batch.request_ids == [request_id]
        assert batch.num_extends() == 1
        q = batch.input_lengths[0]
        c = batch.extend_prefix_lens[0]
        r = batch.extend_replay_lens[0]
        chunks.append((q, c, r, s.last_prefill_work_ms()))
        done = q + c == len(tokens)
        complete(s, request_id, [9001] if done else [])
        if done:
            return chunks
    pytest.fail("prefill failed to make bounded progress")


def seed_cache(s):
    prefill(s, "seed", list(range(12)))
    plan = s.next_execution_plan()
    assert plan.forward[0].num_extends() == 0
    complete(s, "seed", [9002])
    finish(s, "seed")
    assert all(not batch.request_ids for batch in s.next_execution_plan().forward)


@pytest.mark.parametrize("budget", [0.001, 4.0, 5.0, 7.0, 8.0, 100.0])
def test_replayed_hit_progress_final_window_and_shared_token_cap(budget):
    s = scheduler(cap=8, replay=True, budget=budget)
    seed_cache(s)
    # Eight shared tokens, then six new tokens: replay the preceding four
    # cached rows and retain a final four-row window in every legal split.
    chunks = prefill(s, "hit", list(range(8)) + list(range(100, 106)))
    assert chunks[0][2] == 4
    assert chunks[0][1] == 4
    assert all(0 < q <= 8 for q, _, _, _ in chunks)
    assert all(cost == q for q, _, _, cost in chunks)
    assert all(r == 0 for _, _, r, _ in chunks[1:])
    assert chunks[-1][0] >= 4
    assert sum(q - r for q, _, r, _ in chunks) == 6


@pytest.mark.parametrize("budget", [0.001, 4.0, 100.0])
def test_quota_respects_hybrid_checkpoint_junction(budget):
    s = scheduler(cap=8, state=True, budget=budget)
    seed_cache(s)
    chunks = prefill(s, "hit", list(range(6)) + list(range(100, 106)))
    assert chunks[0][1] == 0  # KV hit at 6 has no recurrent checkpoint.
    endpoints = [q + c for q, c, _, _ in chunks]
    assert 6 in endpoints
    assert all(q <= 8 for q, _, _, _ in chunks)
    assert endpoints[-1] == 12


@pytest.mark.parametrize(
    "linear,context,quadratic", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.1, 0.03)]
)
def test_native_diagnostics_charge_actual_replayed_input_and_context(
    linear, context, quadratic
):
    s = scheduler(
        cap=8,
        replay=True,
        budget=5,
        linear=linear,
        context=context,
        quadratic=quadratic,
    )
    seed_cache(s)
    chunks = prefill(s, "hit", list(range(8)) + list(range(100, 108)))
    for q, c, _, cost in chunks:
        assert cost == pytest.approx(q * (linear + context * c + quadratic * q))


def test_completed_prefills_share_budget_without_extra_progress_override():
    s = scheduler(cap=8, budget=3)
    submit(s, "a", [1, 2])
    submit(s, "b", [3])
    submit(s, "c", [4])
    batch = s.next_execution_plan().forward[0]
    assert batch.request_ids == ["a", "b"]
    assert batch.input_lengths == [2, 1]
    assert s.last_prefill_work_ms() == 3
    assert s.prefill_work_overruns() == 0
    complete(s, "a", [20])
    complete(s, "b", [21])
    next_batch = s.next_execution_plan().forward[0]
    assert next_batch.request_ids == ["c"]
    assert s.last_prefill_work_ms() == 1


def test_unconfigured_baseline_uses_entire_token_cap():
    cfg = native.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 4
    cfg.max_batch_size = 4
    cfg.num_device_pages = 256
    cfg.num_snapshot_pages = 1
    cfg.cache_groups = [native.CacheGroupConfig("full", 2, 256, native.AttnKind.Full)]
    s = native.Scheduler(cfg)
    submit(s, "a", [1, 2])
    submit(s, "b", [3, 4, 5])
    batch = s.next_execution_plan().forward[0]
    assert batch.request_ids == ["a", "b"]
    assert batch.input_lengths == [2, 2]
    assert s.last_prefill_work_ms() == 0


@pytest.mark.parametrize("budget,expected_prefill", [(0.001, 1), (100.0, 3)])
def test_mixed_decode_consumes_hard_tokens_but_not_prefill_work(
    budget, expected_prefill
):
    s = scheduler(cap=4, budget=budget, mixed=True)
    prefill(s, "decoding", [1, 2])
    submit(s, "prefilling", list(range(10, 16)))
    batch = s.next_execution_plan().forward[0]
    assert set(batch.request_ids) == {"decoding", "prefilling"}
    lengths = dict(zip(batch.request_ids, batch.input_lengths, strict=True))
    assert lengths == {"decoding": 1, "prefilling": expected_prefill}
    assert sum(lengths.values()) <= 4
    assert s.last_prefill_work_ms() == expected_prefill
