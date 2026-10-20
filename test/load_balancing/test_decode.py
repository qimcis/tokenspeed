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

from dataclasses import replace
from math import isinf

import pytest

from tokenspeed.runtime.load_balancing.decode import DecodePolicy, DecodeRequest
from tokenspeed.runtime.load_balancing.types import Worker, WorkerSnapshot, Workload


def worker(worker_id="d0", **changes):
    return replace(
        Worker(
            worker_id=worker_id,
            incarnation="boot-1",
            role="decode",
            endpoint=f"http://{worker_id}:8000",
            capacity=Workload(
                requests=16,
                decode_context_tokens=65536,
                decode_output_tokens=16384,
                verification_tokens=128,
                kv_tokens=65536,
            ),
            observed_at=100,
            context_limit=8192,
        ),
        **changes,
    )


def snapshot(**changes):
    return WorkerSnapshot(worker(**changes), Workload())


def test_known_output_reserves_whole_worker_context_and_kv():
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 400, verification_tokens=8)
    assert policy.workload(request, snapshot()) == Workload(
        requests=1,
        decode_context_tokens=1400,
        decode_output_tokens=400,
        verification_tokens=8,
        kv_tokens=1400,
    )


def test_progress_reduces_remaining_output_without_losing_context():
    policy = DecodePolicy(unknown_output_tokens=128)
    initial = DecodeRequest(1000, 400)
    progress = replace(initial, context_tokens=1250, generated_tokens=250)
    first = policy.workload(initial, snapshot())
    later = policy.workload(progress, snapshot())
    assert later.requests == first.requests == 1
    assert later.decode_output_tokens == 150
    assert later.decode_context_tokens == first.decode_context_tokens == 1400
    assert later.kv_tokens == first.kv_tokens == 1400


def test_unknown_output_keeps_rolling_horizon_as_context_grows():
    policy = DecodePolicy(unknown_output_tokens=256)
    first = policy.workload(DecodeRequest(1000, None), snapshot())
    later = policy.workload(DecodeRequest(1500, None, 500), snapshot())
    assert first.decode_output_tokens == later.decode_output_tokens == 256
    assert later.decode_context_tokens == first.decode_context_tokens + 500
    assert later.kv_tokens == first.kv_tokens + 500


def test_unknown_output_horizon_clamps_to_context_limit():
    policy = DecodePolicy(unknown_output_tokens=256)
    work = policy.workload(DecodeRequest(8100, None), snapshot())
    assert work.decode_output_tokens == 92
    assert work.kv_tokens == work.decode_context_tokens == 8192
    assert isinf(policy.score(DecodeRequest(8200, None), snapshot()))
    assert isinf(policy.score(DecodeRequest(8100, 100), snapshot()))


def test_cache_reduces_predicted_score_but_not_hard_kv_or_attention_context():
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 100, cache_key="prefix-v1", cached_context_tokens=900)
    hit = snapshot(cache_keys=frozenset({"prefix-v1"}))
    miss = snapshot(cache_keys=frozenset({"another-prefix"}))
    hit_work = policy.workload(request, hit)
    miss_work = policy.workload(request, miss)
    assert hit_work.decode_context_tokens == miss_work.decode_context_tokens == 1100
    assert hit_work.kv_tokens == 1100
    assert miss_work.kv_tokens == 1100
    assert policy.score(request, hit) < policy.score(request, miss)


def test_cache_key_without_token_evidence_does_not_imply_a_prefix_hit():
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 100, cache_key="prefix-v1")
    hit = snapshot(cache_keys=frozenset({"prefix-v1"}))
    assert policy.workload(request, hit).kv_tokens == 1100


@pytest.mark.parametrize(
    "busy",
    [
        Workload(requests=8),
        Workload(decode_context_tokens=30000),
        Workload(decode_output_tokens=8000),
        Workload(verification_tokens=64),
        Workload(kv_tokens=30000),
    ],
)
def test_each_pressure_axis_influences_incremental_score(busy):
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 100, verification_tokens=8)
    idle = snapshot()
    loaded = WorkerSnapshot(idle.worker, busy)
    assert policy.score(request, idle) < policy.score(request, loaded)


def test_long_context_skew_changes_selection_at_equal_concurrency():
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 100)
    short = snapshot(reported=Workload(requests=4, decode_context_tokens=4000))
    long = snapshot(reported=Workload(requests=4, decode_context_tokens=32000))
    assert policy.score(request, short) < policy.score(request, long)


def test_locality_never_overrides_kv_or_concurrency_capacity():
    policy = DecodePolicy(unknown_output_tokens=128)
    request = DecodeRequest(1000, 100, cache_key="prefix", cached_context_tokens=900)
    full_slots = snapshot(
        reported=Workload(requests=16), cache_keys=frozenset({"prefix"})
    )
    full_kv = snapshot(
        reported=Workload(kv_tokens=65400), cache_keys=frozenset({"prefix"})
    )
    assert isinf(policy.score(request, full_slots))
    assert isinf(policy.score(request, full_kv))


def test_unexposed_verification_is_optional_but_exposed_work_needs_capacity():
    policy = DecodePolicy(unknown_output_tokens=128)
    unexposed = snapshot(capacity=replace(worker().capacity, verification_tokens=0))
    assert not isinf(policy.score(DecodeRequest(1000, 100), unexposed))
    assert isinf(
        policy.score(DecodeRequest(1000, 100, verification_tokens=1), unexposed)
    )


@pytest.mark.parametrize("role", ["prefill", "encoder", "complete", "pair"])
def test_policy_does_not_reinterpret_other_roles_as_decode_workers(role):
    assert isinf(
        DecodePolicy(unknown_output_tokens=128).score(
            DecodeRequest(1, 1), snapshot(role=role)
        )
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"context_tokens": -1},
        {"context_tokens": True},
        {"generated_tokens": 101},
        {"max_new_tokens": -1},
        {"max_new_tokens": 1, "generated_tokens": 2},
        {"verification_tokens": 1.5},
        {"cached_context_tokens": 1},
        {"cache_key": ""},
        {"cache_key": "prefix", "cached_context_tokens": 101},
    ],
)
def test_invalid_request_rejected(changes):
    with pytest.raises(ValueError):
        DecodeRequest(**({"context_tokens": 100, "max_new_tokens": 100} | changes))


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_unknown_output_horizon_must_be_explicit_positive_integer(horizon):
    with pytest.raises(ValueError):
        DecodePolicy(unknown_output_tokens=horizon)


def test_advance_preserves_budget_cache_and_accepts_only_monotonic_frontiers():
    policy = DecodePolicy(unknown_output_tokens=128)
    initial = DecodeRequest(
        1000, 400, verification_tokens=8, cache_key="prefix", cached_context_tokens=900
    )
    advanced = policy.advance(initial, generated_tokens=250, verification_tokens=4)
    assert advanced.context_tokens == 1250
    assert advanced.generated_tokens == 250
    assert advanced.max_new_tokens == 400
    assert advanced.cached_context_tokens == 900
    assert advanced.cache_key == "prefix"
    assert advanced.verification_tokens == 4
    assert policy.advance(advanced, generated_tokens=250) == advanced
    with pytest.raises(ValueError, match="regress"):
        policy.advance(advanced, generated_tokens=249)
    with pytest.raises(ValueError, match="original output limit"):
        policy.advance(advanced, generated_tokens=401)


def test_progress_updates_replace_lease_and_completion_releases_all_axes_once():
    from tokenspeed.runtime.load_balancing.registry import WorkerRegistry

    registry = WorkerRegistry(
        freshness_seconds=10, max_workers=8, max_leases=100, history_seconds=10
    )
    registry.upsert(worker())
    policy = DecodePolicy(unknown_output_tokens=128)
    initial = DecodeRequest(1000, None)
    lease = registry.reserve(
        "request",
        "decode",
        lambda state: policy.score(initial, state),
        lambda state: policy.workload(initial, state),
        now=100,
    )
    advanced = policy.advance(initial, generated_tokens=500)
    registry.update_lease(
        lease.lease_id, policy.workload(advanced, registry.get(lease.worker_id))
    )
    assert registry.get(lease.worker_id).reserved == Workload(
        requests=1, decode_context_tokens=1628, decode_output_tokens=128, kv_tokens=1628
    )
    assert registry.release(lease.lease_id, now=101)
    assert not registry.release(lease.lease_id, now=101)
    assert registry.get(lease.worker_id).reserved == Workload()
