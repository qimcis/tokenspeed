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

"""CPU model/policy checks use synthetic durations only as test fixtures."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from tokenspeed.runtime.load_balancing.prefill import (
    PrefillCostModel,
    PrefillObservation,
    PrefillPolicy,
    PrefillRequest,
    PrefixAvailability,
    WorkerPrefix,
    fit_prefill_cost,
)
from tokenspeed.runtime.load_balancing.registry import NoCapacity, WorkerRegistry
from tokenspeed.runtime.load_balancing.types import Worker, Workload


def model(a=1.0, b=0.0, d=0.0):
    return PrefillCostModel(
        linear_ms_per_token=a,
        context_ms_per_token_squared=b,
        quadratic_ms_per_token_squared=d,
    )


def policy(**kwargs):
    return PrefillPolicy(
        cost_model=model(),
        cache_max_age_seconds=5.0,
        context_tiers=kwargs.get("tiers", ()),
        max_tier_wait_seconds=kwargs.get("wait", 0.0),
    )


def worker(name, **kwargs):
    return Worker(
        worker_id=name,
        incarnation="v1",
        role="prefill",
        endpoint="http://localhost:8000",
        capacity=kwargs.pop("capacity", Workload(requests=100, prefill_work=10000.0)),
        observed_at=10.0,
        **kwargs,
    )


def registry(*workers):
    result = WorkerRegistry(
        freshness_seconds=10.0, max_workers=10, max_leases=100, history_seconds=60.0
    )
    for w in workers:
        result.upsert(w)
    return result


def request(tokens=100, **kwargs):
    return PrefillRequest(
        prompt_tokens=tokens,
        replay_tokens=kwargs.get("replay", 0),
        queued_at=10.0,
        prefixes=kwargs.get("prefixes", ()),
    )


def prefix(name, kind, tokens, transfer=0.0, at=10.0, incarnation="v1"):
    return WorkerPrefix(
        worker_id=name,
        availability=PrefixAvailability(
            kind=kind,
            cached_prefix_tokens=tokens,
            transfer_ms=transfer,
            observed_at=at,
            worker_incarnation=incarnation,
        ),
    )


@pytest.mark.parametrize(
    "values",
    [
        (-1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (float("nan"), 1.0, 1.0),
        (1.0, float("inf"), 0.0),
        (True, 0.0, 0.0),
    ],
)
def test_invalid_model(values):
    with pytest.raises(ValidationError):
        model(*values)


def test_cost_monotonic_units_and_integer_budget():
    cost = model(0.2, 0.01, 0.1)
    assert cost.predict_ms(10, 20) == pytest.approx(14.0)
    for c in (0, 10, 1000):
        values = [cost.predict_ms(q, c) for q in range(30)]
        assert values == sorted(values)
        for budget in (0.0, 0.01, 0.3, 10.0, 10000.0):
            q = cost.tokens_for_budget(
                budget_ms=budget, cached_prefix_tokens=c, token_cap=100
            )
            assert cost.predict_ms(q, c) <= budget
            assert q == 100 or cost.predict_ms(q + 1, c) > budget
    assert cost.predict_ms(1, 100) > cost.predict_ms(1, 1)
    assert cost.predict_ms(0, 100) == 0


@pytest.mark.parametrize("q,c", [(True, 0), (1, -1), (1.0, 0), (2**31, 0)])
def test_cost_rejects_non_integer_or_invalid_tokens(q, c):
    with pytest.raises(ValueError):
        model().predict_ms(q, c)


def test_fit_nonnegative_with_heldout_residuals():
    cost = model(0.5, 0.02, 0.001)

    def sample(q, c, noise=0):
        return PrefillObservation(
            uncached_tokens=q,
            cached_prefix_tokens=c,
            elapsed_ms=cost.predict_ms(q, c) + noise,
        )

    fitted = fit_prefill_cost(
        [sample(10, 0), sample(20, 0), sample(10, 100), sample(100, 50)],
        [sample(30, 30, 2), sample(50, 0, -1)],
    )
    assert fitted.cost_model.coefficients == pytest.approx(cost.coefficients)
    assert fitted.heldout_residuals_ms == pytest.approx((-2, 1))
    assert fitted.heldout_mean_absolute_error_ms == pytest.approx(1.5)
    assert fitted.heldout_root_mean_squared_error_ms == pytest.approx(2.5**0.5)
    assert fitted.heldout_max_absolute_error_ms == pytest.approx(2)


def test_fit_rejects_unidentifiable_or_missing_holdout():
    observation = PrefillObservation(
        uncached_tokens=10, cached_prefix_tokens=0, elapsed_ms=5.0
    )
    with pytest.raises(ValueError, match="identify"):
        fit_prefill_cost([observation] * 3, [observation])
    with pytest.raises(ValueError, match="heldout"):
        fit_prefill_cost([observation] * 3, [])


def test_fit_constrains_negative_unconstrained_solution():
    samples = [
        PrefillObservation(
            uncached_tokens=q, cached_prefix_tokens=c, elapsed_ms=float(10 * q - q * c)
        )
        for q, c in [(1, 0), (2, 0), (1, 1), (3, 2)]
    ]
    result = fit_prefill_cost(samples, samples[:1])
    assert all(x >= 0 for x in result.cost_model.coefficients)
    assert result.cost_model.context_ms_per_token_squared == 0


def test_routes_reserved_work_instead_of_request_count():
    reg = registry(
        worker("many", reported=Workload(requests=9, prefill_work=9.0)),
        worker("few", reported=Workload(requests=1, prefill_work=150.0)),
    )
    p = policy()
    first = p.reserve(reg, request_id="one", request=request(), now=10.0)
    assert first.worker_id == "many"
    assert first.workload.prefill_work == 100
    second = p.reserve(reg, request_id="two", request=request(), now=10.0)
    assert second.worker_id == "many"
    third = p.reserve(reg, request_id="three", request=request(), now=10.0)
    assert third.worker_id == "few"
    assert reg.release(first.lease_id)
    assert not reg.release(first.lease_id)


def test_remote_hit_retains_transfer_and_locality():
    reg = registry(worker("local"), worker("remote"), worker("miss"))
    req = request(
        prefixes=(
            prefix("local", "resident", 90),
            prefix("remote", "remote", 99, transfer=80.0),
        )
    )
    p = policy()
    assert p.predicted_work_ms(req, reg.get("local").worker, now=10.0) == 11
    assert p.predicted_work_ms(req, reg.get("remote").worker, now=10.0) == 82
    assert p.predicted_work_ms(req, reg.get("miss").worker, now=10.0) == 100
    assert p.reserve(reg, request_id="x", request=req, now=10.0).worker_id == "local"


def test_full_hit_requires_final_token_and_replay_is_charged():
    p = policy()
    w = worker("a")
    hit = (prefix("a", "resident", 100),)
    assert p.predicted_work_ms(request(prefixes=hit), w, now=10.0) == 1
    assert p.predicted_work_ms(request(prefixes=hit, replay=16), w, now=10.0) == 16


@pytest.mark.parametrize("at,incarnation", [(0.0, "v1"), (11.0, "v1"), (10.0, "old")])
def test_stale_future_and_old_incarnation_cache_are_misses(at, incarnation):
    req = request(
        prefixes=(prefix("a", "resident", 90, at=at, incarnation=incarnation),)
    )
    assert policy().predicted_work_ms(req, worker("a"), now=10.0) == 100


@pytest.mark.parametrize(
    "state",
    [
        dict(healthy=False),
        dict(ready=False),
        dict(draining=True),
        dict(capacity=Workload(requests=10, prefill_work=99.0)),
    ],
)
def test_ineligible_or_full_worker_cannot_receive_lease(state):
    reg = registry(worker("a", **state))
    with pytest.raises(NoCapacity):
        policy().reserve(reg, request_id="x", request=request(), now=10.0)


def test_stale_worker_cannot_receive_lease():
    with pytest.raises(NoCapacity):
        policy().reserve(
            registry(worker("a")), request_id="x", request=request(), now=21.0
        )


def test_context_tier_wait_is_bounded_and_hard_limit_remains():
    p = policy(tiers=(128, 1024), wait=2.0)
    reg = registry(
        worker("large", context_limit=1024), worker("small", context_limit=32)
    )
    with pytest.raises(NoCapacity):
        p.reserve(reg, request_id="x", request=request(), now=10.0)
    assert (
        p.reserve(reg, request_id="x", request=request(), now=12.0).worker_id == "large"
    )
    with pytest.raises(NoCapacity):
        p.reserve(reg, request_id="huge", request=request(2048), now=12.0)


def test_cache_validation_and_tier_validation():
    with pytest.raises(ValidationError):
        prefix("a", "miss", 1)
    with pytest.raises(ValidationError):
        prefix("a", "resident", 10, transfer=1.0)
    with pytest.raises(ValidationError):
        request(prefixes=(prefix("a", "resident", 101),))
    with pytest.raises(ValidationError):
        request(prefixes=(prefix("a", "resident", 10), prefix("a", "resident", 10)))
    with pytest.raises(ValidationError):
        policy(tiers=(128, 128))


def test_new_incarnation_does_not_inherit_old_cache_or_reservations():
    reg = registry(worker("a"))
    p = policy()
    req = request(prefixes=(prefix("a", "resident", 90),))
    old = p.reserve(reg, request_id="old", request=req, now=10.0)
    reg.upsert(replace(worker("a"), incarnation="v2"))
    new = p.reserve(reg, request_id="new", request=req, now=10.0)
    assert new.workload.prefill_work == 100
    reg.release(old.lease_id)
    assert reg.get("a").reserved.prefill_work == 100
