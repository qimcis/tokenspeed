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

"""Whole-worker decode routing using incremental normalized load pressure.

Scores are dimensionless scheduling estimates, not predicted milliseconds. The
registry owns atomic selection/reservation and lease lifetime; this policy never
replays requests or moves an active stream to another worker.
"""

from dataclasses import dataclass, replace

from tokenspeed.runtime.load_balancing.types import WorkerSnapshot, Workload


def _nonnegative_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class DecodeRequest:
    """Current accepted context and output budget for one decode stream.

    ``max_new_tokens`` is the original output limit; ``None`` means unknown.
    ``generated_tokens`` counts accepted generated tokens already in context.
    Verification tokens are the current outstanding speculative verification
    width, when exposed by the runtime. Cache evidence identifies an immutable
    prompt prefix; a key alone without its explicit token count saves no KV.
    """

    context_tokens: int
    max_new_tokens: int | None
    generated_tokens: int = 0
    verification_tokens: int = 0
    cache_key: str | None = None
    cached_context_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("context_tokens", self.context_tokens),
            ("generated_tokens", self.generated_tokens),
            ("verification_tokens", self.verification_tokens),
            ("cached_context_tokens", self.cached_context_tokens),
        ):
            _nonnegative_integer(name, value)
        if self.context_tokens < self.generated_tokens:
            raise ValueError("generated tokens must already be included in context")
        if self.max_new_tokens is not None:
            _nonnegative_integer("max_new_tokens", self.max_new_tokens)
            if self.generated_tokens > self.max_new_tokens:
                raise ValueError("generated tokens exceed the original output limit")
        if self.cached_context_tokens > self.context_tokens - self.generated_tokens:
            raise ValueError("cached prefix cannot include newly generated tokens")
        if self.cache_key is not None and (
            not isinstance(self.cache_key, str) or not self.cache_key
        ):
            raise ValueError("cache_key must be a nonempty string or None")
        if self.cached_context_tokens and self.cache_key is None:
            raise ValueError("cached prefix tokens require an immutable cache key")


class DecodePolicy:
    """Reserve concurrency, context, output, verification and worst-case KV.

    Unknown output lengths retain a rolling ``unknown_output_tokens`` reserve
    after every progress update. The configured horizon is an explicit capacity
    estimate, not a maximum generation length. Context/indexer pressure includes
    the entire projected context even for cached prefixes. Worker capacities
    and reported loads must use the same token units as the reservations.
    """

    def __init__(self, *, unknown_output_tokens: int) -> None:
        _nonnegative_integer("unknown_output_tokens", unknown_output_tokens)
        if unknown_output_tokens == 0:
            raise ValueError("unknown_output_tokens must be positive")
        self.unknown_output_tokens = unknown_output_tokens

    def advance(
        self,
        request: DecodeRequest,
        *,
        generated_tokens: int,
        verification_tokens: int | None = None,
    ) -> DecodeRequest:
        """Advance accepted output without changing the original budget/cache.

        Repeated frontiers are idempotent; older frontiers are rejected. Context
        grows by the accepted-token delta, never by SSE text or drafted tokens.
        The adapter publishes this state only after its lease update succeeds.
        """
        _nonnegative_integer("generated_tokens", generated_tokens)
        if generated_tokens < request.generated_tokens:
            raise ValueError("decode progress cannot regress the accepted frontier")
        return replace(
            request,
            context_tokens=request.context_tokens
            + generated_tokens
            - request.generated_tokens,
            generated_tokens=generated_tokens,
            verification_tokens=(
                request.verification_tokens
                if verification_tokens is None
                else verification_tokens
            ),
        )

    def workload(self, request: DecodeRequest, snapshot: WorkerSnapshot) -> Workload:
        """Return replacement lease work at the supplied accepted-token frontier.

        This is a whole-worker demand; TP size never divides any component.
        Cache hints influence score but cannot reduce this hard KV reservation.
        """
        worker = snapshot.worker
        if worker.role != "decode":
            raise ValueError("decode routing requires a complete decode worker")
        remaining = (
            self.unknown_output_tokens
            if request.max_new_tokens is None
            else request.max_new_tokens - request.generated_tokens
        )
        if worker.context_limit is not None:
            headroom = worker.context_limit - request.context_tokens
            if headroom < 0:
                raise ValueError("request context exceeds the worker context limit")
            if request.max_new_tokens is None:
                remaining = min(remaining, headroom)
            elif remaining > headroom:
                raise ValueError(
                    "request output reserve exceeds the worker context limit"
                )
        projected_context = request.context_tokens + remaining
        return Workload(
            requests=1,
            decode_context_tokens=projected_context,
            decode_output_tokens=remaining,
            verification_tokens=request.verification_tokens,
            kv_tokens=projected_context,
        )

    def score(self, request: DecodeRequest, snapshot: WorkerSnapshot) -> float:
        """Score incremental squared capacity pressure; lower is preferred.

        An unavailable axis or insufficient remaining capacity excludes the
        worker. Identical scores leave deterministic worker ordering to registry.
        Freshness, health and drain checks are exclusively registry-owned.
        """
        try:
            demand = self.workload(request, snapshot)
        except ValueError:
            return float("inf")
        total = snapshot.total
        capacity = snapshot.worker.capacity
        # Cache keys are unpinned hints: eviction must not violate hard capacity.
        if not (total + demand).fits(capacity):
            return float("inf")
        cached = (
            request.cached_context_tokens
            if request.cache_key is not None
            and request.cache_key in snapshot.worker.cache_keys
            else 0
        )
        score = 0.0
        for incoming, existing, limit in (
            (demand.requests, total.requests, capacity.requests),
            (
                demand.decode_context_tokens,
                total.decode_context_tokens,
                capacity.decode_context_tokens,
            ),
            (
                demand.decode_output_tokens,
                total.decode_output_tokens,
                capacity.decode_output_tokens,
            ),
            (
                demand.verification_tokens,
                total.verification_tokens,
                capacity.verification_tokens,
            ),
            (demand.kv_tokens - cached, total.kv_tokens, capacity.kv_tokens),
        ):
            if incoming == 0:
                continue
            if limit <= 0 or existing + incoming > limit:
                return float("inf")
            # Factored difference of squares avoids cancellation at small deltas.
            score += (incoming / limit) * ((2 * existing + incoming) / limit)
        return score
