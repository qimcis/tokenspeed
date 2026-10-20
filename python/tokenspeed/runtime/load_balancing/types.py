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

"""Immutable controller identities and explicitly dimensioned work counters."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Workload:
    requests: int = 0
    prefill_work: float = 0.0
    decode_context_tokens: int = 0
    verification_tokens: int = 0
    encoder_patches: int = 0
    decode_output_tokens: int = 0
    kv_tokens: int = 0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be numeric")
            if value < 0 or value > 2**63 - 1 or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite and nonnegative")
            if field.name != "prefill_work" and not isinstance(value, int):
                raise ValueError(f"{field.name} must be an integer")

    def __add__(self, other: Workload) -> Workload:
        return Workload(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def fits(self, capacity: Workload) -> bool:
        return all(
            getattr(self, f.name) <= getattr(capacity, f.name) for f in fields(self)
        )


@dataclass(frozen=True)
class Worker:
    """One complete stage worker (all TP ranks), or a complete HTTP frontend.

    ``reported`` excludes controller-owned leases. ``observed_at`` is receipt
    time in the controller's monotonic clock, never a remote wall clock.
    Token, patch and concurrency capacities are hard admission limits.
    ``prefill_work`` is an admission budget in predicted milliseconds, not a
    physical runtime guarantee. Every axis requires explicit nonzero capacity.
    """

    worker_id: str
    incarnation: str
    role: str
    endpoint: str
    capacity: Workload
    observed_at: float
    sequence: int = 0
    ready: bool = True
    healthy: bool = True
    draining: bool = False
    reported: Workload = Workload()
    cache_keys: frozenset[str] = frozenset()
    context_limit: int | None = None

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v or len(v) > 256
            for v in (self.worker_id, self.incarnation)
        ):
            raise ValueError(
                "worker_id and incarnation must be 1..256 character strings"
            )
        if self.role not in {"encoder", "prefill", "decode", "complete", "pair"}:
            raise ValueError("unsupported worker role")
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.query
            or url.fragment
            or url.username
        ):
            raise ValueError(
                "endpoint must be an HTTP(S) base URL without credentials/query"
            )
        if (
            isinstance(self.observed_at, bool)
            or not math.isfinite(self.observed_at)
            or self.observed_at < 0
        ):
            raise ValueError("observed_at must be finite and nonnegative")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a nonnegative integer")
        if not isinstance(self.capacity, Workload) or not isinstance(
            self.reported, Workload
        ):
            raise ValueError("capacity and reported must be Workload")
        if (
            not isinstance(self.cache_keys, frozenset)
            or len(self.cache_keys) > 4096
            or any(not isinstance(k, str) or len(k) > 256 for k in self.cache_keys)
        ):
            raise ValueError("cache_keys must be a frozenset of strings")
        if any(
            type(value) is not bool
            for value in (self.ready, self.healthy, self.draining)
        ):
            raise ValueError("worker state flags must be boolean")
        if self.context_limit is not None and (
            type(self.context_limit) is not int or self.context_limit <= 0
        ):
            raise ValueError("context_limit must be a positive integer")


@dataclass(frozen=True)
class WorkerSnapshot:
    worker: Worker
    reserved: Workload

    @property
    def total(self) -> Workload:
        return self.worker.reported + self.reserved


@dataclass(frozen=True)
class Lease:
    lease_id: str
    request_id: str
    worker_id: str
    incarnation: str
    workload: Workload
    created_at: float


@dataclass(frozen=True)
class CompatiblePath:
    """An operator-verified SMG frontend with fixed E/P/D worker membership.

    The gateway must actually use precisely these stage workers. This contract
    cannot configure or verify arbitrary external SMG stage-selection policies.
    """

    path_id: str
    endpoint: str
    workers: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not self.path_id or not self.workers or not isinstance(self.workers, tuple):
            raise ValueError("path identity and immutable worker membership required")
        if len(set(self.workers)) != len(self.workers):
            raise ValueError("duplicate path worker")
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.query
            or url.fragment
            or url.username
        ):
            raise ValueError("path endpoint must be an HTTP(S) base URL")
