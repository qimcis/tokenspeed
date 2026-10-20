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

"""Bounded, plain-value route observations from completed decode work."""

from __future__ import annotations

from dataclasses import dataclass

MAX_EXPERT_KEYS = 256


@dataclass(frozen=True)
class RouteObservation:
    request_id: str
    generation: int
    computed_frontier: int
    expert_keys: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.generation <= 0 or self.computed_frontier <= 0:
            raise ValueError(
                "route observations require a live generation and frontier"
            )
        if not 0 < len(self.expert_keys) <= MAX_EXPERT_KEYS:
            raise ValueError("route observations require 1..256 expert keys")
        if tuple(sorted(set(self.expert_keys))) != self.expert_keys:
            raise ValueError("expert keys must be unique and sorted")
        if any(key < 0 or key >= 2**64 for key in self.expert_keys):
            raise ValueError("expert keys must fit uint64")


class RouteCaptureCadence:
    """Count only eligible pure decode forwards; startup and prefill do not count."""

    def __init__(self, *, interval: int, max_requests: int) -> None:
        if interval <= 0 or max_requests <= 0:
            raise ValueError(
                "route capture interval and request bound must be positive"
            )
        self.interval = interval
        self.max_requests = max_requests
        self._decodes = 0

    def begin(self, *, num_extends: int, input_lengths: tuple[int, ...]) -> bool:
        if num_extends or not input_lengths:
            return False
        if len(input_lengths) > self.max_requests or any(n != 1 for n in input_lengths):
            raise ValueError("route capture requires bounded, single-token decode rows")
        self._decodes += 1
        return (self._decodes - 1) % self.interval == 0


def observations_from_rows(
    *,
    request_ids: tuple[str, ...],
    generations: tuple[int, ...],
    computed_frontiers: tuple[int, ...],
    layer_rows: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
) -> tuple[RouteObservation, ...]:
    """Pack already-landed top-k IDs; negative padding/sentinel IDs carry no hint."""
    size = len(request_ids)
    if not size == len(generations) == len(computed_frontiers):
        raise ValueError("route identity fields must have equal lengths")
    if len(set(request_ids)) != size:
        raise ValueError("route identity rows must be unique")
    if len({layer for layer, _ in layer_rows}) != len(layer_rows):
        raise ValueError("route layers must be unique")
    keys: list[set[int]] = [set() for _ in request_ids]
    for layer, rows in layer_rows:
        if layer < 0 or layer >= 2**32 or len(rows) != size:
            raise ValueError("invalid route layer or row count")
        for target, experts in zip(keys, rows):
            for expert in experts:
                if expert >= 2**32:
                    raise ValueError("expert ID must fit uint32")
                if expert >= 0:
                    target.add((layer << 32) | expert)
    return tuple(
        RouteObservation(rid, generation, frontier, tuple(sorted(support)))
        for rid, generation, frontier, support in zip(
            request_ids, generations, computed_frontiers, keys
        )
        if support
    )
