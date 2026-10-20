# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Bounded, deterministic placement search using per-forward logical route vectors.

Costs are predictions under an even split among physical replicas. They are not
latency measurements. This implementation shares no source with external EPLB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TracePlacementOptions:
    """Explicit limits and objective units for one asynchronous update.

    ``max_moved_slots`` is a whole-update budget, across all layers.
    ``movement_penalty`` is predicted objective units per overwritten slot.
    Measured costs must be supplied separately in units per routed token.
    """

    max_steps: int
    max_moved_slots: int
    movement_penalty: float
    min_relative_improvement: float
    objective: Literal["token_count", "measured_cost"]

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 4096:
            raise ValueError("trace max_steps must be in [1, 4096]")
        if self.max_moved_slots < 0:
            raise ValueError("trace max_moved_slots must be nonnegative")
        if not math.isfinite(self.movement_penalty) or self.movement_penalty < 0:
            raise ValueError("trace movement_penalty must be finite and nonnegative")
        if (
            not math.isfinite(self.min_relative_improvement)
            or not 0 <= self.min_relative_improvement < 1
        ):
            raise ValueError("trace min_relative_improvement must be in [0, 1)")
        if self.objective not in ("token_count", "measured_cost"):
            raise ValueError("unknown trace objective")


@dataclass(frozen=True)
class TracePlacementResult:
    physical_to_logical: torch.Tensor
    before: float
    after: float
    moved_slots: int
    reason: str


def validate_placement(
    rows: torch.Tensor, *, layers: int, experts: int, ep_size: int
) -> None:
    if rows.device.type != "cpu" or rows.ndim != 2 or rows.shape[0] != layers:
        raise ValueError("placement must be a host [layers, physical] tensor")
    if rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("placement must contain integer logical IDs")
    if ep_size <= 0 or rows.shape[1] < experts or rows.shape[1] % ep_size:
        raise ValueError("placement has insufficient slots or invalid rank capacity")
    if bool(((rows < 0) | (rows >= experts)).any()):
        raise ValueError("placement contains invalid logical IDs")
    for row in rows:
        if torch.bincount(row.long(), minlength=experts).min() == 0:
            raise ValueError("placement must cover every logical expert")


def predicted_peak_cost(trace: torch.Tensor, rows: torch.Tensor, ep_size: int) -> float:
    """Sum each step/layer's maximum rank cost, evenly splitting replicas."""
    total = 0.0
    for layer, row in enumerate(rows.long()):
        counts = torch.bincount(row, minlength=trace.shape[-1]).double()
        slots = trace[:, layer, row].double() / counts[row]
        loads = slots.reshape(trace.shape[0], ep_size, -1).sum(-1)
        total += float(loads.max(-1).values.sum())
    return total


def _pack(vectors: torch.Tensor, old: torch.Tensor, ep_size: int) -> torch.Tensor:
    """Allocate copies by per-step expert peaks, then greedily place their vectors."""
    experts, slots = vectors.shape[1], old.numel()
    if not bool(vectors.any()):
        return old.clone()
    copies = torch.ones(experts, dtype=torch.int64)
    for _ in range(slots - experts):
        per_copy = vectors / copies
        if experts == 1:
            other_peak = torch.zeros_like(per_copy)
        else:
            peaks, indices = per_copy.topk(2, dim=1)
            other_peak = peaks[:, :1].expand_as(per_copy).clone()
            other_peak.scatter_(1, indices[:, :1], peaks[:, 1:2])
        scores = torch.maximum(other_peak, vectors / (copies + 1)).sum(0)
        copies[int(scores.argmin())] += 1
    jobs = [expert for expert in range(experts) for _ in range(int(copies[expert]))]
    jobs.sort(
        key=lambda expert: (-float(vectors[:, expert].sum() / copies[expert]), expert)
    )
    capacity = slots // ep_size
    loads = torch.zeros((vectors.shape[0], ep_size), dtype=torch.float64)
    assigned: list[list[int]] = [[] for _ in range(ep_size)]
    for expert in jobs:
        vector = vectors[:, expert] / copies[expert]
        scores = []
        for rank in range(ep_size):
            if len(assigned[rank]) == capacity:
                scores.append(float("inf"))
                continue
            candidate = loads.clone()
            candidate[:, rank] += vector
            scores.append(float(candidate.max(1).values.sum()))
        rank = min(range(ep_size), key=lambda rank: (scores[rank], rank))
        assigned[rank].append(expert)
        loads[:, rank] += vector
    # Preserve the old physical slot when the expert remains on its rank.
    result = old.clone()
    for rank, remaining in enumerate(assigned):
        first = rank * capacity
        vacant = []
        for slot in range(first, first + capacity):
            expert = int(old[slot])
            if expert in remaining:
                remaining.remove(expert)
            else:
                vacant.append(slot)
        for slot, expert in zip(vacant, sorted(remaining)):
            result[slot] = expert
    return result


def validate_trace(
    logical_trace: torch.Tensor,
    *,
    options: TracePlacementOptions,
    expert_costs: torch.Tensor | None,
) -> torch.Tensor:
    """Validate a bounded window on every rank before asynchronous computation."""
    if logical_trace.device.type != "cpu" or logical_trace.ndim != 3:
        raise ValueError("logical_trace must be host [steps, layers, logical]")
    steps, layers, experts = logical_trace.shape
    if not 1 <= steps <= options.max_steps or layers < 1 or experts < 1:
        raise ValueError("trace is empty or exceeds its configured window")
    if not bool(torch.isfinite(logical_trace).all()) or bool((logical_trace < 0).any()):
        raise ValueError("trace loads must be finite and nonnegative")
    trace = logical_trace.double()
    if options.objective == "measured_cost":
        if (
            expert_costs is None
            or expert_costs.device.type != "cpu"
            or tuple(expert_costs.shape) != (layers, experts)
        ):
            raise ValueError(
                "measured_cost requires host [layers, logical] cost coefficients"
            )
        if not bool(torch.isfinite(expert_costs).all()) or bool(
            (expert_costs < 0).any()
        ):
            raise ValueError("expert costs must be finite and nonnegative")
        trace = trace * expert_costs.double()
    elif expert_costs is not None:
        raise ValueError("token_count does not accept measured cost coefficients")
    if not bool(torch.isfinite(trace).all()) or not bool(torch.isfinite(trace.sum())):
        raise ValueError("trace aggregate cost overflowed")
    return trace


def compute_trace_placement(
    logical_trace: torch.Tensor,
    *,
    old_map: torch.Tensor,
    ep_size: int,
    options: TracePlacementOptions,
    expert_costs: torch.Tensor | None = None,
) -> TracePlacementResult:
    """Optimize bounded logical history while retaining coverage and slot addresses.

    Full greedy proposals exceeding the budget are approached by legal one-slot
    replica replacements or two-slot swaps. Only improving changes are accepted;
    no mutation can transiently lose the last copy of a logical expert.
    """
    trace = validate_trace(logical_trace, options=options, expert_costs=expert_costs)
    _, layers, experts = trace.shape
    validate_placement(old_map, layers=layers, experts=experts, ep_size=ep_size)
    old = old_map.to(torch.int32).clone()
    before = predicted_peak_cost(trace, old, ep_size)
    if before == 0 or options.max_moved_slots == 0:
        return TracePlacementResult(
            old, before, before, 0, "empty_load" if before == 0 else "movement_budget"
        )
    proposal = torch.stack(
        [_pack(trace[:, layer], row, ep_size) for layer, row in enumerate(old)]
    )
    current = old.clone()
    current_score = before
    if int((proposal != old).sum()) <= options.max_moved_slots:
        current = proposal
        current_score = predicted_peak_cost(trace, current, ep_size)
    else:
        # One bounded pass toward the proposal. Score only the changing layer,
        # keeping cost linear in layer count rather than rescanning the model.
        moved_total = 0
        for layer in range(layers):
            layer_trace = trace[:, layer : layer + 1]
            layer_score = predicted_peak_cost(
                layer_trace, current[layer : layer + 1], ep_size
            )
            for slot in range(current.shape[1]):
                row = current[layer]
                wanted = int(proposal[layer, slot])
                if int(row[slot]) == wanted:
                    continue
                copies = torch.bincount(row.long(), minlength=experts)
                partners = [-1] if copies[int(row[slot])] > 1 else []
                partners += [
                    p
                    for p in range(row.numel())
                    if int(row[p]) == wanted and int(row[p]) != int(proposal[layer, p])
                ]
                best_row = None
                best_score = layer_score
                best_moved = moved_total
                threshold = layer_score + options.movement_penalty * moved_total
                layer_moved = int((row != old[layer]).sum())
                for partner in partners:
                    candidate = row.clone()
                    candidate[slot] = wanted
                    if partner >= 0:
                        candidate[partner] = row[slot]
                    moved = (
                        moved_total - layer_moved + int((candidate != old[layer]).sum())
                    )
                    if moved > options.max_moved_slots:
                        continue
                    score = predicted_peak_cost(
                        layer_trace, candidate.unsqueeze(0), ep_size
                    )
                    penalized = score + options.movement_penalty * moved
                    if penalized < threshold:
                        best_row, best_score, best_moved = candidate, score, moved
                        threshold = penalized
                if best_row is not None:
                    current[layer] = best_row
                    current_score += best_score - layer_score
                    layer_score, moved_total = best_score, best_moved
    moved = int((current != old).sum())
    gain = before - current_score - options.movement_penalty * moved
    if gain <= 0 or gain / before < options.min_relative_improvement:
        return TracePlacementResult(
            old, before, before, 0, "hysteresis_or_movement_cost"
        )
    validate_placement(current, layers=layers, experts=experts, ep_size=ep_size)
    return TracePlacementResult(current, before, current_score, moved, "improved")
