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

"""Converge completed host observations before redundant schedulers consume them."""

from __future__ import annotations

import torch.distributed as dist

from tokenspeed.runtime.moe.route_capture import CapturedRoutes
from tokenspeed.runtime.moe.route_feedback import RouteCaptureCadence, RouteObservation


class RouteFeedbackHooks:
    """One completed batch at most; scheduler mutation stays explicit in the loop."""

    def __init__(
        self, *, interval: int, max_requests: int, leader: bool, group, source_rank: int
    ) -> None:
        self.cadence = RouteCaptureCadence(interval=interval, max_requests=max_requests)
        self.leader = leader
        self.group = group
        self.source_rank = source_rank
        self._pending: tuple[RouteObservation, ...] = ()

    def collect(
        self,
        *,
        num_extends: int,
        input_lengths: tuple[int, ...],
        captured: CapturedRoutes | None,
    ) -> None:
        if not self.cadence.begin(num_extends=num_extends, input_lengths=input_lengths):
            return
        if self._pending:
            raise RuntimeError(
                "route feedback must drain before the next completed batch"
            )
        if self.leader and captured is None:
            raise RuntimeError("selected route capture did not produce a result")
        payload = [captured.observations() if self.leader else None]
        if self.group.size() > 1:
            dist.broadcast_object_list(payload, src=self.source_rank, group=self.group)
        self._pending = payload[0]

    def drain(self) -> tuple[RouteObservation, ...]:
        pending, self._pending = self._pending, ()
        return pending


class PlacementFeedbackHooks:
    """Publish one complete CPU placement model to the native scheduler.

    Supported attention-TP layouts replicate every input row on all EP source
    ranks. The production static mapper selects one physical destination per
    source; its destination counts are the work weights, without normalization.
    Logical hints still predict future routes, not measured execution times.
    """

    def __init__(
        self, scheduler, *, layer_ids: tuple[int, ...], ep_rank_nodes: tuple[int, ...]
    ) -> None:
        if (
            not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or any(
                type(layer) is not int or not 0 <= layer < 2**32 for layer in layer_ids
            )
        ):
            raise ValueError("placement layer IDs must be unique uint32 values")
        if (
            not ep_rank_nodes
            or len(ep_rank_nodes) > 256
            or any(
                type(node) is not int or not 0 <= node < 256 for node in ep_rank_nodes
            )
        ):
            raise ValueError("placement requires bounded EP rank node IDs")
        self.scheduler = scheduler
        self.layer_ids = layer_ids
        self.ep_rank_nodes = ep_rank_nodes

    def publish(
        self,
        epoch: int,
        physical_to_logical,
        num_local_experts: int,
        *,
        logical_to_all_physical=None,
    ) -> None:
        """Publish actual static destination counts for replicated source rows.

        Startup passes its live replica table, whose order affects dispatch.
        Applied updates omit it: their production preparation reconstructs
        replicas in ascending physical-slot order from the complete new map.
        """
        import torch

        if isinstance(physical_to_logical, torch.Tensor):
            if physical_to_logical.device.type != "cpu":
                raise ValueError("placement publication requires a CPU snapshot")
            rows = physical_to_logical.tolist()
        else:
            rows = physical_to_logical
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("placement epoch must be positive")
        if type(num_local_experts) is not int or num_local_experts <= 0:
            raise ValueError("placement local expert count must be positive")
        if len(rows) != len(self.layer_ids) or not rows or not rows[0]:
            raise ValueError("placement must contain a row for every layer")
        width = len(rows[0])
        if width % num_local_experts:
            raise ValueError("placement slots must divide evenly across ranks")
        num_ranks = width // num_local_experts
        if num_ranks > 256 or width * len(rows) > 4194304:
            raise ValueError("placement exceeds bounded scheduler geometry")
        if num_ranks != len(self.ep_rank_nodes):
            raise ValueError("placement rank count disagrees with EP topology")
        keys: list[int] = []
        replica_rows: list[list[list[int]]] = []
        for layer, row in zip(self.layer_ids, rows, strict=True):
            if len(row) != width or any(
                type(expert) is not int or not -1 <= expert < 2**32 for expert in row
            ):
                raise ValueError("placement rows require equal width and logical IDs")
            counts: dict[int, list[int]] = {}
            for slot, expert in enumerate(row):
                if expert >= 0:
                    counts.setdefault(expert, []).append(slot)
            if not counts or any(
                expert != index for index, expert in enumerate(sorted(counts))
            ):
                raise ValueError("placement must cover contiguous logical expert IDs")
            if (
                len(keys) + len(counts) > 65536
                or (len(keys) + len(counts)) * num_ranks > 4194304
            ):
                raise ValueError("placement exceeds bounded scheduler model")
            replica_rows.append([counts[expert] for expert in sorted(counts)])
            keys.extend((layer << 32) | expert for expert in sorted(counts))
        num_logical = len(replica_rows[0])
        if any(len(layer) != num_logical for layer in replica_rows):
            raise ValueError("placement logical expert count must agree across layers")
        if logical_to_all_physical is not None:
            if isinstance(logical_to_all_physical, torch.Tensor):
                if logical_to_all_physical.device.type != "cpu":
                    raise ValueError("placement replica table requires a CPU snapshot")
                logical_to_all_physical = logical_to_all_physical.tolist()
            if len(logical_to_all_physical) != len(replica_rows):
                raise ValueError("placement replica table must cover every layer")
            ordered = []
            for layer, expected in zip(
                logical_to_all_physical, replica_rows, strict=True
            ):
                if len(layer) != num_logical:
                    raise ValueError("placement replica table must cover every expert")
                ordered_layer = []
                for slots, expected_slots in zip(layer, expected, strict=True):
                    if len(slots) > width or any(
                        type(slot) is not int or slot < -1 for slot in slots
                    ):
                        raise ValueError("placement replica table has invalid slots")
                    valid_slots = [slot for slot in slots if slot >= 0]
                    if sorted(valid_slots) != expected_slots:
                        raise ValueError(
                            "placement replica table disagrees with physical map"
                        )
                    ordered_layer.append(valid_slots)
                ordered.append(ordered_layer)
            replica_rows = ordered
        weights = _replicated_static_rank_work(
            replica_rows, num_local_experts, self.ep_rank_nodes
        )
        self.scheduler.set_expert_placement(epoch, keys, weights)


def _replicated_static_rank_work(replica_rows, num_local_experts, ep_rank_nodes):
    """Count physical owner destinations using the exact production mapper."""
    import torch

    from tokenspeed.runtime.moe.rank_dispatch import (
        compute_logical_to_rank_dispatch_physical_map,
    )

    num_ranks = len(ep_rank_nodes)
    num_layers, num_logical = len(replica_rows), len(replica_rows[0])
    width = max(len(slots) for layer in replica_rows for slots in layer)
    if num_layers * num_logical * width > 4194304:
        raise ValueError("placement replica table exceeds bounded scheduler geometry")
    replicas = torch.tensor(
        [
            [slots + [-1] * (width - len(slots)) for slots in layer]
            for layer in replica_rows
        ],
        dtype=torch.int32,
    )
    counts = torch.zeros((num_layers, num_logical, num_ranks), dtype=torch.float64)
    for source_rank in range(num_ranks):
        destinations = (
            compute_logical_to_rank_dispatch_physical_map(
                replicas, num_local_experts * num_ranks, ep_rank_nodes, source_rank
            ).to(torch.int64)
            // num_local_experts
        )
        counts.scatter_add_(
            -1,
            destinations.unsqueeze(-1),
            torch.ones_like(destinations, dtype=torch.float64).unsqueeze(-1),
        )
    return counts.reshape(-1, num_ranks).tolist()
