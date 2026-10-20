# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the FluentLLM project
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

"""Dependency-light production selection of physical replicas per source rank."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_physical_experts: int,
    ep_rank_nodes: Sequence[int],
    ep_rank: int,
    seed: int = 42,
) -> torch.Tensor:
    """Pick, for every rank, the replica it dispatches each logical expert to.

    Nearest first: a replica on the same GPU, then one on the same node, else a
    seeded fair draw over all replicas so the remote ranks spread evenly. The
    tiers are vectorized over the whole table; only the draws loop, over the
    (layer, expert) pairs that need one, in a fixed order so every rank
    derives the same map.

    Args:
        logical_to_all_physical_map: ``[layers, logical, X]`` replicas, -1 padded.
        num_physical_experts: ``P``, owned contiguously by the EP ranks.
        ep_rank_nodes: The node of every EP rank (its length is the EP size).
        ep_rank: The rank whose ``[layers, logical]`` slice to return.
        seed: Seed of the fair draws.

    Returns:
        ``ep_rank``'s slice, int32 on ``logical_to_all_physical_map``'s device.
    """
    num_gpus = len(ep_rank_nodes)
    if num_gpus <= 0 or num_physical_experts % num_gpus:
        raise ValueError(
            f"{num_physical_experts} physical experts do not divide over "
            f"{num_gpus} EP ranks"
        )
    if not 0 <= ep_rank < num_gpus:
        raise ValueError(f"ep_rank={ep_rank} is outside the {num_gpus} EP ranks")
    num_local = num_physical_experts // num_gpus
    nodes = torch.tensor(list(ep_rank_nodes), dtype=torch.int64)
    num_nodes = int(nodes.max().item()) + 1

    replicas = logical_to_all_physical_map.cpu().to(torch.int64)
    num_layers, num_logical_experts, max_replicas = replicas.shape
    valid = replicas >= 0
    gpu_of = torch.where(valid, replicas // num_local, -1)
    node_of = torch.where(valid, nodes[gpu_of.clamp_min(0)], -1)

    # For every (layer, expert): the column of the FIRST replica on each GPU
    # and on each node, -1 if none. Built by scattering the columns from last
    # to first, so the lowest column wins; max_replicas scatters over
    # [layers, experts, gpus] instead of a loop over the gpus.
    def first_replica_on(owner: torch.Tensor, count: int) -> torch.Tensor:
        first = torch.full(
            (num_layers, num_logical_experts, count), -1, dtype=torch.int64
        )
        for column in reversed(range(max_replicas)):
            index = owner[..., column].clamp_min(0).unsqueeze(-1)
            kept = first.gather(-1, index)
            first.scatter_(
                -1, index, torch.where(valid[..., column : column + 1], column, kept)
            )
        return first

    first_on_gpu = first_replica_on(gpu_of, num_gpus)  # [L, E, G]
    first_on_node = first_replica_on(node_of, num_nodes)[..., nodes]  # [L, E, G]
    column = torch.where(first_on_gpu >= 0, first_on_gpu, first_on_node)
    chosen = replicas.gather(-1, column.clamp_min(0))
    # A single replica serves everyone; otherwise only the nearest tiers.
    single = (valid.sum(-1) == 1).unsqueeze(-1)
    chosen = torch.where(single, replicas[..., :1], chosen)
    output = (
        torch.where((column >= 0) | single, chosen, -1).permute(2, 0, 1).contiguous()
    )

    r = random.Random(seed)
    # Pairs with a rank that has no same-node replica, in row-major order so
    # the draws are identical on every rank.
    for layer_id, logical_expert_id in (output == -1).any(0).nonzero().tolist():
        column = output[:, layer_id, logical_expert_id]
        unassigned = column == -1
        candidates = replicas[layer_id, logical_expert_id][
            valid[layer_id, logical_expert_id]
        ].tolist()
        column[unassigned] = torch.tensor(
            _fair_choices(candidates, k=int(unassigned.sum().item()), r=r),
            dtype=torch.int64,
        )

    return (
        output[ep_rank]
        .to(torch.int32)
        .contiguous()
        .to(logical_to_all_physical_map.device)
    )


def _fair_choices(arr: list, k: int, r: random.Random) -> list:
    quotient, remainder = divmod(k, len(arr))
    choices = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(choices)
    return choices


def dispatch_rank_local_topk_ids(
    topk_ids: torch.Tensor, rank_map: torch.Tensor
) -> torch.Tensor:
    """Select one physical slot for every logical route, retaining dtype and axes."""
    return rank_map[topk_ids].to(topk_ids.dtype)
