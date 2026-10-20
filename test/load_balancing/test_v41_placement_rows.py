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

"""CPU distributed semantic proof of source-rank-preserving placed dispatch.

This exercises the production replica selection and real Gloo exchanges. Expert
arithmetic is a small exact reference, not a claim about the GPU MegaMoE kernel.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.moe.rank_dispatch import (
    compute_logical_to_rank_dispatch_physical_map,
    dispatch_rank_local_topk_ids,
)


def _row_proof(rank, rendezvous, replicated, rows):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        slots = torch.tensor(rows)
        copies = [
            [i for i, e in enumerate(rows) if e == logical] for logical in range(4)
        ]
        width = max(map(len, copies))
        replicas = torch.tensor(
            [[ids + [-1] * (width - len(ids)) for ids in copies]], dtype=torch.int32
        )
        rank_map = compute_logical_to_rank_dispatch_physical_map(
            replicas, len(rows), (0, 0), rank
        )[0]
        x = torch.arange(12, dtype=torch.float64).reshape(3, 4) + (
            0 if replicated else rank * 17
        )
        logical = torch.tensor([[0, 1], [2, 0], [3, 2]], dtype=torch.int64)
        if not replicated and rank:
            logical = logical.flip(0)
        weights = torch.tensor(
            [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]], dtype=torch.float64
        )
        physical = dispatch_rank_local_topk_ids(logical, rank_map)
        assert torch.equal(slots[physical], logical)
        inputs, ids, logical_ids = (
            [torch.empty_like(x) for _ in range(2)],
            [torch.empty_like(physical) for _ in range(2)],
            [torch.empty_like(logical) for _ in range(2)],
        )
        dist.all_gather(inputs, x)
        dist.all_gather(ids, physical)
        dist.all_gather(logical_ids, logical)
        # Preserve source-rank and route-slot axes through dispatch/combine.
        contributions = torch.zeros(2, 3, 2, 4, dtype=torch.float64)
        executions = torch.zeros(2, 3, 2, dtype=torch.int64)
        physical_load = torch.zeros(len(rows), dtype=torch.int64)
        local = len(rows) // 2
        for source_rank in range(2):
            for token in range(3):
                for route in range(2):
                    physical_id = int(ids[source_rank][token, route])
                    if physical_id // local == rank:
                        expert = int(slots[physical_id])
                        # Exact, nonidentical reference expert functions.
                        result = inputs[source_rank][token] * (expert + 1) + expert
                        contributions[source_rank, token, route] = (
                            result * weights[token, route]
                        )
                        executions[source_rank, token, route] += 1
                        physical_load[physical_id] += 1
        dist.all_reduce(contributions)
        dist.all_reduce(executions)
        dist.all_reduce(physical_load)
        assert torch.equal(executions, torch.ones_like(executions))
        assert int(physical_load.sum()) == 2 * 3 * 2
        for source_rank in range(2):
            e = logical_ids[source_rank].unsqueeze(-1)
            expected = (
                (inputs[source_rank].unsqueeze(1) * (e + 1) + e) * weights.unsqueeze(-1)
            ).sum(1)
            assert torch.equal(contributions[source_rank].sum(1), expected)
        if replicated:
            # Two rank-local invocations, one complete output each: outputs
            # agree; combining source-rank outputs again would double them.
            assert torch.equal(contributions[0], contributions[1])
            assert not torch.equal(
                contributions.sum((0, 2)), contributions[rank].sum(1)
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize(
    "rows",
    [
        (0, 1, 2, 3),
        (3, 0, 1, 0, 2, 0),
        (0, 0, 0, 1, 2, 3),
    ],
)
def test_source_rank_invocations_survive_replica_dispatch_exactly(
    tmp_path, replicated, rows
):
    if not dist.is_gloo_available():
        pytest.skip("Gloo CPU collectives unavailable")
    mp.spawn(
        _row_proof,
        args=((tmp_path / "rendezvous").as_uri(), replicated, rows),
        nprocs=2,
        join=True,
    )
