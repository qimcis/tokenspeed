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

"""Explicit opt-in NVIDIA qualification; no speed claim from these tests."""

import os

import pytest
import torch

if os.environ.get("TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS") != "1":
    pytest.skip(
        "set TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS=1 to qualify",
        allow_module_level=True,
    )
if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("requires NVIDIA CUDA", allow_module_level=True)

from tokenspeed_kernel.numerics.dspark import (
    INT64_MIN,
    compact_dspark_candidates_reference,
    pack_dspark_candidates_reference,
)
from tokenspeed_kernel.ops.sampling.triton.dspark_block import (
    _block_compact_candidates_kernel,
    dspark_block_candidate_tiles,
    dspark_block_compact_candidates,
    dspark_block_greedy_resolve,
    dspark_block_greedy_step,
)
from utils import assert_no_triton_compile


@pytest.mark.parametrize("rows", [0, 1, 17, 65])
@pytest.mark.parametrize("tiles", [1, 3, 257])
def test_packed_integer_reduction(rows, tiles):
    scores = torch.randn(rows, tiles, device="cuda", dtype=torch.float32)
    tokens = torch.arange(tiles, device="cuda", dtype=torch.int64).expand(rows, -1)
    partials = pack_dspark_candidates_reference(scores, tokens)
    if rows:
        partials[0].fill_(INT64_MIN)
    if rows > 1:
        partials[1, -1] = 2**63 - 1
    output = torch.empty(rows, 1, dtype=torch.int64, device="cuda")
    dspark_block_compact_candidates(partials, output)
    torch.testing.assert_close(
        output, compact_dspark_candidates_reference(partials), atol=0, rtol=0
    )


def test_special_value_packed_reduction():
    bits = torch.tensor(
        [0, -(2**31), 0x7FC00001, -0x400000, 0x7F800000, -0x800000, 0x7FC00000],
        dtype=torch.int32,
        device="cuda",
    )
    scores = bits.view(torch.float32).expand(5, -1).contiguous()
    tokens = torch.arange(7, dtype=torch.int64, device="cuda").expand(5, -1)
    partials = pack_dspark_candidates_reference(scores, tokens)
    output = torch.empty(5, 1, dtype=torch.int64, device="cuda")
    dspark_block_compact_candidates(partials, output)
    torch.testing.assert_close(
        output, compact_dspark_candidates_reference(partials), atol=0, rtol=0
    )


def test_graph_replay_and_row_count_does_not_compile():
    partials = torch.randint(-10000, 10000, (65, 129), dtype=torch.int64, device="cuda")
    output = torch.empty(65, 1, dtype=torch.int64, device="cuda")
    for rows in (1, 16, 17):
        dspark_block_compact_candidates(partials[:rows], output[:rows])
    with assert_no_triton_compile(_block_compact_candidates_kernel):
        for rows in (2, 31, 32, 65):
            dspark_block_compact_candidates(partials[:rows], output[:rows])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dspark_block_compact_candidates(partials, output)
    partials.fill_(INT64_MIN)
    partials[:, -1] = torch.arange(65, device="cuda", dtype=torch.int64)
    graph.replay()
    torch.testing.assert_close(output, partials.amax(-1, keepdim=True), atol=0, rtol=0)


@pytest.mark.parametrize(
    "tp,rows,local_vocab", [(1, 1, 257), (2, 17, 513), (4, 3, 259)]
)
def test_block_chain_matches_baseline_with_vocab_tail(tp, rows, local_vocab):
    torch.manual_seed(37)
    block, rank = 3, 32
    vocab = tp * local_vocab - 7
    embedding = torch.randn(vocab, rank, dtype=torch.bfloat16, device="cuda")
    projection = torch.randn(tp, local_vocab, rank, dtype=torch.bfloat16, device="cuda")
    bases = torch.randn(
        tp, rows, block, local_vocab, dtype=torch.float32, device="cuda"
    )
    # Include graph-padding-style invalid anchors; both paths clamp identically.
    anchors = torch.arange(rows, dtype=torch.int32, device="cuda") - 1
    tiles = dspark_block_candidate_tiles(local_vocab)
    baseline = torch.empty(tp, rows, tiles, dtype=torch.int64, device="cuda")
    compact = torch.empty(tp, rows, 1, dtype=torch.int64, device="cuda")
    baseline_output = torch.empty(rows, block, dtype=torch.int32, device="cuda")
    compact_output = torch.empty_like(baseline_output)
    baseline_next = torch.empty_like(baseline)
    compact_next = torch.empty_like(compact)
    scratch = torch.empty(rows, tiles, dtype=torch.int64, device="cuda")
    for step in range(block):
        for shard in range(tp):
            args = (bases[shard], step, anchors)
            tail = min(local_vocab, vocab - shard * local_vocab)
            weights = (embedding, projection[shard], shard * local_vocab, tail)
            dspark_block_greedy_step(
                *args, baseline, *weights, baseline_next[shard], baseline_output
            )
            dspark_block_greedy_step(*args, compact, *weights, scratch, compact_output)
            dspark_block_compact_candidates(scratch, compact_next[shard])
        baseline, baseline_next = baseline_next, baseline
        compact, compact_next = compact_next, compact
        torch.testing.assert_close(
            compact, baseline.amax(-1, keepdim=True), atol=0, rtol=0
        )
    dspark_block_greedy_resolve(baseline, baseline_output, block - 1)
    dspark_block_greedy_resolve(compact, compact_output, block - 1)
    torch.testing.assert_close(compact_output, baseline_output, atol=0, rtol=0)


def test_overlap_rejected():
    partials = torch.empty(3, 5, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        dspark_block_compact_candidates(partials, partials.view(-1)[:3].view(3, 1))
