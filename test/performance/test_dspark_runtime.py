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

"""Exercise production runtime control flow using CPU GPU/transport substitutes.

These tests cover exchange layout, autoregressive resolution and buffer reuse.
They do not establish Triton compilation, collective capture support or speed.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]


def _pack(values, ids):
    bits = values.contiguous().view(torch.int32).long()
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return (ordered << 32) | (0xFFFFFFFF - ids)


def _resolve(candidates):
    best = candidates.amax(dim=(0, 2))
    return (0xFFFFFFFF - (best & 0xFFFFFFFF)).long()


def _score(logits, previous, embedding, projection, start, valid, partials):
    values = logits + embedding[previous].float() @ projection.float().T
    ids = start + torch.arange(values.shape[1])
    keys = _pack(values, ids)
    keys[:, valid:] = torch.iinfo(torch.int64).min
    for tile in range(partials.shape[1]):
        partials[:, tile] = keys[:, tile * 256 : (tile + 1) * 256].amax(dim=1)


@pytest.fixture
def heads(monkeypatch):
    # Only the unavailable GPU package and distributed backend are replaced.
    kernel = ModuleType("tokenspeed_kernel.ops.sampling.triton")
    kernel.dspark_block_candidate_tiles = lambda vocab: (vocab + 255) // 256

    def step(
        base, index, bonus, previous, embedding, projection, start, valid, out, tokens
    ):
        last = bonus.long() if index == 0 else _resolve(previous)
        if index > 0:
            tokens[:, index - 1] = last.int()
        _score(base[:, index], last, embedding, projection, start, valid, out)

    kernel.dspark_block_greedy_step = step
    kernel.dspark_block_compact_candidates = lambda src, out: out.copy_(
        src.amax(dim=1, keepdim=True)
    )
    kernel.dspark_block_greedy_resolve = lambda src, out, index: out[:, index].copy_(
        _resolve(src)
    )
    comm = ModuleType("tokenspeed.runtime.distributed.comm_ops")

    def no_group(*args):
        raise AssertionError("TP1 must not communicate")

    comm.all_gather_single = no_group
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    monkeypatch.setitem(sys.modules, comm.__name__, comm)
    return _load_heads()


def _load_heads():
    spec = importlib.util.spec_from_file_location(
        "_dspark_runtime_cpu_test",
        _ROOT / "python/tokenspeed/runtime/models/deepseek_v41_dspark_heads.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Shard:
    org_vocab_start_index: int
    num_org_elements: int
    num_added_elements: int = 0


class _Weight(nn.Module):
    """Synthetic checkpoint weights without a GPU layer backend."""

    def __init__(self, weight, start, valid):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.embedding_dim = weight.shape[1]
        self.tp_size = 1
        self.shard_indices = _Shard(start, valid)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("block", [1, 4])
def test_compact_matches_tiles_and_full_argmax_with_reused_capacity(
    heads, monkeypatch, tp_size, block
):
    torch.manual_seed(76)
    capacity, local_vocab, hidden = 5, 300, 8
    vocab = local_vocab * tp_size - 7
    embedding = torch.randn(vocab, hidden).to(torch.bfloat16)
    projection = torch.randn(tp_size * local_vocab, hidden).to(torch.bfloat16)
    full_logits = torch.randn(capacity, block, tp_size * local_vocab)
    group = tuple(range(tp_size))
    modes = {}
    for mode in ("tiles", "compact"):
        buffers = heads.dspark_greedy_workspace(
            tp_size,
            capacity,
            local_vocab,
            torch.device("cpu"),
            candidate_communication=mode,
        )
        pointers = tuple(x.data_ptr() if x is not None else None for x in buffers)
        outputs = []
        for rows in (capacity, 2, capacity):
            bonus = torch.arange(rows).int()
            expected = torch.empty(rows, block, dtype=torch.int32)
            last = bonus.long()
            for index in range(block):
                values = (
                    full_logits[:rows, index, :vocab]
                    + embedding[last].float() @ projection[:vocab].float().T
                )
                last = values.argmax(dim=1)
                expected[:, index] = last.int()
            transfer_elements = []
            round_index = 0

            def gather(destination, source, actual_group):
                nonlocal round_index
                assert actual_group == group
                assert destination.is_contiguous() and source.is_contiguous()
                per_rank = source.numel() // rows
                gathered = destination.view(tp_size, rows, per_rank)
                gathered[0].copy_(source.view(rows, per_rank))
                previous = (
                    bonus.long()
                    if round_index == 0
                    else expected[:, round_index - 1].long()
                )
                for rank in range(1, tp_size):
                    start = rank * local_vocab
                    partial = torch.empty(rows, 2, dtype=torch.int64)
                    _score(
                        full_logits[:rows, round_index, start : start + local_vocab],
                        previous,
                        embedding,
                        projection[start : start + local_vocab],
                        start,
                        min(local_vocab, vocab - start),
                        partial,
                    )
                    gathered[rank].copy_(
                        partial if per_rank == 2 else partial.amax(dim=1, keepdim=True)
                    )
                transfer_elements.append(source.numel())
                round_index += 1

            if tp_size > 1:
                monkeypatch.setattr(heads, "all_gather_single", gather)
            head = _Weight(projection[:local_vocab], 0, min(vocab, local_vocab))
            markov = heads.DSparkVanillaMarkov(_Weight(embedding, 0, vocab), head)
            out = torch.empty(rows, block, dtype=torch.int32)
            result = heads.sample_dspark_block_greedy(
                full_logits[:rows, :, :local_vocab],
                bonus,
                markov,
                head,
                group,
                buffers[0],
                buffers[1],
                out,
                candidate_communication=mode,
                compact_partials=buffers[2],
            )
            assert result is out
            assert torch.equal(out, expected)
            assert pointers == tuple(
                x.data_ptr() if x is not None else None for x in buffers
            )
            assert sum(transfer_elements) == (
                0 if tp_size == 1 else rows * block * (1 if mode == "compact" else 2)
            )
            outputs.append(out.clone())
        modes[mode] = outputs
    for baseline, compact in zip(modes["tiles"], modes["compact"], strict=True):
        assert torch.equal(baseline, compact)


@pytest.mark.parametrize("mode", ["tiles", "compact"])
def test_zero_rows_and_zero_block_do_not_launch_or_communicate(heads, mode):
    head = _Weight(torch.zeros(300, 8, dtype=torch.bfloat16), 0, 300)
    markov = heads.DSparkVanillaMarkov(head, head)
    buffers = heads.dspark_greedy_workspace(
        1, 4, 300, torch.device("cpu"), candidate_communication=mode
    )
    for rows, block in ((0, 3), (2, 0)):
        output = torch.empty(rows, block, dtype=torch.int32)
        assert (
            heads.sample_dspark_block_greedy(
                torch.empty(rows, block, 300),
                torch.zeros(rows, dtype=torch.int32),
                markov,
                head,
                None,
                buffers[0],
                buffers[1],
                output,
                candidate_communication=mode,
                compact_partials=buffers[2],
            )
            is output
        )


@pytest.mark.parametrize("mode", ["tiles", "compact"])
def test_invalid_workspace_is_rejected_before_scoring(heads, mode):
    head = _Weight(torch.zeros(300, 8, dtype=torch.bfloat16), 0, 300)
    markov = heads.DSparkVanillaMarkov(head, head)
    buffers = heads.dspark_greedy_workspace(
        2, 3, 300, torch.device("cpu"), candidate_communication=mode
    )
    with pytest.raises(ValueError, match="does not cover"):
        heads.sample_dspark_block_greedy(
            torch.zeros(4, 2, 300),
            torch.zeros(4, dtype=torch.int32),
            markov,
            head,
            (0, 1),
            buffers[0],
            buffers[1],
            torch.empty(4, 2, dtype=torch.int32),
            candidate_communication=mode,
            compact_partials=buffers[2],
        )
    with pytest.raises(ValueError, match="must be tiles or compact"):
        heads.dspark_greedy_workspace(
            1, 4, 300, torch.device("cpu"), candidate_communication="unknown"
        )


def test_compact_rejects_tile_workspace(heads):
    head = _Weight(torch.zeros(300, 8, dtype=torch.bfloat16), 0, 300)
    markov = heads.DSparkVanillaMarkov(head, head)
    candidates, partials, compact = heads.dspark_greedy_workspace(
        2, 4, 300, torch.device("cpu"), candidate_communication="tiles"
    )
    with pytest.raises(ValueError, match="does not cover"):
        heads.sample_dspark_block_greedy(
            torch.zeros(4, 2, 300),
            torch.zeros(4, dtype=torch.int32),
            markov,
            head,
            (0, 1),
            candidates,
            partials,
            torch.empty(4, 2, dtype=torch.int32),
            candidate_communication="compact",
            compact_partials=compact,
        )


def test_gpu_compact_runtime_graph_replay():
    import os

    if os.environ.get("TOKENSPEED_RUN_DSPARK_COMPACT_GPU_TESTS") != "1":
        pytest.skip("opt in with TOKENSPEED_RUN_DSPARK_COMPACT_GPU_TESTS=1")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed.runtime.models import deepseek_v41_dspark_heads as runtime

    torch.manual_seed(76)
    rows, block, vocab, rank = 5, 4, 700, 32
    embedding = _Weight(torch.randn(vocab, rank, device="cuda").bfloat16(), 0, vocab)
    head = _Weight(torch.randn(vocab, rank, device="cuda").bfloat16(), 0, vocab)
    markov = runtime.DSparkVanillaMarkov(embedding, head)
    logits = torch.randn(rows, block, vocab, device="cuda")
    # Exercise the drafter's strided bonus column, including multiple rows.
    bonus = torch.arange(rows * (block + 1), device="cuda", dtype=torch.int32).view(
        rows, block + 1
    )[:, 0]
    workspaces = {
        mode: runtime.dspark_greedy_workspace(
            1, rows + 3, vocab, logits.device, candidate_communication=mode
        )
        for mode in ("tiles", "compact")
    }
    outputs = {
        mode: torch.empty(rows, block, device="cuda", dtype=torch.int32)
        for mode in workspaces
    }

    def sample(mode):
        candidates, partials, compact = workspaces[mode]
        return runtime.sample_dspark_block_greedy(
            logits,
            bonus,
            markov,
            head,
            None,
            candidates,
            partials,
            outputs[mode],
            candidate_communication=mode,
            compact_partials=compact,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sample("tiles")
            sample("compact")
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sample("compact")
    for _ in range(2):
        sample("tiles")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs["compact"], outputs["tiles"], rtol=0, atol=0)
        # Captured reads must see the current weights, not stale staged copies.
        head.weight.add_(0.25)
        logits.add_(torch.randn_like(logits))


def test_baseline_kernel_wheel_keeps_tiles_available(heads, monkeypatch):
    kernel = sys.modules["tokenspeed_kernel.ops.sampling.triton"]
    monkeypatch.delattr(kernel, "dspark_block_compact_candidates")
    # Reload production code with the baseline wheel capability surface.
    baseline = _load_heads()
    head = _Weight(torch.zeros(300, 8, dtype=torch.bfloat16), 0, 300)
    markov = baseline.DSparkVanillaMarkov(head, head)
    candidates, partials, compact = baseline.dspark_greedy_workspace(
        1, 3, 300, torch.device("cpu"), candidate_communication="tiles"
    )
    logits = torch.zeros(3, 2, 300)
    logits[:, :, 17] = 1
    output = torch.empty(3, 2, dtype=torch.int32)
    baseline.sample_dspark_block_greedy(
        logits,
        torch.zeros(3, dtype=torch.int32),
        markov,
        head,
        None,
        candidates,
        partials,
        output,
        candidate_communication="tiles",
        compact_partials=compact,
    )
    assert torch.equal(output, torch.full_like(output, 17))
    with pytest.raises(RuntimeError, match="require a tokenspeed-kernel build"):
        baseline.dspark_greedy_workspace(
            1, 3, 300, torch.device("cpu"), candidate_communication="compact"
        )
