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

"""Real cache load lifecycle on CPU; full Engram forward/graph opt in on CUDA."""

import os

import pytest
import torch

from tokenspeed.runtime.models.engram_cache_runtime import EngramCacheRuntime
from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotSetLayer,
    EngramHotSetManifest,
    plan_engram_hot_cache,
)


def _manifest():
    return EngramHotSetManifest(
        1, "checkpoint-a", (EngramHotSetLayer(1, 8, (1, 4, 7)),)
    )


def _runtime(start=0, end=8):
    plan = plan_engram_hot_cache(
        _manifest(),
        layer_id=1,
        model_revision="checkpoint-a",
        total_rows=8,
        row_start=start,
        row_end=end,
        capacity_rows=3,
        embedding_dim=64,
    )
    return EngramCacheRuntime(plan, device="cpu")


def _tables(rows=8):
    weight = (torch.arange(rows * 64).reshape(rows, 64) % 100).to(torch.float8_e4m3fn)
    scale = torch.full((rows, 2), 127, dtype=torch.uint8)
    return weight, scale


def _complete(runtime, weight, scale, order=("weight", "scale")):
    for field in order:
        runtime.begin_table_load(field)
        runtime.complete_table_load(field)
    runtime.finish_loading(weight, scale)


@pytest.mark.parametrize("order", [("weight", "scale"), ("scale", "weight")])
def test_only_paired_completed_loads_publish_cache(order):
    runtime = _runtime()
    weight, scale = _tables()
    runtime.begin_table_load(order[0])
    runtime.complete_table_load(order[0])
    with pytest.raises(RuntimeError, match="both completed"):
        runtime.finish_loading(weight, scale)
    assert runtime.cache is None
    runtime.begin_table_load(order[1])
    runtime.complete_table_load(order[1])
    # A shared-table follower may mark completion before the writer's barrier;
    # no materialization occurs until the model calls this explicit boundary.
    assert runtime.cache is None
    runtime.finish_loading(weight, scale)
    cache = runtime.require_ready()
    assert torch.equal(cache.codes, weight.view(torch.uint8)[[1, 4, 7]])
    assert torch.equal(cache.scales, scale[[1, 4, 7]])


def test_refresh_retains_all_captured_storage_and_updates_both_tensors():
    runtime = _runtime()
    weight, scale = _tables()
    _complete(runtime, weight, scale)
    cache = runtime.require_ready()
    pointers = tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    runtime.begin_update()
    assert not cache.valid
    assert cache.local_row_ids.tolist() == [-1, -1, -1]
    weight.view(torch.uint8).fill_(56)
    scale.fill_(128)
    _complete(runtime, weight, scale, ("scale", "weight"))
    assert runtime.require_ready() is cache
    assert pointers == tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    assert cache.local_row_ids.tolist() == [1, 4, 7]
    assert torch.all(cache.codes == 56)
    assert torch.all(cache.scales == 128)
    assert runtime.generation == 1


def test_failed_and_retried_table_load_does_not_reuse_old_completion():
    runtime = _runtime()
    weight, scale = _tables()
    _complete(runtime, weight, scale)
    runtime.begin_update()
    runtime.complete_table_load("weight")
    runtime.begin_table_load("weight")  # retry fails before completion
    runtime.complete_table_load("scale")
    with pytest.raises(RuntimeError, match="both completed"):
        runtime.finish_loading(weight, scale)
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.require_ready()
    assert runtime.cache.local_row_ids.tolist() == [-1, -1, -1]
    runtime.complete_table_load("weight")
    runtime.finish_loading(weight, scale)
    assert runtime.require_ready().valid


def test_direct_table_update_invalidates_before_completion():
    runtime = _runtime()
    weight, scale = _tables()
    _complete(runtime, weight, scale)
    runtime.begin_table_load("scale")
    assert not runtime.cache.valid
    assert runtime.cache.local_row_ids.tolist() == [-1, -1, -1]
    runtime.complete_table_load("scale")
    with pytest.raises(RuntimeError, match="both completed"):
        runtime.finish_loading(weight, scale)


def test_failed_refresh_cannot_leave_hits_enabled():
    runtime = _runtime()
    weight, scale = _tables()
    _complete(runtime, weight, scale)
    runtime.begin_update()
    runtime.complete_table_load("weight")
    runtime.complete_table_load("scale")
    with pytest.raises(ValueError, match="geometry"):
        runtime.finish_loading(weight, scale[:, :1])
    assert runtime.generation == 0
    assert not runtime.cache.valid
    assert runtime.cache.local_row_ids.tolist() == [-1, -1, -1]


@pytest.mark.parametrize(
    "start,end,owned", [(0, 8, [1, 4, 7]), (0, 4, [1]), (4, 8, [0, 3])]
)
def test_runtime_uses_shared_or_shard_local_rows(start, end, owned):
    runtime = _runtime(start, end)
    weight, scale = _tables(end - start)
    _complete(runtime, weight, scale)
    cache = runtime.require_ready()
    assert cache.local_row_ids.tolist() == owned
    assert torch.equal(cache.codes, weight.view(torch.uint8)[owned])


def test_zero_owned_hot_rows_can_still_finish_loading():
    runtime = _runtime(2, 4)
    weight, scale = _tables(2)
    _complete(runtime, weight, scale)
    cache = runtime.require_ready()
    assert cache.codes.shape == (0, 64)
    ids = torch.tensor([[2, 3, 2, -1, 4]])
    result = cache.lookup_reference(weight, scale, ids)
    assert torch.equal(
        result[0, :3],
        weight.view(torch.uint8)[[0, 1, 0]]
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16),
    )
    assert torch.count_nonzero(result[0, 3:]) == 0


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_RUN_ENGRAM_GPU_TESTS") != "1",
    reason="set TOKENSPEED_RUN_ENGRAM_GPU_TESTS=1 for explicit CUDA execution",
)
def test_real_embedding_candidate_graph_refresh_and_invalidation():
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding

    assert torch.cuda.is_available(), "Opted-in Engram tests require CUDA"
    mapping = Mapping(rank=0, world_size=1)
    baseline = RowShardedEngramEmbedding(8, 64, mapping, "cuda", True, "shared", 1)
    candidate = RowShardedEngramEmbedding(8, 64, mapping, "cuda", True, "shared", 1)
    candidate.configure_hot_cache(
        _manifest(), capacity_rows=3, model_revision="checkpoint-a", device="cuda"
    )
    weight, scale = _tables()
    for embed in (baseline, candidate):
        embed.weight_loader(embed.weight, weight)
        embed.weight_loader(embed.scale, scale)
        embed.finish_hot_cache_loading()
    ids = torch.tensor([[1, 2, 7, 1, -1, 8]], device="cuda")
    torch.testing.assert_close(candidate(ids), baseline(ids), rtol=0, atol=0)
    # Warm the actual serving call on a capture side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = candidate(ids)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, baseline(ids), rtol=0, atol=0)
    pointers = tuple(
        t.data_ptr()
        for t in (
            candidate.hot_cache_runtime.cache.local_row_ids,
            candidate.hot_cache_runtime.cache.codes,
            candidate.hot_cache_runtime.cache.scales,
        )
    )
    # Direct canonical mutations here model a quiesced refresh. Invalidated
    # captured keys must miss, so replay cannot use the old hot-row bytes.
    candidate.begin_hot_cache_update()
    weight.view(torch.uint8).fill_(56)
    scale.fill_(128)
    for embed in (baseline, candidate):
        embed.weight_loader(embed.weight, weight)
        embed.weight_loader(embed.scale, scale)
    # Use a different stream after the invalidated/unfinished load; no success
    # refresh fence has run. The runtime's invalidation boundary must suffice.
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output, baseline(ids), rtol=0, atol=0)
    candidate.finish_hot_cache_loading()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, baseline(ids), rtol=0, atol=0)
    assert pointers == tuple(
        t.data_ptr()
        for t in (
            candidate.hot_cache_runtime.cache.local_row_ids,
            candidate.hot_cache_runtime.cache.codes,
            candidate.hot_cache_runtime.cache.scales,
        )
    )
