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

"""Adversarial byte, ownership and failed-load cases for real Engram classes.

These CPU tests do not establish GPU compilation, UVA access, or stream ordering.
"""

from dataclasses import FrozenInstanceError

import pytest
import torch

from tokenspeed.runtime.models.engram_cache_runtime import EngramCacheRuntime
from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotCache,
    EngramHotSetManifest,
    plan_engram_hot_cache,
)


def _plan(*, start, end, total, hot, width):
    manifest = EngramHotSetManifest.from_dict(
        {
            "schema_version": 1,
            "model_revision": "immutable-revision",
            "layers": [
                {"layer_id": 3, "total_rows": total, "global_row_ids": list(hot)}
            ],
        }
    )
    return plan_engram_hot_cache(
        manifest,
        layer_id=3,
        model_revision="immutable-revision",
        total_rows=total,
        row_start=start,
        row_end=end,
        capacity_rows=len(hot),
        embedding_dim=width,
    )


def _tables(rows, width):
    # Distinct rows expose global/local confusion that repeated row data hides.
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    col = torch.arange(width, dtype=torch.int64)[None, :]
    codes = ((row * 37 + row // 256 + col * 19) % 256).to(torch.uint8)
    scales = ((row * 13 + col[:, : width // 32] * 47) % 256).to(torch.uint8)
    return codes.view(torch.float8_e4m3fn), scales


def _publish(runtime, weight, scale):
    for field in ("weight", "scale"):
        runtime.begin_table_load(field)
        runtime.complete_table_load(field)
    runtime.finish_loading(weight, scale)
    return runtime.require_ready()


@pytest.mark.parametrize("width", [32, 96, 256])
@pytest.mark.parametrize("typed_scales", [False, True])
def test_distinct_shard_rows_extreme_and_noncontiguous_indices(width, typed_scales):
    plan = _plan(start=5, end=12, total=17, hot=[16, 11, 5, 0, 8], width=width)
    weight, scale = _tables(9, width)  # Last two rows are capacity padding.
    if typed_scales:
        scale = scale.view(torch.float8_e8m0fnu)
    cache = EngramHotCache(plan, weight, scale, device="cpu", generation=0)
    indices = torch.tensor(
        [[5, 6, 11, 12], [8, 5, -(2**63), 2**63 - 1], [4, 7, 10, -1]],
        dtype=torch.int64,
    ).T
    assert not indices.is_contiguous()
    codes, scales, owned, hits = cache.gather_bytes_reference(weight, scale, indices)
    for position in ((i, j) for i in range(4) for j in range(3)):
        index = int(indices[position])
        assert bool(owned[position]) == (5 <= index < 12)
        assert bool(hits[position]) == (index in (5, 8, 11))
        if 5 <= index < 12:
            assert torch.equal(codes[position], weight.view(torch.uint8)[index - 5])
            assert torch.equal(scales[position], scale.view(torch.uint8)[index - 5])
        else:
            assert torch.count_nonzero(codes[position]) == 0
            assert torch.all(scales[position] == 127)
    # Padding is never a resident row and cannot leak through extreme indices.
    assert cache.local_row_ids.tolist() == [0, 3, 6]


def test_chunk_boundary_refresh_keeps_sorted_keys_exact_bytes_and_storage():
    start, rows, width = 7, 4103, 32
    hot = list(reversed(range(start, start + rows)))
    plan = _plan(
        start=start, end=start + rows, total=start + rows, hot=hot, width=width
    )
    weight, scale = _tables(rows, width)
    cache = EngramHotCache(plan, weight, scale, device="cpu", generation=0)
    buffers = (cache.local_row_ids, cache.codes, cache.scales)
    pointers = tuple(tensor.data_ptr() for tensor in buffers)
    assert plan.resident_nbytes == sum(t.numel() * t.element_size() for t in buffers)
    assert plan.staging_nbytes < plan.resident_nbytes
    assert cache.local_row_ids.tolist() == list(range(rows))
    assert torch.equal(cache.codes, weight.view(torch.uint8))
    assert torch.equal(cache.scales, scale)
    weight.view(torch.uint8).bitwise_xor_(255)
    scale.bitwise_xor_(255)
    cache.refresh_from_tables(
        weight, scale, model_revision=plan.model_revision, generation=1
    )
    assert pointers == tuple(tensor.data_ptr() for tensor in buffers)
    # Verify every row, including both sides of the 4096-row staging boundary.
    assert torch.equal(cache.codes, weight.view(torch.uint8))
    assert torch.equal(cache.scales, scale)
    assert cache.local_row_ids.tolist() == list(range(rows))


def test_abandoned_update_cannot_publish_a_field_from_the_previous_attempt():
    plan = _plan(start=0, end=4, total=4, hot=[1, 3], width=32)
    weight, scale = _tables(4, 32)
    runtime = EngramCacheRuntime(plan, device="cpu")
    cache = _publish(runtime, weight, scale)
    pointers = tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    runtime.begin_update()
    runtime.begin_table_load("weight")
    weight.view(torch.uint8).fill_(56)
    runtime.complete_table_load("weight")
    # Abandon the first checkpoint attempt before its scale arrives.
    runtime.begin_update()
    runtime.begin_table_load("scale")
    scale.fill_(128)
    runtime.complete_table_load("scale")
    with pytest.raises(RuntimeError, match="both completed"):
        runtime.finish_loading(weight, scale)
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.require_ready()
    assert not cache.valid and torch.all(cache.local_row_ids == -1)
    assert runtime.generation == 0
    runtime.begin_table_load("weight")
    runtime.complete_table_load("weight")
    runtime.finish_loading(weight, scale)
    assert runtime.require_ready() is cache
    assert runtime.generation == cache.generation == 1
    assert pointers == tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    assert torch.all(cache.codes == 56) and torch.all(cache.scales == 128)


def test_first_materialization_failure_can_retry_without_publishing_partial_cache():
    plan = _plan(start=2, end=6, total=6, hot=[2, 5], width=32)
    weight, scale = _tables(4, 32)
    runtime = EngramCacheRuntime(plan, device="cpu")
    for field in ("weight", "scale"):
        runtime.begin_table_load(field)
        runtime.complete_table_load(field)
    with pytest.raises(TypeError, match="raw FP8"):
        runtime.finish_loading(weight.float(), scale)
    assert runtime.cache is None and runtime.generation == -1
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.require_ready()
    runtime.begin_update()
    cache = _publish(runtime, weight, scale)
    assert cache.generation == runtime.generation == 0
    assert torch.equal(cache.codes, weight.view(torch.uint8)[[0, 3]])


@pytest.mark.parametrize(
    "failure", ["revision", "generation", "strided_codes", "strided_scales"]
)
def test_failed_refresh_retry_preserves_identity_and_never_reenables_old_hits(failure):
    plan = _plan(start=0, end=6, total=6, hot=[1, 4], width=64)
    weight, scale = _tables(6, 64)
    cache = EngramHotCache(plan, weight, scale, device="cpu", generation=0)
    pointers = tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    revision, generation = plan.model_revision, 1
    bad_weight, bad_scale = weight, scale
    if failure == "revision":
        revision = "different-checkpoint"
    elif failure == "generation":
        generation = 0
    elif failure == "strided_codes":
        bad_weight = weight.view(torch.uint8).T.contiguous().T.view(torch.float8_e4m3fn)
    else:
        bad_scale = scale.T.contiguous().T
    with pytest.raises(ValueError):
        cache.refresh_from_tables(
            bad_weight, bad_scale, model_revision=revision, generation=generation
        )
    assert not cache.valid and cache.generation == 0
    assert torch.all(cache.local_row_ids == -1)
    with pytest.raises(RuntimeError, match="invalid"):
        cache.gather_bytes_reference(weight, scale, torch.tensor([1, 4]))
    weight.view(torch.uint8).fill_(184)
    scale.fill_(126)
    cache.refresh_from_tables(
        weight, scale, model_revision=plan.model_revision, generation=1
    )
    assert cache.valid and cache.generation == 1
    assert pointers == tuple(
        t.data_ptr() for t in (cache.local_row_ids, cache.codes, cache.scales)
    )
    assert torch.all(cache.codes == 184) and torch.all(cache.scales == 126)


def test_manifest_detaches_mutable_input_and_serialized_output():
    data = {
        "schema_version": 1,
        "model_revision": "r",
        "layers": [{"layer_id": 3, "total_rows": 8, "global_row_ids": [1, 6]}],
    }
    manifest = EngramHotSetManifest.from_dict(data)
    data["layers"][0]["global_row_ids"][0] = 7
    data["model_revision"] = "changed"
    exported = manifest.to_dict()
    exported["layers"][0]["global_row_ids"].append(5)
    assert manifest.model_revision == "r"
    assert manifest.layers[0].global_row_ids == (1, 6)
    with pytest.raises(FrozenInstanceError):
        manifest.model_revision = "changed"
    with pytest.raises(FrozenInstanceError):
        manifest.layers[0].total_rows = 9
