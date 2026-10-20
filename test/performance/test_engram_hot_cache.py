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

"""CPU-only exact-byte and lifecycle tests for the actual cache materializer."""

import copy
import json

import pytest
import torch

from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotCache,
    EngramHotSetManifest,
    build_engram_hot_set_manifest,
    plan_engram_hot_cache,
)


def manifest_data():
    return {
        "schema_version": 1,
        "model_revision": "checkpoint-sha",
        "layers": [
            {"layer_id": 1, "total_rows": 12, "global_row_ids": [0, 3, 5, 8, 11]}
        ],
    }


def plan(*, start=0, end=12, capacity=5):
    return plan_engram_hot_cache(
        EngramHotSetManifest.from_dict(manifest_data()),
        layer_id=1,
        model_revision="checkpoint-sha",
        total_rows=12,
        row_start=start,
        row_end=end,
        capacity_rows=capacity,
        embedding_dim=256,
    )


def tables(rows=12):
    # Every possible FP8 bit pattern occurs, including signed zero and NaNs.
    weight = torch.arange(256).to(torch.uint8).expand(rows, -1).clone()
    scale = (
        torch.tensor([0, 1, 126, 127, 128, 253, 254, 255], dtype=torch.uint8)
        .expand(rows, -1)
        .clone()
    )
    return weight.view(torch.float8_e4m3fn), scale


def direct(weight, scale, ids):
    values = weight.view(torch.uint8)[ids].view(torch.float8_e4m3fn).float()
    exponents = scale[ids].view(torch.float8_e8m0fnu).float()
    return (
        (values.unflatten(-1, (-1, 32)) * exponents.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


def test_manifest_roundtrip_file_and_global_to_local_translation(tmp_path):
    data = manifest_data()
    path = tmp_path / "hot.json"
    path.write_text(json.dumps(data))
    assert EngramHotSetManifest.load(path).to_dict() == data
    shard = plan(start=4, end=9, capacity=2)
    assert shard.global_row_ids == (5, 8)
    assert shard.local_row_ids == (1, 4)
    assert shard.resident_nbytes == 2 * (256 + 8 + 8)
    assert shard.staging_nbytes == shard.resident_nbytes
    # Shared tables copy these same bytes separately on every rank.
    assert plan().resident_nbytes == 5 * 272
    assert plan(capacity=0).local_row_ids == ()
    assert plan(capacity=0).resident_nbytes == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("model_revision", ""),
        ("model_revision", None),
        ("layers", {}),
    ],
)
def test_reject_manifest_header(field, value):
    data = manifest_data()
    data[field] = value
    with pytest.raises(ValueError):
        EngramHotSetManifest.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_id", True),
        ("layer_id", -1),
        ("total_rows", 0),
        ("global_row_ids", [0, 0]),
        ("global_row_ids", [True]),
        ("global_row_ids", [-1]),
        ("global_row_ids", [12]),
        ("global_row_ids", [1.0]),
        ("global_row_ids", "3"),
    ],
)
def test_reject_manifest_layer(field, value):
    data = manifest_data()
    data["layers"][0][field] = value
    with pytest.raises(ValueError):
        EngramHotSetManifest.from_dict(data)


def test_reject_duplicate_layers_unknown_fields_and_duplicate_json_keys(tmp_path):
    data = manifest_data()
    data["layers"].append(copy.deepcopy(data["layers"][0]))
    with pytest.raises(ValueError, match="layer IDs"):
        EngramHotSetManifest.from_dict(data)
    data = manifest_data()
    data["typo"] = 1
    with pytest.raises(ValueError, match="exactly"):
        EngramHotSetManifest.from_dict(data)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        EngramHotSetManifest.load(path)


@pytest.mark.parametrize(
    "replacement",
    [
        {"layer_id": 2},
        {"model_revision": "other"},
        {"total_rows": 13},
        {"capacity_rows": 4},
        {"capacity_rows": -1},
        {"row_start": -1},
        {"row_end": 13},
        {"row_start": 10, "row_end": 9},
        {"embedding_dim": 33},
    ],
)
def test_plan_rejects_identity_geometry_and_capacity_mismatch(replacement):
    kwargs = dict(
        layer_id=1,
        model_revision="checkpoint-sha",
        total_rows=12,
        row_start=0,
        row_end=12,
        capacity_rows=5,
        embedding_dim=256,
    )
    kwargs.update(replacement)
    with pytest.raises(ValueError):
        plan_engram_hot_cache(EngramHotSetManifest.from_dict(manifest_data()), **kwargs)


def test_frequency_builder_uses_deterministic_top_counts():
    result = build_engram_hot_set_manifest(
        model_revision="r",
        layer_rows={2: 9, 1: 9},
        frequencies={1: {5: 9, 4: 9, 0: 0, 1: 8}, 2: {8: 1}},
        capacity_rows=2,
    )
    assert [layer.global_row_ids for layer in result.layers] == [(4, 5), (8,)]
    empty = build_engram_hot_set_manifest(
        model_revision="r", layer_rows={1: 9}, frequencies={}, capacity_rows=0
    )
    assert empty.layers[0].global_row_ids == ()
    for frequencies in ({9: {}}, {1: {9: 2}}, {1: {1: -1}}, {1: {True: 1}}):
        with pytest.raises(ValueError):
            build_engram_hot_set_manifest(
                model_revision="r",
                layer_rows={1: 9},
                frequencies=frequencies,
                capacity_rows=2,
            )


@pytest.mark.parametrize(
    "start,end,capacity", [(0, 12, 5), (4, 9, 2), (0, 12, 0), (6, 6, 1)]
)
def test_exact_bytes_and_bf16_for_hits_misses_repeats_and_shards(start, end, capacity):
    cache_plan = plan(start=start, end=end, capacity=capacity)
    weight, scale = tables(end - start)
    cache = EngramHotCache(cache_plan, weight, scale, device="cpu", generation=0)
    ids = torch.tensor([[-1, 0, 1, 3, 5], [5, 8, 11, 12, 100]], dtype=torch.int64)
    codes, scales, owned, hits = cache.gather_bytes_reference(weight, scale, ids)
    assert torch.equal(owned, (ids >= start) & (ids < end))
    assert torch.equal(
        hits,
        torch.isin(ids, torch.tensor(cache_plan.global_row_ids, dtype=torch.int64)),
    )
    assert torch.equal(codes[owned], weight.view(torch.uint8)[ids[owned] - start])
    assert torch.equal(scales[owned], scale[ids[owned] - start])
    assert not codes[~owned].any()
    assert (scales[~owned] == 127).all()
    expected = torch.zeros((*ids.shape, 256), dtype=torch.bfloat16)
    expected[owned] = direct(weight, scale, ids[owned] - start)
    actual = cache.lookup_reference(weight, scale, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    # Byte comparison also checks negative zero, which numeric equality misses.
    finite = ~expected.isnan()
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


def test_all_e8m0_bytes_cross_all_fp8_bytes():
    weight, _ = tables(256)
    scales = torch.arange(256).to(torch.uint8).unsqueeze(-1).expand(-1, 8).clone()
    manifest = build_engram_hot_set_manifest(
        model_revision="r",
        layer_rows={1: 256},
        frequencies={1: {row: 1 for row in range(0, 256, 2)}},
        capacity_rows=128,
    )
    cache_plan = plan_engram_hot_cache(
        manifest,
        layer_id=1,
        model_revision="r",
        total_rows=256,
        row_start=0,
        row_end=256,
        capacity_rows=128,
        embedding_dim=256,
    )
    cache = EngramHotCache(cache_plan, weight, scales, device="cpu", generation=0)
    ids = torch.arange(256)
    result = cache.lookup_reference(weight, scales, ids)
    torch.testing.assert_close(
        result, direct(weight, scales, ids), rtol=0, atol=0, equal_nan=True
    )


def test_hits_do_not_read_canonical_bytes_and_misses_do():
    weight, scale = tables()
    cache = EngramHotCache(plan(), weight, scale, device="cpu", generation=0)
    saved_codes = weight.view(torch.uint8)[0].clone()
    saved_scales = scale[0].clone()
    # Deliberate divergence is test-only: production writes must invalidate first.
    weight.view(torch.uint8).fill_(1)
    scale.fill_(129)
    codes, scales, _, hits = cache.gather_bytes_reference(
        weight, scale, torch.tensor([0, 1, 0])
    )
    assert hits.tolist() == [True, False, True]
    assert torch.equal(codes[0], saved_codes) and torch.equal(codes[2], saved_codes)
    assert torch.equal(scales[0], saved_scales)
    assert (codes[1] == 1).all() and (scales[1] == 129).all()


def test_refresh_is_pointer_stable_and_failed_refresh_disables_device_hits():
    weight, scale = tables()
    cache = EngramHotCache(plan(), weight, scale, device="cpu", generation=0)
    pointers = [
        tensor.data_ptr() for tensor in (cache.local_row_ids, cache.codes, cache.scales)
    ]
    cache.invalidate()
    assert (cache.local_row_ids == -1).all()
    with pytest.raises(RuntimeError, match="invalid"):
        cache.lookup_reference(weight, scale, torch.tensor([0]))
    weight.view(torch.uint8).fill_(56)
    scale.fill_(127)
    cache.refresh_from_tables(
        weight, scale, model_revision="checkpoint-sha", generation=1
    )
    assert cache.valid and cache.generation == 1
    assert [
        tensor.data_ptr() for tensor in (cache.local_row_ids, cache.codes, cache.scales)
    ] == pointers
    assert cache.local_row_ids.tolist() == list(cache.plan.local_row_ids)
    assert (cache.codes == 56).all() and (cache.scales == 127).all()
    for revision, generation in [("different", 2), ("checkpoint-sha", 1)]:
        with pytest.raises(ValueError):
            cache.refresh_from_tables(
                weight, scale, model_revision=revision, generation=generation
            )
        assert not cache.valid and (cache.local_row_ids == -1).all()
    with pytest.raises(ValueError, match="geometry"):
        cache.refresh_from_tables(
            weight, scale[:, :-1], model_revision="checkpoint-sha", generation=2
        )
    assert not cache.valid


def test_empty_and_scalar_index_shapes_and_int32():
    weight, scale = tables()
    cache = EngramHotCache(plan(), weight, scale, device="cpu", generation=0)
    for shape in [(0,), (2, 0), ()]:
        ids = torch.zeros(shape, dtype=torch.int32)
        assert cache.lookup_reference(weight, scale, ids).shape == (*shape, 256)
    with pytest.raises(TypeError, match="int32"):
        cache.lookup_reference(weight, scale, torch.tensor([0.0]))
