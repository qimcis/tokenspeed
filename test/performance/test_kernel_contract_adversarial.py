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

"""Adversarial CPU leaf contracts; these tests do not execute GPU kernels."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.models.engram_hot_cache import (
    EngramHotCache,
    EngramHotSetLayer,
    EngramHotSetManifest,
    plan_engram_hot_cache,
)

# Import the actual CPU reference without the optional GPU package bootstrap.
_path = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel/python/tokenspeed_kernel/numerics/dspark.py"
)
_spec = importlib.util.spec_from_file_location("_adversarial_dspark_reference", _path)
_dspark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dspark)


def test_packed_finite_order_across_every_fp32_exponent_boundary():
    # Include the last mantissa and first mantissa in each finite exponent,
    # both signs and the smallest subnormal. Exclude zero/NaN: those retain
    # DSpark's separate bitwise ordering rather than numeric argmax semantics.
    positive = torch.tensor(
        [
            1,
            *[
                value
                for exponent in range(1, 255)
                for value in ((exponent << 23) - 1, exponent << 23)
            ],
            0x7F7FFFFF,
        ],
        dtype=torch.int32,
    )
    signed_bits = torch.cat((positive, positive | -(1 << 31)))
    scores = signed_bits.view(torch.float32)
    ids = torch.arange(scores.numel(), dtype=torch.int64)
    keys = _dspark.pack_dspark_candidates_reference(scores, ids)
    # Numeric sorting is an independent oracle for finite, nonzero values.
    assert torch.equal(keys.argsort(stable=True), scores.argsort(stable=True))


def test_compaction_preserves_highest_supported_token_id_and_cross_rank_tie():
    max_id = (1 << 31) - 1
    scores = torch.tensor([[[1.0, 1.0]], [[0.0, 1.0]]])
    ids = torch.tensor([[[max_id, max_id - 1]], [[0, max_id - 2]]])
    keys = _dspark.pack_dspark_candidates_reference(scores, ids)
    assert _dspark.resolve_dspark_candidates_reference(keys).item() == max_id - 2
    compact = _dspark.compact_dspark_candidates_reference(keys)
    assert _dspark.resolve_dspark_candidates_reference(compact).item() == max_id - 2
    one = _dspark.pack_dspark_candidates_reference(scores[:1, :, :1], ids[:1, :, :1])
    assert _dspark.resolve_dspark_candidates_reference(one).item() == max_id


@pytest.mark.parametrize("token_id", [-1, 1 << 31, (1 << 63) - 1])
def test_pack_rejects_token_ids_that_cannot_roundtrip_through_int32(token_id):
    with pytest.raises(ValueError, match="nonnegative int32"):
        _dspark.pack_dspark_candidates_reference(
            torch.tensor(1.0), torch.tensor(token_id, dtype=torch.int64)
        )


@pytest.mark.parametrize("start", [0, (1 << 32) + 7, (1 << 63) - 9])
def test_engram_global_ids_keep_int64_precision_with_strided_repeated_indices(start):
    # Only four local rows are allocated even when global IDs exceed int32.
    manifest = EngramHotSetManifest(
        1, "boundary", (EngramHotSetLayer(0, start + 4, (start, start + 2)),)
    )
    plan = plan_engram_hot_cache(
        manifest,
        layer_id=0,
        model_revision="boundary",
        total_rows=start + 4,
        row_start=start,
        row_end=start + 4,
        capacity_rows=2,
        embedding_dim=32,
    )
    # E4M3 codes 56,64,68,72 are exactly 1,2,3,4 at neutral scale.
    codes = torch.tensor([56, 64, 68, 72], dtype=torch.uint8)[:, None].repeat(1, 32)
    scales = torch.full((4, 1), 127, dtype=torch.uint8)
    cache = EngramHotCache(plan, codes, scales, device="cpu", generation=0)
    indices = torch.tensor(
        [[start - 1, start, start + 1], [start + 2, start + 3, start + 4]],
        dtype=torch.int64,
    ).T
    assert not indices.is_contiguous()
    result = cache.lookup_reference(codes, scales, indices)
    expected = torch.tensor([[0, 3], [1, 4], [2, 0]], dtype=torch.bfloat16)
    assert torch.equal(result, expected[..., None].expand(3, 2, 32))
    repeated = indices[1:2].expand(5, -1)
    actual_codes, actual_scales, owned, hits = cache.gather_bytes_reference(
        codes, scales, repeated
    )
    assert actual_codes.shape == (5, 2, 32)
    assert (actual_codes[:, 0] == 56).all()
    assert (actual_codes[:, 1] == 72).all()
    assert (actual_scales == 127).all() and owned.all()
    assert torch.equal(hits, torch.tensor([[True, False]]).expand(5, -1))
