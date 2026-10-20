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

"""CPU algebra tests only; GPU qualification lives in tokenspeed-kernel/test."""

import importlib.util
from pathlib import Path

import pytest
import torch

# The package bootstrap imports the GPU-only tokenspeed_triton distribution.
# Load the real dependency-light reference directly on CPU-only hosts.
_reference_path = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel/python/tokenspeed_kernel/numerics/dspark.py"
)
_spec = importlib.util.spec_from_file_location("dspark_reference", _reference_path)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
INT64_MIN = _reference.INT64_MIN
compact_dspark_candidates_reference = _reference.compact_dspark_candidates_reference
pack_dspark_candidates_reference = _reference.pack_dspark_candidates_reference
resolve_dspark_candidates_reference = _reference.resolve_dspark_candidates_reference


@pytest.mark.parametrize(
    "tp,rows,tiles", [(1, 0, 1), (1, 17, 1), (2, 7, 3), (8, 33, 257)]
)
def test_compact_exchange_matches_tile_exchange(tp, rows, tiles):
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn((tp, rows, tiles), generator=generator)
    tokens = torch.arange(tp * tiles).reshape(tp, 1, tiles).expand(tp, rows, tiles)
    packed = pack_dspark_candidates_reference(scores, tokens)
    compact = compact_dspark_candidates_reference(packed)
    assert compact.shape == (tp, rows, 1)
    torch.testing.assert_close(
        resolve_dspark_candidates_reference(compact),
        resolve_dspark_candidates_reference(packed),
        atol=0,
        rtol=0,
    )


def test_ties_across_ranks_choose_lowest_global_token():
    scores = torch.full((3, 2, 3), -7.0)
    tokens = (
        torch.tensor([8, 4, 2, 99, 17, 11, 5, 1, 6]).reshape(3, 1, 3).expand(3, 2, 3)
    )
    packed = pack_dspark_candidates_reference(scores, tokens)
    assert resolve_dspark_candidates_reference(packed).tolist() == [1, 1]
    assert resolve_dspark_candidates_reference(
        compact_dspark_candidates_reference(packed)
    ).tolist() == [1, 1]


def test_signed_zero_preserves_upstream_bit_order():
    scores = torch.tensor([[[-0.0, 0.0]]], dtype=torch.float32)
    tokens = torch.tensor([[[0, 1]]], dtype=torch.int64)
    packed = pack_dspark_candidates_reference(scores, tokens)
    assert packed.tolist() == [[[-1, 4294967294]]]
    assert resolve_dspark_candidates_reference(packed).item() == 1


def test_nan_payloads_and_infinities_preserve_upstream_order():
    # Exact bit patterns include negative quiet NaN and two positive payloads.
    bits = torch.tensor(
        [0xFFC00000 - 2**32, 0xFF800000 - 2**32, 0x7F800000, 0x7FC00000, 0x7FC00001],
        dtype=torch.int32,
    )
    scores = bits.view(torch.float32).reshape(1, 1, -1)
    tokens = torch.arange(5, dtype=torch.int64).reshape_as(scores)
    packed = pack_dspark_candidates_reference(scores, tokens)
    assert torch.all(packed[..., 1:] > packed[..., :-1])
    assert resolve_dspark_candidates_reference(packed).item() == 4
    assert (
        resolve_dspark_candidates_reference(
            compact_dspark_candidates_reference(packed)
        ).item()
        == 4
    )


def test_padding_cannot_beat_negative_infinity_and_empty_rows_stay_sentinel():
    packed = torch.full((2, 2, 5), INT64_MIN, dtype=torch.int64)
    packed[1, 0, 3] = pack_dspark_candidates_reference(
        torch.tensor(-float("inf")), torch.tensor(123, dtype=torch.int64)
    )
    assert resolve_dspark_candidates_reference(packed).tolist() == [123, -1]
    compact = compact_dspark_candidates_reference(packed)
    assert compact[0, 1, 0].item() == INT64_MIN
    assert resolve_dspark_candidates_reference(compact).tolist() == [123, -1]


def test_arbitrary_signed_int64_keys_preserve_max_associativity():
    keys = torch.tensor(
        [
            [[INT64_MIN, INT64_MIN + 1, -1], [0, 2**63 - 1, -2]],
            [[-7, 2**62, 99], [2**63 - 2, 42, INT64_MIN]],
        ],
        dtype=torch.int64,
    )
    compact = compact_dspark_candidates_reference(keys)
    torch.testing.assert_close(compact.amax(0).squeeze(-1), keys.amax((0, 2)))


@pytest.mark.parametrize(
    "bad", [torch.empty(2, 0, dtype=torch.int64), torch.ones(2, 3)]
)
def test_invalid_candidate_shape_or_dtype_rejected(bad):
    with pytest.raises(ValueError):
        compact_dspark_candidates_reference(bad)
