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

"""CPU checks of the real C2 public validation contract, not CUDA execution."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the dependency-light API without importing optional GPU backends.
_path = (
    Path(__file__).resolve().parents[2] / "tokenspeed-kernel/python/"
    "tokenspeed_kernel/ops/embedding/cached.py"
)
_spec = importlib.util.spec_from_file_location("engram_cached_api", _path)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)


def _inputs():
    return dict(
        weight=torch.empty((4, 256), dtype=torch.uint8),
        scales=torch.empty((4, 8), dtype=torch.uint8),
        indices=torch.tensor([0, 2], dtype=torch.int64),
        hot_ids=torch.tensor([0, 2], dtype=torch.int64),
        hot_codes=torch.empty((2, 256), dtype=torch.uint8),
        hot_scales=torch.empty((2, 8), dtype=torch.uint8),
        row_start=0,
        row_end=4,
        out=torch.empty((2, 256), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("weight", torch.empty((4, 33), dtype=torch.uint8), "weight"),
        ("weight", torch.empty((4, 256)), "weight"),
        ("scales", torch.empty((4, 8)), "scales"),
        ("row_end", 5, "shard range"),
        ("row_start", -1, "shard range"),
        ("hot_ids", torch.tensor([0, 2], dtype=torch.int32), "hot IDs"),
        ("hot_codes", torch.empty((1, 256), dtype=torch.uint8), "hot codes/scales"),
        ("hot_scales", torch.empty((2, 7), dtype=torch.uint8), "hot codes/scales"),
        ("indices", torch.tensor([0.0, 2.0]), "indices"),
        ("out", torch.empty((2, 256)), "output"),
        ("out", torch.empty((256, 2), dtype=torch.bfloat16).T, "output"),
    ],
)
def test_reject_invalid_metadata_before_launch(field, value, reason):
    args = _inputs()
    args[field] = value
    with pytest.raises(ValueError, match=reason):
        _api.engram_cached_embedding(**args)


def test_cpu_tensors_do_not_silently_select_a_different_execution_path():
    with pytest.raises(ValueError, match="one GPU device"):
        _api.engram_cached_embedding(**_inputs())
