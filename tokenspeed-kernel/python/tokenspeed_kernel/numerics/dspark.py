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

"""CPU-capable references for DSpark's existing signed packed-key contract.

These references establish integer reduction algebra, not CUDA compilation,
collective correctness or equality of BF16 GEMM accumulation orders.
"""

import torch

INT64_MIN = -(1 << 63)
LOW32 = (1 << 32) - 1


def pack_dspark_candidates_reference(
    scores: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Pack FP32 scores and same-shaped int64 token IDs exactly as DSpark.

    IDs must fit nonnegative int32, as consumed by the greedy step. Signed
    zeros remain distinct and NaNs retain their bitwise ordering; this is
    intentionally the upstream packing, not torch.argmax's NaN semantics.
    """
    if scores.dtype != torch.float32 or token_ids.dtype != torch.int64:
        raise ValueError("scores must be FP32 and token_ids must be int64")
    if scores.shape != token_ids.shape or scores.device != token_ids.device:
        raise ValueError("scores and token_ids must have the same shape and device")
    if torch.any((token_ids < 0) | (token_ids > (1 << 31) - 1)):
        raise ValueError("token IDs must fit nonnegative int32")
    bits = scores.contiguous().view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    return (ordered << 32) | (LOW32 - token_ids)


def compact_dspark_candidates_reference(partials: torch.Tensor) -> torch.Tensor:
    """Return signed maxima along a nonempty tile axis, retaining that axis."""
    if partials.ndim < 2 or partials.shape[-1] <= 0:
        raise ValueError("partials must have a nonempty final tile axis")
    if partials.dtype != torch.int64:
        raise ValueError("partials must be int64")
    return partials.amax(dim=-1, keepdim=True)


def resolve_dspark_candidates_reference(candidates: torch.Tensor) -> torch.Tensor:
    """Resolve ``[tp, rows, tiles]`` keys to int32 IDs, including sentinel -1."""
    if candidates.ndim != 3 or candidates.shape[0] <= 0:
        raise ValueError("candidates must be [positive tp, rows, tiles]")
    best = compact_dspark_candidates_reference(candidates).amax(dim=0).squeeze(-1)
    return (LOW32 - (best & LOW32)).to(torch.int32)
