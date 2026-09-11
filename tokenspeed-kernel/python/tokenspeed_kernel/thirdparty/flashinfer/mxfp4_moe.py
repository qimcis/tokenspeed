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

from __future__ import annotations

from flashinfer import (
    mxfp8_quantize,
    nvfp4_block_scale_interleave,
    trtllm_fp4_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
)

try:
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType
except ImportError:
    trtllm_fp4_block_scale_routed_moe = None
    ActivationType = None


def routed_moe_unavailable_reason() -> str | None:
    if trtllm_fp4_block_scale_routed_moe is None or ActivationType is None:
        return "FlashInfer is missing the routed FP4 MoE API"
    return None


__all__ = [
    "ActivationType",
    "get_w2_permute_indices_with_cache",
    "maybe_get_cached_w3_w1_permute_indices",
    "mxfp8_quantize",
    "nvfp4_block_scale_interleave",
    "routed_moe_unavailable_reason",
    "trtllm_fp4_block_scale_moe",
    "trtllm_fp4_block_scale_routed_moe",
]
