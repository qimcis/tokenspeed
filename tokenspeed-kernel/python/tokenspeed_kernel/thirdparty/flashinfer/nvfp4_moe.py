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
    ActivationType,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

fp4_quantize = None
convert_sf_to_mma_layout = None

try:
    from flashinfer import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fused_moe.cute_dsl import CuteDslMoEWrapper
except ImportError:
    CuteDslMoEWrapper = None

__all__ = [
    "ActivationType",
    "CuteDslMoEWrapper",
    "convert_sf_to_mma_layout",
    "fp4_quantize",
    "grouped_gemm_nt_masked",
    "scaled_fp4_grouped_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
]
