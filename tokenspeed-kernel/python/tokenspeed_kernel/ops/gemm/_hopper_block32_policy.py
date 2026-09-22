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

from types import MappingProxyType

from tokenspeed_kernel.ops.gemm.hopper_block32 import HopperBlock32Config
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo

# Exact measured (M, N, K) regimes only. Candidates enter this table after
# independent correctness and paired complete-call measurements on H20.
_H20_BLOCK32_CONFIGS = MappingProxyType(
    {
        (1, 1792, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (2, 1792, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (4, 1792, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (8, 1792, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (1, 4096, 1280): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (2, 4096, 1280): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (4, 4096, 1280): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (8, 4096, 1280): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (1, 5120, 1024): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (2, 5120, 1024): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (4, 5120, 1024): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (8, 5120, 1024): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (1, 576, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (2, 576, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (4, 576, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
        (8, 576, 5120): HopperBlock32Config(16, 64, 8, True, 1, 4, 3),
    }
)


def get_hopper_block32_config(
    platform: PlatformInfo, m: int, n: int, k: int
) -> HopperBlock32Config | None:
    """Return an offline-validated H20 tactic, or retain the portable kernel.

    Args:
        platform: Detected execution device, including its SM90 capability.
        m: Flattened activation row count.
        n: Output column count.
        k: Reduction dimension.

    Returns:
        The immutable measured tactic, or None for an unmeasured device/shape.
    """
    if (
        not platform.is_nvidia
        or platform.arch_version != ArchVersion(9, 0)
        or platform.device_name != "NVIDIA H20"
    ):
        return None
    return _H20_BLOCK32_CONFIGS.get((m, n, k))
