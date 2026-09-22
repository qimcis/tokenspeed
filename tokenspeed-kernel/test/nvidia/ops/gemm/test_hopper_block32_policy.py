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

from types import MappingProxyType, SimpleNamespace

import pytest
from tokenspeed_kernel.ops.gemm import _hopper_block32_policy as policy
from tokenspeed_kernel.ops.gemm.hopper_block32 import HopperBlock32Config
from tokenspeed_kernel.platform import ArchVersion


@pytest.mark.parametrize(
    "is_nvidia,arch,name,shape,expected",
    [
        (True, (9, 0), "NVIDIA H20", (1, 1792, 5120), True),
        (True, (9, 0), "NVIDIA H20", (2, 1792, 5120), False),
        (True, (9, 0), "NVIDIA H20", (1, 1793, 5120), False),
        (True, (9, 0), "NVIDIA H20", (1, 1792, 5119), False),
        (True, (9, 0), "NVIDIA H100", (1, 1792, 5120), False),
        (True, (9, 0), "NVIDIA H20-3e", (1, 1792, 5120), False),
        (True, (9, 1), "NVIDIA H20", (1, 1792, 5120), False),
        (True, (10, 0), "NVIDIA H20", (1, 1792, 5120), False),
        (False, (9, 0), "NVIDIA H20", (1, 1792, 5120), False),
    ],
)
def test_exact_measured_device_and_shape_only(
    monkeypatch: pytest.MonkeyPatch,
    is_nvidia: bool,
    arch: tuple[int, int],
    name: str,
    shape: tuple[int, int, int],
    expected: bool,
) -> None:
    config = HopperBlock32Config(16, 64, 4, True, 1, 4, 3)
    monkeypatch.setattr(
        policy, "_H20_BLOCK32_CONFIGS", MappingProxyType({(1, 1792, 5120): config})
    )
    platform = SimpleNamespace(
        is_nvidia=is_nvidia, arch_version=ArchVersion(*arch), device_name=name
    )
    result = policy.get_hopper_block32_config(platform, *shape)
    assert (result is config) if expected else (result is None)
