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

"""CPU-testable in-place MegaMoE packing and processed storage contract."""

from dataclasses import dataclass

import torch

_MXFP4_BLOCK_SIZE = 32


@dataclass(frozen=True)
class _DeepGemmMegaMoEState:
    l1_weights: tuple[torch.Tensor, torch.Tensor]
    l2_weights: tuple[torch.Tensor, torch.Tensor]
    device: torch.device

    @property
    def routed_expert_weights(self) -> tuple[torch.Tensor, ...]:
        """Contiguous slot views of the exact bytes consumed by MegaMoE.

        Kernel scales are column-major; transpose only the view, never the
        bytes. The generic updater then moves every packed scale with its
        expert while preserving all captured kernel pointers and strides.
        """
        return (
            self.l1_weights[0],
            self.l1_weights[1].transpose(-1, -2),
            self.l2_weights[0],
            self.l2_weights[1].transpose(-1, -2),
        )


def _expected_shapes(
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        (num_local_experts, 2 * intermediate_size, hidden_size // 2),
        (
            num_local_experts,
            2 * intermediate_size,
            hidden_size // _MXFP4_BLOCK_SIZE,
        ),
        (num_local_experts, hidden_size, intermediate_size // 2),
        (
            num_local_experts,
            hidden_size,
            intermediate_size // _MXFP4_BLOCK_SIZE,
        ),
    )


def _interleave_gate_up_(weight: torch.Tensor, granularity: int) -> torch.Tensor:
    """Interleave gate/up blocks in place with one-expert scratch space."""
    squeeze_group_dim = weight.ndim == 2
    if squeeze_group_dim:
        weight = weight.unsqueeze(0)
    groups, rows, *rest = weight.shape
    half = rows // 2
    blocks = half // granularity
    for group in range(groups):
        source = weight[group].clone().view(2, blocks, granularity, *rest)
        weight[group].view(blocks, 2, granularity, *rest).copy_(source.transpose(0, 1))
    return weight.squeeze(0) if squeeze_group_dim else weight


def _pack_ue8m0_scale_(scale: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 bytes into DeepGEMM's layout with one-expert scratch."""
    squeeze_group_dim = scale.ndim == 2
    if squeeze_group_dim:
        scale = scale.unsqueeze(0)
    groups, rows, columns = scale.shape
    if columns % 4:
        raise ValueError("MegaMoE UE8M0 scale columns must be divisible by four")
    packed_columns = columns // 4
    for group in range(groups):
        source = scale[group].clone().view(rows, packed_columns, 4)
        scale[group].view(packed_columns, rows, 4).copy_(source.permute(1, 0, 2))
    packed = scale.view(torch.int32)
    output = torch.as_strided(
        packed,
        size=(groups, rows, packed_columns),
        stride=(rows * packed_columns, 1, rows),
    )
    return output.squeeze(0) if squeeze_group_dim else output


def _reorder_scale_rows_(
    scale: torch.Tensor, interleave_gate_up: bool, granularity: int
) -> torch.Tensor:
    """Apply MegaMoE gate/up and UTCCP row permutations in place."""
    grouped = scale if scale.ndim == 3 else scale.unsqueeze(0)
    groups, rows, packed_columns = grouped.shape
    if rows % 128:
        raise ValueError("MegaMoE scale rows must be divisible by 128")
    for group in range(groups):
        source = grouped[group].contiguous()
        if interleave_gate_up:
            half = rows // 2
            blocks = half // granularity
            source = source.view(2, blocks, granularity, packed_columns)
            source = source.transpose(0, 1).reshape(rows, packed_columns)
        source = source.view(-1, 4, 32, packed_columns)
        source = source.transpose(1, 2).reshape(rows, packed_columns)
        grouped[group].copy_(source)
    return scale
