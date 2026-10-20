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

"""Dependency-light contracts used by the V4.1 placed checkpoint path."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class ExpertPlacement(Protocol):
    num_logical_experts: int
    num_physical_experts: int
    num_layers: int
    ep_size: int
    ep_rank: int

    def local_slot_logical_experts(self, layer_id: int, ep_rank: int) -> list[int]: ...


def validate_v41_placement(
    *,
    num_logical_experts: int,
    layer_index: int,
    ep_size: int,
    ep_rank: int,
    tp_size: int,
    use_mega_moe: bool,
    expert_dtype: str | None,
    dispatch_algorithm: str,
    placement: ExpertPlacement,
    eager: bool,
    prefill_graph_disabled: bool,
    overlap_disabled: bool,
) -> None:
    """Refuse geometry/backend combinations lacking a complete placed path."""
    if not eager or not prefill_graph_disabled or not overlap_disabled:
        raise ValueError(
            "V4.1 expert placement requires eager execution, disabled prefill graphs and disabled overlap"
        )
    if not use_mega_moe or ep_size <= 1 or tp_size != 1:
        raise ValueError("V4.1 expert placement requires MegaMoE EP>1 and MoE TP=1")
    if expert_dtype != "fp4":
        raise ValueError("V4.1 expert placement requires serialized FP4 experts")
    if dispatch_algorithm not in ("static", "dynamic"):
        raise ValueError("V4.1 expert placement supports static or dynamic dispatch")
    if placement.num_logical_experts != num_logical_experts:
        raise ValueError("V4.1 placement logical expert count does not match the model")
    if placement.ep_size != ep_size or placement.ep_rank != ep_rank:
        raise ValueError("V4.1 placement EP ownership does not match the model")
    if not 0 <= layer_index < placement.num_layers:
        raise ValueError("V4.1 layer is outside the expert placement")
    if (
        placement.num_physical_experts < num_logical_experts
        or placement.num_physical_experts % ep_size
    ):
        raise ValueError(
            "V4.1 physical experts must cover logical experts and divide over EP"
        )


def local_logical_experts(
    placement: ExpertPlacement | None,
    *,
    layer_id: int,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> tuple[int, ...]:
    """Logical checkpoint IDs for every local slot, retaining replicas."""
    if placement is not None:
        return tuple(placement.local_slot_logical_experts(layer_id, ep_rank))
    count, remainder = divmod(num_experts, ep_size)
    if remainder:
        raise ValueError("unplaced experts must divide over EP")
    return tuple(range(ep_rank * count, (ep_rank + 1) * count))


def pad_expert_tensor(
    tensor: torch.Tensor,
    *,
    projection: str,
    is_scale: bool,
    intermediate_size: int,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """Preserve serialized FP4/E8M0 bytes and add exact zero intermediate lanes."""
    if projection not in ("w1", "w2", "w3"):
        raise ValueError("unknown expert projection")
    divisor = 32 if is_scale else 2
    extra = padded_intermediate_size - intermediate_size
    if extra < 0 or extra % 32:
        raise ValueError("expert padding must add whole 32-element quantization blocks")
    tensor = tensor.view(torch.uint8)
    if not extra:
        return tensor
    padding = (0, extra // divisor) if projection == "w2" else (0, 0, 0, extra)
    return F.pad(tensor, padding, value=127 if is_scale else 0)
