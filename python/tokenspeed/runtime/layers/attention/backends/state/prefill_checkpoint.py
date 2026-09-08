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

"""Host-planned ragged routing for intermediate recurrent-state checkpoints."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class StateCheckpointSplitPlan:
    """Packed-token routing for one-forward intermediate state snapshots."""

    checkpoint_mask: torch.Tensor
    checkpoint_rows: torch.Tensor
    checkpoint_slots: torch.Tensor
    phase1_token_indices: torch.Tensor | None
    phase2_token_indices: torch.Tensor | None
    phase1_query_start_loc: torch.Tensor
    phase2_query_start_loc: torch.Tensor
    phase1_seq_lens_cpu: torch.Tensor
    phase2_seq_lens_cpu: torch.Tensor
    phase1_cu_seqlens_cpu: torch.Tensor
    phase2_cu_seqlens_cpu: torch.Tensor
    single_boundary: int | None
    total_tokens: int
    _routing_staging_cpu: torch.Tensor = field(repr=False)

    def select_tokens(
        self, tensor: torch.Tensor | None, phase: int, token_dim: int
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if self.single_boundary is not None:
            if phase == 1:
                return tensor.narrow(token_dim, 0, self.single_boundary)
            return tensor.narrow(
                token_dim,
                self.single_boundary,
                self.total_tokens - self.single_boundary,
            )
        indices = self.phase1_token_indices if phase == 1 else self.phase2_token_indices
        return tensor.index_select(token_dim, indices)

    def merge_tokens(
        self, phase1: torch.Tensor, phase2: torch.Tensor, token_dim: int
    ) -> torch.Tensor:
        output_shape = list(phase1.shape)
        output_shape[token_dim] = self.total_tokens
        output = phase1.new_empty(output_shape)
        if self.single_boundary is not None:
            output.narrow(token_dim, 0, self.single_boundary).copy_(phase1)
            output.narrow(
                token_dim,
                self.single_boundary,
                self.total_tokens - self.single_boundary,
            ).copy_(phase2)
            return output
        output.index_copy_(token_dim, self.phase1_token_indices, phase1)
        output.index_copy_(token_dim, self.phase2_token_indices, phase2)
        return output


def _host_prefix_sum(lengths: torch.Tensor) -> torch.Tensor:
    bounds = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    torch.cumsum(lengths.to(torch.int64), dim=0, out=bounds[1:])
    return bounds


def _upload_checkpoint_routing(
    tensors: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pack host routing into one allocation and enqueue one nonblocking copy."""
    offsets = {}
    total_bytes = 0
    for name, tensor in tensors.items():
        offsets[name] = total_bytes
        # Align every field for the int64 index views, including fields after
        # the bool mask and int32 sequence boundaries.
        total_bytes += (tensor.nbytes + 7) // 8 * 8
    staging = torch.empty(
        total_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    for name, tensor in tensors.items():
        staging.narrow(0, offsets[name], tensor.nbytes).view(tensor.dtype).copy_(tensor)
    # Fresh storage per plan: another batch must never overwrite an in-flight
    # H2D source. The plan retains it; the pinned allocator also tracks copy
    # completion if the plan is released before the GPU finishes the upload.
    uploaded = staging.to(device=device, non_blocking=True)
    return staging, {
        name: uploaded.narrow(0, offsets[name], tensor.nbytes).view(tensor.dtype)
        for name, tensor in tensors.items()
    }


def build_state_checkpoint_split_plan(
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens_cpu: torch.Tensor,
    state_checkpoint_lens_cpu: torch.Tensor,
    checkpoint_granularity: int,
    device: torch.device | str,
) -> StateCheckpointSplitPlan | None:
    """Build two packed ragged phases without reading any GPU metadata."""
    if checkpoint_granularity <= 0:
        raise ValueError("checkpoint granularity must be positive")
    for tensor in (
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        state_checkpoint_lens_cpu,
    ):
        if tensor.device.type != "cpu" or tensor.ndim != 1:
            raise ValueError(
                "state-checkpoint metadata must be one-dimensional CPU tensors"
            )
    prefixes = [int(x) for x in extend_prefix_lens_cpu]
    lengths = [int(x) for x in extend_seq_lens_cpu]
    checkpoints = [int(x) for x in state_checkpoint_lens_cpu]
    if not (len(prefixes) == len(lengths) == len(checkpoints)):
        raise ValueError("state-checkpoint metadata length mismatch")
    if any(value < 0 for value in checkpoints + prefixes) or any(
        value <= 0 for value in lengths
    ):
        raise ValueError("invalid state-checkpoint lengths")
    checkpoint_rows = [i for i, value in enumerate(checkpoints) if value > 0]
    if not checkpoint_rows:
        return None

    single_row = len(lengths) == 1
    phase1_lengths: list[int] = []
    phase2_lengths: list[int] = []
    phase1_indices: list[int] = []
    phase2_indices: list[int] = []
    offset = 0
    for row, (prefix, length, checkpoint) in enumerate(
        zip(prefixes, lengths, checkpoints)
    ):
        if length <= 0:
            raise ValueError(f"state checkpoint row {row} has no input tokens")
        if checkpoint > 0:
            endpoint = prefix + length
            if not prefix < checkpoint < endpoint:
                raise ValueError(
                    f"state checkpoint {checkpoint} must lie inside row {row}'s "
                    f"extend interval ({prefix}, {endpoint})"
                )
            if checkpoint % checkpoint_granularity != 0:
                raise ValueError(
                    f"state checkpoint {checkpoint} is not aligned to "
                    f"granularity {checkpoint_granularity}"
                )
            phase1_len = checkpoint - prefix
            phase2_len = endpoint - checkpoint
            phase2_lengths.append(phase2_len)
            if not single_row:
                phase2_indices.extend(range(offset + phase1_len, offset + length))
        else:
            phase1_len = length
        phase1_lengths.append(phase1_len)
        if not single_row:
            phase1_indices.extend(range(offset, offset + phase1_len))
        offset += length

    phase1_lens_cpu = torch.tensor(phase1_lengths, dtype=torch.int32)
    phase2_lens_cpu = torch.tensor(phase2_lengths, dtype=torch.int32)
    phase1_cu_cpu = _host_prefix_sum(phase1_lens_cpu)
    phase2_cu_cpu = _host_prefix_sum(phase2_lens_cpu)
    routing_cpu = {
        "checkpoint_mask": torch.tensor([value > 0 for value in checkpoints]),
        "checkpoint_rows": torch.tensor(checkpoint_rows, dtype=torch.int64),
        "checkpoint_slots": torch.tensor(
            [max((value - 1) // checkpoint_granularity, 0) for value in checkpoints],
            dtype=torch.int64,
        ),
        "phase1_query_start_loc": phase1_cu_cpu.to(dtype=torch.int32),
        "phase2_query_start_loc": phase2_cu_cpu.to(dtype=torch.int32),
    }
    if not single_row:
        routing_cpu["phase1_token_indices"] = torch.tensor(
            phase1_indices, dtype=torch.int64
        )
        routing_cpu["phase2_token_indices"] = torch.tensor(
            phase2_indices, dtype=torch.int64
        )
    staging, routing = _upload_checkpoint_routing(routing_cpu, torch.device(device))
    return StateCheckpointSplitPlan(
        checkpoint_mask=routing["checkpoint_mask"],
        checkpoint_rows=routing["checkpoint_rows"],
        checkpoint_slots=routing["checkpoint_slots"],
        phase1_token_indices=routing.get("phase1_token_indices"),
        phase2_token_indices=routing.get("phase2_token_indices"),
        phase1_query_start_loc=routing["phase1_query_start_loc"],
        phase2_query_start_loc=routing["phase2_query_start_loc"],
        phase1_seq_lens_cpu=phase1_lens_cpu,
        phase2_seq_lens_cpu=phase2_lens_cpu,
        phase1_cu_seqlens_cpu=phase1_cu_cpu,
        phase2_cu_seqlens_cpu=phase2_cu_cpu,
        single_boundary=phase1_lengths[0] if single_row else None,
        total_tokens=offset,
        _routing_staging_cpu=staging,
    )
