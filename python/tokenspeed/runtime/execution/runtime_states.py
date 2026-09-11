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
"""Runtime state tensors shared by the model executor."""

import torch


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index."""

    def __init__(
        self,
        req_pool_size: int,
        vocab_size: int,
        output_length: int,
        device: str = "cuda",
    ):
        self.device = device
        self.vocab_size = vocab_size

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        self.remote_spec_candidate_ready = torch.zeros(
            req_pool_size + 1, dtype=torch.bool, device=device
        )

    def update_valid_cache_length(
        self, req_pool_indices: torch.Tensor, increment_lengths: torch.Tensor
    ) -> None:
        self.valid_cache_lengths.index_add_(0, req_pool_indices, increment_lengths)

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        self.remote_spec_candidate_ready[extend_request_pool_indices] = False

    def write_remote_spec_candidate_ids(
        self, req_pool_idx: int, candidate_ids: list[int]
    ) -> None:
        """Install one full candidate prefix, including its confirmed anchor.

        Args:
            req_pool_idx: Live request-pool slot, excluding the null slot.
            candidate_ids: Anchor followed by proposals, within map capacity.
        """
        width = self.future_input_map.shape[1]
        if not 1 <= len(candidate_ids) <= width:
            raise RuntimeError(
                f"remote spec candidate width mismatch: got {len(candidate_ids)}, maximum {width}"
            )
        if not 0 < req_pool_idx < self.future_input_map.shape[0]:
            raise ValueError(
                "remote candidate request slot is outside the request pool"
            )
        if any(token < 0 or token >= self.vocab_size for token in candidate_ids):
            raise ValueError("remote candidate token is outside the target vocabulary")
        width = len(candidate_ids)
        ids = torch.tensor(
            candidate_ids,
            dtype=torch.int32,
            device="cpu",
            pin_memory=torch.device(self.device).type == "cuda",
        ).to(self.device, non_blocking=True)
        self.future_input_map[req_pool_idx, :width] = ids
        self.remote_spec_candidate_ready[req_pool_idx] = True

    def update_next_anchors(
        self,
        req_pool_indices: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
    ) -> None:
        """Retain the final accepted output, for packed prefill/verify results.

        Prefill outputs occupy one row each; decode outputs occupy a uniform
        width inferred from the actual output, independently of map capacity.
        """
        bs = req_pool_indices.numel()
        num_decodes = bs - num_extends
        if not bs:
            return
        if output_tokens.numel() < num_extends:
            raise ValueError("output does not contain every prefill request")
        if num_decodes:
            decode_tokens = output_tokens.numel() - num_extends
            if decode_tokens % num_decodes:
                raise ValueError("decode outputs are not a homogeneous window")
            width = decode_tokens // num_decodes
            if not 1 <= width <= self.future_input_map.shape[1]:
                raise ValueError("decode output width exceeds candidate capacity")
            offsets = (
                num_extends
                + torch.arange(num_decodes, device=output_tokens.device) * width
            )
            indices = offsets + accept_lengths[num_extends:].to(torch.int64) - 1
            anchors = output_tokens.reshape(-1).index_select(0, indices)
            self.future_input_map[req_pool_indices[num_extends:], 0] = anchors.to(
                torch.int32
            )
        if num_extends:
            self.future_input_map[req_pool_indices[:num_extends], 0] = output_tokens[
                :num_extends
            ].to(torch.int32)
        self.remote_spec_candidate_ready[req_pool_indices] = False

    def gather_candidate_ids(
        self, req_pool_indices: torch.Tensor, decode_width: int
    ) -> torch.Tensor:
        """Gather the active columns using the map's real per-request stride."""
        if not 1 <= decode_width <= self.future_input_map.shape[1]:
            raise ValueError("active decode width exceeds candidate capacity")
        return self.future_input_map[req_pool_indices, :decode_width].reshape(-1)
