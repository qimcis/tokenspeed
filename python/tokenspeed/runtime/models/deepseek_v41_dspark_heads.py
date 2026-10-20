# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Tensor-parallel DSpark Markov head and greedy block sampler.

The Markov head keeps the checkpoint's BF16 dtypes: its bigram table is
replicated on every rank so a lookup is a plain gather, and its rank-``R``
projection is sharded over the vocabulary exactly like the LM head, so the
bias of a shard column lands on the rank that owns that column's base logit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from tokenspeed_kernel.ops.sampling.triton import (
    dspark_block_candidate_tiles,
    dspark_block_greedy_resolve,
    dspark_block_greedy_step,
)
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_gather_single

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )


class DSparkVanillaMarkov(nn.Module):
    """Low-rank token-bigram correction over a vocab-sharded output."""

    def __init__(
        self, embedding: VocabParallelEmbedding, projection: ParallelLMHead
    ) -> None:
        super().__init__()
        if embedding.tp_size != 1:
            raise ValueError("DSpark Markov bigram table must be replicated")
        if embedding.embedding_dim != projection.embedding_dim:
            raise ValueError("DSpark Markov table and projection ranks differ")
        self.embedding = embedding
        self.projection = projection


DSparkCandidateCommunication = Literal["tiles", "compact"]


def _compact_candidates_kernel():
    # Existing kernel wheels remain sufficient for the default tiles path.
    # Workspace setup calls this before graph warmup/capture when opted in.
    try:
        from tokenspeed_kernel.ops.sampling.triton import (
            dspark_block_compact_candidates,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Compact DSpark candidates require a tokenspeed-kernel build with "
            "dspark_block_compact_candidates; install the matching kernel build "
            "or select --dspark-candidate-communication tiles"
        ) from exc
    return dspark_block_compact_candidates


def dspark_greedy_workspace(
    tp_size: int,
    max_rows: int,
    local_vocab: int,
    device: torch.device,
    *,
    candidate_communication: DSparkCandidateCommunication,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Allocate stable packed-candidate buffers before graph capture.

    Args:
        tp_size: Ranks the LM head is sharded over.
        max_rows: Most rows one call will sample.
        local_vocab: Columns of this rank's head shard.
        device: Device of the draft.
        candidate_communication: Gather every vocabulary tile (``tiles``), or
            reduce each rank's tiles before gathering (``compact``).

    Returns:
        ``(candidates, partials, compact_partials)``. Candidates are an int64
        double buffer ``[2, tp_size, max_rows, communicated_tiles]``. Steps
        alternate slots so scoring never overwrites the previous candidates
        it is still resolving. communicated_tiles is one in compact mode.
        Partials hold local tile scores; only single-rank tiles mode writes
        directly into candidates.
        Compact partials hold ``[max_rows, 1]`` local maxima for a multi-rank
        compact gather. Single-rank compact mode reduces directly into its
        current candidate slot. No allocation is needed during sampling.
    """
    if candidate_communication not in ("tiles", "compact"):
        raise ValueError("DSpark candidate communication must be tiles or compact")
    if tp_size < 1 or max_rows < 0 or local_vocab < 1:
        raise ValueError("Invalid DSpark greedy workspace dimensions")
    tiles = dspark_block_candidate_tiles(local_vocab)
    compact = candidate_communication == "compact"
    if compact:
        _compact_candidates_kernel()
    candidates = torch.empty(
        (2, tp_size, max_rows, 1 if compact else tiles),
        dtype=torch.int64,
        device=device,
    )
    partials = (
        torch.empty((max_rows, tiles), dtype=torch.int64, device=device)
        if compact or tp_size > 1
        else None
    )
    compact_partials = (
        torch.empty((max_rows, 1), dtype=torch.int64, device=device)
        if compact and tp_size > 1
        else None
    )
    return candidates, partials, compact_partials


def sample_dspark_block_greedy(
    local_base_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    markov_head: DSparkVanillaMarkov,
    lm_head: ParallelLMHead,
    tp_group,
    candidates: torch.Tensor,
    partials: torch.Tensor | None,
    output: torch.Tensor,
    *,
    candidate_communication: DSparkCandidateCommunication,
    compact_partials: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the trained Markov correction and greedily sample a fixed block.

    Args:
        local_base_logits: ``[rows, block, local_vocab]`` FP32 logits of this
            rank's head shard for every block position.
        bonus_token_ids: ``[rows]`` token each block continues from.
        markov_head: Bigram table and projection shard.
        lm_head: Head whose shard geometry the logits follow.
        tp_group: Tensor-parallel group of the head.
        candidates: Gathered-candidate double buffer from
            ``dspark_greedy_workspace``.
        partials: This rank's tile buffer from ``dspark_greedy_workspace``.
        candidate_communication: Explicit tile or compact exchange selection.
        compact_partials: Compact local maxima from ``dspark_greedy_workspace``.
        output: ``[rows, block]`` int32 destination for the block tokens.

    Returns:
        ``output``, filled in place.
    """
    rows, block, local_vocab = local_base_logits.shape
    shard = lm_head.shard_indices
    if shard.num_added_elements != 0:
        raise ValueError("DSpark greedy sampling needs a head without added vocabulary")
    projection = markov_head.projection
    if (
        projection.shard_indices.org_vocab_start_index != shard.org_vocab_start_index
        or projection.shard_indices.num_org_elements != shard.num_org_elements
    ):
        raise ValueError("DSpark Markov projection is not sharded like the LM head")
    if candidate_communication not in ("tiles", "compact"):
        raise ValueError("DSpark candidate communication must be tiles or compact")
    compact = candidate_communication == "compact"
    compact_kernel = _compact_candidates_kernel() if compact else None
    if candidates.ndim != 4 or candidates.shape[0] != 2:
        raise ValueError(
            "DSpark greedy candidates must be a [2, tp, rows, tiles] buffer"
        )
    _, tp_size, capacity, tiles = candidates.shape
    score_tiles = dspark_block_candidate_tiles(local_vocab)
    needs_partials = compact or tp_size > 1
    if (
        tp_size < 1
        or capacity < rows
        or tiles != (1 if compact else score_tiles)
        or (partials is not None) != needs_partials
        or (compact_partials is not None) != (compact and tp_size > 1)
    ):
        raise ValueError("DSpark greedy workspace does not cover the batch")
    if partials is not None and partials.shape != (capacity, score_tiles):
        raise ValueError("DSpark greedy workspace does not cover the batch")
    if compact_partials is not None and compact_partials.shape != (capacity, 1):
        raise ValueError("DSpark greedy compact workspace does not cover the batch")
    for buffer in (candidates, partials, compact_partials):
        if buffer is not None and (
            buffer.dtype != torch.int64
            or buffer.device != local_base_logits.device
            or not buffer.is_contiguous()
        ):
            raise ValueError(
                "DSpark workspace needs contiguous int64 buffers on the logits device"
            )
    if rows == 0 or block == 0:
        return output
    slots = [
        candidates[slot].view(-1)[: tp_size * rows * tiles].view(tp_size, rows, tiles)
        for slot in range(2)
    ]
    for step in range(block):
        previous, current = slots[(step - 1) % 2], slots[step % 2]
        scored = current[0] if partials is None else partials[:rows]
        dspark_block_greedy_step(
            local_base_logits,
            step,
            bonus_token_ids,
            previous,
            markov_head.embedding.weight,
            projection.weight,
            shard.org_vocab_start_index,
            shard.num_org_elements,
            scored,
            output,
        )
        communicated = scored
        if compact:
            communicated = (
                current[0] if compact_partials is None else compact_partials[:rows]
            )
            compact_kernel(scored, communicated)
        if tp_size > 1:
            all_gather_single(current.view(-1), communicated.view(-1), tp_group)
    dspark_block_greedy_resolve(slots[(block - 1) % 2], output, block - 1)
    return output
