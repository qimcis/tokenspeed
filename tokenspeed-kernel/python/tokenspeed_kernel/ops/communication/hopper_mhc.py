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

"""Prepared single-lane TP8 SM90 allreduce adapter for V4.1 HC4 epilogues.

This module is deliberately not registered as a general allreduce backend. All
ranks must launch the same row geometry and collective order on one ordered
lane. Prepare collectively before capture, under an external startup watchdog;
PyTorch symmetric-memory rendezvous uses the process group's timeout. A device
handshake timeout traps the owned CUDA context instead of returning partial
results. A workspace used by a captured graph must outlive that graph.
"""

from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.triton import _alloc_symm, _peer_ptrs_dev
from tokenspeed_kernel.ops.residual.hopper_v41 import _v41_hc_epilogue


@dataclass(frozen=True)
class V41AllReduceWorkspace:
    """Stable peer buffers owned by one serialized forward executor.

    ``round_mode`` is explicitly ``bf16_step`` (rank-ordered BF16 additions) or
    ``fp32`` (rank-ordered FP32 additions followed by one BF16 cast). No mode is
    claimed bitwise equivalent to NCCL or a different allreduce topology.
    """

    group: dist.ProcessGroup
    device: torch.device
    rank: int
    max_rows: int
    round_mode: str
    timeout_ns: int
    data: torch.Tensor
    signals: torch.Tensor
    data_handle: Any
    signals_handle: Any
    data_ptrs: torch.Tensor
    signal_ptrs: torch.Tensor

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate host-visible geometry.

        The forward executor owns ordering across eager calls, captures and
        replays. Different streams require external event ordering; overlapping
        use is unsupported. Python cannot inspect later CUDA graph replays.
        """
        if (
            x.device != self.device
            or x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[1] != 5120
            or x.shape[0] not in (1, 2, 4, 8)
            or x.shape[0] > self.max_rows
            or not x.is_contiguous()
        ):
            raise ValueError("require contiguous BF16 [1/2/4/8,5120] in this workspace")


def prepare_v41_allreduce(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    round_mode: str,
    timeout_ns: int,
    collective_timeout_s: float,
) -> V41AllReduceWorkspace:
    """Collectively prepare TP8 peer storage before graph capture.

    Args:
        group: The exact eight-rank, single-host TP process group. All ranks
            must already have the correct current CUDA device and share the
            same CUDA_VISIBLE_DEVICES ordinal namespace.
        device: A valid explicit local CUDA device; all ranks use distinct GPUs.
            Invalid local devices can fail before collective metadata agreement.
        max_rows: One of 1, 2, 4, 8, agreed across ranks, never grown in forward.
        round_mode: Explicit ``bf16_step`` or ``fp32`` reduction arithmetic.
        timeout_ns: Per-handshake device deadline in [1 ms, 10 s].
        collective_timeout_s: Bounded wait for the final initialization barrier.
            The group/rendezvous timeout and external startup watchdog must
            separately bound metadata agreement and peer-memory rendezvous.

    Returns:
        Owning workspace including peer handles and tables. Launch/capture and
        graph replay must remain ordered by one owning forward executor. Stream
        changes require external event ordering; no forward synchronization or
        capture-specific state is inserted here.
    """
    if not isinstance(group, dist.ProcessGroup) or group.size() != 8:
        raise ValueError("V4.1 fused allreduce requires an eight-rank process group")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare V4.1 allreduce collectively before capture")
    device = torch.device(device)
    valid_device = device.type == "cuda" and device.index is not None
    capability = torch.cuda.get_device_capability(device) if valid_device else None
    descriptor = (
        socket.gethostname(),
        device.index,
        capability,
        max_rows,
        round_mode,
        timeout_ns,
        collective_timeout_s,
    )
    descriptors: list[Any] = [None] * 8
    dist.all_gather_object(descriptors, descriptor, group=group)
    agreed = all(item[2:] == descriptor[2:] for item in descriptors)
    valid = (
        agreed
        and len({item[0] for item in descriptors}) == 1
        and len({item[1] for item in descriptors}) == 8
        and all(item[1] is not None for item in descriptors)
        and capability == (9, 0)
        and type(max_rows) is int
        and max_rows in (1, 2, 4, 8)
        and round_mode in ("bf16_step", "fp32")
        and type(timeout_ns) is int
        and 1_000_000 <= timeout_ns <= 10_000_000_000
        and math.isfinite(collective_timeout_s)
        and 0 < collective_timeout_s <= 120
    )
    if not valid:
        raise ValueError("incompatible TP8 SM90 allreduce preparation across ranks")
    rank = dist.get_rank(group)
    data_shape = (max_rows, 5120)
    signal_shape = (2, max_rows, 8)
    data, data_handle = _alloc_symm(data_shape, torch.bfloat16, device, group)
    signals, signals_handle = _alloc_symm(signal_shape, torch.uint32, device, group)
    signals.zero_()
    data_ptrs = _peer_ptrs_dev(data_handle, data_shape, torch.bfloat16, 8, device)
    signal_ptrs = _peer_ptrs_dev(signals_handle, signal_shape, torch.uint32, 8, device)
    torch.cuda.current_stream(device).synchronize()
    work = dist.barrier(group=group, async_op=True)
    if not work.wait(timeout=timedelta(seconds=collective_timeout_s)):
        raise RuntimeError("V4.1 allreduce initialization barrier timed out")
    return V41AllReduceWorkspace(
        group,
        device,
        rank,
        max_rows,
        round_mode,
        timeout_ns,
        data,
        signals,
        data_handle,
        signals_handle,
        data_ptrs,
        signal_ptrs,
    )


@triton.jit
def _v41_peer_barrier(
    SIGNAL_PTRS,
    row,
    PHASE: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    RANK: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    # Give every thread a distinct fence operand (8 warps x 32 lanes), then use
    # Triton CTA barriers outside scalar inline asm. Scalar asm may execute only
    # on its canonical lane; putting bar.sync inside it would be unsafe.
    # Distinct phase storage separates publication from read completion. CAS
    # consumes each source's signal before that source may publish again.
    tl.inline_asm_elementwise(
        "{ fence.sc.sys; mov.u32 $0, $1; }",
        "=r,r,~{memory}",
        [tl.arange(0, 256)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    offset = (PHASE * MAX_ROWS + row) * 8 * 4
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred p_thread, p_done, p_retry, p_timeout;
            .reg .u32 tid, peer, old;
            .reg .u64 start, now, elapsed, table, slot, base, peer_offset;
            mov.u32 $0, 0;
            mov.u32 tid, %tid.x;
            setp.ne.u32 p_thread, tid, 0;
            @p_thread bra barrier_done;
            mov.u64 start, %globaltimer;
            mov.u32 peer, 0;
        send_peer:
            mul.wide.u32 peer_offset, peer, 8;
            add.u64 table, $1, peer_offset;
            ld.global.u64 base, [table];
            add.u64 slot, base, $2;
            add.u64 slot, slot, $3;
        send_retry:
            atom.global.release.sys.cas.b32 old, [slot], 0, 1;
            setp.eq.u32 p_retry, old, 0;
            @p_retry bra sent;
            mov.u64 now, %globaltimer;
            sub.u64 elapsed, now, start;
            setp.ge.u64 p_timeout, elapsed, $4;
            @p_timeout trap;
            bra send_retry;
        sent:
            add.u32 peer, peer, 1;
            setp.lt.u32 p_done, peer, 8;
            @p_done bra send_peer;
            add.u64 table, $1, $5;
            ld.global.u64 base, [table];
            add.u64 base, base, $2;
            mov.u32 peer, 0;
        wait_peer:
            mul.wide.u32 peer_offset, peer, 4;
            add.u64 slot, base, peer_offset;
        wait_retry:
            atom.global.acquire.sys.cas.b32 old, [slot], 1, 0;
            setp.eq.u32 p_retry, old, 1;
            @p_retry bra received;
            mov.u64 now, %globaltimer;
            sub.u64 elapsed, now, start;
            setp.ge.u64 p_timeout, elapsed, $4;
            @p_timeout trap;
            bra wait_retry;
        received:
            add.u32 peer, peer, 1;
            setp.lt.u32 p_done, peer, 8;
            @p_done bra wait_peer;
        barrier_done:
        }
        """,
        "=r,l,l,l,l,l,~{memory}",
        [
            SIGNAL_PTRS,
            offset.to(tl.uint64),
            tl.full((), RANK * 4, tl.uint64),
            tl.full((), TIMEOUT_NS, tl.uint64),
            tl.full((), RANK * 8, tl.uint64),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def v41_allreduce_load(
    X,
    DATA_PTRS,
    SIGNAL_PTRS,
    row,
    d,
    H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    RANK: tl.constexpr,
    ROUND_MODE: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    """Publish one row, sum peers, return FP32 values after BF16 rounding.

    One CTA owns this row throughout publication and consumption. Call exactly
    once per CTA, with all CTA threads active, before the HC4 epilogue. The
    kernel grid must contain only the 1/2/4/8 rows of this collective.
    """
    tl.static_assert(H == 5120)
    tl.static_assert(ROUND_MODE == "bf16_step" or ROUND_MODE == "fp32")
    mask = d < H
    local = tl.load(DATA_PTRS + RANK).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(X + row * H + d, mask, other=0)
    tl.store(local + row * H + d, value, mask)
    tl.debug_barrier()
    _v41_peer_barrier(SIGNAL_PTRS, row, 0, MAX_ROWS, RANK, TIMEOUT_NS)
    first = tl.load(DATA_PTRS).to(tl.pointer_type(tl.bfloat16))
    reduced = tl.load(first + row * H + d, mask, other=0, cache_modifier=".cv").to(
        tl.float32
    )
    for peer in tl.static_range(1, 8):
        pointer = tl.load(DATA_PTRS + peer).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(pointer + row * H + d, mask, other=0, cache_modifier=".cv").to(
            tl.float32
        )
        reduced += value
        if ROUND_MODE == "bf16_step":
            reduced = reduced.to(tl.bfloat16).to(tl.float32)
    reduced = reduced.to(tl.bfloat16).to(tl.float32)
    tl.debug_barrier()
    _v41_peer_barrier(SIGNAL_PTRS, row, 1, MAX_ROWS, RANK, TIMEOUT_NS)
    return reduced


@triton.jit
def _v41_allreduce_kernel(
    X,
    OUT,
    DATA_PTRS,
    SIGNAL_PTRS,
    MAX_ROWS: tl.constexpr,
    RANK: tl.constexpr,
    ROUND_MODE: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 8192)
    value = v41_allreduce_load(
        X, DATA_PTRS, SIGNAL_PTRS, row, d, 5120, MAX_ROWS, RANK, ROUND_MODE, TIMEOUT_NS
    )
    tl.store(OUT + row * 5120 + d, value, d < 5120)


@triton.jit
def _v41_allreduce_epilogue_kernel(
    X,
    R,
    POST,
    COMB,
    PRE,
    W,
    HC,
    NORMALIZED,
    Q,
    S,
    DATA_PTRS,
    SIGNAL_PTRS,
    MAX_ROWS: tl.constexpr,
    RANK: tl.constexpr,
    ROUND_MODE: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
    EPS: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 8192)
    value = v41_allreduce_load(
        X, DATA_PTRS, SIGNAL_PTRS, row, d, 5120, MAX_ROWS, RANK, ROUND_MODE, TIMEOUT_NS
    )
    _v41_hc_epilogue(
        value,
        row,
        d,
        d < 5120,
        R,
        POST,
        COMB,
        PRE,
        W,
        HC,
        NORMALIZED,
        Q,
        S,
        5120,
        EPS,
        QUANTIZE,
        8192,
    )


def v41_allreduce(workspace: V41AllReduceWorkspace, x: torch.Tensor) -> torch.Tensor:
    """Reduce a BF16 [rows,5120] input using the prepared TP8 lane.

    Returns a fresh BF16 tensor. This entry point isolates protocol validation
    from the fused epilogue; it has the same workspace/order/timeout contract.
    """
    workspace.validate_input(x)
    out = torch.empty_like(x)
    _v41_allreduce_kernel[(x.shape[0],)](
        x,
        out,
        workspace.data_ptrs,
        workspace.signal_ptrs,
        workspace.max_rows,
        workspace.rank,
        workspace.round_mode,
        workspace.timeout_ns,
        num_warps=8,
    )
    return out


def v41_allreduce_post_pre_norm_quant(
    workspace: V41AllReduceWorkspace,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    quantize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Allreduce then HC4 post/collapse/norm/optional group32 quant in one CTA.

    Args:
        workspace: Prepared TP8 SM90 single-lane storage.
        x: Contiguous rank-local BF16 [rows,5120], rows in 1/2/4/8.
        residual: Contiguous replicated BF16 [rows,4,5120] previous HC state.
        post: Contiguous replicated FP32 [rows,4] post coefficients.
        comb: Contiguous replicated FP32 [rows,inputHC,outputHC] coefficients.
        pre: Contiguous replicated FP32 [rows,4] previous-sublayer coefficients.
        norm_weight: Contiguous replicated BF16/FP32 [5120] norm weights.
        eps: Explicit finite positive RMSNorm epsilon.
        quantize: Explicit bool enabling FP8 E4M3 codes/row-major UE8M0 scales.

    Returns:
        Fresh HC state, normalized BF16, optional FP8 codes and uint8 scales.
        Only prepared scratch is mutated; input tensors are not modified.
    """
    workspace.validate_input(x)
    rows = x.shape[0]
    if residual.shape != (rows, 4, 5120) or residual.dtype != torch.bfloat16:
        raise ValueError("residual must be BF16 [rows,4,5120]")
    for value, shape in ((post, (rows, 4)), (comb, (rows, 4, 4)), (pre, (rows, 4))):
        if value.shape != shape or value.dtype != torch.float32:
            raise ValueError("HC coefficients must be FP32 with matching HC4 shapes")
    if norm_weight.shape != (5120,) or norm_weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("invalid RMSNorm weight")
    if not math.isfinite(eps) or eps <= 0 or type(quantize) is not bool:
        raise ValueError("require finite positive epsilon and explicit bool quantize")
    if any(
        t.device != x.device or not t.is_contiguous()
        for t in (residual, post, comb, pre, norm_weight)
    ):
        raise ValueError("all inputs must be contiguous on the workspace device")
    hc = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn) if quantize else None
    scales = (
        torch.empty((rows, 160), dtype=torch.uint8, device=x.device)
        if quantize
        else None
    )
    _v41_allreduce_epilogue_kernel[(rows,)](
        x,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        hc,
        normalized,
        codes,
        scales,
        workspace.data_ptrs,
        workspace.signal_ptrs,
        workspace.max_rows,
        workspace.rank,
        workspace.round_mode,
        workspace.timeout_ns,
        eps,
        quantize,
        num_warps=8,
    )
    return hc, normalized, codes, scales
