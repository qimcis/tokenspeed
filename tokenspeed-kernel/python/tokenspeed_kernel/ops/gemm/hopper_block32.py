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

"""Explicit-tactic SM90 FP8 GEMM with one scale per row and 32 K elements.

The caller chooses the tactic offline and supplies output/split scratch. There
is no autotuner, allocation, quantization, weight conversion, or retained tensor
cache here. Small-M tactics can transpose the MMA axes and divide K among CTAs;
large-M tactics reuse each tile across more rows. Every FP8 dot covers exactly
one scale group, including a masked final group. Split partials and reduction
stay FP32 until the single final output cast.

This is an independent implementation of the split-K/transposed-MMA design also
used by SGLang's Hopper block-FP8 path. Its mixed float/E8M0 scales, arbitrary
batch strides, explicit scratch, and bounded architecture contract are specific
to this implementation. The existing optional CuTe split-K adapter targets SM100
BF16, not this SM90 per32-scaled FP8 contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton


@dataclass(frozen=True)
class HopperBlock32Config:
    """Offline launch tactic; all seven fields must be explicitly supplied.

    ``block_m``/``block_n`` tile logical output rows/columns regardless of
    ``swap_ab``. ``split_k`` partitions contiguous groups of 32 elements;
    one means direct output. ``group_m`` orders output tiles for weight reuse.
    ``num_warps``/``num_stages`` control compilation. No value is a claim of a
    measured optimum; integration owns its validated selection policy.
    """

    block_m: int
    block_n: int
    split_k: int
    swap_ab: bool
    group_m: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in (
            "block_m",
            "block_n",
            "split_k",
            "group_m",
            "num_warps",
            "num_stages",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if type(self.swap_ab) is not bool:
            raise TypeError("swap_ab must be a boolean")
        if self.block_m not in (16, 32, 64, 128):
            raise ValueError("block_m must be 16, 32, 64, or 128")
        if self.block_n not in (16, 32, 64, 128, 256):
            raise ValueError("block_n must be 16, 32, 64, 128, or 256")
        if self.split_k not in (1, 2, 4, 8, 16, 32):
            raise ValueError("split_k must be a power of two from 1 through 32")
        if self.group_m < 1:
            raise ValueError("group_m must be positive")
        if self.num_warps not in (4, 8):
            raise ValueError("num_warps must be 4 or 8")
        if not 1 <= self.num_stages <= 5:
            raise ValueError("num_stages must be between 1 and 5")


def block32_workspace_shape(
    A: torch.Tensor, B: torch.Tensor, config: HopperBlock32Config
) -> tuple[int, int, int] | None:
    """Return required FP32 scratch shape, or None for a direct-output tactic.

    Args:
        A: Activation with shape ``[..., K]`` and at least two dimensions.
        B: Weight with shape ``[N, K]``.
        config: Explicit launch tactic.

    Returns:
        ``(split_k, flattened_M, N)`` if split-K is enabled; otherwise None.
        This metadata-only helper does not allocate or inspect tensor values.
    """
    if not isinstance(config, HopperBlock32Config):
        raise TypeError("config must be HopperBlock32Config")
    if A.ndim < 2 or B.ndim != 2 or A.shape[-1] != B.shape[-1]:
        raise ValueError("expected A[..., K] and B[N, K] with matching K")
    if A.shape[-1] < 1:
        raise ValueError("K must be positive")
    if config.split_k == 1:
        return None
    return config.split_k, math.prod(A.shape[:-1]), B.shape[0]


def _positive_nonoverlapping(tensor: torch.Tensor) -> bool:
    # Sufficient metadata-only condition supporting transpose, slices, padding,
    # and arbitrary leading batch strides. Reject overlapping/as_strided output.
    extent = 1
    for stride, size in sorted(
        (stride, size)
        for stride, size in zip(tensor.stride(), tensor.shape)
        if size > 1
    ):
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


@triton.jit
def _row_offsets(rows, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offsets = tl.full(rows.shape, 0, tl.int64)
    remaining = rows
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offsets += (remaining % SHAPE[axis]).to(tl.int64) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    return offsets


@triton.jit
def _scale_components(scale, UE8M0: tl.constexpr):
    if UE8M0:
        mantissa = tl.where(scale == 255, float("nan"), 1.0)
        exponent = scale.to(tl.int32) - 127
    else:
        # Decompose FP32 using bits, including subnormals without a potentially
        # flush-to-zero arithmetic operation on the original tiny float.
        bits = scale.to(tl.uint32, bitcast=True)
        encoded_exp = (bits >> 23) & 255
        fraction = bits & 0x7FFFFF
        mantissa = ((bits & 0x807FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
        tiny_mantissa = fraction.to(tl.float32) * 1.1920928955078125e-7
        tiny_mantissa = tl.where((bits >> 31) != 0, -tiny_mantissa, tiny_mantissa)
        mantissa = tl.where(encoded_exp == 0, tiny_mantissa, mantissa)
        exponent = tl.where(encoded_exp == 0, -126, encoded_exp.to(tl.int32) - 127)
        mantissa = tl.where(encoded_exp == 255, scale, mantissa)
        exponent = tl.where(encoded_exp == 255, 0, exponent)
    return mantissa, exponent


@triton.jit
def _scaled_group_dot(
    dot, a_scale, b_scale, A_UE8M0: tl.constexpr, B_UE8M0: tl.constexpr
):
    # Apply binary exponents to the dot, avoiding intermediate under/overflow
    # for reciprocal E8M0 scales. ldexp also handles exponent sums outside the
    # representable scale range without turning an exact zero dot into NaN.
    a_mantissa, a_exponent = _scale_components(a_scale, A_UE8M0)
    b_mantissa, b_exponent = _scale_components(b_scale, B_UE8M0)
    return libdevice.ldexp(dot * (a_mantissa * b_mantissa), a_exponent + b_exponent)


@triton.jit
def _hopper_block32_dot(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_SHAPE: tl.constexpr,
    A_ROW_STRIDES: tl.constexpr,
    AS_ROW_STRIDES: tl.constexpr,
    C_ROW_STRIDES: tl.constexpr,
    A_K_STRIDE: tl.constexpr,
    B_N_STRIDE: tl.constexpr,
    B_K_STRIDE: tl.constexpr,
    AS_K_STRIDE: tl.constexpr,
    BS_N_STRIDE: tl.constexpr,
    BS_K_STRIDE: tl.constexpr,
    C_N_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    group = tile // (GROUP_M * n_tiles)
    first_m = group * GROUP_M
    group_rows = tl.minimum(m_tiles - first_m, GROUP_M)
    m_tile = first_m + tile % group_rows
    n_tile = (tile % (GROUP_M * n_tiles)) // group_rows
    rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    # Invalid rows map to zero before batch-coordinate arithmetic, then remain
    # masked in every load/store. No modulo wrapping duplicates valid work.
    safe_rows = tl.where(rows < M, rows, 0)
    a_rows = _row_offsets(safe_rows, ROW_SHAPE, A_ROW_STRIDES)
    as_rows = _row_offsets(safe_rows, ROW_SHAPE, AS_ROW_STRIDES)
    k_offsets = tl.arange(0, 32)
    groups = tl.cdiv(K, 32)
    per_split = tl.cdiv(groups, SPLIT_K)
    first_group = split * per_split
    stop_group = tl.minimum(first_group + per_split, groups)
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kg in range(first_group, stop_group):
        ks = kg * 32 + k_offsets
        a = tl.load(
            A + a_rows[:, None] + ks[None, :] * A_K_STRIDE,
            (rows[:, None] < M) & (ks[None, :] < K),
            0.0,
        )
        b = tl.load(
            B + cols[None, :] * B_N_STRIDE + ks[:, None] * B_K_STRIDE,
            (cols[None, :] < N) & (ks[:, None] < K),
            0.0,
        )
        a_s = tl.load(As + as_rows + kg * AS_K_STRIDE, rows < M, 0)
        b_s = tl.load(Bs + cols * BS_N_STRIDE + kg * BS_K_STRIDE, cols < N, 0)
        # Every finite E4M3 value is exactly representable in FP16. Native
        # Hopper FP8 WGMMA has reduced accumulation precision even with an
        # FP32 result, so convert only the loaded tiles before FP32-accumulating
        # MMA. Keep the original FP8 storage and each 32-element scale group.
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        if SWAP_AB:
            dot = tl.dot(tl.trans(b), tl.trans(a), out_dtype=tl.float32)
            scaled = _scaled_group_dot(
                dot,
                a_s[None, :],
                b_s[:, None],
                As.dtype.element_ty == tl.uint8,
                Bs.dtype.element_ty == tl.uint8,
            )
        else:
            dot = tl.dot(a, b, out_dtype=tl.float32)
            scaled = _scaled_group_dot(
                dot,
                a_s[:, None],
                b_s[None, :],
                As.dtype.element_ty == tl.uint8,
                Bs.dtype.element_ty == tl.uint8,
            )
        accumulator += scaled
    if SWAP_AB:
        accumulator = tl.trans(accumulator)
    if SPLIT_K == 1:
        c_rows = _row_offsets(safe_rows, ROW_SHAPE, C_ROW_STRIDES)
        c_ptrs = C + c_rows[:, None] + cols[None, :] * C_N_STRIDE
    else:
        c_ptrs = C + split.to(tl.int64) * M * N + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, accumulator, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _hopper_block32_reduce(
    Parts,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_SHAPE: tl.constexpr,
    OUT_ROW_STRIDES: tl.constexpr,
    OUT_N_STRIDE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    elements = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    partials = tl.load(
        Parts + splits[:, None].to(tl.int64) * M * N + elements[None, :],
        elements[None, :] < M * N,
        0,
    )
    result = tl.sum(partials, axis=0)
    rows = tl.where(elements < M * N, elements // N, 0)
    out_rows = _row_offsets(rows, ROW_SHAPE, OUT_ROW_STRIDES)
    tl.store(Out + out_rows + (elements % N) * OUT_N_STRIDE, result, elements < M * N)


def gemm_fp8_block32(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out: torch.Tensor,
    config: HopperBlock32Config,
    workspace: torch.Tensor | None,
) -> torch.Tensor:
    """Compute per32-scaled FP8 ``A @ B.T`` on NVIDIA SM90.

    Args:
        A: FP8 E4M3FN activation ``[..., K]`` with at least two dimensions.
        B: FP8 E4M3FN weight ``[N, K]``. K need not be divisible by 32.
        As: Row/group scales ``[..., ceil(K/32)]``, uint8 E8M0 or FP32.
        Bs: Row/group scales ``[N, ceil(K/32)]``, uint8 E8M0 or FP32.
            Scale dtypes are independent; E8M0 codes0..254 denote
            ``2**(code-127)`` and reserved code255 propagates NaN. No packed
            int32 scale layouts or N-grouped scales are accepted.
        out: Caller-owned BF16, FP16, or FP32 ``[..., N]`` tensor. Positive
            nonoverlapping strides are supported for every tensor, including
            independently strided leading batch axes. Output must not share
            storage with inputs, scales, or scratch.
        config: Explicit offline-selected tactic, never a runtime autotune.
        workspace: None for split_k=1; otherwise caller-owned contiguous FP32
            tensor with the shape returned by ``block32_workspace_shape``.
            It must not share storage with another argument and must not be
            reused concurrently by different streams/graph executions.

    Returns:
        The supplied ``out`` tensor. Inputs/scales remain live at every launch
        and CUDA graph replay; this function retains no tensor references.
        Split-K changes FP32 summation order, so bitwise equivalence to a
        sequential-K kernel is not promised. As with the baseline, the sum of
        scaled group contributions must be representable in FP32.
    """
    scratch_shape = block32_workspace_shape(A, B, config)
    if not A.is_cuda or torch.version.hip is not None:
        raise ValueError("Hopper Block32 GEMM requires an NVIDIA CUDA device")
    if torch.cuda.get_device_capability(A.device) != (9, 0):
        raise ValueError("Hopper Block32 GEMM is restricted to SM90")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("A and B must use torch.float8_e4m3fn")
    if As.dtype not in (torch.uint8, torch.float32) or Bs.dtype not in (
        torch.uint8,
        torch.float32,
    ):
        raise TypeError("scales must use uint8 E8M0 or float32")
    if out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("out must use bfloat16, float16, or float32")
    rows = tuple(A.shape[:-1])
    n, k = B.shape
    scale_k = triton.cdiv(k, 32)
    if tuple(As.shape) != rows + (scale_k,) or tuple(Bs.shape) != (n, scale_k):
        raise ValueError("scales must provide one value per row and 32 K elements")
    if tuple(out.shape) != rows + (n,):
        raise ValueError("out shape must equal A.shape[:-1] + (B.shape[0],)")
    tensors = (A, B, As, Bs, out)
    if any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(not _positive_nonoverlapping(t) for t in tensors):
        raise ValueError("tensor strides must be positive and nonoverlapping")
    input_storages = {
        t.untyped_storage().data_ptr() for t in (A, B, As, Bs) if t.numel()
    }
    if out.numel() and out.untyped_storage().data_ptr() in input_storages:
        raise ValueError("out must not share storage with inputs or scales")
    if scratch_shape is None:
        if workspace is not None:
            raise ValueError("workspace must be None for split_k=1")
        destination = out
    else:
        if workspace is None or tuple(workspace.shape) != scratch_shape:
            raise ValueError(f"workspace must have shape {scratch_shape}")
        if (
            workspace.dtype != torch.float32
            or workspace.device != A.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace must be contiguous FP32 on the input device")
        if (
            workspace.numel()
            and workspace.untyped_storage().data_ptr()
            in input_storages | {out.untyped_storage().data_ptr()}
        ):
            raise ValueError("workspace must not share storage with another argument")
        destination = workspace
    m = math.prod(rows)
    if m == 0 or n == 0:
        return out
    grid = (
        triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),
        config.split_k,
    )
    _hopper_block32_dot[grid](
        A,
        B,
        As,
        Bs,
        destination,
        m,
        n,
        k,
        rows,
        tuple(A.stride()[:-1]),
        tuple(As.stride()[:-1]),
        tuple(out.stride()[:-1]),
        A.stride(-1),
        B.stride(0),
        B.stride(1),
        As.stride(-1),
        Bs.stride(0),
        Bs.stride(1),
        out.stride(-1),
        config.block_m,
        config.block_n,
        config.group_m,
        config.split_k,
        config.swap_ab,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    if config.split_k > 1:
        _hopper_block32_reduce[(triton.cdiv(m * n, 256),)](
            workspace,
            out,
            m,
            n,
            rows,
            tuple(out.stride()[:-1]),
            out.stride(-1),
            config.split_k,
            256,
            num_warps=4,
            num_stages=1,
        )
    return out
