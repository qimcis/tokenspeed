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

"""HC4 fusion: independent rounded math, original chain, and exact quantization.

The numerical gate is fixed before GPU validation: relative L2 <= 0.006 for
every token's HC state and normalized output. Quantization of the candidate's
normalized BF16 output must match the original quantizer byte for byte.
Kernel imports are fixture-local so CPU reference checks need no CUDA runtime.
"""

from __future__ import annotations

import pytest
import torch

RELATIVE_L2_LIMIT = 0.006
EPS = 1.0e-6


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM90 CUDA GPU")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_group32_ue8m0,
    )
    from tokenspeed_kernel.ops.residual.hopper_v41 import v41_post_pre_norm_quant
    from tokenspeed_kernel.ops.residual.triton import (
        mhc_pre_layer_norm_hc4,
        triton_mhc_post,
    )

    return (
        v41_post_pre_norm_quant,
        triton_mhc_post,
        mhc_pre_layer_norm_hc4,
        triton_quantize_fp8_group32_ue8m0,
    )


def _inputs(rows, hidden, weight_dtype, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, hidden), generator=generator).to(torch.bfloat16)
    residual = torch.randn((rows, 4, hidden), generator=generator).to(torch.bfloat16)
    post = torch.rand((rows, 4), generator=generator) + 0.1
    # Deliberately asymmetric: transposing input/output HC axes must fail.
    comb = torch.rand((rows, 4, 4), generator=generator)
    pre = torch.rand((rows, 4), generator=generator) + 0.1
    weight = (1 + 0.1 * torch.randn((hidden,), generator=generator)).to(weight_dtype)
    return tuple(t.to(device) for t in (x, residual, post, comb, pre, weight))


def _reference(inputs, eps):
    x, residual, post, comb, pre, weight = (
        value.detach().cpu().float() for value in inputs
    )
    hc = torch.empty_like(residual, dtype=torch.bfloat16)
    for output_hc in range(4):
        accumulator = post[:, output_hc, None] * x
        for input_hc in range(4):
            accumulator = accumulator + (
                comb[:, input_hc, output_hc, None] * residual[:, input_hc]
            )
        hc[:, output_hc] = accumulator.to(torch.bfloat16)
    collapsed = torch.zeros_like(x)
    for hc_index in range(4):
        collapsed = collapsed + pre[:, hc_index, None] * hc[:, hc_index].float()
    collapsed = collapsed.to(torch.bfloat16).float()
    inv_rms = torch.rsqrt((collapsed * collapsed).mean(dim=-1, keepdim=True) + eps)
    normalized = (collapsed * inv_rms * weight).to(torch.bfloat16)
    return hc, normalized


def _assert_relative_l2(actual, reference):
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype == torch.bfloat16
    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(reference).all()
    if not actual.numel():
        return
    actual = actual.flatten(start_dim=1)
    reference = reference.flatten(start_dim=1)
    error = torch.linalg.vector_norm(actual - reference, dim=1)
    norm = torch.linalg.vector_norm(reference, dim=1)
    # Zero reference permits only an exactly zero residual; no epsilon floor.
    assert torch.all(error[norm == 0] == 0)
    ratios = error[norm > 0] / norm[norm > 0]
    assert torch.all(ratios <= RELATIVE_L2_LIMIT), ratios.tolist()


def _check(inputs, result, kernels, quantize):
    _, post_kernel, norm_kernel, quantizer = kernels
    hc, normalized, codes, scales = result
    x, residual, post, comb, pre, weight = inputs
    assert hc.shape == residual.shape and normalized.shape == x.shape
    assert hc.is_contiguous() and normalized.is_contiguous()
    reference_hc, reference_normalized = _reference(inputs, EPS)
    old_hc = post_kernel(x, residual, post, comb)
    old_normalized = torch.empty_like(x)
    norm_kernel(pre, old_hc, weight, old_normalized, eps=EPS)
    for expected_hc, expected_normalized in (
        (reference_hc, reference_normalized),
        (old_hc, old_normalized),
    ):
        _assert_relative_l2(hc, expected_hc)
        _assert_relative_l2(normalized, expected_normalized)
    if quantize:
        expected_codes, expected_scales = quantizer(
            normalized, "token_group", 32, "ue8m0", False
        )
        assert codes.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.uint8
        assert codes.shape == x.shape
        assert scales.shape == (x.shape[0], x.shape[1] // 32)
        assert codes.is_contiguous() and scales.is_contiguous()
        assert torch.isfinite(codes.float()).all()
        assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
        assert torch.equal(scales, expected_scales)
    else:
        assert codes is None and scales is None


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_production_shapes(kernels, rows, weight_dtype):
    inputs = _inputs(rows, 5120, weight_dtype, 391 + rows, "cuda")
    saved = tuple(t.clone() for t in inputs)
    result = kernels[0](*inputs, EPS, True)
    _check(inputs, result, kernels, True)
    assert all(torch.equal(a, b) for a, b in zip(inputs, saved))
    assert all(t.data_ptr() not in {x.data_ptr() for x in inputs} for t in result)
    # Caller ownership is retained across successive eager calls.
    second = kernels[0](*inputs, EPS, True)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(result, second))


def test_masked_hidden(kernels):
    inputs = _inputs(2, 96, torch.float32, 407, "cuda")
    _check(inputs, kernels[0](*inputs, EPS, True), kernels, True)


def test_without_quantization(kernels):
    inputs = _inputs(4, 5120, torch.bfloat16, 409, "cuda")
    _check(inputs, kernels[0](*inputs, EPS, False), kernels, False)


@pytest.mark.parametrize("quantize", [False, True])
def test_empty(kernels, quantize):
    inputs = _inputs(0, 5120, torch.bfloat16, 419, "cuda")
    _check(inputs, kernels[0](*inputs, EPS, quantize), kernels, quantize)


@pytest.mark.parametrize("case", ["zero", "cancellation"])
def test_zero_and_cancellation(kernels, case):
    inputs = _inputs(2, 5120, torch.bfloat16, 421, "cuda")
    x, residual, post, comb, pre, _ = inputs
    x.zero_()
    post.zero_()
    if case == "zero":
        residual.zero_()
    else:
        # Exactly representable opposite HC streams leave a small nonzero term.
        residual[:, 0].fill_(16)
        residual[:, 1].fill_(-16)
        residual[:, 2].mul_(1 / 128)
        residual[:, 3].zero_()
        comb.copy_(torch.tensor([1, 0.5, 2, 0.25], device="cuda").expand_as(comb))
        pre.copy_(torch.tensor([1, -1, 1, 1], device="cuda").expand_as(pre))
    result = kernels[0](*inputs, EPS, True)
    _check(inputs, result, kernels, True)
    if case == "zero":
        assert torch.count_nonzero(result[0]) == 0
        assert torch.count_nonzero(result[1]) == 0
        assert torch.count_nonzero(result[2].float()) == 0
    else:
        assert torch.count_nonzero(result[1]) > 0


def test_graph_replay_reads_mutated_inputs(kernels):
    inputs = _inputs(2, 5120, torch.bfloat16, 431, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernels[0](*inputs, EPS, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = kernels[0](*inputs, EPS, True)
    pointers = tuple(t.data_ptr() for t in result)
    graph.replay()
    _check(inputs, result, kernels, True)
    previous = result[1].clone()
    replacement = _inputs(2, 5120, torch.bfloat16, 433, "cuda")
    for destination, source in zip(inputs, replacement):
        destination.copy_(source)
    graph.replay()
    _check(inputs, result, kernels, True)
    assert tuple(t.data_ptr() for t in result) == pointers
    assert not torch.equal(result[1], previous)


@pytest.mark.parametrize("category", ["shape", "dtype", "stride", "options_device"])
def test_rejected_inputs(kernels, category):
    inputs = _inputs(2, 96, torch.bfloat16, 439, "cuda")
    invalid = []
    if category == "shape":
        for index, value in (
            (0, inputs[0].unsqueeze(0)),
            (0, inputs[0][:, :31].contiguous()),
            (0, inputs[0][:, :33].contiguous()),
            (0, torch.empty((2, 8224), device="cuda", dtype=torch.bfloat16)),
            (1, inputs[1][:, :3].contiguous()),
            (2, inputs[2][:1].contiguous()),
            (3, inputs[3][:, :, :3].contiguous()),
            (4, inputs[4][:, :3].contiguous()),
            (5, inputs[5][:-1].contiguous()),
        ):
            changed = list(inputs)
            changed[index] = value
            invalid.append((*changed, EPS, True))
    elif category == "dtype":
        for index in range(6):
            changed = list(inputs)
            changed[index] = inputs[index].to(torch.float16)
            invalid.append((*changed, EPS, True))
    elif category == "stride":
        for index, original in enumerate(inputs):
            storage = torch.empty(
                (*original.shape[:-1], original.shape[-1] * 2),
                device="cuda",
                dtype=original.dtype,
            )
            view = storage[..., ::2]
            view.copy_(original)
            assert not view.is_contiguous()
            changed = list(inputs)
            changed[index] = view
            invalid.append((*changed, EPS, True))
    else:
        invalid.extend(
            (*inputs, eps, True) for eps in (0, -1, float("nan"), float("inf"))
        )
        invalid.append((*inputs, EPS, 1))
        for index in range(6):
            changed = list(inputs)
            changed[index] = inputs[index].cpu()
            invalid.append((*changed, EPS, True))
    for arguments in invalid:
        with pytest.raises(ValueError):
            kernels[0](*arguments)


def test_cpu_reference_hc_axis_and_strict_zero_gate():
    inputs = _inputs(1, 32, torch.float32, 443, "cpu")
    x, residual, post, comb, pre, weight = inputs
    x.zero_()
    post.zero_()
    residual.copy_(torch.tensor([1, 2, 4, 8]).view(1, 4, 1).expand_as(residual))
    comb.copy_(torch.arange(16, dtype=torch.float32).view(1, 4, 4))
    pre.copy_(torch.tensor([[1, 0, 0, 0]], dtype=torch.float32))
    weight.fill_(1)
    hc, normalized = _reference(inputs, EPS)
    expected_hc = torch.tensor([136, 151, 166, 181], dtype=torch.bfloat16)
    assert torch.equal(hc, expected_hc.view(1, 4, 1).expand_as(hc))
    assert torch.equal(normalized, torch.ones_like(normalized))
    zero = torch.zeros((1, 32), dtype=torch.bfloat16)
    _assert_relative_l2(zero, zero)
    with pytest.raises(AssertionError):
        _assert_relative_l2(torch.ones_like(zero), zero)
