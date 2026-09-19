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

"""CPU selection/routing tests: every computational leaf is replaced.

Queries expose host-visible geometry only. Metadata deliberately refuses host
reads, so adding a device-dependent Python dispatch breaks these tests without
allocating tensors, initializing a CUDA context, or launching a GPU kernel.
"""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.selection as selection
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import flash_mla, hopper
from tokenspeed_kernel.platform import ArchVersion
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


class DeviceMetadata:
    def item(self):
        raise AssertionError("routing must not read device metadata")

    def cpu(self):
        raise AssertionError("routing must not transfer device metadata")

    def tolist(self):
        raise AssertionError("routing must not materialize device metadata")

    def __bool__(self):
        raise AssertionError("routing must not branch on device metadata")


def _arguments(tokens, heads, combined, prefill):
    return (
        SimpleNamespace(ndim=3, shape=(tokens, heads, 512), dtype=torch.bfloat16),
        object(),
        DeviceMetadata(),
        DeviceMetadata(),
        object() if combined else None,
        DeviceMetadata() if combined else None,
        DeviceMetadata() if combined else None,
        object(),
        512**-0.5,
        object(),
        256,
        object(),
        object() if prefill else None,
        DeviceMetadata() if prefill else None,
    )


@pytest.fixture
def leaves(monkeypatch):
    calls = []

    def make_leaf(name):
        def leaf(
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            softmax_scale,
            out,
            query_chunk_size,
            schedule,
            prefill_kv,
            prefill_indices,
        ):
            arguments = (
                q,
                swa_cache,
                swa_slots,
                swa_lens,
                global_cache,
                global_slots,
                global_lens,
                attn_sink,
                softmax_scale,
                out,
                query_chunk_size,
                schedule,
                prefill_kv,
                prefill_indices,
            )
            calls.append((name, arguments, {}))
            return out

        return leaf

    def tiled(
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        out,
        query_chunk_size,
        schedule,
        prefill_kv,
        prefill_indices,
        *,
        pv_mode,
        tile_k,
        num_warps,
    ):
        arguments = (
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            softmax_scale,
            out,
            query_chunk_size,
            schedule,
            prefill_kv,
            prefill_indices,
        )
        calls.append(
            (
                "tiled",
                arguments,
                {"pv_mode": pv_mode, "tile_k": tile_k, "num_warps": num_warps},
            )
        )
        return out

    monkeypatch.setattr(hopper, "_portable_attention", make_leaf("portable"))
    monkeypatch.setattr(hopper, "_gathered_attention", make_leaf("gathered"))
    monkeypatch.setattr(hopper, "hopper_tiled_selected_attention", tiled)
    return calls


def _assert_call(calls, arguments, name, options):
    assert len(calls) == 1
    actual_name, forwarded, actual_options = calls[0]
    assert actual_name == name
    assert len(forwarded) == 14
    assert all(actual is expected for actual, expected in zip(forwarded, arguments))
    assert actual_options == options


def test_public_hopper_signature_requires_exactly_fourteen_arguments():
    parameters = list(inspect.signature(hopper.selected_attention).parameters.values())
    assert len(parameters) == 14
    assert [parameter.name for parameter in parameters] == [
        "q",
        "swa_cache",
        "swa_slots",
        "swa_lens",
        "global_cache",
        "global_slots",
        "global_lens",
        "attn_sink",
        "softmax_scale",
        "out",
        "query_chunk_size",
        "schedule",
        "prefill_kv",
        "prefill_indices",
    ]
    assert all(parameter.default is inspect.Parameter.empty for parameter in parameters)
    assert all(
        parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    )


@pytest.mark.parametrize("tokens", [1, 4, 8, 15, 16, 64, 256])
@pytest.mark.parametrize("available", [False, True])
def test_combined_crossover_and_optional_native_fallback(
    monkeypatch, leaves, tokens, available
):
    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", lambda: available)
    arguments = _arguments(tokens, 8, True, False)
    assert hopper.selected_attention(*arguments) is arguments[9]
    expected = "gathered" if available and tokens >= 16 else "portable"
    _assert_call(leaves, arguments, expected, {})


@pytest.mark.parametrize(
    "tokens,tile_k,num_warps",
    [(1, 64, 8), (8, 64, 8), (64, 64, 8), (65, 32, 4), (128, 32, 4), (256, 32, 4)],
)
@pytest.mark.parametrize("available", [False, True])
def test_swa_has_direct_tiles_without_optional_native_dependency(
    monkeypatch, leaves, tokens, tile_k, num_warps, available
):
    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", lambda: available)
    arguments = _arguments(tokens, 8, False, False)
    assert hopper.selected_attention(*arguments) is arguments[9]
    _assert_call(
        leaves,
        arguments,
        "tiled",
        {"pv_mode": "bf16", "tile_k": tile_k, "num_warps": num_warps},
    )


@pytest.mark.parametrize("tokens", [1, 128, 1024])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_prefill_workspace_takes_priority_and_native_is_optional(
    monkeypatch, leaves, tokens, combined, available
):
    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", lambda: available)
    arguments = _arguments(tokens, 8, combined, True)
    assert hopper.selected_attention(*arguments) is arguments[9]
    _assert_call(leaves, arguments, "gathered" if available else "portable", {})


@pytest.mark.parametrize("heads", [1, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "combined,prefill", [(False, False), (True, False), (False, True)]
)
def test_unmeasured_head_counts_keep_portable_without_native_probe(
    monkeypatch, leaves, heads, combined, prefill
):
    def unexpected_probe():
        pytest.fail("unmeasured heads must not query the optional native API")

    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", unexpected_probe)
    arguments = _arguments(64, heads, combined, prefill)
    assert hopper.selected_attention(*arguments) is arguments[9]
    _assert_call(leaves, arguments, "portable", {})


@pytest.mark.parametrize("shape", [(0, 8, 512), (8, 512), (1, 8, 576)])
def test_empty_or_unsupported_geometry_delegates_validation(monkeypatch, leaves, shape):
    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", lambda: True)
    arguments = list(_arguments(1, 8, False, False))
    arguments[0] = SimpleNamespace(ndim=len(shape), shape=shape, dtype=torch.bfloat16)
    assert hopper.selected_attention(*arguments) is arguments[9]
    _assert_call(leaves, arguments, "portable", {})


def test_orphan_prefill_indices_are_forwarded_for_portable_validation(
    monkeypatch, leaves
):
    monkeypatch.setattr(hopper, "is_flash_mla_v41_available", lambda: True)
    arguments = list(_arguments(64, 8, False, False))
    arguments[-1] = DeviceMetadata()
    assert hopper.selected_attention(*arguments) is arguments[9]
    _assert_call(leaves, arguments, "portable", {})


@pytest.fixture
def isolated_selection(monkeypatch):
    monkeypatch.delenv(
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_SELECTED_ATTENTION", raising=False
    )
    monkeypatch.setattr(selection, "_global_overrides", {})
    KernelRegistry.get().clear_cache()
    yield
    KernelRegistry.get().clear_cache()


def test_registration_is_bf16_nvidia_sm90_only(
    h100_platform, b200_platform, b300_platform
):
    specs = KernelRegistry.get().get_for_operator(
        "attention",
        "dsv41_selected_attention",
        features=None,
        platform=None,
        format_signature=None,
        tags=None,
        solution="hopper",
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "hopper_dsv41_selected_attention"
    assert spec.priority == Priority.PERFORMANT
    assert spec.capability.min_arch_version == ArchVersion(9, 0)
    assert spec.capability.max_arch_version == ArchVersion(9, 0)
    assert spec.capability.vendors == frozenset({"nvidia"})
    assert spec.format_signatures == frozenset(
        {format_signature(x=dense_tensor_format(torch.bfloat16))}
    )
    assert spec.capability.satisfied_by(h100_platform)
    assert not spec.capability.satisfied_by(b200_platform)
    assert not spec.capability.satisfied_by(b300_platform)
    assert not spec.capability.satisfied_by(replace(h100_platform, vendor="amd"))
    assert not spec.capability.satisfied_by(
        replace(h100_platform, arch_version=ArchVersion(9, 1))
    )


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("has_native_inputs", [False, True])
def test_hopper_selector_keeps_real_heads(
    monkeypatch, isolated_selection, h100_platform, available, has_native_inputs
):
    monkeypatch.setattr(selection, "current_platform", lambda: h100_platform)
    monkeypatch.setattr(flash_mla, "is_flash_mla_v41_available", lambda: available)
    q = _arguments(1, 8, True, False)[0]
    selected = dsv41._selected_attention_kernel(q, has_native_inputs)
    assert selected.name == "hopper_dsv41_selected_attention"
    assert dsv41.prefers_padded_query(q) is False


@pytest.mark.parametrize("platform_name", ["b200_platform", "b300_platform"])
@pytest.mark.parametrize("available", [False, True])
def test_blackwell_selection_and_padding_guard_are_unchanged(
    monkeypatch, isolated_selection, request, platform_name, available
):
    platform = request.getfixturevalue(platform_name)
    monkeypatch.setattr(selection, "current_platform", lambda: platform)
    monkeypatch.setattr(flash_mla, "is_flash_mla_v41_available", lambda: available)
    q = _arguments(1, 8, True, False)[0]
    selected = dsv41._selected_attention_kernel(q, True)
    assert selected.name == (
        "flashmla_dsv41_selected_attention"
        if available
        else "triton_dsv41_selected_attention"
    )
    assert dsv41.prefers_padded_query(q) is available
    without_inputs = dsv41._selected_attention_kernel(q, False)
    assert without_inputs.name == "triton_dsv41_selected_attention"
