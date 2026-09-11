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

"""Execute GLM's actual decode-window helpers without optional GPU imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def attention_type():
    path = (
        Path(__file__).resolve().parents[3] / "python/tokenspeed/runtime/models/glm5.py"
    )
    tree = ast.parse(path.read_text())
    window = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "GlmDsaDecodeWindow"
    )
    attention = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "GlmMoeDsaAttention"
    )
    methods = {
        "_resolve_decode_q_len",
        "_resolve_num_decode_tokens",
        "_resolve_decode_req_count",
        "_resolve_decode_window",
        "_check_decode_q_len_per_req",
    }
    attention.bases = []
    attention.body = [
        node for node in attention.body if getattr(node, "name", None) in methods
    ]
    # The tensor-free helper bodies and dataclass are unchanged; only imports
    # and unrelated GPU layers are omitted from this CPU unit harness.
    namespace = {"__name__": __name__, "dataclass": dataclass}
    exec(
        compile(
            ast.Module(body=[window, attention], type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    return namespace["GlmMoeDsaAttention"]


def _context(width, num_extends, num_decodes, backend):
    return SimpleNamespace(
        bs=num_extends + num_decodes,
        num_extends=num_extends,
        decode_input_tokens=width,
        attn_backend=backend,
    )


def _metadata(num_extends, num_decodes):
    return SimpleNamespace(
        num_extends=num_extends,
        seq_lens_k=torch.empty(num_extends + num_decodes, dtype=torch.int32),
        page_table=torch.empty(num_extends + num_decodes, 2, dtype=torch.int32),
    )


@pytest.mark.parametrize("width,num_decodes", [(1, 1), (1, 3), (6, 1), (6, 3)])
def test_mixed_prefill_boundary_follows_active_query_width(
    attention_type, width, num_decodes
):
    ctx = _context(width, 1, num_decodes, SimpleNamespace(spec_num_tokens=6))
    total = 10 + width * num_decodes
    window = attention_type._resolve_decode_window(
        ctx,
        _metadata(1, num_decodes),
        total_tokens=total,
    )
    assert window.start == 10
    assert window.end == total
    assert window.num_tokens == width * num_decodes
    assert window.num_reqs == num_decodes
    assert window.q_len_per_req == width


def test_context_width_precedes_previously_prepared_backend_width(attention_type):
    backend = SimpleNamespace(spec_num_tokens=6, prepared_decode_width=1)
    ctx = _context(6, 0, 2, backend)
    window = attention_type._resolve_decode_window(
        ctx, _metadata(0, 2), total_tokens=12
    )
    assert (window.start, window.num_tokens, window.q_len_per_req) == (0, 12, 6)


@pytest.mark.parametrize(
    "backend,expected",
    [
        (SimpleNamespace(spec_num_tokens=6, prepared_decode_width=1), 1),
        (SimpleNamespace(spec_num_tokens=6), 6),
        (SimpleNamespace(), 1),
    ],
)
def test_legacy_context_uses_prepared_then_configured_width(
    attention_type, backend, expected
):
    ctx = _context(None, 1, 1, backend)
    del ctx.decode_input_tokens
    window = attention_type._resolve_decode_window(
        ctx, _metadata(1, 1), total_tokens=10 + expected
    )
    assert window.start == 10
    assert window.num_tokens == expected


def test_pure_prefill_keeps_all_rows_in_prefill(attention_type):
    ctx = _context(1, 2, 0, SimpleNamespace(spec_num_tokens=6))
    window = attention_type._resolve_decode_window(
        ctx, _metadata(2, 0), total_tokens=17
    )
    assert (window.start, window.end, window.num_tokens, window.num_reqs) == (
        17,
        17,
        0,
        0,
    )


def test_native_eight_does_not_expand_target_kernel_support(attention_type):
    with pytest.raises(NotImplementedError, match="1-6 query tokens"):
        attention_type._check_decode_q_len_per_req(8)
