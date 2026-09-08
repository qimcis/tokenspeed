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

"""CPU-only routing and state-seam checks (not CUDA kernel validation)."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_STATE = _ROOT / "python/tokenspeed/runtime/layers/attention/backends/state"
# Import the dependency-free routing module directly: importing the backend
# registry also imports optional vendor kernel packages, even for CPU tests.
_spec = importlib.util.spec_from_file_location(
    "_checkpoint_routing_test", _STATE / "prefill_checkpoint.py"
)
_routing = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _routing
_spec.loader.exec_module(_routing)
build = _routing.build_state_checkpoint_split_plan


def _plan(prefixes, lengths, checkpoints):
    return build(
        torch.tensor(prefixes, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        torch.tensor(checkpoints, dtype=torch.int32),
        4,
        "cpu",
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(
    "prefixes,lengths,checkpoints",
    [
        ([0], [7], [4]),
        ([4, 0, 0], [7, 2, 5], [8, 0, 4]),
        ([0, 4, 0], [2, 7, 2], [0, 8, 0]),
    ],
)
def test_select_merge_roundtrip_with_padding(prefixes, lengths, checkpoints, axis):
    plan = _plan(prefixes, lengths, checkpoints)
    shape = [2, 3, 4]
    shape[axis] = sum(lengths) + 5
    tensor = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
    phase1 = plan.select_tokens(tensor, 1, axis)
    phase2 = plan.select_tokens(tensor, 2, axis)
    torch.testing.assert_close(
        plan.merge_tokens(phase1, phase2, axis),
        tensor.narrow(axis, 0, sum(lengths)),
    )
    assert plan.select_tokens(None, 1, axis) is None
    assert plan.phase1_cu_seqlens_cpu[-1] + plan.phase2_cu_seqlens_cpu[-1] == sum(
        lengths
    )
    assert plan.checkpoint_slots.dtype == torch.int64
    if len(lengths) == 1:
        assert (
            phase1.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize(
    "prefixes,lengths,checkpoints",
    [
        ([0], [7], [8]),
        ([4], [7], [4]),
        ([0], [7], [3]),
        ([0], [0], [4]),
        ([0], [7], [-1]),
        ([0], [7, 7], [4]),
    ],
)
def test_invalid_checkpoint_metadata_rejected(prefixes, lengths, checkpoints):
    with pytest.raises(ValueError):
        _plan(prefixes, lengths, checkpoints)


def test_no_checkpoint_allocates_no_routing_plan():
    assert _plan([0, 4], [4, 2], [0, 0]) is None
    assert _plan([], [], []) is None


def _seams(namespace):
    # Execute the real orchestration methods with CPU reference seams, without
    # importing CUDA-only dependencies. GPU tests separately exercise dispatch.
    tree = ast.parse((_STATE / "mamba.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MambaAttnBackend"
    )
    names = {"_checkpointed_prefill_conv", "_checkpointed_prefill_scan"}
    functions = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    unit = ast.Module(body=functions, type_ignores=[])
    namespace.update(
        torch=torch, StateCheckpointSplitPlan=_routing.StateCheckpointSplitPlan
    )
    exec(compile(unit, str(_STATE / "mamba.py"), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("ragged", [False, True])
def test_checkpoint_scan_rounds_stored_state_and_keeps_other_rows(ragged):
    lengths = [7, 2, 6] if ragged else [7]
    checkpoints = [4, 0, 4] if ragged else [4]
    plan = _plan([0] * len(lengths), lengths, checkpoints)
    data = torch.arange(sum(lengths), dtype=torch.float32) / 17 + 0.013
    pool = torch.full((10, 1, 1, 1), -7, dtype=torch.bfloat16)
    phase1_ids = torch.tensor([1, 2, 3][: len(lengths)], dtype=torch.int32)
    final_ids = torch.tensor([4, 2, 5][: len(lengths)], dtype=torch.int32)

    def scan(q, k, v, initial, starts, **kwargs):
        assert kwargs["seq_len"] == q.shape[1] == int(starts[-1])
        assert torch.equal(starts.to(torch.int64), kwargs["cu_seqlens_cpu"])
        assert torch.equal(q.flatten(), kwargs["g_raw"].flatten())
        output = []
        states = []
        for row, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
            state = initial[row].float()
            for value in q[0, lo:hi]:
                state = state * 0.7 + value
                output.append(state.clone())
            states.append(state)
        return torch.stack(output), torch.stack(states)

    namespace = _seams({})
    backend = SimpleNamespace(_prefill_scan=scan)
    query = data.reshape(1, -1, 1, 1)
    out = namespace["_checkpointed_prefill_scan"](
        backend,
        query,
        query,
        query,
        torch.zeros(len(lengths), 1, 1, 1),
        pool,
        phase1_ids,
        final_ids,
        plan,
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
        a=None,
        b=None,
        g_raw=data[:, None],
        f_a_out=None,
        f_b_weight=None,
        beta_raw=None,
        lower_bound=None,
    )
    expected = []
    offset = 0
    for row, (length, boundary) in enumerate(zip(lengths, checkpoints)):
        state = torch.zeros(1, 1, 1)
        for i in range(length):
            state = state * 0.7 + data[offset + i]
            expected.append(state.clone())
            if i + 1 == boundary:
                torch.testing.assert_close(
                    pool[phase1_ids[row]], state.to(torch.bfloat16), rtol=0, atol=0
                )
                state = state.to(torch.bfloat16).float()
        torch.testing.assert_close(
            pool[final_ids[row]], state.to(torch.bfloat16), rtol=0, atol=0
        )
        offset += length
    torch.testing.assert_close(out, torch.stack(expected), rtol=0, atol=0)
    assert torch.all(pool[0] == -7)


@pytest.mark.parametrize("ragged", [False, True])
def test_checkpoint_conv_preserves_intermediate_window(ragged):
    lengths = [7, 2, 6] if ragged else [7]
    checkpoints = [4, 0, 4] if ragged else [4]
    plan = _plan([0] * len(lengths), lengths, checkpoints)
    data = torch.arange(sum(lengths), dtype=torch.float32).view(-1, 1) + 1
    pool = torch.zeros(8, 1, 3)
    first = torch.tensor([1, 2, 3][: len(lengths)], dtype=torch.int32)
    final = torch.tensor([4, 2, 5][: len(lengths)], dtype=torch.int32)

    def conv(
        inputs,
        weights,
        bias,
        *,
        activation,
        conv_states,
        has_initial_state,
        cache_indices,
        query_start_loc,
        seq_lens_cpu
    ):
        assert torch.equal(query_start_loc[1:] - query_start_loc[:-1], seq_lens_cpu)
        output = inputs.clone()
        for row, (lo, hi) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
            page = int(cache_indices[row])
            state = (
                conv_states[page].clone()
                if has_initial_state[row]
                else torch.zeros(1, 3)
            )
            for token in range(int(lo), int(hi)):
                value = inputs[:, token].clone()
                output[:, token] = value + state.sum()
                state = torch.cat((state[:, 1:], value[:, None]), 1)
            conv_states[page] = state
        inputs.copy_(output)  # the real conv also overwrites its input
        return inputs

    namespace = _seams({"causal_conv1d_fn": conv})
    actual = namespace["_checkpointed_prefill_conv"](
        None,
        data.clone(),
        torch.empty(0),
        None,
        None,
        pool,
        torch.zeros(len(lengths), dtype=torch.bool),
        first,
        final,
        plan,
    )
    expected = []
    offset = 0
    for row, (length, boundary) in enumerate(zip(lengths, checkpoints)):
        state = torch.zeros(1, 3)
        for i in range(length):
            value = data[offset + i]
            expected.append(value + state.sum())
            state = torch.cat((state[:, 1:], value[:, None]), 1)
            if i + 1 == boundary:
                torch.testing.assert_close(pool[first[row]], state, rtol=0, atol=0)
        torch.testing.assert_close(pool[final[row]], state, rtol=0, atol=0)
        offset += length
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    assert torch.count_nonzero(pool[0]) == 0


def test_capability_requires_every_state_leaf():
    path = _STATE.parent / "support.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "supports_prefill_state_checkpoints"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    resolve = namespace[fn.name]

    def node(state, capable, children):
        return SimpleNamespace(
            cache_consumer_families={"state"} if state else {"history"},
            supports_prefill_state_checkpoints=capable,
            child_backends=lambda: children,
        )

    history = node(False, False, ())
    kda = node(True, True, ())
    unsupported = node(True, False, ())
    assert not resolve(history)
    assert resolve(node(False, False, (history, kda)))
    assert not resolve(node(False, False, (history, kda, unsupported)))
    assert resolve(kda, history, None)
    assert not resolve(kda, unsupported)
