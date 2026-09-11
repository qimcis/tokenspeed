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
"""CPU regressions for immutable remote verification and accepted anchors."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    resolve_decode_input_tokens,
)


def _op(lengths, num_extends, width):
    return SimpleNamespace(
        input_lengths=lengths,
        num_extends=lambda: num_extends,
        decode_input_tokens=width,
    )


def test_width_comes_from_the_reserved_operation():
    assert resolve_decode_input_tokens(_op([1, 1], 0, 1), 6) == 1
    assert resolve_decode_input_tokens(_op([6, 6], 0, 6), 6) == 6
    assert resolve_decode_input_tokens(_op([17, 6, 6], 1, 6), 6) == 6
    assert resolve_decode_input_tokens(_op([17, 8], 2, 0), 6) == 6
    with pytest.raises(ValueError, match="planned decode width"):
        resolve_decode_input_tokens(_op([1, 1], 0, 6), 6)
    with pytest.raises(ValueError, match="homogeneous"):
        resolve_decode_input_tokens(_op([1, 6], 0, 1), 6)
    with pytest.raises(ValueError, match="homogeneous"):
        resolve_decode_input_tokens(_op([8], 0, 8), 6)


def test_fallback_uses_each_requests_anchor_at_the_real_row_stride():
    state = RuntimeStates(5, 1000, 6, "cpu")
    state.future_input_map.copy_(torch.arange(36).reshape(6, 6))
    rows = torch.tensor([4, 1, 3])
    assert state.gather_candidate_ids(rows, 1).tolist() == [24, 6, 18]
    assert state.gather_candidate_ids(rows, 6).tolist() == [
        24,
        25,
        26,
        27,
        28,
        29,
        6,
        7,
        8,
        9,
        10,
        11,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_accept_reject_and_fallback_rounds_preserve_the_last_accepted_anchor():
    state = RuntimeStates(4, 1000, 6, "cpu")
    rows = torch.tensor([3, 1])
    state.write_remote_spec_candidate_ids(3, [10, 11, 12, 13, 14, 15])
    state.write_remote_spec_candidate_ids(1, [20, 21, 22, 23, 24, 25])
    state.update_next_anchors(
        rows,
        torch.tensor([11, 12, 91, 0, 0, 0, 21, 22, 23, 24, 25, 92]),
        torch.tensor([3, 6]),
        0,
    )
    assert state.gather_candidate_ids(rows, 1).tolist() == [91, 92]
    assert not state.remote_spec_candidate_ready[rows].any()
    # The next round is truly one output per row, independent of map capacity.
    state.update_next_anchors(rows, torch.tensor([93, 94]), torch.tensor([1, 1]), 0)
    assert state.gather_candidate_ids(rows, 1).tolist() == [93, 94]
    state.write_remote_spec_candidate_ids(3, [93, 31, 32, 33, 34, 35])
    assert state.gather_candidate_ids(rows[:1], 6).tolist() == [93, 31, 32, 33, 34, 35]


def test_mixed_prefill_outputs_and_verify_outputs_have_different_strides():
    state = RuntimeStates(4, 1000, 6, "cpu")
    rows = torch.tensor([1, 2, 3])
    state.update_next_anchors(
        rows,
        torch.tensor([40, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65]),
        torch.tensor([1, 2, 4]),
        1,
    )
    assert state.gather_candidate_ids(rows, 1).tolist() == [40, 51, 63]
    state.update_next_anchors(
        rows, torch.tensor([70, 80, 90]), torch.ones(3, dtype=torch.int32), 3
    )
    assert state.gather_candidate_ids(rows, 1).tolist() == [70, 80, 90]


def test_invalid_remote_candidate_does_not_publish_readiness():
    state = RuntimeStates(3, 20, 6, "cpu")
    for candidates in ([], [1] * 7, [1, -1, 2], [1, 20, 2]):
        with pytest.raises((ValueError, RuntimeError)):
            state.write_remote_spec_candidate_ids(1, candidates)
        assert not state.remote_spec_candidate_ready[1]
    with pytest.raises(ValueError):
        state.write_remote_spec_candidate_ids(0, [1, 2])


def test_submitted_dp_width_consensus_cannot_be_reassigned():
    metadata = DpForwardMetadata(
        global_num_tokens=[6, 1],
        global_batch_size=[1, 1],
        global_forward_mode=[2, 2],
        all_decode_or_idle=True,
        all_extend=False,
        need_idle_forward=False,
        global_decode_input_tokens=[6, 1],
        decode_graph_width=None,
    )
    with pytest.raises(FrozenInstanceError):
        metadata.decode_graph_width = 6


def _runner_type():
    pytest.importorskip("tokenspeed_kernel")
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner

    return ForwardStepRunner


def test_graphs_for_equal_batch_sizes_keep_separate_verification_widths():
    runner_type = _runner_type()
    runner = runner_type.__new__(runner_type)
    runner.sampling_backend = None
    runner.max_tokens_per_req = 6
    runner.graphs = {("default", 1, 4): object(), ("default", 6, 4): object()}
    key_one = runner._cuda_graph_key(4, 1)
    key_six = runner._cuda_graph_key(4, 6)
    assert key_one != key_six
    assert key_one[1:] == (1, 4)
    assert key_six[1:] == (6, 4)


def test_one_fallback_cohort_forces_all_eight_ep_ranks_to_eager():
    runner_type = _runner_type()
    runner = runner_type.__new__(runner_type)
    runner.disable = False
    runner.disable_padding = False
    runner.dp_size = 8
    runner.max_tokens_per_req = 6
    runner.decode_widths = (1, 6)
    runner.max_capture_bs = 32
    for width in (1, 6):
        ctx = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            decode_input_tokens=width,
            global_decode_input_tokens=[1, 6, 6, 6, 6, 6, 6, 6],
            decode_graph_width=None,
            all_decode_or_idle=True,
            global_num_tokens=[4, 24, 24, 24, 24, 24, 24, 24],
        )
        assert not runner._can_use_graph(4, ctx)
    ctx.decode_graph_width = 1
    ctx.global_decode_input_tokens = [1] * 7 + [0]
    ctx.global_num_tokens = [4] * 7 + [0]
    ctx.decode_input_tokens = 1
    assert runner._can_use_graph(0, ctx)
    assert runner._global_graph_bs(ctx) == 4


def test_graph_padding_uses_dummy_request_slots_for_both_widths():
    runner_type = _runner_type()
    runner = runner_type.__new__(runner_type)
    runner.config = SimpleNamespace(spec_algo="DFLASH", max_req_pool_size=9)
    runner.input_buffers = SimpleNamespace(
        state_write_req_pool_indices_buf=torch.full((5,), -1, dtype=torch.int64)
    )
    active = torch.tensor([2, 4], dtype=torch.int64)
    for width in (1, 6):
        padded = runner._pad_graph_req_pool_indices(active, 5)
        assert padded.tolist() == [2, 4, 9, 9, 9]
        runner._set_graph_state_write_indices(active, 5)
        assert runner.input_buffers.state_write_req_pool_indices_buf.tolist() == [
            2,
            4,
            9,
            9,
            9,
        ]
