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

"""Single-forward checkpoint admission and immediate first-replay publication."""

import pytest
from conftest import (
    K3_STATE_GROUPS,
    _advance,
    _find_forward_op,
    _finish,
    _make_k3_config,
    _spec,
)

ts = pytest.importorskip("tokenspeed_scheduler")


def _config():
    cfg = _make_k3_config()
    cfg.prefill_state_checkpoints = True
    return cfg


@pytest.mark.parametrize("checkpoint_enabled", [False, True])
@pytest.mark.parametrize("prefix_enabled", [False, True])
def test_one_forward_only_when_capability_and_prefix_cache_enabled(
    checkpoint_enabled, prefix_enabled
):
    cfg = _config()
    cfg.prefill_state_checkpoints = checkpoint_enabled
    cfg.disable_prefix_cache = not prefix_enabled
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("r", [1, 2, 3, 4, 5])])
    op = _find_forward_op(scheduler.next_execution_plan())
    split = prefix_enabled and not checkpoint_enabled
    assert op.input_lengths == [4 if split else 5]
    assert op.state_checkpoint_lens == [
        4 if prefix_enabled and checkpoint_enabled else 0
    ]
    if checkpoint_enabled and prefix_enabled:
        for group in K3_STATE_GROUPS:
            row = op.block_tables[group][0]
            assert row[0] == 0  # no uninitialized snapshots materialized
            assert row[1] > 0 and row[2] > 0 and row[1] != row[2]


@pytest.mark.parametrize("do_decode", [False, True])
def test_first_replay_hits_checkpoint_without_a_second_cold_prefill(do_decode):
    scheduler = ts.Scheduler(_config())
    tokens = [1, 2, 3, 4, 5]
    scheduler.submit_requests([_spec("cold", tokens)])
    first = _find_forward_op(scheduler.next_execution_plan())
    assert first.input_lengths == [5] and first.state_checkpoint_lens == [4]
    checkpoint_pages = {g: first.block_tables[g][0][1] for g in K3_STATE_GROUPS}
    _advance(scheduler, "cold", [6])
    if do_decode:
        decode = _find_forward_op(scheduler.next_execution_plan())
        assert decode.extend_prefix_lens == []
        assert decode.state_checkpoint_lens == []
        _advance(scheduler, "cold", [7])
    _finish(scheduler, "cold")

    scheduler.submit_requests([_spec("replay", tokens)])
    replay = _find_forward_op(scheduler.next_execution_plan())
    assert replay.extend_prefix_lens == [4]
    assert replay.input_ids == [5]
    assert replay.state_checkpoint_lens == [0]
    for group in K3_STATE_GROUPS:
        row = replay.block_tables[group][0]
        assert row[1] == checkpoint_pages[group]
        assert row[2] > 0 and row[2] != row[1]  # private writable tail


def test_chunked_prefill_carries_checkpoint_only_on_final_chunk():
    cfg = _config()
    cfg.max_scheduled_tokens = 4
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("r", list(range(7)))])
    first = _find_forward_op(scheduler.next_execution_plan())
    assert first.input_lengths == [4] and first.state_checkpoint_lens == [0]
    _advance(scheduler, "r", [])
    last = _find_forward_op(scheduler.next_execution_plan())
    assert last.extend_prefix_lens == [4]
    assert last.input_lengths == [3] and last.state_checkpoint_lens == [6]
    _advance(scheduler, "r", [90])
    _finish(scheduler, "r")
    scheduler.submit_requests([_spec("replay", list(range(7)))])
    replay = _find_forward_op(scheduler.next_execution_plan())
    assert replay.extend_prefix_lens == [6] and replay.input_lengths == [1]


@pytest.mark.parametrize("length", [1, 2, 4])
def test_short_and_aligned_prompts_need_no_intermediate_checkpoint(length):
    scheduler = ts.Scheduler(_config())
    scheduler.submit_requests([_spec("r", list(range(length)))])
    op = _find_forward_op(scheduler.next_execution_plan())
    assert op.input_lengths == [length] and op.state_checkpoint_lens == [0]


def test_packed_rows_keep_their_own_checkpoint_boundaries():
    scheduler = ts.Scheduler(_config())
    scheduler.submit_requests([_spec("a", [1, 2, 3]), _spec("b", [9, 8])])
    op = _find_forward_op(scheduler.next_execution_plan())
    assert op.request_ids == ["a", "b"]
    assert op.input_lengths == [3, 2]
    assert op.state_checkpoint_lens == [2, 0]


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("decode_width", [1, 8])
def test_growth_horizon_matches_split_body_reservation(overlap, decode_width):
    def first_tables(enabled):
        cfg = _config()
        cfg.prefill_state_checkpoints = enabled
        cfg.overlap_schedule_depth = overlap
        cfg.decode_input_tokens = decode_width
        # Width-8 overlap protects two verify windows; the tiny default pool
        # deliberately cannot admit that shape, independently of this change.
        cfg.num_device_pages = 129
        for group in cfg.cache_groups:
            group.total_pages = 129
        scheduler = ts.Scheduler(cfg)
        scheduler.submit_requests([_spec("r", [1, 2, 3, 4, 5])])
        op = _find_forward_op(scheduler.next_execution_plan())
        return {
            g: len([p for p in op.block_tables[g][0] if p > 0]) for g in K3_STATE_GROUPS
        }

    assert first_tables(True) == first_tables(False)


def test_finer_state_geometry_keeps_the_split_fallback():
    cfg = _config()
    groups = cfg.cache_groups
    for group in groups:
        if group.family == ts.CacheGroupFamily.State:
            group.block_granularity = 1
    cfg.cache_groups = groups
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("r", [1, 2, 3])])
    op = _find_forward_op(scheduler.next_execution_plan())
    assert op.input_lengths == [2] and op.state_checkpoint_lens == [0]


@pytest.mark.parametrize("role", ["P", "D"])
def test_pd_roles_keep_checkpoint_ownership_local_to_prefill(role):
    cfg = _config()
    cfg.role = getattr(ts.SchedulerConfig.Role, role)
    for group in cfg.cache_groups:
        group.transfer_policy = (
            ts.CacheTransferPolicy.LatestSnapshot
            if group.group_id in K3_STATE_GROUPS
            else ts.CacheTransferPolicy.FullSuffix
        )
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("r", [1, 2, 3, 4, 5])])
    scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("r")))
    plan = scheduler.next_execution_plan()
    if role == "D":
        assert plan.remote_prefill.state_checkpoint_lens == [0]
        assert _find_forward_op(plan) is None
    else:
        prefill = _find_forward_op(plan)
        assert prefill.input_lengths == [5]
        assert prefill.state_checkpoint_lens == [4]
        _advance(scheduler, "r", [9])
        transfer = scheduler.next_execution_plan()
        assert transfer.remote_decode.request_ids == ["r"]
        scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.SucceededEvent("r")))
        scheduler.submit_requests([_spec("replay", [1, 2, 3, 4, 5])])
        scheduler.advance(
            ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("replay"))
        )
        replay = _find_forward_op(scheduler.next_execution_plan())
        assert replay.extend_prefix_lens == [4]


@pytest.mark.parametrize("overlap", [0, 1])
def test_unaligned_prompts_fill_pool_without_losing_decode_progress(overlap):
    cfg = _config()
    cfg.overlap_schedule_depth = overlap
    scheduler = ts.Scheduler(cfg)
    requests = []
    for i in range(6):
        spec = _spec(f"r{i}", [i * 10 + j for j in range(3)])
        spec.max_new_tokens = 4
        requests.append(spec)
    scheduler.submit_requests(requests)
    produced = {spec.request_id: 0 for spec in requests}
    finished = set()
    for _ in range(100):
        plan = scheduler.next_execution_plan()
        for op in plan.forward:
            for row, rid in enumerate(op.request_ids):
                produced[rid] += 1
                _advance(scheduler, rid, [100 + produced[rid]])
                if produced[rid] == 4:
                    _finish(scheduler, rid)
                    finished.add(rid)
                elif row >= len(op.extend_prefix_lens):
                    reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                    reserve.request_id = rid
                    reserve.reserve_num_tokens_in_next_schedule_event = 1
                    scheduler.advance(ts.ExecutionEvent().add_event(reserve))
        if len(finished) == 6:
            break
    assert len(finished) == 6
    assert scheduler.active_kv_pages() == 0
