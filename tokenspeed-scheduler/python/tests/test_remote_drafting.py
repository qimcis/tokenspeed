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

"""Remote scheduling keeps confirmed prefixes and cache reservation separate."""

import pytest
from tokenspeed_scheduler import (
    CacheGroupConfig,
    CacheRetention,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def event(kind, **values):
    result = getattr(ForwardEvent, kind)()
    for key, value in values.items():
        setattr(result, key, value)
    return result


def advance(scheduler, *events):
    update = ExecutionEvent()
    for item in events:
        update.add_event(item)
    scheduler.advance(update)


def config(remote, draft_width):
    result = SchedulerConfig()
    result.prefix_granularity = 8
    result.max_scheduled_tokens = 128
    result.max_batch_size = 8
    result.num_device_pages = 256
    result.num_host_pages = 0
    result.disable_l2_cache = True
    result.decode_input_tokens = 6
    result.draft_input_tokens = draft_width
    result.remote_draft_enabled = remote
    result.remote_draft_min_ready = 2
    result.remote_draft_max_defer_ms = 100
    result.remote_draft_feature_group = "dflash2_projected_features"
    result.cache_groups = [
        CacheGroupConfig("full_attention", 8, 256),
        CacheGroupConfig(
            "dflash2_projected_features",
            8,
            256,
            CacheRetention.SlidingWindow,
            32,
        ),
    ]
    return result


def seed(count, prompt_length):
    scheduler = Scheduler(config(True, 0))
    requests = []
    for index in range(count):
        request = RequestSpec()
        request.request_id = f"r{index}"
        request.tokens = list(range(10 * index, 10 * index + prompt_length))
        request.max_new_tokens = 100
        requests.append(request)
    scheduler.submit_requests(requests)
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.decode_input_tokens == 0
    for index in range(count):
        advance(
            scheduler,
            event("ExtendResult", request_id=f"r{index}", tokens=[100 + index]),
        )
    return scheduler


def descriptor(scheduler, request_id):
    return next(
        item
        for item in scheduler.remote_draft_requests()
        if item.request_id == request_id
    )


def pending(scheduler, request_id, session_id, now_ms):
    current = descriptor(scheduler, request_id)
    advance(
        scheduler,
        event(
            "RemoteDraftPending",
            request_id=request_id,
            session_id=session_id,
            endpoint=current.endpoint,
            anchor_id=current.anchor_id,
            now_ms=now_ms,
        ),
    )
    return current.endpoint, current.anchor_id


def ready(scheduler, request_id, session_id, prefix):
    endpoint, anchor = prefix
    advance(
        scheduler,
        event(
            "RemoteDraftReady",
            request_id=request_id,
            session_id=session_id,
            endpoint=endpoint,
            anchor_id=anchor,
            candidate_ids=[201, 202, 203, 204, 205],
        ),
    )


def test_pending_prefix_survives_other_request_work():
    scheduler = seed(2, 7)
    prefix = pending(scheduler, "r0", "s0", 0)
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.request_ids == ["r1"]
    assert batch.decode_input_tokens == 1
    assert batch.decode_input_ids == [101]
    assert descriptor(scheduler, "r0").endpoint == prefix[0]
    ready(scheduler, "r0", "s0", prefix)
    next_batch = scheduler.next_execution_plan().forward[0]
    assert next_batch.request_ids == ["r0"]
    assert next_batch.decode_input_tokens == 6
    assert next_batch.spec_candidate_ids == [[100, 201, 202, 203, 204, 205]]
    # The already-submitted width-one batch is immutable.
    assert batch.decode_input_tokens == 1


def test_one_escape_survives_overlap_without_releasing_every_pending_prefix():
    scheduler = seed(3, 7)
    for index in range(3):
        pending(scheduler, f"r{index}", f"s{index}", 0)
    first = scheduler.next_execution_plan().forward[0]
    assert first.request_ids == ["r0"]
    assert first.decode_input_tokens == 1
    assert not scheduler.next_execution_plan().forward[0].request_ids
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[150]))
    assert not descriptor(scheduler, "r0").admission_allowed
    pending(scheduler, "r0", "premature", 0)
    assert descriptor(scheduler, "r0").session_id == "s0"
    second = scheduler.next_execution_plan().forward[0]
    assert second.request_ids == ["r0"]
    assert second.decode_input_ids == [150]
    assert descriptor(scheduler, "r1").status == "pending"


def test_expired_pending_gets_service_before_unassigned_work():
    scheduler = seed(2, 7)
    pending(scheduler, "r0", "s0", 0)
    advance(scheduler, event("RemoteDraftTick", now_ms=100))
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.request_ids[0] == "r0"
    assert batch.decode_input_tokens == 1


def test_ready_minimum_selects_one_homogeneous_batch():
    scheduler = seed(3, 7)
    for index in range(2):
        prefix = pending(scheduler, f"r{index}", f"s{index}", 0)
        ready(scheduler, f"r{index}", f"s{index}", prefix)
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.request_ids == ["r0", "r1"]
    assert batch.input_lengths == [6, 6]
    assert batch.decode_input_tokens == 6


@pytest.mark.parametrize("accepted", [1, 3, 6])
def test_alternating_widths_keep_computed_frontier_and_next_anchor(accepted):
    scheduler = seed(1, 7)
    prefix = pending(scheduler, "r0", "s0", 0)
    ready(scheduler, "r0", "s0", prefix)
    wide = scheduler.next_execution_plan().forward[0]
    assert wide.decode_input_tokens == 6
    state = descriptor(scheduler, "r0")
    assert (state.computed_endpoint, state.reserved_endpoint) == (7, 13)
    assert state.results_in_flight == 1
    assert not scheduler.next_execution_plan().forward[0].request_ids
    accepted_ids = list(range(300, 300 + accepted))
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=accepted_ids))
    narrow = scheduler.next_execution_plan().forward[0]
    assert narrow.decode_input_tokens == 1
    assert narrow.decode_input_ids == [accepted_ids[-1]]
    state = descriptor(scheduler, "r0")
    assert state.computed_endpoint == 7 + accepted
    assert state.reserved_endpoint == max(13, 8 + accepted)
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[400]))
    new_prefix = pending(scheduler, "r0", "s0", 0)
    ready(scheduler, "r0", "s0", new_prefix)
    wide_again = scheduler.next_execution_plan().forward[0]
    assert wide_again.decode_input_tokens == 6
    assert wide_again.decode_input_ids == [400]
    assert descriptor(scheduler, "r0").reserved_endpoint == 14 + accepted
    # The page containing the last possible target query is admitted.
    assert len(wide_again.block_tables["full_attention"][0]) * 8 >= 14 + accepted


def test_stale_and_duplicate_candidates_cannot_mutate_a_submitted_forward():
    scheduler = seed(1, 7)
    prefix = pending(scheduler, "r0", "s0", 0)
    ready(scheduler, "r0", "s0", prefix)
    wide = scheduler.next_execution_plan().forward[0]
    ready(scheduler, "r0", "s0", prefix)
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[250]))
    ready(scheduler, "r0", "s0", prefix)
    assert descriptor(scheduler, "r0").status == "unavailable"
    assert wide.spec_candidate_ids == [[100, 201, 202, 203, 204, 205]]
    assert scheduler.next_execution_plan().forward[0].decode_input_tokens == 1


def test_export_snapshot_pin_survives_abort_until_explicit_release():
    cfg = config(True, 0)
    groups = cfg.cache_groups
    groups[1].cache_blocks_per_lcm_block = 4
    groups[1].total_pages = 1 + (cfg.num_device_pages - 1) * 4
    cfg.cache_groups = groups
    scheduler = Scheduler(cfg)
    request = RequestSpec()
    request.request_id = "r0"
    request.tokens = list(range(17))
    scheduler.submit_requests([request])
    prefill = scheduler.next_execution_plan().forward[0]
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[100]))
    endpoint, anchor = pending(scheduler, "r0", "s0", 0)
    # Worker admission precedes the explicit export request.
    assert scheduler.remote_draft_snapshots() == []
    advance(
        scheduler,
        event(
            "RemoteDraftExport",
            request_id="r0",
            session_id="s0",
            endpoint=endpoint,
            anchor_id=anchor,
            start=0,
        ),
    )
    snapshots = scheduler.remote_draft_snapshots()
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert (snapshot.start, snapshot.endpoint, snapshot.anchor_id) == (0, 17, 100)
    feature_pages = snapshot.block_tables["dflash2_projected_features"]
    assert len(feature_pages) >= 3
    feature_parents = {1 + (page - 1) // 4 for page in feature_pages if page > 0}
    target_parents = {
        page for page in prefill.block_tables["full_attention"][0] if page > 0
    }
    assert feature_parents.isdisjoint(target_parents)
    assert len({1 + (page - 1) // 4 for page in feature_pages[:3]}) == 1
    advance(scheduler, event("Abort", request_id="r0"))
    scheduler.next_execution_plan()
    available_with_pin = scheduler.available_kv_pages()
    advance(
        scheduler, event("ReleaseRemoteDraftSnapshot", ticket_id=snapshot.ticket_id)
    )
    assert scheduler.available_kv_pages() > available_with_pin
    # Release is idempotent and stale network replies cannot resurrect it.
    advance(
        scheduler, event("ReleaseRemoteDraftSnapshot", ticket_id=snapshot.ticket_id)
    )
    ready(scheduler, "r0", "s0", (endpoint, anchor))
    assert scheduler.remote_draft_requests() == []


def test_expired_history_export_is_rejected():
    scheduler = seed(1, 40)
    endpoint, anchor = pending(scheduler, "r0", "s0", 0)
    with pytest.raises(ValueError, match="retained confirmed history"):
        advance(
            scheduler,
            event(
                "RemoteDraftExport",
                request_id="r0",
                session_id="s0",
                endpoint=endpoint,
                anchor_id=anchor,
                start=0,
            ),
        )


def test_local_native_eight_reserves_real_post_verification_write_slots():
    scheduler = Scheduler(config(False, 8))
    request = RequestSpec()
    request.request_id = "local"
    request.tokens = list(range(7))
    scheduler.submit_requests([request])
    scheduler.next_execution_plan()
    advance(scheduler, event("ExtendResult", request_id="local", tokens=[100]))
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.decode_input_tokens == 6
    # Anchor E7, at most six accepted outputs, then eight native draft writes:
    # the last possible write is position20, crossing the 16-token boundary.
    assert len(batch.block_tables["full_attention"][0]) * 8 >= 7 + 6 + 8


def test_admission_opportunities_rotate_after_served_request():
    scheduler = seed(2, 7)
    pending(scheduler, "r0", "s0", 0)
    assert scheduler.remote_draft_requests()[0].request_id == "r1"


def test_request_id_reuse_rejects_old_session_candidates():
    scheduler = seed(1, 7)
    old_prefix = pending(scheduler, "r0", "old-session", 0)
    advance(scheduler, event("Abort", request_id="r0"))
    scheduler.next_execution_plan()
    replacement = RequestSpec()
    replacement.request_id = "r0"
    replacement.tokens = list(range(7))
    scheduler.submit_requests([replacement])
    scheduler.next_execution_plan()
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[100]))
    new_prefix = pending(scheduler, "r0", "new-session", 0)
    assert new_prefix == old_prefix
    ready(scheduler, "r0", "old-session", old_prefix)
    assert descriptor(scheduler, "r0").status == "pending"
    ready(scheduler, "r0", "new-session", new_prefix)
    assert scheduler.next_execution_plan().forward[0].decode_input_tokens == 6


def test_rejected_speculative_rows_never_publish_as_prefix_history():
    scheduler = seed(1, 7)
    prefix = pending(scheduler, "r0", "s0", 0)
    ready(scheduler, "r0", "s0", prefix)
    scheduler.next_execution_plan()
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[999]))
    advance(scheduler, event("Finish", request_id="r0"))
    scheduler.next_execution_plan()
    replacement = RequestSpec()
    replacement.request_id = "reuse"
    replacement.tokens = [*range(7), 100, 999]
    scheduler.submit_requests([replacement])
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.extend_prefix_lens == [8]
    assert batch.input_ids == [999]


def test_old_timeout_cannot_clear_a_newer_job_in_same_session():
    scheduler = seed(1, 7)
    old = pending(scheduler, "r0", "s0", 0)
    scheduler.next_execution_plan()
    advance(scheduler, event("ExtendResult", request_id="r0", tokens=[150]))
    # Worker failure resolves the held admission; its healthy session can update.
    advance(
        scheduler,
        event(
            "RemoteDraftUnavailable",
            request_id="r0",
            session_id="s0",
            endpoint=old[0],
            anchor_id=old[1],
        ),
    )
    new = pending(scheduler, "r0", "s0", 1)
    advance(
        scheduler,
        event(
            "RemoteDraftUnavailable",
            request_id="r0",
            session_id="s0",
            endpoint=old[0],
            anchor_id=old[1],
        ),
    )
    assert descriptor(scheduler, "r0").status == "pending"
    ready(scheduler, "r0", "s0", new)
    assert scheduler.next_execution_plan().forward[0].decode_input_tokens == 6


@pytest.mark.parametrize("defer_ms", [0, -1])
def test_zero_deferral_is_immediate_fallback_and_negative_is_rejected(defer_ms):
    cfg = config(True, 0)
    cfg.remote_draft_max_defer_ms = defer_ms
    if defer_ms < 0:
        with pytest.raises(ValueError, match="non-negative deferral"):
            Scheduler(cfg)
        return
    scheduler = Scheduler(cfg)
    request = RequestSpec()
    request.request_id = "zero"
    request.tokens = list(range(7))
    scheduler.submit_requests([request])
    scheduler.next_execution_plan()
    advance(scheduler, event("ExtendResult", request_id="zero", tokens=[100]))
    pending(scheduler, "zero", "s0", 0)
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.request_ids == ["zero"]
    assert batch.decode_input_tokens == 1
    assert batch.decode_input_ids == [100]


@pytest.mark.parametrize("width", [1, 6, 8])
def test_local_mixed_batch_reports_actual_decode_width(width):
    cfg = config(False, 8 if width == 6 else 0)
    cfg.decode_input_tokens = width
    cfg.enable_mixed_prefill_decode = True
    scheduler = Scheduler(cfg)
    decoding = RequestSpec()
    decoding.request_id = "decode"
    decoding.tokens = list(range(7))
    scheduler.submit_requests([decoding])
    assert scheduler.next_execution_plan().forward[0].decode_input_tokens == 0
    advance(scheduler, event("ExtendResult", request_id="decode", tokens=[100]))
    prefill = RequestSpec()
    prefill.request_id = "prefill"
    prefill.tokens = [50, 51, 52]
    scheduler.submit_requests([prefill])
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.request_ids == ["prefill", "decode"]
    assert batch.input_lengths == [3, width]
    assert batch.decode_input_tokens == width
