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

"""Actual C++ scheduler/controller/TCP integration with CPU feature inference."""

import threading
import time
from test.runtime.test_remote_drafting_acceptance import pool  # noqa: F401

import torch
from tokenspeed_scheduler import (
    CacheGroupConfig,
    CacheRetention,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

from tokenspeed.runtime.draft_pool.controller import RemoteDraftController
from tokenspeed.runtime.draft_pool.features import (
    CompletedFeatureExport,
    FeatureExportDescriptor,
)


class SnapshotDevice:
    """GPU gather/copy is replaced; C++ pins and the complete wire path remain."""

    def __init__(self):
        self.completed = []
        self.released = []
        self.cancelled = set()

    def queue_remote_draft_exports(self, snapshots):
        for snapshot in snapshots:
            assert snapshot.block_tables["dflash2_projected_features"]
            self.completed.append(
                CompletedFeatureExport(
                    FeatureExportDescriptor(
                        ticket_id=snapshot.ticket_id,
                        session_id=snapshot.session_id,
                        start=snapshot.start,
                        end=snapshot.endpoint,
                        anchor_token=snapshot.anchor_id,
                    ),
                    torch.ones(
                        snapshot.endpoint - snapshot.start, 4, dtype=torch.bfloat16
                    ),
                )
            )
            self.released.append(snapshot.ticket_id)

    def poll_remote_draft_exports(self):
        result = self.completed, self.released
        self.completed, self.released = [], []
        return result

    def cancel_remote_draft_exports(self, session_id):
        self.cancelled.add(session_id)

    def close_remote_draft_exports(self):
        self.completed.clear()


def advance(scheduler, events):
    update = ExecutionEvent()
    for event in events:
        update.add_event(event)
    scheduler.advance(update)


def result(request_id, tokens):
    event = ForwardEvent.ExtendResult()
    event.request_id, event.tokens = request_id, tokens
    return event


def make_scheduler():
    config = SchedulerConfig()
    config.prefix_granularity = 8
    config.max_scheduled_tokens = 128
    config.max_batch_size = 3
    config.num_device_pages = 256
    config.num_host_pages = 0
    config.disable_l2_cache = True
    config.decode_input_tokens = 6
    config.remote_draft_enabled = True
    config.remote_draft_min_ready = 2
    config.remote_draft_max_defer_ms = 5000
    config.remote_draft_feature_group = "dflash2_projected_features"
    config.cache_groups = [
        CacheGroupConfig("full_attention", 8, 256),
        CacheGroupConfig(
            "dflash2_projected_features", 8, 256, CacheRetention.SlidingWindow, 8
        ),
    ]
    scheduler = Scheduler(config)
    requests = []
    for index in range(3):
        request = RequestSpec()
        request.request_id = f"r{index}"
        request.tokens = list(range(index * 10, index * 10 + 7))
        request.max_new_tokens = 100
        requests.append(request)
    scheduler.submit_requests(requests)
    assert scheduler.next_execution_plan().forward[0].num_extends() == 3
    advance(scheduler, [result(f"r{i}", [100 + i]) for i in range(3)])
    return scheduler


def test_confirmed_remote_prefix_survives_other_ar_work_and_enters_real_verifier(pool):
    start, connect = pool
    service = start("tcp://127.0.0.1:*", True)
    inbox = connect(service.service.bound_endpoint, b"controller")
    device, scheduler = SnapshotDevice(), make_scheduler()
    controller = RemoteDraftController(
        enabled=True,
        device=device,
        client=inbox.client,
        contract=service.engines[0].contract,
        max_sessions=2,
        max_exports=2,
        max_defer_ms=5000,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        leader_rank=0,
    )
    active = {"r0", "r1", "r2"}
    ar_batches = []

    def step():
        advance(scheduler, controller.poll_ready_events(scheduler, active, False))
        controller.queue_exports(scheduler)
        batch = scheduler.next_execution_plan().forward[0]
        if batch.request_ids and batch.decode_input_tokens == 1:
            ar_batches.append(tuple(batch.request_ids))
            advance(scheduler, [result(rid, [106]) for rid in batch.request_ids])
            controller.after_commit(active)
        return batch

    deadline = time.monotonic() + 5
    while not service.engines[0].started.is_set():
        assert time.monotonic() < deadline
        step()
        threading.Event().wait(0.002)
    prefixes = {
        d.request_id: d.endpoint
        for d in scheduler.remote_draft_requests()
        if d.status == "pending"
    }
    assert prefixes
    for _ in range(5):
        step()
    current = {d.request_id: d.endpoint for d in scheduler.remote_draft_requests()}
    assert all(current[rid] == endpoint for rid, endpoint in prefixes.items())
    assert ar_batches  # target used other requests while the worker was blocked
    service.engines[0].release.set()
    verified = None
    while verified is None:
        assert time.monotonic() < deadline
        batch = step()
        if batch.request_ids and batch.decode_input_tokens == 6:
            verified = batch
        threading.Event().wait(0.002)
    assert all(len(row) == 6 for row in verified.spec_candidate_ids)
    assert verified.decode_input_ids == [row[0] for row in verified.spec_candidate_ids]
    advance(
        scheduler,
        [
            result(rid, [row[1], 107])
            for rid, row in zip(
                verified.request_ids, verified.spec_candidate_ids, strict=True
            )
        ],
    )
    controller.after_commit(active)
    controller.close()
