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

"""Encoder telemetry keeps units, bounded state and counter delta semantics."""

from dataclasses import dataclass

import pytest

from tokenspeed.runtime.load_balancing.telemetry import EncoderLoadMetrics
from tokenspeed.runtime.metrics.events import MetricEvent


@dataclass(frozen=True)
class Observation:
    queued_patch_work: int = 0
    unknown_patch_items: int = 0
    in_flight_patch_work: int = 0
    cache_hits: int = 0
    completed_patch_work: int = 0
    failed_patch_work: int = 0


class RecordingSink:
    def __init__(self):
        self.events: list[MetricEvent] = []

    def emit(self, event):
        self.events.append(event)

    def close(self, timeout):
        pass


def test_actual_queue_and_patch_deltas_go_through_metric_reporter():
    sink = RecordingSink()
    metrics = EncoderLoadMetrics(
        {"engine_rank": "0"}, enabled=True, reporters=["jsonl"], event_sink=sink
    )
    metrics.observe(Observation(queued_patch_work=64, unknown_patch_items=2))
    metrics.observe(Observation(in_flight_patch_work=64, cache_hits=1))
    metrics.observe(Observation(cache_hits=1, completed_patch_work=64))
    metrics.observe(
        Observation(cache_hits=1, completed_patch_work=80, failed_patch_work=8)
    )
    completed = [
        e.value for e in sink.events if e.metric.endswith("completed_patch_work_total")
    ]
    assert completed == [64, 16]
    hits = [e.value for e in sink.events if e.metric.endswith("cache_hits_total")]
    assert hits == [1]
    assert all(e.labels == {"engine_rank": "0"} for e in sink.events)
    assert any(
        e.metric.endswith("unknown_patch_items") and e.value == 2 for e in sink.events
    )


def test_counter_reset_rejected_before_any_partial_emission():
    sink = RecordingSink()
    metrics = EncoderLoadMetrics({}, enabled=True, reporters=["jsonl"], event_sink=sink)
    metrics.observe(Observation(cache_hits=4))
    count = len(sink.events)
    with pytest.raises(ValueError, match="decreased"):
        metrics.observe(Observation(queued_patch_work=7, cache_hits=3))
    assert len(sink.events) == count


def test_disabled_observer_emits_nothing():
    sink = RecordingSink()
    metrics = EncoderLoadMetrics(
        {}, enabled=False, reporters=["jsonl"], event_sink=sink
    )
    metrics.observe(Observation(queued_patch_work=4, cache_hits=2))
    assert sink.events == []
