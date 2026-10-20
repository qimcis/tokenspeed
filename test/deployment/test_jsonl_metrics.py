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

"""CPU-only behavioral checks for asynchronous JSONL metric delivery."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tokenspeed.runtime.metrics.events import MetricEvent
from tokenspeed.runtime.metrics.jsonl import JsonlMetricSink, validate_config


def event(index=0, *, labels=None, operation="inc", weight=1):
    return MetricEvent(
        timestamp_ns=index,
        metric="tokenspeed:test_total",
        operation=operation,
        value=0.125,
        labels={} if labels is None else labels,
        weight=weight,
    )


def records(sink):
    return [json.loads(line) for line in sink.path.read_text().splitlines()]


def test_preserves_observation_semantics_and_label_snapshot(tmp_path):
    sink = JsonlMetricSink({"output_dir": str(tmp_path)})
    labels = {"role": "prefill"}
    sink.emit(event(1, labels=labels, operation="observe", weight=3))
    labels["role"] = "decode"
    sink.emit(event(2, labels=labels, operation="set"))
    sink.emit(event(3))
    sink.close(2)
    assert records(sink) == [
        {
            "timestamp_ns": 1,
            "metric": "tokenspeed:test_total",
            "operation": "observe",
            "value": 0.125,
            "labels": {"role": "prefill"},
            "weight": 3,
        },
        {
            "timestamp_ns": 2,
            "metric": "tokenspeed:test_total",
            "operation": "set",
            "value": 0.125,
            "labels": {"role": "decode"},
            "weight": 1,
        },
        {
            "timestamp_ns": 3,
            "metric": "tokenspeed:test_total",
            "operation": "inc",
            "value": 0.125,
            "labels": {},
            "weight": 1,
        },
    ]
    assert sink.stats.written_events == 3
    assert sink.stats.dropped_events == 0
    assert sink.stats.closed


def test_independent_producers_do_not_overwrite_files(tmp_path):
    first = JsonlMetricSink({"output_dir": str(tmp_path)})
    second = JsonlMetricSink({"output_dir": str(tmp_path)})
    first.emit(event(1))
    second.emit(event(2))
    first.close(2)
    second.close(2)
    assert first.path != second.path
    assert records(first)[0]["timestamp_ns"] == 1
    assert records(second)[0]["timestamp_ns"] == 2


def test_concurrent_overflow_accounts_for_every_event(tmp_path, caplog):
    sink = JsonlMetricSink({"output_dir": str(tmp_path), "queue_size": 1})
    barrier = threading.Barrier(8)

    def produce(worker):
        barrier.wait()
        for index in range(1000):
            sink.emit(event(worker * 1000 + index))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(produce, range(8)))
    sink.close(2)
    stats = sink.stats
    assert stats.written_events + stats.dropped_events == 8000
    assert stats.dropped_events > 0
    assert stats.queued_events == 0
    assert stats.write_errors == 0
    written = records(sink)
    assert len(written) == stats.written_events
    assert len({row["timestamp_ns"] for row in written}) == len(written)
    assert "dropped=" in caplog.text


def test_close_races_with_emit_and_is_idempotent(tmp_path):
    sink = JsonlMetricSink({"output_dir": str(tmp_path), "queue_size": 64})
    barrier = threading.Barrier(5)

    def produce(worker):
        barrier.wait()
        for index in range(500):
            sink.emit(event(worker * 500 + index))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(produce, worker) for worker in range(4)]
        barrier.wait()
        sink.close(2)
        for future in futures:
            future.result()
    sink.close(2)
    assert sink.stats.written_events + sink.stats.dropped_events == 2000
    before = sink.path.read_text()
    sink.emit(event(2000))
    sink.close(2)
    assert sink.path.read_text() == before
    assert sink.stats.written_events + sink.stats.dropped_events == 2001


def test_zero_budget_close_returns_without_drain_wait(tmp_path):
    sink = JsonlMetricSink({"output_dir": str(tmp_path), "queue_size": 8192})
    for index in range(1000):
        sink.emit(event(index))
    started = time.monotonic()
    sink.close(0)
    assert time.monotonic() - started < 0.5
    assert sink.stats.closed
    # Await eventual cleanup only to leave no worker behind in the test process.
    sink._thread.join(2)
    assert not sink._thread.is_alive()
    assert sink.stats.written_events + sink.stats.dropped_events == 1000


def test_output_failure_is_observable_and_does_not_reopen(tmp_path, caplog):
    output = tmp_path / "not-a-directory"
    output.write_text("unchanged")
    sink = JsonlMetricSink({"output_dir": str(output)})
    sink.emit(event(1))
    sink.close(2)
    assert sink.stats.write_errors == 1
    assert sink.stats.dropped_events == 1
    sink.emit(event(2))
    sink.close(2)
    assert sink.stats.dropped_events == 2
    assert output.read_text() == "unchanged"
    assert "JSONL metric output failed" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"output_dir": "relative/path"},
        {"output_dir": "/tmp", "queue_size": True},
        {"output_dir": "/tmp", "queue_size": 0},
        {"output_dir": "/tmp", "queue_size": 1.5},
        {"output_dir": "/tmp", "queue_size": 1_000_001},
        {"output_dir": "/tmp", "unknown": "ignored"},
        {"output_dir": "/tmp\x00"},
    ],
)
def test_config_rejects_invalid_or_unsupported_settings(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_validation_has_no_filesystem_side_effects(tmp_path):
    output = tmp_path / "not-created"
    assert validate_config({"output_dir": str(output), "queue_size": 4}) == {
        "output_dir": str(output),
        "queue_size": 4,
    }
    assert not output.exists()
