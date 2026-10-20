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

"""Bounded, process-local JSONL delivery for existing metric observations."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from tokenspeed.runtime.metrics.events import MetricConfig, MetricEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JsonlStats:
    """Process-local delivery counters, including events rejected after close."""

    written_events: int
    dropped_events: int
    write_errors: int
    queued_events: int
    closed: bool


def validate_config(config: MetricConfig) -> MetricConfig:
    """Validate output_dir and optional queue_size without allocating resources."""
    if type(config) is not dict or set(config) - {"output_dir", "queue_size"}:
        raise ValueError("JSONL metrics accepts only output_dir and queue_size")
    output_dir = config.get("output_dir")
    if (
        type(output_dir) is not str
        or not output_dir.strip()
        or "\x00" in output_dir
        or not Path(output_dir).is_absolute()
    ):
        raise ValueError("JSONL output_dir must be an absolute directory path")
    queue_size = config.get("queue_size", 8192)
    if type(queue_size) is not int or not 1 <= queue_size <= 1_000_000:
        raise ValueError("JSONL queue_size must be an integer in [1, 1000000]")
    return {"output_dir": output_dir, "queue_size": queue_size}


class JsonlMetricSink:
    """Write one uniquely named file per producer, with best-effort delivery.

    The worker owns filesystem access and JSON serialization. close() bounds the
    caller's wait, but cannot interrupt a blocked filesystem write; the daemon
    worker releases its file when that write returns. No resources reopen after
    closing begins. Files contain MetricEvent fields without aggregation.
    """

    def __init__(self, config: MetricConfig) -> None:
        normalized = validate_config(config)
        self.path = Path(normalized["output_dir"]) / (
            f"metrics-{os.getpid()}-{uuid.uuid4().hex}.jsonl"
        )
        self._queue: queue.Queue[MetricEvent] = queue.Queue(normalized["queue_size"])
        self._lock = threading.Lock()
        self._closing = False
        self._deadline = math.inf
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._reported_dropped = 0
        self._reported_errors = 0
        self._last_report = -math.inf
        self._thread = threading.Thread(
            target=self._run, name="tokenspeed-jsonl-metrics", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> JsonlStats:
        """Return a consistent counter snapshot; queued excludes an active write."""
        with self._lock:
            return JsonlStats(
                self._written,
                self._dropped,
                self._errors,
                self._queue.qsize(),
                self._closing,
            )

    def emit(self, event: MetricEvent) -> None:
        """Snapshot labels and enqueue without serialization, I/O, or backpressure."""
        with self._lock:
            if self._closing:
                self._dropped += 1
                return
            snapshot = replace(event, labels=dict(event.labels))
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                self._dropped += 1

    def close(self, timeout: float) -> None:
        """Stop admission and drain until the first close deadline; safe to repeat."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("JSONL close timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._lock:
            if not self._closing:
                self._closing = True
                self._deadline = deadline
        self._thread.join(max(0.0, deadline - time.monotonic()))

    def _discard_pending(self) -> None:
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        with self._lock:
            self._dropped += count

    def _report(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_report < 5.0:
            return
        stats = self.stats
        if (
            stats.dropped_events == self._reported_dropped
            and stats.write_errors == self._reported_errors
        ):
            return
        self._reported_dropped = stats.dropped_events
        self._reported_errors = stats.write_errors
        self._last_report = now
        logger.warning(
            f"JSONL metrics: dropped={stats.dropped_events}, "
            f"write_errors={stats.write_errors}, written={stats.written_events}"
        )

    def _run(self) -> None:
        active: MetricEvent | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8", buffering=1) as output:
                while True:
                    with self._lock:
                        closing = self._closing
                        deadline = self._deadline
                    if closing and (
                        time.monotonic() >= deadline or self._queue.empty()
                    ):
                        break
                    try:
                        active = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        self._report()
                        continue
                    # No conversion or aggregation: observation weights and
                    # operation semantics remain those of the producer.
                    record = {
                        "timestamp_ns": active.timestamp_ns,
                        "metric": active.metric,
                        "operation": active.operation,
                        "value": active.value,
                        "labels": active.labels,
                        "weight": active.weight,
                    }
                    output.write(json.dumps(record, allow_nan=False) + "\n")
                    with self._lock:
                        self._written += 1
                    active = None
                    self._report()
        except Exception:
            with self._lock:
                self._errors += 1
                self._dropped += int(active is not None)
                self._closing = True
                self._deadline = time.monotonic()
            logger.exception("JSONL metric output failed; this producer stops writing")
        finally:
            self._discard_pending()
            self._report(force=True)


def create_sink(config: MetricConfig) -> JsonlMetricSink:
    """Create one process-local asynchronous writer from validated settings."""
    return JsonlMetricSink(config)


def register() -> None:
    """Declare the JSONL factory through the existing runtime plugin interface."""
    from tokenspeed.runtime.metrics.registry import (
        METRIC_REPORTER_API_VERSION,
        register_metric_reporter,
    )

    if METRIC_REPORTER_API_VERSION != 1:
        raise RuntimeError("Unsupported metric reporter API for JSONL")
    register_metric_reporter(
        "jsonl",
        validate=validate_config,
        create_sink=create_sink,
        start_node=None,
        override=False,
    )
