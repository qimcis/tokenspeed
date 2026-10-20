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

"""Bounded, observed worker load exported through existing metric reporters.

These measurements retain their actual units. Patch counts are not latency
predictions; no rate, prefix hit length, or free capacity is inferred here.
"""

from __future__ import annotations

from typing import Protocol

from tokenspeed.runtime.metrics.events import MetricEventSink
from tokenspeed.runtime.metrics.reporter import MetricFactory


class EncoderObservation(Protocol):
    queued_patch_work: int
    unknown_patch_items: int
    in_flight_patch_work: int
    cache_hits: int
    completed_patch_work: int
    failed_patch_work: int


class EncoderLoadMetrics:
    """One encoder TP worker's actual queue and completed-work observations.

    Construct only on the reporting rank and reuse for that worker incarnation.
    The observer retains six scalars, regardless of serving duration. Counter
    resets are rejected rather than silently attributing a new incarnation's
    work to an old one. The existing reporter owns I/O queues and backpressure.
    """

    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        reporters: list[str],
        event_sink: MetricEventSink | None = None,
        registry=None,
    ) -> None:
        self.enabled = enabled
        factory = MetricFactory(
            enabled=enabled,
            reporters=reporters,
            event_sink=event_sink,
            registry=registry,
        )
        self._gauges = {
            name: factory.gauge(
                name=f"tokenspeed:encoder_{name}",
                documentation=documentation,
                labelnames=tuple(labels),
                multiprocess_mode="livesum",
            ).labels(**labels)
            for name, documentation in (
                ("queued_patch_work", "Known input patches waiting for encoding."),
                (
                    "unknown_patch_items",
                    "Queued or active items without measured input patch counts.",
                ),
                (
                    "in_flight_patch_work",
                    "Known input patches in the active encoder forward.",
                ),
            )
        }
        self._counters = {
            name: factory.counter(
                name=f"tokenspeed:encoder_{name}_total",
                documentation=documentation,
                labelnames=tuple(labels),
            ).labels(**labels)
            for name, documentation in (
                ("cache_hits", "Encoder items served by the embedding cache."),
                ("completed_patch_work", "Known input patches successfully encoded."),
                ("failed_patch_work", "Known input patches whose encoding failed."),
            )
        }
        self._last_counters = (0, 0, 0)

    def observe(self, observation: EncoderObservation) -> None:
        """Publish one actual snapshot, retaining no per-request history."""
        if not self.enabled:
            return
        gauges = (
            observation.queued_patch_work,
            observation.unknown_patch_items,
            observation.in_flight_patch_work,
        )
        counters = (
            observation.cache_hits,
            observation.completed_patch_work,
            observation.failed_patch_work,
        )
        if any(type(value) is not int or value < 0 for value in (*gauges, *counters)):
            raise ValueError("encoder load observations must be nonnegative integers")
        if any(now < old for now, old in zip(counters, self._last_counters)):
            raise ValueError("encoder counters decreased within one incarnation")
        for metric, value in zip(self._gauges.values(), gauges):
            metric.set(value)
        for metric, now, old in zip(
            self._counters.values(), counters, self._last_counters
        ):
            if now != old:
                metric.inc(now - old)
        self._last_counters = counters
