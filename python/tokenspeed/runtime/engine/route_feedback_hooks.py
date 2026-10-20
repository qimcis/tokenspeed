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

"""Converge completed host observations before redundant schedulers consume them."""

from __future__ import annotations

import torch.distributed as dist

from tokenspeed.runtime.moe.route_capture import CapturedRoutes
from tokenspeed.runtime.moe.route_feedback import RouteCaptureCadence, RouteObservation


class RouteFeedbackHooks:
    """One completed batch at most; scheduler mutation stays explicit in the loop."""

    def __init__(
        self, *, interval: int, max_requests: int, leader: bool, group, source_rank: int
    ) -> None:
        self.cadence = RouteCaptureCadence(interval=interval, max_requests=max_requests)
        self.leader = leader
        self.group = group
        self.source_rank = source_rank
        self._pending: tuple[RouteObservation, ...] = ()

    def collect(
        self,
        *,
        num_extends: int,
        input_lengths: tuple[int, ...],
        captured: CapturedRoutes | None,
    ) -> None:
        if not self.cadence.begin(num_extends=num_extends, input_lengths=input_lengths):
            return
        if self._pending:
            raise RuntimeError(
                "route feedback must drain before the next completed batch"
            )
        if self.leader and captured is None:
            raise RuntimeError("selected route capture did not produce a result")
        payload = [captured.observations() if self.leader else None]
        if self.group.size() > 1:
            dist.broadcast_object_list(payload, src=self.source_rank, group=self.group)
        self._pending = payload[0]

    def drain(self) -> tuple[RouteObservation, ...]:
        pending, self._pending = self._pending, ()
        return pending
