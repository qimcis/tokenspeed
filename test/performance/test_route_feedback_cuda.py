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

"""Explicitly opt-in CUDA qualification; never part of CPU implementation tests."""

import os

import pytest
import torch

from tokenspeed.runtime.moe.route_capture import RouteCapture

pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_RUN_ROUTE_FEEDBACK_GPU") != "1",
    reason="set TOKENSPEED_RUN_ROUTE_FEEDBACK_GPU=1 on an authorized GPU host",
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pinned_route_copy_uses_existing_completion_event(dtype):
    if not torch.cuda.is_available():
        pytest.fail("explicit GPU route qualification requires CUDA")
    capture = RouteCapture(
        layer_widths=((3, 2),), interval=1, max_requests=2, pin_memory=True
    )
    pointer = capture.buffers[3].data_ptr()
    stream = torch.cuda.Stream()
    previous_debug = torch.cuda.get_sync_debug_mode()
    for offset in (0, 2):
        with torch.cuda.stream(stream):
            # All allocations and the source's H2D happen before the guarded path.
            ids = torch.tensor(
                [[1 + offset, 2], [3, 4 + offset]], device="cuda", dtype=dtype
            )
            torch.cuda.set_sync_debug_mode("error")
            try:
                capture.begin(num_extends=0, input_lengths=(1, 1))
                capture.observe(3, ids)
                result = capture.finish(
                    request_ids=("a", "b"),
                    generations=(1, 2),
                    computed_frontiers=(10 + offset, 20 + offset),
                )
                completion = torch.cuda.Event()
                completion.record()
            finally:
                torch.cuda.set_sync_debug_mode(previous_debug)
        # Exactly the existing commit wait; capture does not add one of its own.
        completion.synchronize()
        observed = result.observations()
        assert observed[0].expert_keys == tuple(
            sorted({(3 << 32) | (1 + offset), (3 << 32) | 2})
        )
        assert observed[1].expert_keys == tuple(
            sorted({(3 << 32) | 3, (3 << 32) | (4 + offset)})
        )
        assert capture.buffers[3].is_pinned()
        assert capture.buffers[3].data_ptr() == pointer
