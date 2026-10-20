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

"""CPU checks use event-gated fake transport with the real fanout barrier."""

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tokenspeed.runtime.epd.entities import EmbeddingChunk, EmbeddingTransferInfo
from tokenspeed.runtime.epd.mooncake.fanout import notify_fanout, transfer_fanout


def receivers():
    first = EmbeddingTransferInfo(
        1, "host", 1, "first", 1, 0, 1, 1, "torch.bfloat16", False, 2, span=1
    )
    return [first, dataclasses.replace(first, mooncake_session_id="second")]


def chunk():
    return EmbeddingChunk(1, 1, 1, 1, "torch.bfloat16", 2)


@pytest.mark.parametrize("raises", [True, False])
def test_failed_receiver_cannot_release_source_while_sibling_write_runs(raises):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def send(info, payload):
        if info.mooncake_session_id == "first":
            if raises:
                raise RuntimeError("transport failed")
            return -1
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test did not release blocked transfer")
        finished.set()
        return 0

    with (
        ThreadPoolExecutor(max_workers=2) as writes,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        result = caller.submit(transfer_fanout, writes, send, receivers(), chunk())
        try:
            assert entered.wait(5)
            assert not result.done()
            assert not finished.is_set()
        finally:
            release.set()
        failures = result.result(timeout=5)
        assert len(failures) == 1
        assert "first" in failures[0]
        assert finished.is_set()
        # One failed room does not poison the queue's shared executor.
        assert (
            transfer_fanout(writes, lambda info, payload: 0, receivers(), chunk()) == []
        )


def test_failed_submission_is_reported_without_raising():
    writes = ThreadPoolExecutor(max_workers=2)
    writes.shutdown()
    failures = transfer_fanout(writes, lambda info, payload: 0, receivers(), chunk())
    assert len(failures) == 2
    assert all("submit failed" in failure for failure in failures)


def test_status_failure_does_not_skip_other_receivers():
    delivered = []

    def notify(info, status):
        if info.mooncake_session_id == "first":
            raise OSError("receiver disconnected")
        delivered.append((info.mooncake_session_id, status))

    failures = notify_fanout(notify, receivers(), 3)
    assert len(failures) == 1
    assert "first" in failures[0]
    assert delivered == [("second", 3)]
