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

"""Cross-thread notification bounds and lifetime independent of GPU execution."""

import threading

import zmq

from tokenspeed.runtime.draft_pool.transport import _PollWakeup


def test_notifications_coalesce_and_rearm_without_stale_readiness():
    wakeup = _PollWakeup()
    poller = zmq.Poller()
    poller.register(wakeup.fileno(), zmq.POLLIN)
    try:
        for _ in range(10000):
            wakeup.notify()
        assert dict(poller.poll(0))[wakeup.fileno()] & zmq.POLLIN
        wakeup.drain()
        assert poller.poll(0) == []
        wakeup.notify()
        assert dict(poller.poll(0))[wakeup.fileno()] & zmq.POLLIN
        wakeup.drain()
        wakeup.drain()
        assert poller.poll(0) == []
    finally:
        wakeup.close()


def test_cross_thread_notification_interrupts_long_poll():
    wakeup = _PollWakeup()
    started = threading.Event()
    result = []

    def wait():
        poller = zmq.Poller()
        poller.register(wakeup.fileno(), zmq.POLLIN)
        started.set()
        result.extend(poller.poll(10000))

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        assert started.wait(1)
        wakeup.notify()
        thread.join(1)
        assert not thread.is_alive()
        assert dict(result)[wakeup.fileno()] & zmq.POLLIN
    finally:
        wakeup.notify()
        thread.join(1)
        wakeup.close()


def test_shutdown_can_race_with_completion_notification():
    wakeup = _PollWakeup()
    started = threading.Event()
    stop = threading.Event()
    errors = []

    def notify():
        try:
            started.set()
            while not stop.is_set():
                wakeup.notify()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=notify)
    thread.start()
    try:
        assert started.wait(1)
        wakeup.close()
        wakeup.notify()
        wakeup.drain()
        wakeup.close()
    finally:
        stop.set()
        thread.join(1)
        wakeup.close()
    assert not thread.is_alive()
    assert not errors
