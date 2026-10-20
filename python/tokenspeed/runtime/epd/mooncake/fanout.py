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

"""Completion barrier for synchronous embedding writes to multiple receivers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future

from tokenspeed.runtime.epd.entities import EmbeddingChunk, EmbeddingTransferInfo


def transfer_fanout(
    executor: Executor,
    send: Callable[[EmbeddingTransferInfo, EmbeddingChunk], int],
    infos: Sequence[EmbeddingTransferInfo],
    chunk: EmbeddingChunk,
) -> list[str]:
    """Drain every submitted synchronous write before returning failure reasons.

    A terminal room status releases its source ring slot. An exception from one
    receiver must therefore never skip waiting for the other in-flight writes.
    The transport owns write timeouts; this barrier must not time out and release
    memory while a synchronous write still accesses it.
    """
    pending: list[tuple[EmbeddingTransferInfo, Future[int]]] = []
    failures: list[str] = []
    for info in infos:
        try:
            pending.append((info, executor.submit(send, info, chunk)))
        except Exception as exc:
            failures.append(f"{info.mooncake_session_id}: submit failed: {exc}")
    for info, future in pending:
        try:
            result = future.result()
            if result != 0:
                failures.append(
                    f"{info.mooncake_session_id}: mooncake transfer ret={result}"
                )
        except Exception as exc:
            failures.append(f"{info.mooncake_session_id}: transfer raised: {exc}")
    return failures


def notify_fanout(
    notify: Callable[[EmbeddingTransferInfo, int], None],
    infos: Sequence[EmbeddingTransferInfo],
    status: int,
) -> list[str]:
    """Attempt every receiver notification, collecting errors without raising."""
    failures: list[str] = []
    for info in infos:
        try:
            notify(info, status)
        except Exception as exc:
            failures.append(
                f"{info.mooncake_session_id}: status notification failed: {exc}"
            )
    return failures
