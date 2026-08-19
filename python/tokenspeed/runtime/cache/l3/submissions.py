# Copyright (c) 2026 LightSeek Foundation

"""Asynchronous L3 submission, result, and presence state."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.cache.l3.errors import (
    L3Error,
    L3ShutdownError,
    L3SubmissionError,
)
from tokenspeed.runtime.cache.l3.store_io import StoreObjectKey
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _future_result(future: Future, context: str):
    """Normalize arbitrary worker failures at the asynchronous boundary."""
    try:
        return future.result()
    except L3Error:
        raise
    except Exception as exc:
        raise L3SubmissionError(
            f"{context} raised {type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class _PendingLoad:
    hashes: tuple[str, ...]
    future: Future
    status: str = "pending"
    error: str | None = None


@dataclass
class _PendingWrite:
    objects: tuple[StoreObjectKey, ...]
    future: Future
    nbytes: int


@dataclass
class _PresenceProbe:
    hashes: tuple[str, ...]
    future: Future


@dataclass
class _ObjectProbe:
    hashes: tuple[str, ...]
    future: Future


class L3SubmissionTracker:
    """Own futures and cache-index observations, independent of Store I/O."""

    def __init__(
        self,
        *,
        io_workers: int,
        presence_ttl: float,
        max_pending_writes: int,
        max_pending_write_bytes: int,
    ) -> None:
        io_pool = None
        try:
            io_pool = ThreadPoolExecutor(
                max_workers=io_workers, thread_name_prefix="tokenspeed-l3-io"
            )
            write_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="tokenspeed-l3-write"
            )
        except (RuntimeError, ValueError) as exc:
            if io_pool is not None:
                io_pool.shutdown(wait=True, cancel_futures=False)
            raise L3SubmissionError("failed to create L3 worker pools") from exc
        self._io_pool = io_pool
        self._write_pool = write_pool
        self._loads: dict[tuple[int, ...], _PendingLoad] = {}
        self._writes: list[_PendingWrite] = []
        self._outcomes: dict[StoreObjectKey, bool] = {}
        self._presence: dict[str, tuple[bool, float]] = {}
        self._probe: _PresenceProbe | None = None
        self._object_probe: _ObjectProbe | None = None
        self._presence_ttl = float(presence_ttl)
        self._max_pending_writes = int(max_pending_writes)
        self._max_pending_write_bytes = int(max_pending_write_bytes)
        self._pending_write_bytes = 0

    def submit_load(
        self,
        key: tuple[int, ...],
        hashes: tuple[str, ...],
        function: Callable[[], Any],
    ) -> None:
        if key in self._loads:
            raise L3SubmissionError(f"duplicate L3 load submission {key}")
        try:
            future = self._io_pool.submit(function)
        except RuntimeError as exc:
            raise L3SubmissionError(f"failed to submit L3 load {key}") from exc
        self._loads[key] = _PendingLoad(hashes=hashes, future=future)

    def submit_write(
        self,
        objects: tuple[StoreObjectKey, ...],
        function: Callable[[], dict[StoreObjectKey, bool]],
        *,
        nbytes: int,
    ) -> bool:
        if not self.can_accept_write(nbytes):
            return False
        try:
            future = self._write_pool.submit(function)
        except RuntimeError as exc:
            raise L3SubmissionError("failed to submit background L3 write") from exc
        self._writes.append(
            _PendingWrite(objects=objects, future=future, nbytes=int(nbytes))
        )
        self._pending_write_bytes += int(nbytes)
        return True

    def can_accept_write(self, nbytes: int) -> bool:
        """Check queue capacity before allocating a pinned write snapshot."""
        self.poll_writes()
        if nbytes <= 0:
            raise L3SubmissionError("L3 write size must be positive")
        if (
            len(self._writes) >= self._max_pending_writes
            or self._pending_write_bytes + nbytes > self._max_pending_write_bytes
        ):
            logger.warning(
                "dropping L3 write snapshot: pending=%s/%s bytes=%s/%s requested=%s",
                len(self._writes),
                self._max_pending_writes,
                self._pending_write_bytes,
                self._max_pending_write_bytes,
                nbytes,
            )
            return False
        return True

    def load_status(self, key: tuple[int, ...]) -> tuple[str, list[str], str | None]:
        submission = self._loads.get(key)
        if submission is None:
            return "failed", [], f"unknown L3 load submission {key}"
        if submission.status == "pending" and submission.future.done():
            try:
                _future_result(submission.future, f"L3 load {key}")
                submission.status = "succeeded"
                self.record_presence(submission.hashes, present=True)
            except L3Error as exc:
                submission.status = "failed"
                submission.error = str(exc)
        hashes = (
            list(dict.fromkeys(submission.hashes))
            if submission.status == "failed"
            else []
        )
        return submission.status, hashes, submission.error

    def acknowledge_load(self, key: tuple[int, ...]) -> None:
        self._loads.pop(key, None)

    def abort_load(self, key: tuple[int, ...]) -> bool:
        """Drop a submission and return whether it was cancelled before running."""
        submission = self._loads.pop(key, None)
        if submission is None:
            return False
        if submission.future.cancel():
            return True
        try:
            _future_result(submission.future, f"aborted L3 load {key}")
        except L3Error:
            pass
        return False

    def poll_writes(self) -> None:
        pending: list[_PendingWrite] = []
        now = time.monotonic()
        for submission in self._writes:
            if not submission.future.done():
                pending.append(submission)
                continue
            try:
                outcome = _future_result(submission.future, "L3 background Store write")
                if not isinstance(outcome, dict):
                    raise L3SubmissionError(
                        "L3 background Store write returned a non-dict outcome"
                    )
            except L3Error as exc:
                logger.warning("L3 background put failed: %s", exc)
                outcome = {value: False for value in submission.objects}
            try:
                self._outcomes.update(outcome)
                for (_group, content_hash, _offset), present in outcome.items():
                    if not present:
                        self._presence[content_hash] = (
                            False,
                            now + self._presence_ttl,
                        )
            except Exception as exc:
                logger.warning(
                    "invalid L3 background put outcome; marking failed: %s", exc
                )
                self._outcomes.update({value: False for value in submission.objects})
            finally:
                self._pending_write_bytes -= submission.nbytes
        self._writes = pending

    def probe_presence(
        self,
        content_hashes: Sequence[str],
        loader: Callable[[tuple[str, ...]], dict[str, bool]],
    ) -> tuple[str, dict[str, bool], str | None]:
        hashes = tuple(dict.fromkeys(value for value in content_hashes if value))
        if not hashes:
            return "ready", {}, None
        now = time.monotonic()
        probe_error = None
        if self._probe is not None:
            if not self._probe.future.done():
                known = {
                    value: self._presence[value][0]
                    for value in hashes
                    if value in self._presence and self._presence[value][0]
                }
                return (
                    ("ready", known, None)
                    if len(known) == len(hashes)
                    else ("pending", {}, None)
                )
            probe = self._probe
            self._probe = None
            try:
                outcome = _future_result(probe.future, "L3 Store existence probe")
                if not isinstance(outcome, dict):
                    raise L3SubmissionError(
                        "L3 Store existence probe returned a non-dict outcome"
                    )
            except L3Error as exc:
                logger.warning("L3 Store existence probe failed: %s", exc)
                outcome = {value: False for value in probe.hashes}
                probe_error = str(exc)
            for value, present in outcome.items():
                self._presence[value] = (bool(present), now + self._presence_ttl)
        missing = [
            value
            for value in hashes
            if value not in self._presence
            or (not self._presence[value][0] and self._presence[value][1] <= now)
        ]
        if missing:
            try:
                future = self._io_pool.submit(loader, tuple(missing))
            except RuntimeError as exc:
                raise L3SubmissionError("failed to submit L3 presence probe") from exc
            self._probe = _PresenceProbe(tuple(missing), future)
            return "pending", {}, None
        return (
            "ready",
            {value: self._presence[value][0] for value in hashes},
            probe_error,
        )

    def record_presence(self, content_hashes: Sequence[str], *, present: bool) -> None:
        expiry = time.monotonic() + self._presence_ttl
        for value in content_hashes:
            if value:
                self._presence[str(value)] = (bool(present), expiry)

    def probe_objects(
        self,
        content_hashes: Sequence[str],
        loader: Callable[[tuple[str, ...]], dict[StoreObjectKey, bool]],
    ) -> tuple[str, dict[StoreObjectKey, bool], str | None]:
        hashes = tuple(dict.fromkeys(value for value in content_hashes if value))
        if not hashes:
            return "ready", {}, None
        if self._object_probe is None:
            try:
                future = self._io_pool.submit(loader, hashes)
            except RuntimeError as exc:
                raise L3SubmissionError("failed to submit L3 object probe") from exc
            self._object_probe = _ObjectProbe(hashes, future)
            return "pending", {}, None
        probe = self._object_probe
        if not probe.future.done():
            return "pending", {}, None
        self._object_probe = None
        try:
            outcome = _future_result(probe.future, "L3 Store object probe")
            if not isinstance(outcome, dict):
                raise L3SubmissionError(
                    "L3 Store object probe returned a non-dict outcome"
                )
            return (
                "ready",
                outcome,
                None,
            )
        except L3Error as exc:
            logger.warning("L3 Store object probe failed: %s", exc)
            return "ready", {}, str(exc)

    def invalidate_presence(self, content_hashes: Sequence[str]) -> None:
        for value in content_hashes:
            self._presence.pop(value, None)

    def outcomes(self) -> dict[StoreObjectKey, bool]:
        self.poll_writes()
        return dict(self._outcomes)

    def acknowledge_outcomes(self, objects: Sequence[StoreObjectKey]) -> None:
        for value in objects:
            self._outcomes.pop(value, None)

    def record_outcomes(self, outcomes: dict[StoreObjectKey, bool]) -> None:
        self._outcomes.update(
            {value: bool(present) for value, present in outcomes.items()}
        )

    def wait(self) -> None:
        futures = [submission.future for submission in self._loads.values()]
        futures.extend(submission.future for submission in self._writes)
        if self._probe is not None:
            futures.append(self._probe.future)
        if self._object_probe is not None:
            futures.append(self._object_probe.future)
        for future in futures:
            try:
                _future_result(future, "L3 shutdown wait")
            except L3Error:
                pass
        self.poll_writes()

    def shutdown(self) -> None:
        self.wait()
        errors = []
        for name, pool in (("I/O", self._io_pool), ("write", self._write_pool)):
            try:
                pool.shutdown(wait=True, cancel_futures=False)
            except RuntimeError as exc:
                errors.append(f"{name} pool: {exc}")
        if errors:
            raise L3ShutdownError("; ".join(errors))

    def clear(self) -> None:
        self._loads.clear()
        self._writes.clear()
        self._pending_write_bytes = 0
        self._outcomes.clear()
        self._presence.clear()
        self._probe = None
        self._object_probe = None
