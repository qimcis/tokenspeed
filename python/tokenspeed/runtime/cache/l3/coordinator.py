# Copyright (c) 2026 LightSeek Foundation

"""TP-consistent L3 admission, rollback, and Store-index coordination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tokenspeed.runtime.cache.l3.errors import L3Error, L3SubmissionError
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed_scheduler import Cache, ExecutionEvent

logger = get_colorful_logger(__name__)


@dataclass
class _DeferredLoad:
    plan: object
    zero_event: object | None
    op_ids: tuple[int, ...]
    submit_error: str | None


class L3LoadCoordinator:
    """Keep L3 scheduler admission unanimous across attention-TP ranks."""

    def __init__(
        self,
        *,
        executor,
        scheduler,
        tp_rank: int,
        tp_size: int,
        cpu_group,
        publish_kv_events: Callable[[], None],
    ) -> None:
        self.executor = executor
        self.scheduler = scheduler
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.cpu_group = cpu_group
        self.publish_kv_events = publish_kv_events
        self._deferred: _DeferredLoad | None = None

    @property
    def has_deferred_load(self) -> bool:
        return self._deferred is not None

    def _gather(self, local):
        if self.tp_size == 1:
            return [local]
        gathered = [None] * self.tp_size
        try:
            dist.all_gather_object(gathered, local, group=self.cpu_group)
        except RuntimeError as exc:
            raise L3SubmissionError("L3 TP consensus collective failed") from exc
        return gathered

    def _any_rank(self, local_has_work: bool) -> bool:
        if self.tp_size == 1:
            return local_has_work
        flag = torch.tensor([int(local_has_work)], dtype=torch.int32)
        try:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.cpu_group)
        except RuntimeError as exc:
            raise L3SubmissionError("L3 TP work collective failed") from exc
        return bool(flag.item())

    @staticmethod
    def _load_op_ids(plan) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                int(op_id)
                for op in plan.cache
                if isinstance(op, Cache.StoreLoadOp)
                for op_id in op.op_ids
            )
        )

    @staticmethod
    def _load_hashes(plan) -> list[str]:
        return list(
            dict.fromkeys(
                str(value)
                for op in plan.cache
                if isinstance(op, Cache.StoreLoadOp)
                for values in op.content_hashes
                for value in values
                if value
            )
        )

    @staticmethod
    def _write_objects(plan) -> list[tuple[int, str, int]]:
        objects = []
        for op in plan.cache:
            if not isinstance(op, Cache.WriteBackOp):
                continue
            offset_groups = list(op.cache_block_offsets)
            for index, (groups, hashes) in enumerate(
                zip(op.group_ids, op.content_hashes)
            ):
                offsets = (
                    offset_groups[index]
                    if index < len(offset_groups)
                    else [0] * len(hashes)
                )
                objects.extend(
                    (int(group), str(value), int(offset))
                    for group, value, offset in zip(groups, hashes, offsets)
                    if value
                )
        return list(dict.fromkeys(objects))

    def _update_store_keys(
        self, objects: list[tuple[int, str, int]], present: list[bool]
    ) -> None:
        self.scheduler.update_store_keys(
            [value[0] for value in objects],
            [value[1] for value in objects],
            [value[2] for value in objects],
            present,
        )

    def submit(self, plan, zero_event) -> bool:
        """Submit L3 work and return whether execution must be deferred."""
        l3_ops = [
            op
            for op in plan.cache
            if isinstance(op, (Cache.WriteBackOp, Cache.StoreLoadOp))
        ]
        op_ids = self._load_op_ids(plan)
        submit_error = None
        if l3_ops:
            try:
                submitted = self.executor.submit_plan(
                    SimpleNamespace(cache=l3_ops), cache_zero_event=zero_event
                )
                if tuple(submitted) != op_ids:
                    raise L3SubmissionError(
                        "L3 executor returned mismatched Store load op ids: "
                        f"expected={op_ids} submitted={tuple(submitted)}"
                    )
            except L3Error as exc:
                submit_error = str(exc)
                write_objects = self._write_objects(SimpleNamespace(cache=l3_ops))
                if write_objects:
                    self.executor.record_store_index_outcomes(
                        {value: False for value in write_objects}
                    )
                if not op_ids:
                    logger.warning(
                        "L3 background Store write submission failed: %s", exc
                    )
        if not op_ids:
            return False
        self._deferred = _DeferredLoad(plan, zero_event, op_ids, submit_error)
        return True

    def resolve_deferred(self):
        """Return ``(plan, zero_event, admitted_ops)`` once every rank succeeds."""
        deferred = self._deferred
        if deferred is None:
            raise L3SubmissionError("no deferred L3 load to resolve")
        if deferred.submit_error is not None:
            local = {
                "status": "failed",
                "hashes": self._load_hashes(deferred.plan),
                "error": deferred.submit_error,
            }
        else:
            status, hashes, error = self.executor.load_submission_status(
                deferred.op_ids
            )
            local = {"status": status, "hashes": hashes, "error": error}
        gathered = self._gather(local)
        if any(item["status"] == "pending" for item in gathered):
            return None
        if any(item["status"] == "failed" for item in gathered):
            self._rollback(deferred, gathered)
            return None
        self.executor.acknowledge_load_submission(deferred.op_ids)
        self._deferred = None
        return deferred.plan, deferred.zero_event, len(deferred.op_ids)

    def _rollback(self, deferred: _DeferredLoad, gathered) -> None:
        failed_hashes = list(
            dict.fromkeys(
                value for item in gathered for value in item["hashes"] if value
            )
        )
        abort_error = None
        try:
            self.executor.abort_load_submission(deferred.op_ids)
        except L3Error as exc:
            abort_error = exc
        finally:
            if deferred.zero_event is not None:
                try:
                    deferred.zero_event.synchronize()
                except RuntimeError as exc:
                    raise L3SubmissionError(
                        "failed to synchronize cache-page zeroing during rollback"
                    ) from exc
        if abort_error is not None:
            raise L3SubmissionError("L3 rollback cleanup failed") from abort_error
        if failed_hashes:
            self.executor.record_presence(failed_hashes, present=False)
            self.scheduler.update_store_index(
                failed_hashes, [False] * len(failed_hashes)
            )
        failure = ExecutionEvent()
        for op_id in deferred.op_ids:
            event = Cache.StoreLoadFailedEvent()
            event.op_id = op_id
            failure.add_event(event)
        self.scheduler.advance(failure)
        self.publish_kv_events()
        if self.tp_rank == 0:
            errors = [item["error"] for item in gathered if item["error"]]
            logger.warning(
                "L3 Store load failed for op_ids=%s; requeued for recompute: %s",
                deferred.op_ids,
                "; ".join(errors) or "unknown Store error",
            )
        self._deferred = None

    def refresh_store_index(self) -> bool:
        hashes = [str(value) for value in self.scheduler.store_probe_hashes()]
        if not hashes:
            return True
        status, outcome, error = self.executor.probe_store_objects(hashes)
        gathered = self._gather({"status": status, "outcome": outcome, "error": error})
        if any(item["status"] == "pending" for item in gathered):
            return False
        if any(item["error"] for item in gathered):
            self.scheduler.update_store_index(hashes, [False] * len(hashes))
            self.executor.record_presence(hashes, present=False)
            return True
        objects = sorted(set().union(*(item["outcome"] for item in gathered)))
        present = [
            all(bool(item["outcome"].get(value, False)) for item in gathered)
            for value in objects
        ]
        if objects:
            self._update_store_keys(objects, present)
        missing = {value[1] for value, exists in zip(objects, present) if not exists}
        if missing:
            self.executor.record_presence(sorted(missing), present=False)
        return True

    def commit_store_index_outcomes(self) -> None:
        local = self.executor.peek_store_index_outcomes()
        if not self._any_rank(bool(local)):
            return
        gathered = self._gather(local)
        common = set(gathered[0])
        for outcome in gathered[1:]:
            common.intersection_update(outcome)
        if not common:
            return
        objects = sorted(common)
        present = [
            all(bool(outcome[value]) for outcome in gathered) for value in objects
        ]
        self._update_store_keys(objects, present)
        self.executor.acknowledge_store_index_outcomes(objects)
        failed = [value[1] for value, success in zip(objects, present) if not success]
        if failed:
            self.executor.record_presence(failed, present=False)
