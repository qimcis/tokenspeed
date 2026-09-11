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

"""Target projection and independent, bounded feature-export completion.

Request history lives in the LCM feature group. The tensors here are per-forward
metadata or in-flight snapshots; neither is an additional prefix cache. Target
token completion never waits for the snapshot's device-to-host copy.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
    retained_feature_interval,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.layers.attention.kv_cache.projected_features import (
        ProjectedFeatureCache,
    )
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


def committed_feature_interval(endpoint: int, window_left: int) -> tuple[int, int]:
    """Return prior-input positions needed by an anchor at ``endpoint``.

    The correction/bonus token at the endpoint has not passed through the target
    yet. A native attention window of 2048 has ``window_left=2047``.
    """
    return retained_feature_interval(endpoint, window_left)


def load_target_projection(
    config,
    model_path: str,
    revision: str | None,
    load_config: LoadConfig,
    device: str,
    target_model,
):
    """Load only the draft checkpoint's BF16 projector and wire target taps.

    Args:
        config: The pinned DFlash2 checkpoint configuration.
        model_path: Local checkpoint directory or model repository.
        revision: Checkpoint revision selected by the serving configuration.
        load_config: Existing checkpoint loader settings.
        device: Target execution device.
        target_model: Already-loaded target whose capture hooks are configured.

    Returns:
        A frozen ``DFlashTargetProjection`` with no draft layers or vocabulary.
    """
    from tokenspeed.runtime.execution.drafter.dflash import (
        configure_dflash_target_capture,
    )
    from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
    from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
    from tokenspeed.runtime.models.dflash import DFlashTargetProjection

    with set_default_torch_dtype(torch.bfloat16), torch.device(device):
        projector = DFlashTargetProjection(config)
    parameters = dict(projector.named_parameters())

    def projection_name(name: str) -> str:
        return name.removeprefix("model.")

    loader = DefaultModelLoader(load_config)
    source = DefaultModelLoader.Source(
        model_or_path=model_path,
        revision=revision,
        prefix="",
        fall_back_to_pt=False,
    )
    weights = loader._get_weights_iterator(
        source, lambda name: projection_name(name) in parameters
    )
    loaded = set()
    for name, weight in weights:
        name = projection_name(name)
        if name not in parameters:
            continue
        if name in loaded:
            raise ValueError(f"duplicate DFlash2 projection weight {name!r}")
        default_weight_loader(parameters[name], weight)
        loaded.add(name)
    missing = parameters.keys() - loaded
    if missing:
        raise ValueError(f"missing DFlash2 projection weights: {sorted(missing)}")
    projector.eval().requires_grad_(False)
    configure_dflash_target_capture(target_model, config)
    return projector


class RemoteFeatureCapture:
    """Project all evaluated inputs into the scheduler-owned feature cache.

    ``prepare_batch`` refreshes pointer-stable metadata before eager execution or
    graph replay. Capture writes evaluated speculative rows too; the scheduler's
    committed endpoint alone determines which rows can be exported/published.
    """

    def __init__(
        self,
        projector,
        cache: ProjectedFeatureCache,
        input_buffers: InputBuffers,
        max_context_tokens: int,
    ) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if int(projector.hidden_size) != cache.hidden_size:
            raise ValueError("projection and feature cache hidden sizes differ")
        self.projector = projector
        self.cache = cache
        self.input_buffers = input_buffers
        columns = (
            max_context_tokens + cache.block_granularity - 1
        ) // cache.block_granularity
        self.block_table = torch.zeros(
            (input_buffers.max_bs, columns),
            dtype=torch.int32,
            device=input_buffers.device,
        )
        self.valid_requests = torch.zeros(
            input_buffers.max_bs, dtype=torch.bool, device=input_buffers.device
        )

    def prepare_batch(
        self,
        ctx: ForwardContext,
        block_tables: Mapping[str, torch.Tensor],
        actual_bs: int,
    ) -> None:
        """Refresh tables and live-row masking after the runner selects padding."""
        if not 0 <= actual_bs <= ctx.bs <= self.block_table.shape[0]:
            raise ValueError("feature batch exceeds its allocated request capacity")
        self.valid_requests.zero_()
        self.valid_requests[:actual_bs].fill_(True)
        self.block_table.zero_()
        if actual_bs == 0:
            return
        table = block_tables[self.cache.group_id]
        if (
            table.ndim != 2
            or table.dtype != torch.int32
            or table.shape[0] < actual_bs
            or table.shape[1] > self.block_table.shape[1]
        ):
            raise ValueError("invalid projected-feature scheduler table geometry")
        self.block_table[:actual_bs, : table.shape[1]].copy_(
            table[:actual_bs], non_blocking=True
        )
        # A replay does not call the Python cache consumer again. Observe the
        # current L2 generation here, outside the graph, rather than recording
        # a capture-time event/flag as though it covered every future restore.
        self.cache.wait_ready()

    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        """Request canonical whole-tap capture in every target forward mode."""
        from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode

        ctx.capture_hidden_mode = CaptureHiddenMode.FULL

    def capture(
        self, ctx: ForwardContext, logits_output: LogitsProcessorOutput
    ) -> None:
        """Project captured inputs and write their existing LCM allocations."""
        count = ctx.input_num_tokens
        if count == 0:
            return
        hidden = logits_output.hidden_states
        if (
            hidden is None
            or hidden.ndim != 2
            or hidden.shape[0] != count
            or hidden.shape[1] != self.projector.context_in_features
        ):
            raise RuntimeError(
                "target did not return the complete ordered feature taps"
            )
        if ctx.bs <= 0:
            raise RuntimeError("feature input rows require an active batch")
        request_ids = torch.arange(ctx.bs, device=hidden.device, dtype=torch.int64)
        if ctx.num_extends == 0:
            if count % ctx.bs:
                raise RuntimeError("decode feature rows must have homogeneous width")
            request_ids = request_ids.repeat_interleave(count // ctx.bs)
        else:
            request_ids = torch.repeat_interleave(
                request_ids,
                self.input_buffers.input_lengths_buf[: ctx.bs],
                output_size=count,
            )
        projected = self.projector.project_target_hidden(
            hidden.to(dtype=self.projector.context_dtype)
        ).to(dtype=torch.bfloat16)
        self.cache.write(
            positions=self.input_buffers.positions_buf[:count],
            request_indices=request_ids,
            block_table=self.block_table,
            projected_features=projected,
            is_valid_token=self.valid_requests.index_select(0, request_ids),
        )


class FeatureSourceLease(Protocol):
    """Scheduler pin released only once an export's source is no longer read."""

    def release(self) -> None:
        """Queue the scheduler snapshot release exactly once, from either thread."""


@dataclass(frozen=True)
class FeatureExportDescriptor:
    """Immutable confirmed-prefix identity supplied by the C++ scheduler."""

    ticket_id: int
    session_id: str
    start: int
    end: int
    anchor_token: int


@dataclass(frozen=True)
class CompletedFeatureExport:
    """A CPU-only payload whose ownership passes to the bounded transport queue."""

    descriptor: FeatureExportDescriptor
    features: torch.Tensor


@dataclass
class _PendingExport:
    descriptor: FeatureExportDescriptor
    source: torch.Tensor
    lease: FeatureSourceLease
    nbytes: int
    cancelled: bool
    event: object | None
    host: torch.Tensor | None


class AsyncFeatureExporter:
    """Bound D2H staging and retire copies independently of token completion.

    Construct and call ``submit``/``shutdown`` on the data-plane thread. ``poll``
    and ``cancel`` may run on the control thread: polling queries completion
    events and transfers CPU ownership only, without tensor or stream operations.
    A cancelled ticket keeps both its source lease and pinned staging until its
    copy completes. Returned CPU payloads belong to the bounded transport queue.
    """

    def __init__(
        self,
        hidden_size: int,
        window_left: int,
        max_pending: int,
        max_pending_bytes: int,
        device_module,
        copy_stream,
    ) -> None:
        if min(hidden_size, window_left, max_pending, max_pending_bytes) <= 0:
            raise ValueError("feature export dimensions and capacity must be positive")
        snapshot_bytes = hidden_size * window_left * torch.bfloat16.itemsize
        if max_pending_bytes < snapshot_bytes:
            raise ValueError("feature export budget cannot hold one complete window")
        self.hidden_size = hidden_size
        self.window_left = window_left
        self.max_pending = max_pending
        self.max_pending_bytes = max_pending_bytes
        self.device_module = device_module
        self.copy_stream = copy_stream
        self._pending: dict[int, _PendingExport] = {}
        self._pending_bytes = 0
        self._accepting = True
        self._lock = threading.Lock()

    @property
    def pending_bytes(self) -> int:
        """Bytes reserved by pending copies, including logically cancelled ones."""
        with self._lock:
            return self._pending_bytes

    @property
    def pending_count(self) -> int:
        """Copies retaining a source lease or host staging allocation."""
        with self._lock:
            return len(self._pending)

    def owns_ticket(self, ticket_id: int) -> bool:
        """Whether a ticket still retains its source lease, including failures."""
        with self._lock:
            return ticket_id in self._pending

    def has_capacity(self, num_rows: int) -> bool:
        """Check bounded staging before the single data-plane producer gathers.

        Only ``submit`` reserves capacity. With one producer, concurrent polling
        can only free capacity between this check and submission.
        """
        if not 0 <= num_rows <= self.window_left:
            raise ValueError("feature row count exceeds the native history window")
        nbytes = num_rows * self.hidden_size * torch.bfloat16.itemsize
        with self._lock:
            return (
                self._accepting
                and len(self._pending) < self.max_pending
                and self._pending_bytes + nbytes <= self.max_pending_bytes
            )

    def submit(
        self,
        descriptor: FeatureExportDescriptor,
        features: torch.Tensor,
        prerequisite_stream,
        lease: FeatureSourceLease,
    ) -> bool:
        """Enqueue one already-gathered snapshot, taking its lease on success.

        Returns ``False`` before any copy/allocation when staging is full or
        shutdown began; the caller retains its lease in that case. A successful
        submission transfers lease ownership even if cancellation races it.
        On an exception, ``owns_ticket`` remains true when retirement could not
        be established; callers must preserve that source pin.
        """
        start, end = descriptor.start, descriptor.end
        earliest, _ = committed_feature_interval(end, self.window_left)
        if start < earliest or start > end or descriptor.anchor_token < 0:
            raise ValueError("feature export interval is outside the confirmed window")
        if (
            features.ndim != 2
            or tuple(features.shape) != (end - start, self.hidden_size)
            or features.dtype != torch.bfloat16
            or not features.is_contiguous()
        ):
            raise ValueError("feature exports require contiguous BF16 projected rows")
        nbytes = features.numel() * features.element_size()
        pending = _PendingExport(descriptor, features, lease, nbytes, False, None, None)
        with self._lock:
            if descriptor.ticket_id in self._pending:
                raise ValueError("duplicate pending feature export ticket")
            if (
                not self._accepting
                or len(self._pending) >= self.max_pending
                or self._pending_bytes + nbytes > self.max_pending_bytes
            ):
                return False
            self._pending[descriptor.ticket_id] = pending
            self._pending_bytes += nbytes
        try:
            host, event = self._copy_to_host(pending, prerequisite_stream)
        except BaseException:
            # A partially enqueued copy must retire before its tensors/pin are
            # released. If retirement itself fails, retain the ticket and fail
            # closed; shutdown/process teardown remains its only safe owner.
            try:
                self.copy_stream.synchronize()
            except BaseException:
                with self._lock:
                    self._accepting = False
                raise
            with self._lock:
                self._pending.pop(descriptor.ticket_id)
                self._pending_bytes -= nbytes
            lease.release()
            raise
        with self._lock:
            pending.host = host
            pending.event = event
        return True

    def _copy_to_host(self, pending: _PendingExport, prerequisite_stream):
        features = pending.source
        host = torch.empty(
            features.shape, dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        # Keep staging alive even if a launch/event-record raises after the
        # asynchronous copy has already begun.
        pending.host = host
        event = self.device_module.Event()
        with self.device_module.stream(self.copy_stream):
            self.copy_stream.wait_stream(prerequisite_stream)
            host.copy_(features, non_blocking=True)
            event.record(self.copy_stream)
        return host, event

    def cancel(self, session_id: str) -> None:
        """Invalidate delivery without freeing storage still read by a copy."""
        with self._lock:
            for pending in self._pending.values():
                if pending.descriptor.session_id == session_id:
                    pending.cancelled = True

    def poll(self) -> list[CompletedFeatureExport]:
        """Return completed CPU snapshots without waiting for pending transfers."""
        completed = []
        retired = []
        with self._lock:
            for ticket_id, pending in tuple(self._pending.items()):
                if pending.event is None or not pending.event.query():
                    continue
                self._pending.pop(ticket_id)
                self._pending_bytes -= pending.nbytes
                retired.append(pending)
                if not pending.cancelled:
                    assert pending.host is not None
                    completed.append(
                        CompletedFeatureExport(pending.descriptor, pending.host)
                    )
        for pending in retired:
            pending.lease.release()
        return completed

    def shutdown(self) -> None:
        """Stop admissions and retire all copies before releasing their owners."""
        with self._lock:
            self._accepting = False
            for pending in self._pending.values():
                pending.cancelled = True
        # Called on the data plane after submission has stopped. Synchronizing
        # this dedicated stream cannot become part of ordinary token completion.
        self.copy_stream.synchronize()
        with self._lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
            self._pending_bytes = 0
        for ticket in pending:
            ticket.lease.release()
