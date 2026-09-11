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

"""Private independently batched DFlash2 worker launch interface."""

import argparse
import signal
from dataclasses import dataclass

from tokenspeed.runtime.draft_pool.config import (
    validate_remote_revision,
    validate_remote_tcp_endpoint,
)


@dataclass(frozen=True)
class DraftWorkerArgs:
    target_model_path: str
    target_revision: str
    draft_model_path: str
    draft_revision: str
    listen_address: str
    device_id: int
    max_resident_sessions: int
    max_batch_size: int
    staging_slots: int
    max_queued_jobs: int
    max_peers: int
    max_header_bytes: int
    max_feature_bytes: int
    max_host_memory_bytes: int
    lease_ms: int
    heartbeat_ms: int
    poll_ms: int
    linger_ms: int

    def __post_init__(self) -> None:
        for name in ("target_model_path", "draft_model_path"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty checkpoint path")
        validate_remote_revision(self.target_revision, "--target-revision")
        validate_remote_revision(self.draft_revision, "--draft-revision")
        validate_remote_tcp_endpoint(self.listen_address, allow_wildcard=True)
        for name in (
            "max_resident_sessions",
            "max_batch_size",
            "staging_slots",
            "max_queued_jobs",
            "max_peers",
            "max_header_bytes",
            "max_feature_bytes",
            "max_host_memory_bytes",
            "lease_ms",
            "heartbeat_ms",
            "poll_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.device_id < 0 or self.linger_ms < 0:
            raise ValueError("device_id and linger_ms must be nonnegative")
        if self.max_batch_size > self.max_resident_sessions:
            raise ValueError("max_batch_size cannot exceed max_resident_sessions")
        if not self.max_batch_size <= self.staging_slots <= self.max_resident_sessions:
            raise ValueError(
                "staging_slots must be between max_batch_size and max_resident_sessions"
            )
        if self.heartbeat_ms >= self.lease_ms:
            raise ValueError("heartbeat_ms must be shorter than lease_ms")
        if self.max_feature_bytes > 24 * 1024 * 1024:
            raise ValueError(
                "max_feature_bytes cannot exceed the native 24 MiB feature window"
            )


def add_draft_worker_args(parser: argparse.ArgumentParser) -> None:
    """Register worker-only arguments without importing GPU runtime modules."""
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Full GLM-5.3 target checkpoint; only vocabulary weights are loaded.",
    )
    parser.add_argument(
        "--target-revision",
        required=True,
        help="Immutable 40-character target checkpoint commit ID.",
    )
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Matching BF16 GLM-5.3 DFlash2 checkpoint.",
    )
    parser.add_argument(
        "--draft-revision",
        required=True,
        help="Immutable 40-character draft checkpoint commit ID.",
    )
    parser.add_argument(
        "--listen-address",
        required=True,
        help="Private tcp://host:port worker bind endpoint.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index for the independent TP1 worker.",
    )
    parser.add_argument(
        "--max-resident-sessions",
        type=int,
        required=True,
        help="Maximum sessions with worker KV state.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        required=True,
        help="Maximum independently batched native-eight draft jobs.",
    )
    parser.add_argument(
        "--staging-slots",
        type=int,
        default=8,
        help="Maximum admitted feature transfers held in host staging.",
    )
    parser.add_argument("--max-queued-jobs", type=int, default=64)
    parser.add_argument("--max-peers", type=int, default=8)
    parser.add_argument("--max-header-bytes", type=int, default=65536)
    parser.add_argument("--max-feature-bytes", type=int, default=24 * 1024 * 1024)
    parser.add_argument(
        "--max-host-memory-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Hard configuration budget for bounded transport/staging host memory.",
    )
    parser.add_argument(
        "--lease-ms",
        type=int,
        default=30000,
        help="Finite lease reclaiming orphaned worker sessions.",
    )
    parser.add_argument("--heartbeat-ms", type=int, default=1000)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--linger-ms", type=int, default=1000)


def run_draft_worker_from_args(args: argparse.Namespace) -> None:
    """Validate CLI configuration and launch the private worker service.

    The service constructs the engine on its execution thread, so parsing,
    socket IO and signal handling never access CUDA.
    """
    names = DraftWorkerArgs.__dataclass_fields__
    config = DraftWorkerArgs(**{name: getattr(args, name) for name in names})

    from tokenspeed.runtime.draft_pool.transport import (
        DraftPoolServiceConfig,
        run_draft_worker,
    )
    from tokenspeed.runtime.draft_pool.worker import WorkerModelConfig

    service_config = DraftPoolServiceConfig(
        listen_endpoint=config.listen_address,
        resident_limit=config.max_resident_sessions,
        staging_limit=config.staging_slots,
        max_queued_jobs=config.max_queued_jobs,
        max_batch_size=config.max_batch_size,
        max_peers=config.max_peers,
        max_header_bytes=config.max_header_bytes,
        max_feature_bytes=config.max_feature_bytes,
        max_host_memory_bytes=config.max_host_memory_bytes,
        lease_ms=config.lease_ms,
        heartbeat_ms=config.heartbeat_ms,
        poll_ms=config.poll_ms,
        linger_ms=config.linger_ms,
    )
    model_config = WorkerModelConfig(
        target_model_path=config.target_model_path,
        target_revision=config.target_revision,
        draft_model_path=config.draft_model_path,
        draft_revision=config.draft_revision,
        device_id=config.device_id,
        max_resident_sessions=config.max_resident_sessions,
        max_batch_size=config.max_batch_size,
    )

    # SIGINT already raises KeyboardInterrupt. Give SIGTERM the same bounded
    # service.close() path instead of terminating while GPU tickets are live.
    def terminate(signum, frame) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        run_draft_worker(service_config, model_config)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
