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

"""TP1 native DFlash2 worker, exclusively owned by the service's data thread.

The transport owns admission, leases and bounded CPU staging. This engine
owns weights, fixed resident KV slots and one independently batched forward.
Importing this module does not import torch or initialize a CUDA context.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.draft_pool.worker_cache import WorkerCacheGeometry
from tokenspeed.runtime.draft_pool.worker_weights import (
    TARGET_VOCABULARY_NAMES,
    iter_checkpoint_weights,
    load_checkpoint_config,
    load_complete_draft_weights,
    resolve_checkpoint,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class WorkerModelConfig:
    """Checkpoint identities and fixed per-process device/memory limits."""

    target_model_path: str
    target_revision: str
    draft_model_path: str
    draft_revision: str
    device_id: int
    max_resident_sessions: int
    max_batch_size: int

    def __post_init__(self) -> None:
        if (
            self.device_id < 0
            or not 1 <= self.max_batch_size <= self.max_resident_sessions
        ):
            raise ValueError(
                "Worker requires a valid device and batch <= resident slots."
            )
        if not all(
            (
                self.target_model_path,
                self.target_revision,
                self.draft_model_path,
                self.draft_revision,
            )
        ):
            raise ValueError("Both checkpoint paths and revisions are required.")


@dataclass(frozen=True)
class WorkerDraftJob:
    """A proposal request for an unchanged confirmed prefix and its anchor."""

    session_id: str
    confirmed_endpoint: int
    anchor_token: int


@dataclass(frozen=True)
class WorkerDraftResult:
    """CPU-only full native proposal; the target consumes its first five IDs."""

    session_id: str
    confirmed_endpoint: int
    anchor_token: int
    candidate_ids: tuple[int, ...]


@dataclass
class ResidentSession:
    slot: int
    confirmed_endpoint: int | None


class WorkerSessionTable:
    """Bounded identity/endpoint bookkeeping independent of device tensors."""

    def __init__(self, capacity: int, history_tokens: int) -> None:
        if capacity < 1 or history_tokens < 1:
            raise ValueError("Session capacity and history must be positive.")
        self._free = list(range(capacity))
        self._sessions: dict[str, ResidentSession] = {}
        self.history_tokens = history_tokens

    def open(self, session_id: str) -> ResidentSession:
        if not session_id or session_id in self._sessions:
            raise ValueError("A new worker session needs a unique nonempty identity.")
        if not self._free:
            raise RuntimeError("Worker resident session capacity is exhausted.")
        session = ResidentSession(heapq.heappop(self._free), None)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> ResidentSession:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ValueError("Worker session is missing or has been closed.") from None

    def validate_update(
        self, session_id: str, start: int, endpoint: int, is_snapshot: bool
    ) -> ResidentSession:
        """Validate before writes; installed endpoint changes only after success."""
        session = self.get(session_id)
        if not 0 <= start <= endpoint or endpoint - start > self.history_tokens:
            raise ValueError(
                "Feature interval is invalid or exceeds the history bound."
            )
        if is_snapshot:
            if session.confirmed_endpoint is not None:
                raise ValueError(
                    "Resetting an installed context requires a new session."
                )
            if start != max(0, endpoint - self.history_tokens):
                raise ValueError(
                    "Snapshot must contain the entire retained feature interval."
                )
        elif session.confirmed_endpoint is None or start != session.confirmed_endpoint:
            raise ValueError("Feature delta must start at the installed endpoint.")
        return session

    def close(self, session_id: str) -> int | None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        heapq.heappush(self._free, session.slot)
        return session.slot


class DFlash2WorkerEngine:
    """Load and execute the supported draft pair on one CUDA execution thread.

    All methods except reading ``contract`` must run on the constructing
    thread. The service creates the engine there, not on its ROUTER thread.
    No distributed process group or target model is constructed.
    """

    def __init__(self, config: WorkerModelConfig) -> None:
        import torch

        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.draft_pool.config import (
            validate_remote_model_pair,
            validate_remote_revision,
        )
        from tokenspeed.runtime.draft_pool.protocol import build_contract
        from tokenspeed.runtime.draft_pool.worker_cache import (
            WorkerAttentionBackend,
            WorkerKVPool,
            validate_worker_attention,
        )
        from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
        from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

        self._thread_id = threading.get_ident()
        self._closed = False
        self.config = config
        validate_remote_revision(config.target_revision, "target_revision")
        validate_remote_revision(config.draft_revision, "draft_revision")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The DFlash2 worker requires a CUDA device with BF16 support."
            )
        torch.cuda.set_device(config.device_id)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The draft worker requires native CUDA BF16 support.")
        self.device = f"cuda:{config.device_id}"
        draft_checkpoint = resolve_checkpoint(
            config.draft_model_path, config.draft_revision, None
        )
        target_checkpoint = resolve_checkpoint(
            config.target_model_path, config.target_revision, TARGET_VOCABULARY_NAMES
        )
        draft_hf_config = load_checkpoint_config(
            draft_checkpoint, config.draft_revision, True
        )
        target_hf_config = load_checkpoint_config(
            target_checkpoint, config.target_revision, False
        )
        draft_dict = draft_hf_config.to_dict()
        target_dict = target_hf_config.to_dict()
        validate_remote_model_pair(target_dict, draft_dict)
        self.contract = build_contract(
            target_model=config.target_model_path,
            target_revision=config.target_revision,
            draft_model=config.draft_model_path,
            draft_revision=config.draft_revision,
            target_hf_config=target_dict,
            draft_hf_config=draft_dict,
            target_verify_tokens=6,
        )
        self.model_config = draft_hf_config
        self.max_position = int(draft_dict["max_position_embeddings"])
        self.geometry = WorkerCacheGeometry(
            config.max_resident_sessions,
            self.contract.window_tokens,
            self.contract.native_block_tokens,
        )
        self.sessions = WorkerSessionTable(
            config.max_resident_sessions, self.geometry.history_tokens
        )
        mapping = Mapping(
            rank=0,
            world_size=1,
            attn_tp_size=1,
            attn_cp_size=1,
            attn_dp_size=1,
            dense_tp_size=1,
            dense_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            moe_dp_size=1,
            vision_tp_size=1,
            vision_dp_size=1,
            linear_attn_tp_size=1,
            pp_size=1,
            pp_layer_partition=None,
            nprocs_per_node=1,
            nnodes=1,
            base_gpu_id=config.device_id,
            gpu_id_step=1,
        )
        with set_default_torch_dtype(torch.bfloat16), torch.device(
            self.device
        ), torch.no_grad():
            self.model = DFlash2DraftModel(
                config=self.model_config, mapping=mapping, quant_config=None, prefix=""
            )
            validate_worker_attention(
                self.model,
                self.geometry,
                int(draft_dict["num_key_value_heads"]),
                int(draft_dict["head_dim"]),
            )
            load_complete_draft_weights(self.model, draft_checkpoint)
            vocabulary_args = dict(
                num_embeddings=self.contract.vocab_size,
                embedding_dim=self.contract.feature_width,
                params_dtype=torch.bfloat16,
                org_num_embeddings=self.contract.vocab_size,
                padding_size=64,
                quant_config=None,
                prefix="",
                tp_rank=0,
                tp_size=1,
                tp_group=(0,),
                use_presharded_weights=False,
            )
            self.embed_tokens = VocabParallelEmbedding(**vocabulary_args)
            self.lm_head = ParallelLMHead(**vocabulary_args, bias=False)
            vocabulary = {
                "model.embed_tokens.weight": self.embed_tokens,
                "lm_head.weight": self.lm_head,
            }
            for name, tensor in iter_checkpoint_weights(target_checkpoint):
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(
                        "Target vocabulary weights must be unquantized floating tensors."
                    )
                if tuple(tensor.shape) != (
                    self.contract.vocab_size,
                    self.contract.feature_width,
                ):
                    raise ValueError(
                        f"Incompatible target vocabulary shape for {name}."
                    )
                module = vocabulary[name]
                module.weight_loader(module.weight, tensor)
            for module in self.model.modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
            self.model.eval()
            self.logits_processor = LogitsProcessor(
                self.model_config,
                skip_all_gather=False,
                do_argmax=False,
                logit_scale=float(
                    draft_dict["dflash_config"].get("output_multiplier", 1.0)
                ),
                tp_rank=0,
                tp_size=1,
                tp_group=(0,),
            )
            softcapping = float(
                draft_dict["dflash_config"].get("final_logit_softcapping") or 0.0
            )
            self.logits_processor.final_logit_softcapping = (
                softcapping if softcapping > 0 else None
            )
            kv_bytes = self.geometry.kv_bytes(
                int(draft_dict["num_hidden_layers"]),
                int(draft_dict["num_key_value_heads"]),
                int(draft_dict["head_dim"]),
                2,
            )
            # Reserve bounded logits and transient layer work in addition to
            # resident KV; fail at startup rather than admitting unstable load.
            transient_bytes = (
                2 * 1024**3
                + config.max_batch_size
                * self.contract.native_block_tokens
                * self.contract.vocab_size
                * 4
            )
            available, _ = torch.cuda.mem_get_info(config.device_id)
            if kv_bytes + transient_bytes > available:
                raise ValueError(
                    "Worker resident/batch limits exceed available GPU memory; reduce them."
                )
            self.pool = WorkerKVPool(
                self.geometry,
                int(draft_dict["num_hidden_layers"]),
                int(draft_dict["num_key_value_heads"]),
                int(draft_dict["head_dim"]),
                self.device,
            )
            self.backend = WorkerAttentionBackend(self.geometry, self.device)
        torch.cuda.synchronize(config.device_id)

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError(
                "Draft model access is restricted to its execution thread."
            )
        if self._closed:
            raise RuntimeError("Draft worker engine is closed.")

    def open_session(self, session_id: str) -> None:
        """Reserve a preallocated resident KV slot before snapshot transfer."""
        self._check_thread()
        self.sessions.open(session_id)

    def install_features(
        self,
        session_id: str,
        feature_start: int,
        confirmed_endpoint: int,
        features: torch.Tensor,
        is_snapshot: bool,
    ) -> None:
        """Install CPU BF16 projected context at its original absolute positions.

        The endpoint excludes the anchor token. A failure invalidates the
        session: partial context writes may have overwritten an older ring.
        """
        import torch

        self._check_thread()
        session = self.sessions.validate_update(
            session_id, feature_start, confirmed_endpoint, is_snapshot
        )
        if confirmed_endpoint + self.geometry.native_block_tokens > self.max_position:
            raise ValueError("Draft positions exceed the checkpoint's RoPE range.")
        if (
            features.device.type != "cpu"
            or features.dtype != torch.bfloat16
            or features.ndim != 2
        ):
            raise ValueError("Worker context must be a CPU BF16 matrix.")
        if tuple(features.shape) != (
            confirmed_endpoint - feature_start,
            self.contract.feature_width,
        ):
            raise ValueError(
                "Feature tensor does not match its context interval/schema."
            )
        try:
            with torch.inference_mode():
                projected = features.to(device=self.device, non_blocking=False)
                positions = torch.arange(
                    feature_start,
                    confirmed_endpoint,
                    dtype=torch.int64,
                    device=self.device,
                )
                locations = torch.tensor(
                    self.geometry.context_locations(
                        session.slot, feature_start, confirmed_endpoint
                    ),
                    dtype=torch.int64,
                    device=self.device,
                )
                if confirmed_endpoint > feature_start:
                    self.model.write_context_kv(
                        projected, positions, locations, self.pool
                    )
                # ACK is an installed-context acknowledgement, not merely an
                # issued-kernel acknowledgement. This waits only this worker.
                torch.cuda.current_stream(self.config.device_id).synchronize()
        except Exception:
            self.close_session(session_id)
            raise
        session.confirmed_endpoint = confirmed_endpoint

    def draft_batch(self, jobs: Sequence[WorkerDraftJob]) -> list[WorkerDraftResult]:
        """Execute full native-eight blocks and selectors for distinct sessions."""
        import torch

        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.drafter.dflash2 import select_dflash2_block
        from tokenspeed.runtime.execution.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from tokenspeed.runtime.layers.logits_processor import LogitsMetadata

        self._check_thread()
        if not 1 <= len(jobs) <= self.config.max_batch_size:
            raise ValueError("Worker batch is empty or exceeds its fixed limit.")
        if len({job.session_id for job in jobs}) != len(jobs):
            raise ValueError("A session can appear only once in a worker batch.")
        states = [self.sessions.get(job.session_id) for job in jobs]
        for job, state in zip(jobs, states, strict=True):
            if state.confirmed_endpoint != job.confirmed_endpoint:
                raise ValueError(
                    "Proposal requested for an uninstalled or stale endpoint."
                )
            if not 0 <= job.anchor_token < self.contract.vocab_size:
                raise ValueError("Anchor token is outside the target vocabulary.")
        with torch.inference_mode():
            width = self.geometry.native_block_tokens
            bs = len(jobs)
            anchors = torch.tensor(
                [job.anchor_token for job in jobs],
                dtype=torch.int64,
                device=self.device,
            )
            endpoints = [job.confirmed_endpoint for job in jobs]
            self.backend.prepare([state.slot for state in states], endpoints)
            positions = (
                torch.tensor(endpoints, dtype=torch.int64, device=self.device)[:, None]
                + torch.arange(width, dtype=torch.int64, device=self.device)[None, :]
            )
            block_ids = torch.full(
                (bs, width),
                int(self.model_config.dflash_config["mask_token_id"]),
                dtype=torch.int64,
                device=self.device,
            )
            block_ids[:, 0].copy_(anchors)
            embeddings = self.embed_tokens(block_ids.flatten(), reduce_results=False)
            context = ForwardContext(
                attn_backend=self.backend,
                token_to_kv_pool=self.pool,
                bs=bs,
                num_extends=bs,
                input_num_tokens=bs * width,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            output = self.model(
                ctx=context,
                input_ids=block_ids.flatten(),
                positions=positions.flatten(),
                input_lengths=None,
                input_embeds=embeddings,
                kv_sync_event=None,
            )
            hidden = output.hidden_states.view(bs, width, self.contract.feature_width)
            logits = self.logits_processor._get_logits(
                hidden[:, 1:].reshape(-1, self.contract.feature_width),
                self.lm_head,
                LogitsMetadata(forward_mode=ForwardMode.DECODE),
            )
            top_k = self.model.candidate_selector.top_k
            unary, candidates = torch.topk(logits, top_k, dim=-1, sorted=False)
            tokens = torch.empty((bs, width), dtype=torch.int32, device=self.device)
            select_dflash2_block(
                self.model.candidate_selector,
                candidates.view(bs, width - 1, top_k),
                unary.float().view(bs, width - 1, top_k),
                hidden,
                anchors,
                tokens,
                self.contract.vocab_size,
            )
            # The service receives only bounded host IDs. It never touches
            # model tensors or synchronizes CUDA on its network thread.
            rows = tokens[:, 1:].cpu().tolist()
        return [
            WorkerDraftResult(
                job.session_id, job.confirmed_endpoint, job.anchor_token, tuple(row)
            )
            for job, row in zip(jobs, rows, strict=True)
        ]

    def close_session(self, session_id: str) -> None:
        """Release a session after earlier execution-thread operations retire."""
        self._check_thread()
        slot = self.sessions.close(session_id)
        if slot is not None:
            self.pool.clear_slot(slot)

    def close(self) -> None:
        """Retire device work before releasing resident storage and weights."""
        import torch

        self._check_thread()
        torch.cuda.synchronize(self.config.device_id)
        self.pool = None
        self.backend = None
        self.model = None
        self.embed_tokens = None
        self.lm_head = None
        self._closed = True
