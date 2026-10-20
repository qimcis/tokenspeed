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

"""CPU-only validation and command construction for pinned V4.1 deployments."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tokenspeed.runtime.performance_options import (
    PerformanceCapabilities,
    PerformanceOptions,
    validate_performance_options,
)

Positive = Annotated[int, Field(gt=0)]
Role = Literal["aggregate", "encode", "prefill", "decode"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Worker(StrictConfig):
    role: Role
    gpus: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    attention_tp: Literal[1, 2, 4, 8]

    @model_validator(mode="after")
    def validate_layout(self):
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("a worker cannot claim the same GPU twice")
        if len(self.gpus) not in (1, 2, 4, 8):
            raise ValueError("worker size must be 1, 2, 4 or 8")
        # The initial recipe deliberately does not imply unqualified attention DP.
        if self.attention_tp != len(self.gpus):
            raise ValueError("this recipe requires attention TP equal to worker size")
        return self


class Engine(StrictConfig):
    speculation: Literal["none", "dspark"]
    engram: Literal["gpu", "host_shared", "host_sharded"]
    max_model_len: Positive
    max_total_tokens: Positive
    max_num_seqs: Positive
    chunked_prefill_size: Positive
    graph_max_seqs: Positive
    gpu_memory_utilization: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def validate_capacity(self):
        if self.max_model_len > 1048576:
            raise ValueError("context exceeds the checkpoint's architectural limit")
        if self.max_total_tokens < self.max_model_len:
            raise ValueError("token budget cannot admit one maximum-context request")
        if self.max_total_tokens % 128:
            raise ValueError("token budget must be aligned to 128-token prefix pages")
        if self.chunked_prefill_size > self.max_total_tokens:
            raise ValueError("prefill chunk exceeds the total token budget")
        if self.chunked_prefill_size < 256:
            raise ValueError("prefill needs a replay window plus one prefix page")
        if self.graph_max_seqs > self.max_num_seqs:
            raise ValueError("graph sequence capacity exceeds admission capacity")
        return self


class Performance(StrictConfig):
    # Selecting this section requires explicit path choices.
    expert_affinity: bool
    execution: Literal["default", "eager_serial"]
    dspark_candidate_communication: Literal["tiles", "compact"]
    max_decode_batch_size: Annotated[int, Field(ge=0)] = 0
    expert_affinity_lookahead: int = 256
    expert_affinity_max_deferrals: int = 8
    expert_affinity_max_hint_age_tokens: int = 64
    expert_affinity_capture_interval: int = 16
    expert_affinity_max_layers: int = 4
    engram_hot_cache_manifest: str | None = None
    engram_hot_cache_capacity: int = 0

    def options(self, revision: str) -> PerformanceOptions:
        return PerformanceOptions(
            **self.model_dump(exclude={"execution"}),
            engram_hot_cache_model_revision=(
                revision if self.engram_hot_cache_capacity else None
            ),
        )


class Transport(StrictConfig):
    encode_cache_mib: Annotated[int, Field(ge=0)]
    encode_host_cache_mib: Annotated[int, Field(ge=0)]
    encode_slots: Positive
    encode_slot_mib: Positive
    receive_slots: Positive
    receive_slot_mib: Positive

    @model_validator(mode="after")
    def validate_slot(self):
        if self.receive_slot_mib < self.encode_slot_mib:
            raise ValueError("receive slot must cover the largest encode payload")
        return self


class Metrics(StrictConfig):
    reporter: Literal["none", "jsonl"]
    queue_size: Annotated[int, Field(gt=0, le=1000000)]


class Deployment(StrictConfig):
    mode: Literal["aggregate", "pd", "epd"]
    model: Literal["deepseek-ai/DeepSeek-V4.1-Flash-1017"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    served_model_name: Annotated[str, Field(min_length=1)]
    gpu_budget: Positive
    base_port: Annotated[int, Field(ge=1024, le=64000)]
    engine: Engine
    performance: Performance | None = None
    transport: Transport
    metrics: Metrics
    workers: list[Worker] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_topology(self):
        roles = [worker.role for worker in self.workers]
        expected = {
            "aggregate": ["aggregate"],
            "pd": ["prefill", "decode"],
            "epd": ["encode", "prefill", "decode"],
        }[self.mode]
        if sorted(roles) != sorted(expected):
            raise ValueError(f"{self.mode} requires exactly these roles: {expected}")
        gpus = [gpu for worker in self.workers for gpu in worker.gpus]
        if len(set(gpus)) != len(gpus):
            raise ValueError("GPU assignments overlap across workers")
        if len(gpus) > self.gpu_budget:
            raise ValueError("workers exceed the total GPU budget")
        if self.mode == "epd":
            sizes = {worker.role: worker.attention_tp for worker in self.workers}
            if sizes["prefill"] % sizes["encode"]:
                raise ValueError("prefill TP must be a multiple of encode TP")
        if self.performance is not None:
            options = self.performance.options(self.revision)
            options.validate()
            if options.max_decode_batch_size > self.engine.max_num_seqs:
                raise ValueError("max_decode_batch_size exceeds max_num_seqs")
            if (
                options.expert_affinity
                and self.engine.max_num_seqs > options.expert_affinity_lookahead
            ):
                raise ValueError("expert_affinity_lookahead must cover max_num_seqs")
            for worker in self.workers:
                if worker.role in ("encode", "prefill"):
                    continue
                validate_performance_options(
                    options,
                    PerformanceCapabilities(
                        model_is_deepseek_v41=True,
                        nvidia_cuda=True,
                        role=worker.role,
                        speculation=(
                            "DSPARK" if self.engine.speculation == "dspark" else None
                        ),
                        speculative_sampling=False,
                        pipeline_size=1,
                        attention_dp=1,
                        attention_tp=worker.attention_tp,
                        world_size=len(worker.gpus),
                        query_context_size=1,
                        kv_parallel_size=1,
                        eager=self.performance.execution == "eager_serial",
                        prefill_graph_disabled=self.performance.execution
                        == "eager_serial",
                        overlap_disabled=self.performance.execution == "eager_serial",
                        mixed_batch=False,
                        expert_rebalancing=False,
                        custom_expert_placement=False,
                        routing_simulation=False,
                        host_engram=self.engine.engram != "gpu",
                        host_engram_layout=self.engine.engram.removeprefix("host_"),
                        load_format="auto",
                    ),
                )
            if options.engram_hot_cache_capacity:
                _hot_cache_ledger(self)
        return self

    @classmethod
    def load(cls, path: Path) -> Deployment:
        with path.open("rb") as source:
            data = tomllib.load(source)
        performance = data.get("performance")
        if isinstance(performance, dict) and performance.get(
            "engram_hot_cache_manifest"
        ):
            manifest = Path(performance["engram_hot_cache_manifest"])
            if not manifest.is_absolute():
                performance["engram_hot_cache_manifest"] = str(
                    (path.parent / manifest).resolve()
                )
        return cls.model_validate(data)


def _hot_cache_ledger(config: Deployment) -> dict[str, list[dict[str, int]]]:
    """Account actual selected rows on every TP rank, including metadata/staging."""
    if config.performance is None or not config.performance.engram_hot_cache_capacity:
        return {}
    from tokenspeed.runtime.models.engram_hot_cache import (
        EngramHotSetManifest,
        plan_engram_hot_cache,
    )

    manifest = EngramHotSetManifest.load(config.performance.engram_hot_cache_manifest)
    expected = {1: 384006168, 14: 384016682}
    if {layer.layer_id: layer.total_rows for layer in manifest.layers} != expected:
        raise ValueError("Engram manifest layers must match the pinned V4.1 checkpoint")
    ledger = {}
    for worker in config.workers:
        if worker.role == "encode":
            continue
        rank_entries = []
        for rank in range(worker.attention_tp):
            payload = metadata = staging = 0
            for layer_id, rows in expected.items():
                shard = (rows + worker.attention_tp - 1) // worker.attention_tp
                start, end = (
                    (rank * shard, min((rank + 1) * shard, rows))
                    if config.engine.engram == "host_sharded"
                    else (0, rows)
                )
                plan = plan_engram_hot_cache(
                    manifest,
                    layer_id=layer_id,
                    model_revision=config.revision,
                    total_rows=rows,
                    row_start=start,
                    row_end=end,
                    capacity_rows=config.performance.engram_hot_cache_capacity,
                    embedding_dim=256,
                )
                payload += len(plan.local_row_ids) * 264
                metadata += len(plan.local_row_ids) * 8
                staging = max(staging, plan.staging_nbytes)
            rank_entries.append(
                {
                    "rank": rank,
                    "codes_and_scales": payload,
                    "lookup_metadata": metadata,
                    "host_staging_peak": staging,
                }
            )
        ledger[worker.role] = rank_entries
    return ledger


def memory_lower_bound(config: Deployment) -> dict:
    """Known tensor payloads, NOT a fit prediction or an allocator measurement.

    Global histories are replicated within attention TP in these recipes. The
    sequence cap applies separately to every LM worker. Shared-host Engram is
    shared within a worker, not across the independent P and D processes.
    """
    engram = (384006168 + 384016682) * (256 + 8)
    experts = 40 * 384 * 3 * 5120 * 2304 // 2
    rows = []
    hot_caches = _hot_cache_ledger(config)
    for worker in config.workers:
        size = len(worker.gpus)
        if worker.role == "encode":
            rows.append(
                {
                    "role": worker.role,
                    "gpus": worker.gpus,
                    "components_per_rank": {
                        "embedding_send_ring": config.transport.encode_slots
                        * config.transport.encode_slot_mib
                        * 2**20,
                        "embedding_device_cache_budget": config.transport.encode_cache_mib
                        * 2**20,
                    },
                    "known_device_bytes_per_rank": (
                        config.transport.encode_slots * config.transport.encode_slot_mib
                        + config.transport.encode_cache_mib
                    )
                    * 2**20,
                    "known_host_bytes_per_worker": config.transport.encode_host_cache_mib
                    * 2**20
                    * size,
                }
            )
            continue
        global_cache = config.engine.max_total_tokens * 890
        swa = config.engine.max_num_seqs * 128 * 40 * 528
        tails = config.engine.max_num_seqs * 3 * 2 * 2 * 512 * 4
        draft_windows = (
            config.engine.max_num_seqs * 128 * 3 * 512 * 2
            if config.engine.speculation == "dspark"
            else 0
        )
        gpu_engram = (engram + size - 1) // size if config.engine.engram == "gpu" else 0
        receive = (
            config.transport.receive_slots * config.transport.receive_slot_mib * 2**20
            if config.mode == "epd" and worker.role == "prefill"
            else 0
        )
        components = {
            "routed_expert_values": (experts + size - 1) // size,
            "engram_codes_and_scales": gpu_engram,
            "global_cache_payload": global_cache,
            "swa_payload": swa,
            "compressor_tails": tails,
            "draft_windows": draft_windows,
            "embedding_receive_pool": receive,
        }
        route_capture_host = (
            config.engine.max_num_seqs
            * config.performance.expert_affinity_max_layers
            * 6
            * 8
            if config.performance is not None
            and config.performance.expert_affinity
            and worker.role in ("aggregate", "decode")
            else 0
        )
        hot_ranks = hot_caches.get(worker.role, [])
        if hot_ranks:
            components["engram_hot_codes_and_scales_max"] = max(
                row["codes_and_scales"] for row in hot_ranks
            )
            components["engram_hot_lookup_metadata_max"] = max(
                row["lookup_metadata"] for row in hot_ranks
            )
        rows.append(
            {
                "role": worker.role,
                "gpus": worker.gpus,
                **(
                    {"expert_route_capture_pinned_host_bytes": route_capture_host}
                    if route_capture_host
                    else {}
                ),
                **(
                    {
                        "engram_hot_cache_per_rank": hot_ranks,
                        "engram_hot_host_staging_peak_per_worker": sum(
                            row["host_staging_peak"] for row in hot_ranks
                        ),
                    }
                    if hot_ranks
                    else {}
                ),
                "components_per_rank": components,
                "known_device_bytes_per_rank": sum(components.values()),
                "known_host_bytes_per_worker": (
                    (0 if config.engine.engram == "gpu" else engram)
                    + route_capture_host
                ),
            }
        )
    return {
        "status": "analytical_lower_bound_not_a_fit_check",
        "workers": rows,
        "known_host_bytes": sum(row["known_host_bytes_per_worker"] for row in rows),
        "excluded": [
            "expert scales and padding",
            "dense/shared/vision/draft weights",
            "cache packing and allocator slack",
            "graph and kernel workspaces",
            "load peaks (Engram hot-cache staging peak is reported separately)",
            "communication buffers",
            "host process overhead",
            "embedding payloads outside configured pools",
            "in-flight replay and verify rows",
        ],
    }


def build_manifest(
    config: Deployment,
    *,
    snapshot: str,
    python: str = sys.executable,
    output_directory: Path | None = None,
) -> dict:
    """Build argv vectors without importing CUDA, downloading or spawning anything."""
    commands = []
    endpoints = {}
    generated_files = {}
    output = (
        str(output_directory.resolve()) if output_directory is not None else "<output>"
    )
    for index, worker in enumerate(config.workers):
        port = config.base_port + 100 * (index + 1)
        bootstrap = port + 1
        endpoints[worker.role] = (port, bootstrap)
        argv = [
            python,
            "-m",
            "smg_grpc_servicer.tokenspeed",
            "--model",
            snapshot,
            "--served-model-name",
            config.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--nnodes",
            "1",
            "--node-rank",
            "0",
            "--world-size",
            str(len(worker.gpus)),
            "--tensor-parallel-size",
            str(worker.attention_tp),
            "--dist-init-addr",
            f"127.0.0.1:{port + 20}",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--load-format",
            "auto",
            "--reasoning-parser",
            "deepseek_v41",
            "--max-num-seqs",
            str(config.engine.max_num_seqs),
            "--chunked-prefill-size",
            str(config.engine.chunked_prefill_size),
            "--max-prefill-tokens",
            str(config.engine.chunked_prefill_size),
            "--gpu-memory-utilization",
            str(config.engine.gpu_memory_utilization),
        ]
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, worker.gpus)),
            "TOKENSPEED_SKIP_GRPC_WARMUP": "1",
        }
        if config.metrics.reporter == "jsonl":
            reporter_name = f"{worker.role}-metrics.json"
            generated_files[reporter_name] = {
                "jsonl": {
                    "output_dir": f"{output}/metrics/{worker.role}",
                    "queue_size": config.metrics.queue_size,
                }
            }
            argv += [
                "--enable-metrics",
                "--metrics-reporters",
                "jsonl",
                "--metrics-reporter-config",
                f"{output}/{reporter_name}",
            ]
        if worker.role != "aggregate":
            argv += [
                "--disaggregation-mode",
                worker.role,
                "--disaggregation-transfer-backend",
                "mooncake",
                "--disaggregation-layerwise-interval",
                "0",
            ]
        if worker.role in ("encode", "prefill"):
            argv += ["--disaggregation-bootstrap-port", str(bootstrap)]
        if worker.role == "encode":
            env.update(
                {
                    "TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB": str(
                        config.transport.encode_cache_mib
                    ),
                    "TOKENSPEED_EPD_ENCODE_EMBED_CACHE_DRAM_MB": str(
                        config.transport.encode_host_cache_mib
                    ),
                    "TOKENSPEED_EPD_ENCODE_RING_SLOTS": str(
                        config.transport.encode_slots
                    ),
                    "TOKENSPEED_EPD_ENCODE_RING_SLOT_MB": str(
                        config.transport.encode_slot_mib
                    ),
                }
            )
        else:
            engine = config.engine
            argv += [
                "--attention-backend",
                "deepseek_v41",
                "--moe-backend",
                "mega_moe",
                "--sampling-backend",
                "flashinfer",
                "--enable-expert-parallel",
                "--disable-kvstore",
                "--max-model-len",
                str(engine.max_model_len),
                "--max-total-tokens",
                str(engine.max_total_tokens),
                "--prefix-granularity",
                "128",
                "--max-cudagraph-capture-size",
                str(engine.graph_max_seqs),
            ]
            if engine.engram != "gpu":
                argv += [
                    "--engram-host-table",
                    "--engram-host-table-layout",
                    engine.engram.removeprefix("host_"),
                ]
            if engine.speculation == "dspark":
                argv += ["--speculative-algorithm", "DSPARK"]
            if config.performance is not None:
                performance = config.performance
                if performance.engram_hot_cache_capacity:
                    argv += [
                        "--engram-hot-cache-manifest",
                        performance.engram_hot_cache_manifest,
                        "--engram-hot-cache-capacity",
                        str(performance.engram_hot_cache_capacity),
                        "--engram-hot-cache-model-revision",
                        config.revision,
                    ]
                if worker.role in ("aggregate", "decode"):
                    argv += [
                        "--dspark-candidate-communication",
                        performance.dspark_candidate_communication,
                        "--max-decode-batch-size",
                        str(performance.max_decode_batch_size),
                    ]
                    if performance.execution == "eager_serial":
                        argv += [
                            "--enforce-eager",
                            "--disable-prefill-graph",
                            "--disable-overlap-schedule",
                        ]
                    if performance.expert_affinity:
                        argv += ["--expert-affinity"]
                        for key, value in performance.model_dump().items():
                            if key.startswith("expert_affinity_"):
                                argv += ["--" + key.replace("_", "-"), str(value)]
            if worker.role == "decode":
                argv += ["--disable-prefix-caching", "--disable-prefill-graph"]
            if worker.role == "prefill" and config.mode == "epd":
                env.update(
                    {
                        "TOKENSPEED_EPD_RECV_POOL_SLOTS": str(
                            config.transport.receive_slots
                        ),
                        "TOKENSPEED_EPD_RECV_POOL_SLOT_MB": str(
                            config.transport.receive_slot_mib
                        ),
                    }
                )
        commands.append({"role": worker.role, "port": port, "argv": argv, "env": env})
    gateway = [
        python,
        "-m",
        "smg",
        "launch",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.base_port),
        "--prometheus-port",
        str(config.base_port + 1),
        "--model-path",
        snapshot,
        "--tokenizer-path",
        snapshot,
        "--reasoning-parser",
        "deepseek_v41",
        "--tool-call-parser",
        "deepseek_v41",
        "--disable-retries",
        "--max-concurrent-requests",
        str(config.engine.max_num_seqs),
        "--queue-size",
        str(config.engine.max_num_seqs * 8),
    ]
    if config.mode == "aggregate":
        gateway += ["--worker-urls", f"grpc://127.0.0.1:{endpoints['aggregate'][0]}"]
    else:
        gateway += [
            "--epd-disaggregation" if config.mode == "epd" else "--pd-disaggregation"
        ]
        for role in ("encode", "prefill", "decode"):
            if role in endpoints:
                port, bootstrap = endpoints[role]
                gateway += [f"--{role}", f"grpc://127.0.0.1:{port}"]
                if role != "decode":
                    gateway.append(str(bootstrap))
    commands.append(
        {"role": "gateway", "port": config.base_port, "argv": gateway, "env": {}}
    )
    return {
        "schema_version": 1,
        "model": config.model,
        "revision": config.revision,
        "config": config.model_dump(),
        "commands": commands,
        "generated_files": generated_files,
        "ports": {
            "gateway": config.base_port,
            "gateway_metrics": config.base_port + 1,
            "workers": {
                role: {
                    "grpc": port,
                    "bootstrap": bootstrap if role in ("encode", "prefill") else None,
                    "rendezvous": port + 20,
                    "control_cluster": [
                        port + offset for offset in (20, 21, 23, 24, 25, 26)
                    ],
                }
                for role, (port, bootstrap) in endpoints.items()
            },
        },
        "memory": memory_lower_bound(config),
        "hardware_validated": False,
        **(
            {
                "performance": {
                    "resolved": {
                        **asdict(config.performance.options(config.revision)),
                        "execution": config.performance.execution,
                    },
                    "validation": "CPU contract only; GPU numerical, graph and performance gates unrun",
                    "decode_policy_roles": [
                        w.role
                        for w in config.workers
                        if w.role in ("aggregate", "decode")
                    ],
                }
            }
            if config.performance is not None
            else {}
        ),
    }
