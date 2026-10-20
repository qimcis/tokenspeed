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

"""Allocation-free capability checks shared by runtime and deployment planning."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PerformanceOptions:
    expert_affinity: bool = False
    expert_affinity_lookahead: int = 256
    expert_affinity_max_deferrals: int = 8
    expert_affinity_max_hint_age_tokens: int = 64
    expert_affinity_capture_interval: int = 16
    expert_affinity_max_layers: int = 4
    max_decode_batch_size: int = 0
    engram_hot_cache_manifest: str | None = None
    engram_hot_cache_capacity: int = 0
    engram_hot_cache_model_revision: str | None = None
    dspark_candidate_communication: str = "tiles"

    @property
    def experimental(self) -> bool:
        return (
            self.expert_affinity
            or self.engram_hot_cache_capacity > 0
            or self.dspark_candidate_communication == "compact"
        )

    def validate(self) -> None:
        bounds = {
            "expert_affinity_lookahead": (2, 256),
            "expert_affinity_max_deferrals": (1, 1024),
            "expert_affinity_max_hint_age_tokens": (1, 4096),
            "expert_affinity_capture_interval": (1, 65536),
            "expert_affinity_max_layers": (1, 32),
            "max_decode_batch_size": (0, 2**31 - 1),
            "engram_hot_cache_capacity": (0, 2**31 - 1),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if type(self.expert_affinity) is not bool:
            raise ValueError("expert_affinity must be a boolean")
        if self.dspark_candidate_communication not in ("tiles", "compact"):
            raise ValueError("dspark_candidate_communication must be tiles or compact")
        if self.engram_hot_cache_capacity:
            if (
                not self.engram_hot_cache_manifest
                or not self.engram_hot_cache_model_revision
            ):
                raise ValueError(
                    "Engram hot cache requires manifest and explicit model revision"
                )
        elif (
            self.engram_hot_cache_manifest is not None
            or self.engram_hot_cache_model_revision is not None
        ):
            raise ValueError(
                "Engram manifest/revision requires positive hot cache capacity"
            )
        if not self.expert_affinity:
            defaults = PerformanceOptions()
            for field in fields(self):
                name = field.name
                if name.startswith("expert_affinity_") and getattr(
                    self, name
                ) != getattr(defaults, name):
                    raise ValueError(f"{name} requires expert_affinity")


@dataclass(frozen=True)
class PerformanceCapabilities:
    model_is_deepseek_v41: bool
    nvidia_cuda: bool
    role: str
    speculation: str | None
    speculative_sampling: bool
    pipeline_size: int
    attention_dp: int
    attention_tp: int
    world_size: int
    query_context_size: int
    kv_parallel_size: int
    eager: bool
    prefill_graph_disabled: bool
    overlap_disabled: bool
    mixed_batch: bool
    expert_rebalancing: bool
    custom_expert_placement: bool
    routing_simulation: bool
    host_engram: bool
    host_engram_layout: str
    load_format: str


def validate_performance_options(
    options: PerformanceOptions, caps: PerformanceCapabilities
) -> None:
    options.validate()
    if not options.experimental:
        return
    if not caps.model_is_deepseek_v41 or not caps.nvidia_cuda:
        raise ValueError(
            "Performance candidates require a verified DeepSeek V4.1 checkpoint and NVIDIA CUDA"
        )
    if caps.role == "encode":
        raise ValueError("Performance candidates do not support the encode role")
    if options.expert_affinity:
        if caps.role not in ("null", "aggregate", "decode"):
            raise ValueError("expert_affinity requires aggregate or decode role")
        if caps.speculation is not None:
            raise ValueError("expert_affinity does not support speculation")
        if (
            not (caps.eager and caps.prefill_graph_disabled and caps.overlap_disabled)
            or caps.mixed_batch
        ):
            raise ValueError(
                "expert_affinity requires eager execution, disabled prefill graphs/overlap and no mixed batches"
            )
        if (
            caps.pipeline_size != 1
            or caps.attention_dp != 1
            or caps.query_context_size != 1
            or caps.kv_parallel_size != 1
            or caps.attention_tp != caps.world_size
        ):
            raise ValueError(
                "expert_affinity requires replicated-input attention TP across the worker; DP/PP/QCP/KVP must be 1"
            )
        if (
            caps.expert_rebalancing
            or caps.custom_expert_placement
            or caps.routing_simulation
        ):
            raise ValueError(
                "expert_affinity does not support EPLB, custom expert placement or routing simulation"
            )
    if options.engram_hot_cache_capacity:
        if not caps.host_engram or caps.host_engram_layout not in ("shared", "sharded"):
            raise ValueError(
                "Engram hot cache requires an explicit shared or sharded host table"
            )
        if caps.load_format not in ("auto", "safetensors"):
            raise ValueError("Engram hot cache requires auto or safetensors loading")
    if options.dspark_candidate_communication == "compact":
        if (
            caps.speculation != "DSPARK"
            or caps.pipeline_size != 1
            or caps.speculative_sampling
        ):
            raise ValueError(
                "compact DSpark requires DSPARK, PP1 and greedy draft verification without speculative sampling"
            )


def require_scheduler_performance_support(
    options: PerformanceOptions, config_type: type, scheduler_type: type
) -> None:
    """Refuse opt-ins on older native bindings while preserving their FIFO path."""
    if options.max_decode_batch_size and not hasattr(
        config_type, "max_decode_batch_size"
    ):
        raise ValueError(
            "max_decode_batch_size requires the updated tokenspeed-scheduler; build/install the scheduler from this source checkout"
        )
    if options.expert_affinity and any(
        not hasattr(scheduler_type, name)
        for name in (
            "configure_expert_affinity",
            "update_expert_route",
            "request_generation",
        )
    ):
        raise ValueError(
            "expert_affinity requires the updated tokenspeed-scheduler; build/install the scheduler from this source checkout"
        )
