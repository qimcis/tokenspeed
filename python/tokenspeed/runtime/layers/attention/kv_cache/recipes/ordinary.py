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

"""The four ordinary cache families: MHA, MLA, DSA, MSA.

One recipe serves all four, and a heterogeneous draft too: what a layer costs
is dispatched on the attention config that owns it, so an MLA target with an
MHA draft is just layers with two different geometries in one plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    MXFP8_KV_SCALE_TILE_TOKENS,
    CacheGroupDeclaration,
    hybrid_slab_group_size,
    layer_group_ids,
)


class OrdinaryRecipe(CacheRecipe):
    """MHA / MLA / DSA / MSA: one cache group per attention structure.

    Capacity comes from the profiled bytes-per-token rather than the parent
    size, and every group packs one CacheBlock per parent -- the identity
    grain is the block span.
    """

    def __init__(self, *, family, **kwargs) -> None:
        super().__init__(**kwargs)
        self.family = family

    # ---- layer vocabulary ----

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        ids = _config_group_ids(self.attn_config, self.num_target_layers)
        if self.draft_attn_config is None:
            return ids
        if self.draft_attn_config.prefix_granularity != self.prefix_granularity:
            raise ValueError("target and draft prefix granularities must match")
        return ids + _config_group_ids(self.draft_attn_config, self.num_draft_layers)

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        """Merged labels, target then draft, always one per layer.

        A side whose config labels cannot align per layer resolves to
        full-history rather than mislabeling a group: plain MLA/DSA configs
        declare no labels at all, and a NextN draft inherits the target
        hf_config's ``layer_types`` (one draft layer against 61 target
        labels).
        """
        target = tuple(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)
        if len(target) != self.num_target_layers:
            target = (FULL_ATTENTION,) * self.num_target_layers
        if self.draft_attn_config is None:
            return target
        draft = tuple(
            self.draft_attn_config.component(SoftmaxAttnConfig).cache_layer_types
        )
        if len(draft) != self.num_draft_layers:
            draft = (FULL_ATTENTION,) * self.num_draft_layers
        return target + draft

    @property
    def remote_feature_cache_enabled(self) -> bool:
        return bool(getattr(self.server_args, "remote_draft_endpoint", None))

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        groups = super().groups()
        if not self.remote_feature_cache_enabled:
            return groups
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
            PROJECTED_FEATURE_BLOCKS_PER_PARENT,
            aliased_projected_feature_group,
        )

        if self.draft_attn_config is not None:
            raise ValueError("remote drafting must not allocate local draft KV layers")
        if self.draft_model_config is None:
            raise ValueError("remote feature cache requires the draft model config")
        if len(groups) != 1:
            raise ValueError("remote features require one target history group")
        return groups + (
            aliased_projected_feature_group(
                target_group=groups[0],
                blocks_per_parent=PROJECTED_FEATURE_BLOCKS_PER_PARENT,
                block_granularity=self.prefix_granularity,
                hidden_size=int(self.model_config.hidden_size),
                sliding_window=int(self.draft_model_config.hf_config.sliding_window),
            ),
        )

    # ---- geometry ----

    @property
    @override
    def alignment(self) -> int:
        return 1

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 1.0

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        """Keep layer blocks at one per parent; derive non-layer packing."""
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
            PROJECTED_FEATURE_BLOCKS_PER_PARENT,
            PROJECTED_FEATURE_GROUP,
        )

        # LCM parents are group-exclusive. Feature blocks alias existing
        # target planes at fixed packing without enlarging target parents.
        return {
            spec.group_id: (
                PROJECTED_FEATURE_BLOCKS_PER_PARENT
                if spec.group_id == PROJECTED_FEATURE_GROUP
                else 1
            )
            for spec, _ in groups
        }

    # ---- fields ----

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id < self.num_target_layers:
            config, local_layer_id = self.attn_config, layer_id
        else:
            config = self.draft_attn_config
            local_layer_id = layer_id - self.num_target_layers
        return _config_layer_fields(
            config,
            layer_id=layer_id,
            local_layer_id=local_layer_id,
            occurrence=occurrence,
        )

    # ---- capacity: profiled bytes per token, not parent size ----

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        if self.remote_feature_cache_enabled:
            # Projected features are another physical consumer: the old
            # attention-only profiled cell size would overrun this budget.
            budgeted = self._budgeted_parents(
                self.cache_budget_bytes - self.workspace_bytes(), layout.lcm_block_bytes
            )
            if self.token_limit is None:
                return budgeted
            return min(budgeted, self.parents_needed(layout, self.token_limit))
        bytes_per_token = self.attn_config.cache_cell_size() * _storage_layers(
            self.attn_config, self.num_target_layers
        )
        if self.draft_attn_config is not None:
            bytes_per_token += (
                self.draft_attn_config.cache_cell_size()
                * _storage_layers(self.draft_attn_config, self.num_draft_layers)
            )
        if bytes_per_token <= 0:
            raise ValueError(
                f"KV cache cell size must be positive, got {bytes_per_token}"
            )
        # Every group packs one CacheBlock per parent, so a parent spans the
        # identity grain and profiled bytes/token size it directly.
        parent_tokens = self.prefix_granularity
        return self._capped_parents(
            self._budgeted_parents(
                self.cache_budget_bytes, bytes_per_token * parent_tokens
            ),
            parent_tokens=parent_tokens,
        )

    @override
    def workspace_bytes(self) -> int:
        if not self.remote_feature_cache_enabled:
            return super().workspace_bytes()
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.projected_features import (
            projected_feature_workspace_bytes,
        )

        decode_tokens = self.attn_config.max_bs * self.decode_input_tokens
        forward_tokens = max(int(self.server_args.chunked_prefill_size), decode_tokens)
        graph_enabled = not self.server_args.enforce_eager
        prefill_graph_enabled = (
            graph_enabled
            and not self.server_args.disable_prefill_graph
            and self.server_args.all2all_backend in (None, "none")
            and self.server_args.prefill_graph_max_tokens != 0
        )
        # The configured chunk is an upper bound on every prefill graph bucket.
        # Avoid depending on execution's default-bucket policy in the cache recipe.
        prefill_graph_tokens = (
            min(
                (
                    int(self.server_args.prefill_graph_max_tokens)
                    if self.server_args.prefill_graph_max_tokens is not None
                    else forward_tokens
                ),
                forward_tokens,
            )
            if prefill_graph_enabled
            else 0
        )
        address_workspace = projected_feature_workspace_bytes(
            hidden_size=int(self.model_config.hidden_size),
            window_left=int(self.draft_model_config.hf_config.sliding_window) - 1,
            max_forward_tokens=forward_tokens,
            max_decode_graph_tokens=decode_tokens if graph_enabled else 0,
            max_prefill_graph_tokens=prefill_graph_tokens,
        )
        feature_table_columns = (
            self.attn_config.context_len + self.prefix_granularity - 1
        ) // self.prefix_granularity
        feature_metadata_bytes = self.attn_config.max_bs * (
            feature_table_columns * 4 + 1
        )
        return address_workspace + feature_metadata_bytes

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        if not self.remote_feature_cache_enabled:
            return super().token_capacity(layout, num_lcm_blocks)
        # Attention history and the feature window acquire distinct group
        # blocks from the same LCM allocator. Charge both demands, including
        # live windows, prefill chunks and overlapped decode reservations.
        upper_bound = super().token_capacity(layout, num_lcm_blocks)
        if self.token_limit is not None:
            upper_bound = min(upper_bound, self.token_limit)
        return self._capacity_from_parents(
            layout, num_lcm_blocks, upper_bound=upper_bound
        )


def _storage_layers(config, num_layers: int) -> int:
    spec = config.component(SoftmaxAttnConfig)
    group_size = hybrid_slab_group_size(
        spec.cache_layer_types,
        sliding_window_tokens=spec.sliding_window_tokens,
    )
    return group_size if group_size is not None else num_layers


def _config_group_ids(config, num_layers: int) -> tuple[str, ...]:
    """Per-layer group ids for one ordinary attention config."""
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, MHAConfig | MSAConfig):
        layer_types = tuple(spec.cache_layer_types)
        if layer_types:
            ids = tuple(
                layer_group_ids(
                    layer_types=layer_types,
                    sliding_window_tokens=spec.sliding_window_tokens,
                )
            )
            if len(ids) != num_layers:
                raise ValueError("cache group ids must cover every layer")
            return ids
    return (FULL_ATTENTION,) * num_layers


def _config_layer_fields(
    config, *, layer_id: int, local_layer_id: int, occurrence: int
) -> tuple[CacheFieldSpec, ...]:
    """What one layer costs, dispatched on the config that owns the layer."""
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, DSAConfig):
        return _mla_layer_fields(config, layer_id, occurrence) + (
            _index_k_field(config, layer_id),
        )
    if isinstance(spec, MSAConfig):
        fields = _mha_layer_fields(config, layer_id, occurrence)
        if local_layer_id in spec.sparse_layer_ids:
            fields += (_index_k_field(config, layer_id),)
        return fields
    if isinstance(spec, MLAConfig):
        return _mla_layer_fields(config, layer_id, occurrence)
    if isinstance(spec, MHAConfig):
        return _mha_layer_fields(config, layer_id, occurrence)
    raise TypeError(f"no ordinary cache recipe for {type(spec).__name__}")


def _mha_layer_fields(config, layer_id: int, occurrence: int):
    """One MHA layer's K/V pages, with mxfp8 scale planes when enabled."""
    spec = config.component(SoftmaxAttnConfig)
    mxfp8 = bool(config.kv_cache_mxfp8)
    if mxfp8 and config.prefix_granularity != MXFP8_KV_SCALE_TILE_TOKENS:
        raise AssertionError(
            "mxfp8 KV cache requires --prefix-granularity "
            f"{MXFP8_KV_SCALE_TILE_TOKENS} (the attention kernel consumes "
            "the interleaved paged scale layout)"
        )
    kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
    head_dim = spec.head_dim
    if config.prefix_granularity <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("MHA full-attention geometry must be positive")
    shape = (config.prefix_granularity, kv_heads, head_dim)
    kv_dtype = (
        # MXFP8 writes go through dtype-aware kernels, so the arena keeps the
        # fp8 view; the scatter-written paths fall back to uint8.
        cache_dtype_name(torch.float8_e4m3fn)
        if mxfp8
        else scatter_stored_dtype_name(config.kv_cache_dtype)
    )
    fields = (
        CacheFieldSpec(f"layer.{layer_id}.k", f"unit.{occurrence}.k", shape, kv_dtype),
        CacheFieldSpec(f"layer.{layer_id}.v", f"unit.{occurrence}.v", shape, kv_dtype),
    )
    if not mxfp8:
        return fields
    return fields + mxfp8_kv_scale_fields(
        layer_id=layer_id,
        occurrence=occurrence,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_granularity=config.prefix_granularity,
    )


def _mla_layer_fields(config, layer_id: int, occurrence: int):
    """One MLA layer's latent page, split into planes when quantized."""
    spec = config.component(SoftmaxAttnConfig)
    if config.prefix_granularity <= 0:
        raise ValueError("MLA full-attention geometry must be positive")
    if config.kv_cache_quant_method != "per_token_head":
        latent_width = spec.kv_lora_rank + spec.qk_rope_head_dim
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.latent_kv",
                f"slot.{occurrence}",
                (config.prefix_granularity, 1, latent_width),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            ),
        )
    return tuple(
        CacheFieldSpec(
            f"layer.{layer_id}.{name}",
            f"layer.{layer_id}.{name}",
            shape,
            dtype,
        )
        for name, shape, dtype in (
            (
                "latent_kv",
                (config.prefix_granularity, 1, spec.kv_lora_rank),
                scatter_stored_dtype_name(config.kv_cache_dtype),
            ),
            (
                "latent_scale",
                (config.prefix_granularity, 1, 1),
                cache_dtype_name(torch.float32),
            ),
            (
                "rope_k",
                (config.prefix_granularity, 1, spec.qk_rope_head_dim),
                cache_dtype_name(config.dtype),
            ),
        )
    )


def _index_k_field(config, layer_id: int) -> CacheFieldSpec:
    """The sparse indexer's key row for one layer (DSA bytes, MSA elements)."""
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        DSAConfig,
        dsa_index_k_row_bytes,
    )

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, DSAConfig):
        return CacheFieldSpec(
            f"layer.{layer_id}.index_k",
            f"layer.{layer_id}.index_k",
            (config.prefix_granularity, dsa_index_k_row_bytes(spec.index_head_dim)),
            "uint8",
        )
    return CacheFieldSpec(
        f"layer.{layer_id}.index_k",
        f"layer.{layer_id}.index_k",
        (config.prefix_granularity, spec.index_head_dim),
        cache_dtype_name(config.dtype),
    )
