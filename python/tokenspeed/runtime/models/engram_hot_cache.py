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

"""Immutable Engram hot rows: strict manifests, bounded byte copies and CPU oracle.

The owner must quiesce forwards and transfers before invalidation/refresh. This
module never replaces resident buffers after construction. Membership changes or
checkpoint revision changes require a new engine, including new CUDA graphs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_ROWS = 4096


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63-1]")
    return value


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class EngramHotSetLayer:
    layer_id: int
    total_rows: int
    global_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.layer_id, "layer_id")
        _integer(self.total_rows, "total_rows", 1)
        if type(self.global_row_ids) is not tuple:
            raise ValueError("global_row_ids must be an immutable tuple")
        for row in self.global_row_ids:
            _integer(row, "global row ID")
            if row >= self.total_rows:
                raise ValueError("Global row ID is outside the layer table")
        if len(set(self.global_row_ids)) != len(self.global_row_ids):
            raise ValueError("Global row IDs must be unique within each layer")


@dataclass(frozen=True)
class EngramHotSetManifest:
    schema_version: int
    model_revision: str
    layers: tuple[EngramHotSetLayer, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Engram manifest schema_version must be 1")
        if type(self.model_revision) is not str or not self.model_revision.strip():
            raise ValueError("Engram manifest requires an explicit model_revision")
        if type(self.layers) is not tuple or not self.layers:
            raise ValueError("Engram manifest requires immutable layer entries")
        if any(not isinstance(layer, EngramHotSetLayer) for layer in self.layers):
            raise ValueError("Invalid Engram manifest layer")
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("Engram manifest layer IDs must be unique")

    @classmethod
    def from_dict(cls, data: object) -> EngramHotSetManifest:
        data = _keys(data, {"schema_version", "model_revision", "layers"}, "manifest")
        if type(data["layers"]) is not list:
            raise ValueError("Manifest layers must be a list")
        layers = []
        for entry in data["layers"]:
            entry = _keys(entry, {"layer_id", "total_rows", "global_row_ids"}, "layer")
            if type(entry["global_row_ids"]) is not list:
                raise ValueError("Manifest global_row_ids must be a list")
            layers.append(
                EngramHotSetLayer(
                    entry["layer_id"],
                    entry["total_rows"],
                    tuple(entry["global_row_ids"]),
                )
            )
        return cls(data["schema_version"], data["model_revision"], tuple(layers))

    @classmethod
    def load(cls, path: str | Path) -> EngramHotSetManifest:
        """Read at most 16 MiB; reject duplicate JSON fields and unknown fields."""
        with Path(path).open("rb") as stream:
            raw = stream.read(_MANIFEST_MAX_BYTES + 1)
        if len(raw) > _MANIFEST_MAX_BYTES:
            raise ValueError("Engram manifest exceeds 16 MiB")
        return cls.from_dict(json.loads(raw, object_pairs_hook=_unique_object))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "layers": [
                {
                    "layer_id": layer.layer_id,
                    "total_rows": layer.total_rows,
                    "global_row_ids": list(layer.global_row_ids),
                }
                for layer in self.layers
            ],
        }


def build_engram_hot_set_manifest(
    *,
    model_revision: str,
    layer_rows: Mapping[int, int],
    frequencies: Mapping[int, Mapping[int, int]],
    capacity_rows: int,
) -> EngramHotSetManifest:
    """Choose up to capacity_rows global rows per layer from offline counts.

    Higher counts win; ties choose the lower global ID. Zero-count rows are
    omitted. Inputs must already be bounded/aggregated by the trace collector;
    this offline builder never runs on the serving thread.
    """
    _integer(capacity_rows, "capacity_rows")
    if set(frequencies) - set(layer_rows):
        raise ValueError("Frequency observations contain an unknown layer")
    layers = []
    for layer_id, rows in sorted(layer_rows.items()):
        _integer(layer_id, "layer_id")
        _integer(rows, "total_rows", 1)
        counts = frequencies.get(layer_id, {})
        for row, count in counts.items():
            _integer(row, "global row ID")
            _integer(count, "frequency")
            if row >= rows:
                raise ValueError("Frequency row is outside the layer table")
        selected = sorted(
            (row for row, count in counts.items() if count),
            key=lambda row: (-counts[row], row),
        )[:capacity_rows]
        layers.append(EngramHotSetLayer(layer_id, rows, tuple(sorted(selected))))
    return EngramHotSetManifest(1, model_revision, tuple(layers))


@dataclass(frozen=True)
class EngramHotCachePlan:
    model_revision: str
    layer_id: int
    total_rows: int
    row_start: int
    row_end: int
    capacity_rows: int
    embedding_dim: int
    global_row_ids: tuple[int, ...]
    local_row_ids: tuple[int, ...]

    @property
    def resident_nbytes(self) -> int:
        """Codes, scales and int64 search keys on this rank (no full-table map)."""
        return len(self.local_row_ids) * (
            self.embedding_dim + self.embedding_dim // 32 + 8
        )

    @property
    def staging_nbytes(self) -> int:
        """Bound on explicit CPU code/scale and int64 index staging per copy chunk."""
        return min(len(self.local_row_ids), _COPY_CHUNK_ROWS) * (
            self.embedding_dim + self.embedding_dim // 32 + 8
        )


def plan_engram_hot_cache(
    manifest: EngramHotSetManifest,
    *,
    layer_id: int,
    model_revision: str,
    total_rows: int,
    row_start: int,
    row_end: int,
    capacity_rows: int,
    embedding_dim: int,
) -> EngramHotCachePlan:
    """Validate identity and convert global manifest rows to sorted owned rows.

    Capacity is a per-layer, per-rank bound after ownership filtering. Shared
    tables use [0,total_rows), so every participating rank owns a resident copy.
    Zero capacity selects no rows and lets the runtime retain its baseline path.
    """
    for name, value in (
        ("layer_id", layer_id),
        ("row_start", row_start),
        ("row_end", row_end),
        ("capacity_rows", capacity_rows),
    ):
        _integer(value, name)
    _integer(total_rows, "total_rows", 1)
    _integer(embedding_dim, "embedding_dim", 32)
    if embedding_dim % 32 or not 0 <= row_start <= row_end <= total_rows:
        raise ValueError("Invalid Engram row ownership or scale geometry")
    if model_revision != manifest.model_revision:
        raise ValueError("Engram manifest model revision mismatch")
    layer = next(
        (entry for entry in manifest.layers if entry.layer_id == layer_id), None
    )
    if layer is None or layer.total_rows != total_rows:
        raise ValueError("Engram manifest layer identity or row count mismatch")
    owned = (
        tuple(sorted(row for row in layer.global_row_ids if row_start <= row < row_end))
        if capacity_rows
        else ()
    )
    if len(owned) > capacity_rows:
        raise ValueError("Engram hot rows exceed this rank's capacity")
    return EngramHotCachePlan(
        model_revision,
        layer_id,
        total_rows,
        row_start,
        row_end,
        capacity_rows,
        embedding_dim,
        owned,
        tuple(row - row_start for row in owned),
    )


class EngramHotCache:
    """Raw immutable resident membership with explicit, pointer-stable refresh.

    Only the CPU reference methods count hits; production counters belong to the
    fused kernel. There is no adaptive replacement or per-forward CPU transfer.
    """

    def __init__(
        self,
        plan: EngramHotCachePlan,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        device: torch.device | str,
        generation: int,
    ):
        self.plan = plan
        self.valid = False
        self.generation = -1
        self._validate_tables(weight, scale)
        self.local_row_ids = torch.empty(
            (len(plan.local_row_ids),), dtype=torch.int64, device=device
        )
        self.codes = torch.empty(
            (len(plan.local_row_ids), plan.embedding_dim),
            dtype=torch.uint8,
            device=device,
        )
        self.scales = torch.empty(
            (len(plan.local_row_ids), plan.embedding_dim // 32),
            dtype=torch.uint8,
            device=device,
        )
        self.refresh_from_tables(
            weight, scale, model_revision=plan.model_revision, generation=generation
        )

    def _validate_tables(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        if weight.dtype not in (
            torch.uint8,
            torch.float8_e4m3fn,
        ) or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError(
                "Engram cache requires raw FP8 E4M3 codes and E8M0 scale bytes"
            )
        if weight.device.type != "cpu" or scale.device.type != "cpu":
            raise ValueError("Engram hot cache requires canonical CPU tables")
        if (
            weight.ndim != 2
            or scale.ndim != 2
            or weight.shape[0] != scale.shape[0]
            or weight.shape[0] < self.plan.row_end - self.plan.row_start
            or weight.shape[1] != self.plan.embedding_dim
            or scale.shape[1] != self.plan.embedding_dim // 32
            or not weight.is_contiguous()
            or not scale.is_contiguous()
        ):
            raise ValueError(
                "Engram cache table geometry or contiguous layout mismatch"
            )

    def invalidate(self) -> None:
        """Disable device hits in place; caller must first quiesce graph replay."""
        self.valid = False
        self.local_row_ids.fill_(-1)

    def refresh_from_tables(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        model_revision: str,
        generation: int,
    ) -> None:
        """Copy both loaded tables at a caller-established quiescent boundary.

        Any failure leaves the cache invalid. The owner must finish shared-host
        publication and synchronize device copies before admitting graph replay.
        """
        self.invalidate()
        _integer(generation, "generation")
        if model_revision != self.plan.model_revision or generation <= self.generation:
            raise ValueError(
                "Engram refresh requires matching revision and newer generation"
            )
        self._validate_tables(weight, scale)
        for start in range(0, len(self.plan.local_row_ids), _COPY_CHUNK_ROWS):
            rows = torch.tensor(
                self.plan.local_row_ids[start : start + _COPY_CHUNK_ROWS],
                dtype=torch.int64,
                device="cpu",
            )
            end = start + rows.numel()
            self.codes[start:end].copy_(weight.view(torch.uint8).index_select(0, rows))
            self.scales[start:end].copy_(scale.view(torch.uint8).index_select(0, rows))
        try:
            for start in range(0, len(self.plan.local_row_ids), _COPY_CHUNK_ROWS):
                rows = torch.tensor(
                    self.plan.local_row_ids[start : start + _COPY_CHUNK_ROWS],
                    dtype=torch.int64,
                    device="cpu",
                )
                self.local_row_ids[start : start + rows.numel()].copy_(rows)
        except Exception:
            self.invalidate()
            raise
        self.generation = generation
        self.valid = True

    def gather_bytes_reference(
        self, weight: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU oracle: return code bytes, scale bytes, owned mask and hit mask.

        Indices are global IDs of arbitrary shape; repeated IDs remain repeated.
        Only misses read canonical host rows. Non-owned IDs yield zero codes and
        neutral scale bytes (127); hit/miss counts exclude these masked rows.
        """
        if not self.valid:
            raise RuntimeError("Engram hot cache is invalid; reload both tables first")
        self._validate_tables(weight, scale)
        if indices.device.type != "cpu" or self.codes.device.type != "cpu":
            raise ValueError("Engram cache reference requires CPU indices and cache")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram indices must be int32 or int64")
        flat = indices.reshape(-1).long()
        owned = (flat >= self.plan.row_start) & (flat < self.plan.row_end)
        local = flat - self.plan.row_start
        slots = torch.searchsorted(self.local_row_ids, local)
        hits = torch.zeros_like(owned)
        if self.local_row_ids.numel():
            safe = slots.clamp(max=self.local_row_ids.numel() - 1)
            hits = (
                owned
                & (slots < self.local_row_ids.numel())
                & (self.local_row_ids[safe] == local)
            )
        codes = torch.zeros((flat.numel(), self.plan.embedding_dim), dtype=torch.uint8)
        scales = torch.full(
            (flat.numel(), self.plan.embedding_dim // 32), 127, dtype=torch.uint8
        )
        codes[hits] = self.codes[slots[hits]]
        scales[hits] = self.scales[slots[hits]]
        misses = owned & ~hits
        codes[misses] = weight.view(torch.uint8)[local[misses]]
        scales[misses] = scale.view(torch.uint8)[local[misses]]
        return (
            codes.reshape(*indices.shape, self.plan.embedding_dim),
            scales.reshape(*indices.shape, self.plan.embedding_dim // 32),
            owned.reshape(indices.shape),
            hits.reshape(indices.shape),
        )

    def lookup_reference(
        self, weight: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """CPU BF16 oracle using precisely the existing FP8/E8M0 torch casts."""
        codes, scales, owned, _ = self.gather_bytes_reference(weight, scale, indices)
        values = codes.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
        exponents = scales.view(torch.float8_e8m0fnu).float()
        result = (values * exponents.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return result.masked_fill(~owned.unsqueeze(-1), 0)
