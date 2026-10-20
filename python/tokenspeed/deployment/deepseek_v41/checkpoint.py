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

"""Inventory checkpoint storage without loading tensors or importing torch."""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path


def tensor_category(name: str) -> str:
    """Group checkpoint constituents using the V4.1 loader's namespaces.

    Draft experts stay with the draft role. Engram counts only table codes and
    scales; its auxiliary projections remain in dense/shared storage.
    """
    name = name.removeprefix("model.")
    if name.startswith("mtp."):
        return "draft"
    if name.endswith((".engram.embed.weight", ".engram.embed.scale")):
        return "engram"
    if name.startswith(("vision.", "aligner.")) or name in (
        "image_start",
        "image_end",
        "image_newline",
    ):
        return "vision"
    if ".ffn.experts." in name:
        return "routed_experts"
    return "dense_shared_and_other"


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(data: str | bytes, label: str) -> dict:
    try:
        result = json.loads(data, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def audit_checkpoint(directory: Path) -> dict:
    """Read indexed safetensors headers and account for actual packed byte spans.

    This reports checkpoint bytes, not runtime residency: sharding, replication,
    conversion and workspace costs belong to the runtime memory ledger. Dtypes
    and shapes are recorded as stored; packed formats are never expanded using
    logical model dimensions. This is a storage inventory, not a weight-loader
    compatibility or complete model-coverage check.
    """
    index = _read_json_object(
        (directory / "model.safetensors.index.json").read_bytes(), "checkpoint index"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index must contain a nonempty weight map")
    for name, file in weight_map.items():
        if (
            not name
            or name == "__metadata__"
            or not isinstance(file, str)
            or Path(file).name != file
            or "\\" in file
            or not file.endswith(".safetensors")
        ):
            raise ValueError("invalid tensor name or checkpoint shard path")
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata must be an object")
    declared = metadata.get("total_size")
    if declared is not None and (type(declared) is not int or declared < 0):
        raise ValueError("checkpoint index total_size must be a nonnegative integer")
    categories = defaultdict(int)
    dtypes = defaultdict(int)
    seen = set()
    for shard in sorted(set(weight_map.values())):
        path = directory / shard
        with path.open("rb") as source:
            size = path.stat().st_size
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"truncated safetensors header: {shard}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 100 * 2**20 or length + 8 > size:
                raise ValueError(f"invalid safetensors header size: {shard}")
            header = _read_json_object(
                source.read(length), f"safetensors header in {shard}"
            )
        header_metadata = header.get("__metadata__", {})
        if not isinstance(header_metadata, dict) or any(
            not isinstance(value, str) for value in header_metadata.values()
        ):
            raise ValueError(f"invalid safetensors metadata in {shard}")
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != shard or name in seen:
                raise ValueError(f"tensor/index mismatch: {name}")
            if not isinstance(tensor, dict):
                raise ValueError(f"invalid tensor descriptor: {name}")
            offsets = tensor.get("data_offsets")
            shape, dtype = tensor.get("shape"), tensor.get("dtype")
            if (
                not isinstance(shape, list)
                or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(dtype, str)
                or not dtype
            ):
                raise ValueError(f"invalid tensor shape or dtype: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length
            ):
                raise ValueError(f"invalid tensor extent: {name}")
            start, end = offsets
            seen.add(name)
            spans.append((start, end))
            categories[tensor_category(name)] += end - start
            dtypes[dtype] += end - start
        previous_end = 0
        for start, end in sorted(spans):
            if start != previous_end:
                raise ValueError(f"overlapping or missing tensor data in {shard}")
            previous_end = end
        if previous_end != size - 8 - length:
            raise ValueError(f"unindexed payload in {shard}")
    if seen != weight_map.keys():
        raise ValueError("checkpoint shards omit indexed tensors")
    total = sum(categories.values())
    if declared is not None and declared != total:
        raise ValueError("checkpoint index total_size disagrees with tensor storage")
    return {
        "status": "checkpoint_storage_not_runtime_residency",
        "tensor_count": len(seen),
        "total_bytes": total,
        "category_bytes": dict(categories),
        "dtype_bytes": dict(dtypes),
    }
