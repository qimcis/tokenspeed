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

"""Safetensors selection for a draft and its separately loaded vocabulary.

Selecting files happens before downloading or opening tensor payloads. In
particular a target checkpoint is never passed to the full model loader.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


TARGET_VOCABULARY_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


@dataclass(frozen=True)
class CheckpointFiles:
    """Resolved configuration, selected shards and optional tensor filter."""

    config_path: Path
    weight_paths: tuple[Path, ...]
    weight_names: tuple[str, ...] | None


def load_checkpoint_config(
    checkpoint: CheckpointFiles, revision: str, is_draft_worker: bool
):
    """Apply the same metadata normalization used by the target ModelConfig.

    The checkpoint is already resolved to an immutable local snapshot. This
    loads metadata only: no model, vocabulary or additional weight shards.
    Native GLM/DFlash field restoration and Transformer defaults must match
    on both ends before their complete effective configurations are hashed.
    """
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    return get_config(
        model=str(checkpoint.config_path.parent),
        trust_remote_code=False,
        revision=revision,
        model_override_args=None,
        is_draft_worker=is_draft_worker,
        speculative_algorithm="DFLASH" if is_draft_worker else None,
    )


def selected_shards(
    weight_map: Mapping[str, str], weight_names: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Return only shards needed by the requested tensors, failing if absent."""
    names = tuple(weight_map) if weight_names is None else weight_names
    missing = set(names).difference(weight_map)
    if missing:
        raise ValueError(f"Checkpoint is missing weights: {sorted(missing)}")
    shards = sorted({weight_map[name] for name in names})
    for shard in shards:
        path = PurePosixPath(shard)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            raise ValueError(f"Invalid safetensors shard path: {shard!r}")
    return tuple(shards)


def resolve_checkpoint(
    model_path: str, revision: str, weight_names: tuple[str, ...] | None
) -> CheckpointFiles:
    """Resolve pinned local/HF safetensors without fetching unrelated shards.

    Args:
        model_path: Local checkpoint directory or Hugging Face repository.
        revision: Explicit checkpoint revision, also used by the wire contract.
        weight_names: Exact tensor names; None selects the whole draft only.

    Returns:
        Paths to the configuration and the minimal selected shard set.
    """
    if not revision:
        raise ValueError("A checkpoint revision is required.")
    root = Path(model_path)
    if root.is_dir():
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            paths = tuple(
                root / name for name in selected_shards(weight_map, weight_names)
            )
        else:
            paths = tuple(sorted(root.glob("*.safetensors")))
            if len(paths) != 1:
                raise ValueError(
                    "A multi-shard checkpoint requires its safetensors index."
                )
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError("Selected checkpoint shards are unavailable.")
        return CheckpointFiles(config_path, paths, weight_names)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    config_path = Path(hf_hub_download(model_path, "config.json", revision=revision))
    try:
        index_path = Path(
            hf_hub_download(
                model_path, "model.safetensors.index.json", revision=revision
            )
        )
    except EntryNotFoundError:
        shards = ("model.safetensors",)
    else:
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = selected_shards(weight_map, weight_names)
    paths = tuple(
        Path(hf_hub_download(model_path, shard, revision=revision)) for shard in shards
    )
    return CheckpointFiles(config_path, paths, weight_names)


def iter_checkpoint_weights(
    checkpoint: CheckpointFiles,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read only selected tensor payloads, on CPU, once each.

    Uses the same safetensors dependency and per-parameter load hooks as the
    runtime loader. Target shards can contain large expert tensors: those
    payloads must never be materialized merely to reach the vocabulary.
    """
    from safetensors import safe_open

    wanted = None if checkpoint.weight_names is None else set(checkpoint.weight_names)
    seen: set[str] = set()
    for path in checkpoint.weight_paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if wanted is not None and name not in wanted:
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                seen.add(name)
                yield name, handle.get_tensor(name)
    if wanted is not None and wanted != seen:
        raise ValueError(f"Checkpoint is missing weights: {sorted(wanted - seen)}")


def canonical_draft_parameter(name: str) -> str:
    """Map a checkpoint's split QKV/MLP tensors to the native parameter name."""
    name = name.removeprefix("model.")
    for source, destination in (
        (".q_proj.", ".qkv_proj."),
        (".k_proj.", ".qkv_proj."),
        (".v_proj.", ".qkv_proj."),
        (".gate_proj.", ".gate_up_proj."),
        (".up_proj.", ".gate_up_proj."),
    ):
        name = name.replace(source, destination)
    return name


def load_complete_draft_weights(model, checkpoint: CheckpointFiles) -> None:
    """Use native weight hooks and reject incomplete GQA draft checkpoints."""
    expected = set(dict(model.named_parameters()))
    seen: set[str] = set()
    source_names: set[str] = set()

    def weights():
        for name, tensor in iter_checkpoint_weights(checkpoint):
            source_names.add(name.removeprefix("model."))
            canonical = canonical_draft_parameter(name)
            if canonical in expected:
                seen.add(canonical)
            elif "rotary_emb.inv_freq" not in name:
                raise ValueError(f"Unexpected draft tensor: {name}")
            yield name, tensor

    model.load_weights(weights())
    missing = expected - seen
    # One Q/K/V shard must not count as a fully initialized stacked tensor.
    for name in expected:
        if ".qkv_proj." in name and name not in source_names:
            for component in ("q_proj", "k_proj", "v_proj"):
                source = name.replace("qkv_proj", component)
                if source not in source_names:
                    missing.add(source)
        if ".gate_up_proj." in name and name not in source_names:
            for component in ("gate_proj", "up_proj"):
                source = name.replace("gate_up_proj", component)
                if source not in source_names:
                    missing.add(source)
    if missing:
        raise ValueError(f"Incomplete draft checkpoint: {sorted(missing)}")
