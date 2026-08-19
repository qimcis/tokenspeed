# Copyright (c) 2026 LightSeek Foundation

"""Stable names and keys for distributed cache objects."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)
_CACHE_ABI_VERSION = "tokenspeed-cache-abi-v2"
_NAMESPACE_DIGEST_HEX = 32


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unknown"


def build_store_namespace(
    *,
    model_id: str | None,
    model_revision: str | None,
    model_fingerprint: str | None,
    cache_abi_fingerprint: str | None,
    extra_tag: str | None,
) -> str:
    if model_id is None or not str(model_id).strip():
        raise ValueError("L3 Store namespace requires a model identifier")
    model_value = str(model_id)
    resolved_revision = model_revision
    root = Path(model_value)
    if root.is_dir():
        if root.parent.name == "snapshots":
            resolved_revision = resolved_revision or root.name
            repository = root.parent.parent.name
            if repository.startswith("models--"):
                model_value = repository[len("models--") :].replace("--", "/")
            else:
                model_value = root.parent.parent.name
        else:
            # Local mount points differ across workers; the deterministic
            # artifact fingerprint below disambiguates same-basename models.
            model_value = root.name
    model = sanitize_component(model_value)
    tag = sanitize_component(extra_tag) if extra_tag else "default"
    # Hash canonical values before sanitizing anything. Sanitization is only
    # for the human-readable key prefix; it must not collapse distinct IDs.
    raw = "\0".join(
        (
            model_value,
            str(resolved_revision or ""),
            str(model_fingerprint or ""),
            str(cache_abi_fingerprint or ""),
            str(extra_tag or ""),
        )
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_NAMESPACE_DIGEST_HEX]
    return f"{tag}_{model}_{digest}"


def fingerprint_model_artifacts(model_id: str | None) -> str | None:
    """Version local weights and processor/config artifacts deterministically."""
    if not model_id:
        return None
    root = Path(model_id)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in (
        "*.json",
        "*.py",
        "*.safetensors.index.json",
        "*.bin.index.json",
        "*.safetensors",
        "*.bin",
        "*.pt",
    ):
        candidates.extend(root.rglob(pattern))
    candidates = sorted(
        set(candidates), key=lambda path: path.relative_to(root).as_posix()
    )
    if not candidates:
        return None
    digest = hashlib.sha256()
    immutable_hf_snapshot = root.parent.name == "snapshots" and bool(root.name)
    for path in candidates:
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "little"))
            digest.update(relative)
            digest.update(stat.st_size.to_bytes(8, "little"))
            with path.open("rb") as handle:
                if immutable_hf_snapshot and path.suffix not in (".json", ".py"):
                    # HF snapshots are commit-addressed and large files are
                    # symlinks into content-addressed blobs. Hash that identity
                    # instead of reading a multi-terabyte checkpoint at startup.
                    resolved = path.resolve()
                    blob_identity = resolved.name.encode("utf-8")
                    digest.update(len(blob_identity).to_bytes(4, "little"))
                    digest.update(blob_identity)
                else:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
        except OSError as exc:
            logger.warning("L3 namespace could not fingerprint %s: %s", path, exc)
            return None
    return digest.hexdigest()[:_NAMESPACE_DIGEST_HEX]


def fingerprint_cache_layout(layout: Any, *, runtime_tags: tuple[str, ...] = ()) -> str:
    parts: list[str] = [_CACHE_ABI_VERSION]
    for group in getattr(layout, "groups", ()):
        fields = getattr(group, "fields", ())
        parts.append(f"group:{group.group_id}:{group.cache_blocks_per_lcm_block}")
        for field in fields:
            parts.append(
                "field:"
                f"{field.field_id}:{field.device_buffer_index}:"
                f"{field.device_block_zero_offset_bytes}:"
                f"{field.block_stride_bytes}:{field.payload_bytes}:"
                f"{tuple(getattr(field, 'shape', ()))}:"
                f"{getattr(field, 'element_size', '')}"
            )
    for consumer in getattr(layout, "consumers", ()):
        parts.append(f"consumer:{','.join(consumer)}")
    parts.extend(f"runtime:{tag}" for tag in runtime_tags)
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_NAMESPACE_DIGEST_HEX]


def store_key(
    content_hash: str,
    group_id: str,
    cache_block_offset: int,
    tp_rank: int | None = None,
    *,
    namespace: str | None = None,
) -> str:
    base = (
        f"{content_hash}_{group_id}"
        if cache_block_offset == 0
        else f"{content_hash}_{group_id}_o{cache_block_offset}"
    )
    if tp_rank is not None:
        base = f"{base}_tp{tp_rank}"
    return f"{namespace}:{base}" if namespace else base
