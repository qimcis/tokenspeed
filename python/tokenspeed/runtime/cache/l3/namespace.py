# Copyright (c) 2026 LightSeek Foundation

"""Stable names and keys for distributed cache objects."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


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
    model = sanitize_component(str(model_id))
    revision = sanitize_component(str(model_revision)) if model_revision else "default"
    abi = (
        sanitize_component(str(cache_abi_fingerprint))
        if cache_abi_fingerprint
        else "unknown-abi"
    )
    fingerprint = (
        sanitize_component(model_fingerprint) if model_fingerprint else "unknown-model"
    )
    tag = sanitize_component(extra_tag) if extra_tag else "default"
    raw = f"{model}@{revision}:{fingerprint}:{abi}:{tag}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{tag}_{model}_{digest}"


def fingerprint_model_artifacts(model_id: str | None) -> str | None:
    """Version a local checkpoint using identity and bounded content samples."""
    if not model_id:
        return None
    root = Path(model_id)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in (
        "config.json",
        "*.safetensors.index.json",
        "*.bin.index.json",
        "*.safetensors",
        "*.bin",
        "*.pt",
    ):
        candidates.extend(root.glob(pattern))
    candidates = sorted(set(candidates), key=lambda path: path.name)
    if not candidates:
        return None
    digest = hashlib.sha256()
    sample_bytes = 1024 * 1024
    for path in candidates:
        try:
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}:".encode("ascii"))
            with path.open("rb") as handle:
                if stat.st_size <= sample_bytes * 2:
                    digest.update(handle.read())
                else:
                    digest.update(handle.read(sample_bytes))
                    handle.seek(-sample_bytes, os.SEEK_END)
                    digest.update(handle.read(sample_bytes))
        except OSError as exc:
            logger.warning("L3 namespace could not fingerprint %s: %s", path, exc)
            return None
    return digest.hexdigest()[:16]


def fingerprint_cache_layout(layout: Any) -> str:
    parts: list[str] = []
    for group in getattr(layout, "groups", ()):
        fields = getattr(group, "fields", ())
        field_ids = ",".join(sorted(getattr(field, "field_id", "") for field in fields))
        payloads = ",".join(
            str(getattr(field, "payload_bytes", "")) for field in fields
        )
        strides = ",".join(
            str(getattr(field, "block_stride_bytes", "")) for field in fields
        )
        parts.append(
            f"{group.group_id}:{group.cache_blocks_per_lcm_block}:"
            f"{field_ids}:{payloads}:{strides}"
        )
    raw = "|".join(parts) if parts else "empty-layout"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
