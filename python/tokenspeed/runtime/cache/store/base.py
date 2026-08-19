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

"""Store abstraction and factory."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenspeed.runtime.cache.store.errors import KVStoreBackendError


class BaseKVStore(ABC):
    extra_backend_tag: str | None = None

    @property
    def supports_device_memory(self) -> bool:
        """Whether registered CUDA pointers are valid Store I/O buffers."""
        return False

    @property
    def supports_buffer_unregistration(self) -> bool:
        return False

    @abstractmethod
    def batch_exists(self, keys: list[str]) -> list[int]:
        """Return 1 if key exists, 0 otherwise, per key."""

    @abstractmethod
    def batch_get_into(
        self,
        keys: list[str],
        buffer_ptrs: list[Any],
        buffer_sizes: list[Any],
    ) -> list[int]:
        """Zero-copy get into host buffers. Returns bytes per key or -1."""

    @abstractmethod
    def batch_put_from(
        self,
        keys: list[str],
        buffer_ptrs: list[Any],
        buffer_sizes: list[Any],
    ) -> list[int]:
        """Zero-copy put from host buffers. Returns 0 on success, -1 on error."""

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return self.batch_exists(keys)

    def register_buffer(self, ptr: int, size: int) -> int:
        """Register a host buffer with the Store client. Returns 0 on success."""
        return 0

    def unregister_buffer(self, ptr: int) -> int:
        """Unregister a previously registered Store I/O buffer."""
        return 0

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class MooncakeStoreConfig:
    master_server_address: str | None = None
    client_server_address: str | None = None
    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    global_segment_size: str = "4gb"
    protocol: str = "tcp"
    device_name: str = ""
    master_metrics_port: int = 9003
    transfer_timeout_seconds: int = 30
    check_server: bool = False
    standalone_storage: bool = False
    enable_ssd_offload: bool = False
    ssd_offload_path: str | None = None
    extra_backend_tag: str | None = None


def _parse_extra_config(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("@"):
        config_path = Path(raw[1:]).expanduser()
        try:
            raw = config_path.read_text()
        except OSError as exc:
            raise ValueError(
                f"cannot read KVStore config {config_path}: {exc}"
            ) from exc
    # Launchers and process supervisors sometimes preserve the shell's matching
    # quote characters as part of argv. Accept that harmless representation,
    # while still requiring strict JSON for the actual payload.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid --kvstore-storage-backend-extra-config JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--kvstore-storage-backend-extra-config must be a JSON object")
    # Allow both flat and nested: {"mooncake_store_config": {...}} or {...}
    if "mooncake_store_config" in parsed and isinstance(
        parsed["mooncake_store_config"], dict
    ):
        return parsed["mooncake_store_config"]
    # Also unwrap {"backend_name":..., "module_path":..., "class_name":...} style
    if "extra_config" in parsed and isinstance(parsed["extra_config"], dict):
        return parsed["extra_config"]
    return parsed


def load_mooncake_store_config(
    extra_config_raw: str | None = None,
) -> MooncakeStoreConfig | None:
    import os

    # 1) extra_config JSON
    extra = _parse_extra_config(extra_config_raw)
    if extra is not None and (
        extra.get("master_server_address") is not None
        or extra.get("client_server_address") is not None
    ):
        return MooncakeStoreConfig(
            master_server_address=extra.get("master_server_address"),
            client_server_address=extra.get("client_server_address"),
            local_hostname=extra.get("local_hostname", "localhost"),
            metadata_server=extra.get("metadata_server", "P2PHANDSHAKE"),
            global_segment_size=extra.get("global_segment_size", "4gb"),
            protocol=extra.get("protocol", "tcp"),
            device_name=extra.get("device_name", ""),
            master_metrics_port=int(extra.get("master_metrics_port", 9003)),
            transfer_timeout_seconds=int(extra.get("transfer_timeout_seconds", 30)),
            check_server=bool(extra.get("check_server", False)),
            standalone_storage=bool(extra.get("standalone_storage", False)),
            enable_ssd_offload=bool(extra.get("enable_ssd_offload", False)),
            ssd_offload_path=extra.get("ssd_offload_path"),
            extra_backend_tag=extra.get("extra_backend_tag"),
        )

    # 2) file path
    cfg_path = os.getenv("TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH")
    if cfg_path:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise KVStoreBackendError(
                f"Failed to load Mooncake config from {cfg_path}: {exc}"
            ) from exc
        if not isinstance(cfg, dict):
            raise ValueError(f"Mooncake config file {cfg_path} must be a JSON object")
        if (
            cfg.get("master_server_address") is None
            and cfg.get("client_server_address") is None
        ):
            raise ValueError(
                f"Mooncake config file {cfg_path} requires master_server_address or client_server_address"
            )
        return MooncakeStoreConfig(
            master_server_address=cfg.get("master_server_address"),
            client_server_address=cfg.get("client_server_address"),
            local_hostname=cfg.get("local_hostname", "localhost"),
            metadata_server=cfg.get("metadata_server", "P2PHANDSHAKE"),
            global_segment_size=cfg.get("global_segment_size", "4gb"),
            protocol=cfg.get("protocol", "tcp"),
            device_name=cfg.get("device_name", ""),
            master_metrics_port=int(cfg.get("master_metrics_port", 9003)),
            transfer_timeout_seconds=int(cfg.get("transfer_timeout_seconds", 30)),
            check_server=bool(cfg.get("check_server", False)),
            standalone_storage=bool(cfg.get("standalone_storage", False)),
            enable_ssd_offload=bool(cfg.get("enable_ssd_offload", False)),
            ssd_offload_path=cfg.get("ssd_offload_path"),
            extra_backend_tag=cfg.get("extra_backend_tag"),
        )

    # 3) env vars
    master = os.getenv("MOONCAKE_MASTER")
    client = os.getenv("MOONCAKE_CLIENT")
    if master is None and client is None:
        return None
    return MooncakeStoreConfig(
        master_server_address=master,
        client_server_address=client,
        local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
        metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        global_segment_size=os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "4gb"),
        protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
        device_name=os.getenv("MOONCAKE_DEVICE", ""),
        master_metrics_port=int(os.getenv("MOONCAKE_MASTER_METRICS_PORT", "9003")),
        transfer_timeout_seconds=int(os.getenv("MC_TRANSFER_TIMEOUT", "30")),
        check_server=os.getenv("MOONCAKE_CHECK_SERVER", "false").lower()
        in ("1", "true", "yes"),
        standalone_storage=os.getenv("MOONCAKE_STANDALONE_STORAGE", "false").lower()
        in ("1", "true", "yes"),
    )


def _load_custom_backend(extra_config_raw: str | None) -> BaseKVStore | None:
    extra = _parse_extra_config(extra_config_raw)
    if extra is None:
        return None
    module_path = extra.get("module_path")
    class_name = extra.get("class_name")
    if not module_path or not class_name:
        return None
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        raise KVStoreBackendError(
            f"Failed to load custom KVStore backend {module_path}:{class_name}: {exc}"
        ) from exc
    try:
        try:
            return cls(extra)  # type: ignore[call-arg]
        except TypeError:
            return cls()  # type: ignore[call-arg]
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise KVStoreBackendError(
            f"Failed to initialize custom KVStore backend "
            f"{module_path}:{class_name}: {exc}"
        ) from exc


def create_kv_store(
    backend: str | None,
    extra_config_raw: str | None = None,
) -> BaseKVStore | None:
    if backend is None:
        return None

    # Custom backend takes precedence if module_path/class_name present
    custom = _load_custom_backend(extra_config_raw)
    if custom is not None:
        return custom

    if backend == "mooncake":
        cfg = load_mooncake_store_config(extra_config_raw)
        if cfg is None:
            raise ValueError(
                "Mooncake Store requires master/client address: set "
                '--kvstore-storage-backend-extra-config \'{"master_server_address": "host:port"}\' '
                "or TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH or MOONCAKE_MASTER env"
            )
        # Defer import so unit tests without mooncake can still import this module
        from tokenspeed.runtime.cache.store.mooncake_store import MooncakeStore

        return MooncakeStore(cfg)

    raise ValueError(
        f"Unknown --kvstore-storage-backend {backend!r}; expected 'mooncake' or custom module_path/class_name"
    )
