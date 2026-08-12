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

"""Mooncake Store backend."""

from __future__ import annotations

import ctypes
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.cache.store.base import BaseKVStore, MooncakeStoreConfig
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MiB
_SETUP_TIMEOUT_S = 600
_WARMUP_RETRIES = 10


def _parse_global_segment_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError("Invalid global_segment_size: missing number before 'gb'")
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


class InMemoryStore(BaseKVStore):
    """In-process dict store for unit tests (no mooncake dependency)."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def batch_exists(self, keys: list[str]) -> list[int]:
        return [1 if k in self._store else 0 for k in keys]

    def batch_get_into(
        self,
        keys: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> list[int]:
        out: list[int] = []
        for k, ptr, size in zip(keys, buffer_ptrs, buffer_sizes):
            v = self._store.get(k)
            if v is None:
                out.append(-1)
                continue
            n = min(len(v), int(size[0]) if isinstance(size, (list, tuple)) else int(size))
            # buffer_ptrs are host pointers; copy via ctypes
            ctypes.memmove(ptr, v, n)
            out.append(n)
        return out

    def batch_put_from(
        self,
        keys: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> list[int]:
        for k, ptr, size in zip(keys, buffer_ptrs, buffer_sizes):
            n = int(size[0]) if isinstance(size, (list, tuple)) else int(size)
            self._store[k] = ctypes.string_at(ptr, n)
        return [0] * len(keys)


class MooncakeStore(BaseKVStore):
    """``mooncake.store.MooncakeDistributedStore`` wrapper."""

    def __init__(self, config: MooncakeStoreConfig) -> None:
        self.config = config
        self.extra_backend_tag: str | None = config.extra_backend_tag
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Please install mooncake (kvcache-ai/Mooncake) to use "
                "--kvstore-storage-backend mooncake: "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
            ) from exc

        self._StoreClass = MooncakeDistributedStore
        self.store = MooncakeDistributedStore()
        self._setup_store()
        self._warmup()

    def _setup_store(self) -> None:
        cfg = self.config
        seg_size = _parse_global_segment_size(cfg.global_segment_size)

        transfer_engine = None
        client_hostname = cfg.local_hostname
        try:
            from tokenspeed.runtime.distributed.parallel_state import (  # type: ignore[import-not-found]
                get_mooncake_transfer_engine,
            )

            shared = get_mooncake_transfer_engine()
            if (
                shared is not None
                and cfg.device_name == shared.get_ib_device()
                and cfg.metadata_server == "P2PHANDSHAKE"
                and cfg.protocol == "rdma"
            ):
                client_hostname = shared.get_session_id()
                transfer_engine = shared.get_engine()
                logger.info("MooncakeStore: reusing shared TransferEngine %s", shared)
        except Exception:
            pass

        setup_kwargs: dict[str, Any] = {}
        if cfg.enable_ssd_offload:
            setup_kwargs["enable_ssd_offload"] = True
        if cfg.ssd_offload_path is not None:
            setup_kwargs["ssd_offload_path"] = cfg.ssd_offload_path

        master = cfg.master_server_address
        client_addr = cfg.client_server_address
        # Standalone dummy mode when explicitly requested
        if cfg.standalone_storage:
            if not client_addr:
                raise ValueError("standalone_storage requires client_server_address")
            ret = self.store.setup_dummy(
                seg_size,
                _DEFAULT_LOCAL_BUFFER_SIZE,
                client_addr,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake Store setup_dummy failed: ret={ret}")
            logger.info("MooncakeStore: setup_dummy ok (seg=%s)", seg_size)
            return

        if not master:
            if client_addr:
                raise ValueError(
                    "client_server_address without master_server_address requires "
                    "standalone_storage=true; set master_server_address or enable standalone"
                )
            raise ValueError("Mooncake Store requires master_server_address")

        while True:
            try:
                ret = self.store.setup(
                    client_hostname,
                    cfg.metadata_server,
                    seg_size,
                    _DEFAULT_LOCAL_BUFFER_SIZE,
                    cfg.protocol,
                    cfg.device_name,
                    master,
                    transfer_engine,
                    **setup_kwargs,
                )
                break
            except TypeError as exc:
                unknown = [k for k in list(setup_kwargs) if k in str(exc)]
                if not unknown:
                    raise
                logger.warning(
                    "Mooncake Store setup() doesn't support %s — retrying without",
                    ", ".join(unknown),
                )
                for k in unknown:
                    setup_kwargs.pop(k, None)

        if ret != 0:
            raise RuntimeError(f"Mooncake Store setup failed: ret={ret} (master={master})")
        logger.info(
            "MooncakeStore: setup ok host=%s master=%s proto=%s seg=%s",
            client_hostname,
            master,
            cfg.protocol,
            seg_size,
        )

    def _warmup(self) -> None:
        key = "tokenspeed_mooncake_l3_warmup_" + uuid.uuid4().hex
        val = bytes(4 * 1024)
        last_ret = -1
        for attempt in range(_WARMUP_RETRIES):
            last_ret = self.store.put(key, val)
            if last_ret == 0:
                break
            logger.warning("MooncakeStore warmup put attempt %s/%s ret=%s", attempt + 1, _WARMUP_RETRIES, last_ret)
            time.sleep(1.0)
        else:
            raise RuntimeError(f"MooncakeStore warmup put failed after {_WARMUP_RETRIES} attempts ret={last_ret}")
        assert self.store.is_exist(key) == 1
        assert self.store.get(key) == val
        logger.info("MooncakeStore: warmup ok")

    def _tag(self, keys: list[str]) -> list[str]:
        if self.extra_backend_tag is None:
            return keys
        return [f"{self.extra_backend_tag}_{k}" for k in keys]

    def batch_exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(self._tag(keys))

    def batch_get_into(
        self,
        keys: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> list[int]:
        tagged = self._tag(keys)
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_get_into_multi_buffers(tagged, buffer_ptrs, buffer_sizes)
        return self.store.batch_get_into(tagged, buffer_ptrs, buffer_sizes)

    def batch_put_from(
        self,
        keys: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> list[int]:
        tagged = self._tag(keys)
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_put_from_multi_buffers(tagged, buffer_ptrs, buffer_sizes)
        return self.store.batch_put_from(tagged, buffer_ptrs, buffer_sizes)

    def register_buffer(self, ptr: int, size: int) -> int:
        return self.store.register_buffer(ptr, size)

    def close(self) -> None:
        pass

    @staticmethod
    def _uses_multi_buffer(ptrs: list[Any]) -> bool:
        return bool(ptrs) and isinstance(ptrs[0], (list, tuple))
