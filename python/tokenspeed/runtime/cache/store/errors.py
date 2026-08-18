# Copyright (c) 2026 LightSeek Foundation

"""Typed failures exposed by distributed cache Store backends."""


class KVStoreError(RuntimeError):
    """Base class for Store backend failures."""


class KVStoreBackendError(KVStoreError):
    """A Store setup or data operation failed."""


class KVStoreShutdownError(KVStoreError):
    """A Store backend could not be closed cleanly."""
