# Copyright (c) 2026 LightSeek Foundation

"""Typed failures used at L3 subsystem boundaries."""


class L3Error(RuntimeError):
    """Base class for recoverable L3 failures."""


class L3BackendError(L3Error):
    """The configured Store backend rejected or failed an operation."""


class L3TransferError(L3Error):
    """Host/device cache movement or CUDA synchronization failed."""


class L3SubmissionError(L3Error):
    """An asynchronous L3 operation could not be submitted or completed."""


class L3ShutdownError(L3Error):
    """L3 resources could not be closed cleanly."""
