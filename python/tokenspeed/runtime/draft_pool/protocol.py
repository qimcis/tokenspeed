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

"""Bounded, CPU-only messages for the private remote draft pool.

The first multipart frame is strictly typed msgpack. Only ``Update`` carries
a second frame: contiguous, little-endian BF16 projected features for the
absolute interval ``[feature_start, confirmed_endpoint)``. The anchor token
at the endpoint has not passed through the target and has no feature row.
ROUTER identities belong to transport and are not part of this codec.

OPEN/OPENED reserve worker resources before an UPDATE snapshot is sent.
ACK reports installed context independently of PROPOSAL eligibility. Session
ownership, admission, leases and ordered GPU installation belong to transport
and the worker, not to this serialization layer.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Union

import msgspec
import torch

from tokenspeed.runtime.draft_pool.config import validate_remote_revision

PROTOCOL_VERSION = 1
FEATURE_SCHEMA = "dflash2.projected.bf16.v1"
MAX_HEADER_BYTES = 64 * 1024
MAX_FEATURE_BYTES = 24 * 1024 * 1024
_MAX_ENDPOINT = (1 << 63) - 1
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_LOADER_METADATA = frozenset(
    {
        "_name_or_path",
        "_commit_hash",
        "transformers_version",
        "_attn_implementation",
        "_attn_implementation_internal",
        "_attn_implementation_autoset",
    }
)


class DraftProtocolError(ValueError):
    """An invalid or incompatible private wire message."""


class DraftProtocolContract(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Pinned model identities and the complete projected-feature contract."""

    protocol_version: int
    target_model: str
    target_revision: str
    draft_model: str
    draft_revision: str
    projection_fingerprint: str
    feature_schema: str
    feature_dtype: Literal["bfloat16"]
    feature_width: int
    vocab_size: int
    window_tokens: int
    native_block_tokens: int
    target_verify_tokens: int


class _Message(
    msgspec.Struct, tag=True, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    pass


class Hello(_Message):
    contract: DraftProtocolContract


class Ready(_Message):
    contract: DraftProtocolContract
    resident_limit: int
    staging_limit: int
    max_batch_size: int
    lease_ms: int


class Open(_Message):
    session_id: str
    confirmed_endpoint: int
    anchor_token: int
    history_start: int


class Opened(_Message):
    session_id: str
    confirmed_endpoint: int
    anchor_token: int


class Update(_Message):
    session_id: str
    confirmed_endpoint: int
    anchor_token: int
    feature_start: int
    is_snapshot: bool


class Ack(_Message):
    session_id: str
    confirmed_endpoint: int
    anchor_token: int


class Proposal(_Message):
    session_id: str
    confirmed_endpoint: int
    anchor_token: int
    candidate_ids: tuple[int, ...]


class Close(_Message):
    session_id: str


class Busy(_Message):
    session_id: str
    reason: str
    retry_after_ms: int


class MissingState(_Message):
    session_id: str
    reason: str


class Failure(_Message):
    session_id: str | None
    code: str
    detail: str


DraftMessage = Union[
    Hello,
    Ready,
    Open,
    Opened,
    Update,
    Ack,
    Proposal,
    Close,
    Busy,
    MissingState,
    Failure,
]
_MESSAGE_TYPES = (
    Hello,
    Ready,
    Open,
    Opened,
    Update,
    Ack,
    Proposal,
    Close,
    Busy,
    MissingState,
    Failure,
)


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    """Decoded metadata and independently owned CPU feature storage, if any."""

    message: DraftMessage
    features: torch.Tensor | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DraftProtocolError(message)


def _text(value: str, maximum: int, name: str) -> None:
    _require(isinstance(value, str) and 0 < len(value) <= maximum, f"invalid {name}")


def _integer(value: int, minimum: int, maximum: int, name: str) -> None:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"invalid {name}",
    )


def validate_contract(contract: DraftProtocolContract) -> None:
    """Validate a contract's versions, shapes and bounded wire identifiers."""
    _require(isinstance(contract, DraftProtocolContract), "invalid contract type")
    _integer(contract.protocol_version, PROTOCOL_VERSION, PROTOCOL_VERSION, "version")
    for name in ("target_model", "target_revision", "draft_model", "draft_revision"):
        _text(getattr(contract, name), 1024, name)
    for name in ("target_revision", "draft_revision"):
        try:
            validate_remote_revision(getattr(contract, name), name)
        except ValueError as exc:
            raise DraftProtocolError(str(exc)) from exc
    _require(
        isinstance(contract.projection_fingerprint, str)
        and _FINGERPRINT.fullmatch(contract.projection_fingerprint) is not None,
        "projection_fingerprint must be a lowercase SHA256 digest",
    )
    _require(contract.feature_schema == FEATURE_SCHEMA, "unsupported feature schema")
    _require(contract.feature_dtype == "bfloat16", "only BF16 features are supported")
    _integer(contract.feature_width, 1, MAX_FEATURE_BYTES // 2, "feature width")
    _integer(contract.vocab_size, 1, (1 << 31) - 1, "vocabulary size")
    _integer(contract.window_tokens, 2, 2048, "window tokens")
    _integer(contract.native_block_tokens, 8, 8, "native block tokens")
    _integer(contract.target_verify_tokens, 6, 6, "target verify tokens")
    _require(
        (contract.window_tokens - 1) * contract.feature_width * 2 <= MAX_FEATURE_BYTES,
        "feature window exceeds the maximum snapshot size",
    )


def validate_compatibility(
    *, expected: DraftProtocolContract, received: DraftProtocolContract
) -> None:
    """Raise on any mismatch between local and HELLO/READY model contracts."""
    validate_contract(expected)
    validate_contract(received)
    differences = [
        name
        for name in DraftProtocolContract.__struct_fields__
        if getattr(expected, name) != getattr(received, name)
    ]
    _require(not differences, f"incompatible draft contract: {', '.join(differences)}")


def projection_fingerprint(
    *,
    target_layer_ids: tuple[int, ...],
    hidden_size: int,
    rms_norm_eps: float,
    draft_revision: str,
) -> str:
    """Hash tap order and projection facts, with weights pinned by draft revision.

    ``target_layer_ids`` retains capture order. ``hidden_size`` is the projected
    feature width, ``rms_norm_eps`` the hidden norm epsilon, and
    ``draft_revision`` the resolved immutable revision containing fc/norm weights.
    Returns the lowercase SHA256 digest used by the wire contract.
    """
    _require(type(target_layer_ids) is tuple and bool(target_layer_ids), "invalid taps")
    for layer_id in target_layer_ids:
        _integer(layer_id, 0, (1 << 31) - 1, "target layer ID")
    _require(
        len(set(target_layer_ids)) == len(target_layer_ids), "duplicate target taps"
    )
    _integer(hidden_size, 1, MAX_FEATURE_BYTES // 2, "hidden size")
    _require(
        type(rms_norm_eps) in (float, int)
        and math.isfinite(rms_norm_eps)
        and rms_norm_eps > 0,
        "invalid RMS norm epsilon",
    )
    _text(draft_revision, 1024, "draft revision")
    try:
        validate_remote_revision(draft_revision, "draft revision")
    except ValueError as exc:
        raise DraftProtocolError(str(exc)) from exc
    canonical = msgspec.json.encode(
        [
            FEATURE_SCHEMA,
            list(target_layer_ids),
            hidden_size,
            float(rms_norm_eps),
            draft_revision,
        ]
    )
    return hashlib.sha256(canonical).hexdigest()


def _canonical_config(value: object) -> object:
    """Normalize JSON ordering, retaining all configuration except loader metadata."""
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            _require(
                type(key) in (str, int),
                "configuration keys must be strings or integers",
            )
            name = str(key)
            if name in _CONFIG_LOADER_METADATA:
                continue
            _require(name not in result, "configuration has ambiguous JSON keys")
            result[name] = _canonical_config(child)
        return {name: result[name] for name in sorted(result)}
    if type(value) in (list, tuple):
        return [_canonical_config(child) for child in value]
    if type(value) is float:
        _require(math.isfinite(value), "configuration numbers must be finite")
        return value
    _require(
        value is None or type(value) in (str, int, bool),
        "configuration must contain only JSON values",
    )
    return value


def _checkpoint_identity(config: Mapping[str, object], revision: str) -> str:
    committed = config.get("_commit_hash")
    if committed is not None:
        _require(
            isinstance(committed, str) and committed.lower() == revision,
            "loaded configuration commit differs from its pinned revision",
        )
    canonical = _canonical_config(config)
    digest = hashlib.sha256(msgspec.json.encode(canonical)).hexdigest()
    model_type = config.get("model_type", "checkpoint")
    _text(model_type, 256, "configuration model type")
    return f"{model_type}:{digest}"


def build_contract(
    *,
    target_model: str,
    target_revision: str,
    draft_model: str,
    draft_revision: str,
    target_hf_config: Mapping[str, object],
    draft_hf_config: Mapping[str, object],
    target_verify_tokens: int,
) -> DraftProtocolContract:
    """Build the same handshake facts from resolved target and worker configs.

    ``target_model`` and ``draft_model`` are loading locations, which may differ
    between machines. Wire identities use model type plus the complete normalized
    configuration digest instead of those paths. Immutable revisions identify
    weights. Configuration arguments must come from the SAME TokenSpeed
    ``get_config(...).to_dict()`` path on both peers; raw checkpoint JSON does not
    contain the loader's materialized defaults and restored fields. This is CPU
    metadata processing; constructing a model or inspecting tensors is unnecessary. The
    verification width is explicit and must be six for this protocol version.
    Returns a validated contract whose digest binds checkpoint identities,
    ordered target taps and canonical projection/normalization facts.
    """
    _require(isinstance(target_hf_config, Mapping), "invalid target config")
    _require(isinstance(draft_hf_config, Mapping), "invalid draft config")
    _text(target_model, 4096, "target loading location")
    _text(draft_model, 4096, "draft loading location")
    _text(target_revision, 1024, "target revision")
    _text(draft_revision, 1024, "draft revision")
    target_revision = target_revision.lower()
    draft_revision = draft_revision.lower()
    target = target_hf_config.get("text_config", target_hf_config)
    _require(isinstance(target, Mapping), "invalid target text config")
    nested = draft_hf_config.get("dflash_config", {})
    _require(isinstance(nested, Mapping), "invalid dflash config")
    try:
        hidden_size = draft_hf_config["hidden_size"]
        _require(
            target["hidden_size"] == hidden_size, "target and feature widths differ"
        )
        vocab_size = target["vocab_size"]
        _require(
            draft_hf_config["vocab_size"] == vocab_size,
            "target and draft vocabularies differ",
        )
        taps = nested.get("target_layer_ids", draft_hf_config.get("target_layer_ids"))
        _require(type(taps) in (list, tuple) and bool(taps), "missing target layer IDs")
        target_layer_ids = tuple(taps)
        layer_count = target["num_hidden_layers"]
        _integer(layer_count, 1, (1 << 31) - 1, "target layer count")
        for layer_id in target_layer_ids:
            _integer(layer_id, 0, layer_count - 1, "target layer ID")
        native_block_tokens = nested.get(
            "block_size", draft_hf_config.get("block_size")
        )
        window_tokens = draft_hf_config["sliding_window"]
        # Match the existing DFlashTargetProjection's checkpoint convention.
        rms_norm_eps = draft_hf_config.get("rms_norm_eps", 1e-6)
    except (KeyError, TypeError) as exc:
        raise DraftProtocolError(
            f"incomplete draft checkpoint configuration: {exc}"
        ) from exc
    projection_digest = projection_fingerprint(
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        rms_norm_eps=rms_norm_eps,
        draft_revision=draft_revision,
    )
    target_identity = _checkpoint_identity(target_hf_config, target_revision)
    draft_identity = _checkpoint_identity(draft_hf_config, draft_revision)
    fingerprint = hashlib.sha256(
        msgspec.json.encode(
            [
                target_identity,
                target_revision,
                draft_identity,
                draft_revision,
                projection_digest,
            ]
        )
    ).hexdigest()
    contract = DraftProtocolContract(
        protocol_version=PROTOCOL_VERSION,
        target_model=target_identity,
        target_revision=target_revision,
        draft_model=draft_identity,
        draft_revision=draft_revision,
        projection_fingerprint=fingerprint,
        feature_schema=FEATURE_SCHEMA,
        feature_dtype="bfloat16",
        feature_width=hidden_size,
        vocab_size=vocab_size,
        window_tokens=window_tokens,
        native_block_tokens=native_block_tokens,
        target_verify_tokens=target_verify_tokens,
    )
    validate_contract(contract)
    return contract


def required_history_start(*, confirmed_endpoint: int, window_tokens: int) -> int:
    """Return the earliest required input feature for an anchor at the endpoint."""
    _integer(confirmed_endpoint, 0, _MAX_ENDPOINT, "confirmed endpoint")
    _integer(window_tokens, 2, 2048, "window tokens")
    return max(0, confirmed_endpoint - (window_tokens - 1))


def matches_confirmed_prefix(
    *,
    message: Ack | Proposal,
    session_id: str,
    confirmed_endpoint: int,
    anchor_token: int,
) -> bool:
    """Check exact eligibility; a context ACK may remain useful after this is false."""
    return (
        message.session_id == session_id
        and message.confirmed_endpoint == confirmed_endpoint
        and message.anchor_token == anchor_token
    )


def _validate_message(message: DraftMessage, contract: DraftProtocolContract) -> None:
    _require(type(message) in _MESSAGE_TYPES, "unsupported message type")
    if isinstance(message, (Hello, Ready)):
        validate_compatibility(expected=contract, received=message.contract)
    if isinstance(message, Ready):
        for name in ("resident_limit", "staging_limit", "max_batch_size", "lease_ms"):
            _integer(getattr(message, name), 1, (1 << 31) - 1, name)
        _require(
            message.staging_limit <= message.resident_limit, "too many staging slots"
        )
        _require(
            message.max_batch_size <= message.resident_limit, "batch exceeds residents"
        )
    if hasattr(message, "session_id"):
        session_id = message.session_id
        _require(
            (isinstance(message, Failure) and session_id is None)
            or (
                isinstance(session_id, str)
                and _SESSION_ID.fullmatch(session_id) is not None
            ),
            "invalid session ID",
        )
    if isinstance(message, (Open, Opened, Update, Ack, Proposal)):
        _integer(message.confirmed_endpoint, 0, _MAX_ENDPOINT, "confirmed endpoint")
        _integer(message.anchor_token, 0, contract.vocab_size - 1, "anchor token")
    if isinstance(message, Open):
        _require(
            type(message.history_start) is int
            and message.history_start
            == required_history_start(
                confirmed_endpoint=message.confirmed_endpoint,
                window_tokens=contract.window_tokens,
            ),
            "OPEN must reserve the complete retained feature window",
        )
    if isinstance(message, Update):
        _integer(message.feature_start, 0, message.confirmed_endpoint, "feature start")
        _require(type(message.is_snapshot) is bool, "invalid snapshot flag")
        first_required = required_history_start(
            confirmed_endpoint=message.confirmed_endpoint,
            window_tokens=contract.window_tokens,
        )
        _require(
            message.feature_start >= first_required, "feature transfer exceeds window"
        )
        if message.is_snapshot:
            _require(
                message.feature_start == first_required, "snapshot misses required rows"
            )
    if isinstance(message, Proposal):
        _require(
            type(message.candidate_ids) is tuple
            and len(message.candidate_ids) == contract.native_block_tokens - 1,
            "proposal must contain the native seven candidate IDs",
        )
        for candidate_id in message.candidate_ids:
            _integer(candidate_id, 0, contract.vocab_size - 1, "candidate token")
    if isinstance(message, (Busy, MissingState)):
        _text(message.reason, 256, "reason")
    if isinstance(message, Busy):
        _integer(message.retry_after_ms, 0, (1 << 31) - 1, "retry delay")
    if isinstance(message, Failure):
        _text(message.code, 128, "failure code")
        _text(message.detail, 2048, "failure detail")


class DraftMessageCodec:
    """Copying multipart codec; rejects every non-CPU tensor before touching data.

    ``contract`` supplies exact feature shape and checkpoint compatibility.
    ``max_header_bytes`` and ``max_feature_bytes`` may tighten, never relax,
    the protocol hard limits. Transport must also configure socket and queue
    bounds: this codec cannot undo allocation performed by a socket receive.
    """

    def __init__(
        self,
        *,
        contract: DraftProtocolContract,
        max_header_bytes: int,
        max_feature_bytes: int,
    ) -> None:
        validate_contract(contract)
        _integer(max_header_bytes, 1, MAX_HEADER_BYTES, "header byte limit")
        _integer(max_feature_bytes, 1, MAX_FEATURE_BYTES, "feature byte limit")
        _require(
            sys.byteorder == "little", "BF16 wire format requires little-endian CPU"
        )
        self.contract = contract
        self.max_header_bytes = max_header_bytes
        self.max_feature_bytes = max_feature_bytes
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(type=DraftMessage, strict=True)

    def encode(
        self, message: DraftMessage, features: torch.Tensor | None
    ) -> list[bytes]:
        """Return owned byte frames; features must be contiguous CPU BF16 or None."""
        # This check must precede every tensor operation, including conversion.
        if features is not None:
            _require(
                isinstance(features, torch.Tensor), "features must be a CPU tensor"
            )
            _require(
                features.device.type == "cpu", "non-CPU feature tensors are forbidden"
            )
        _validate_message(message, self.contract)
        try:
            header = self._encoder.encode(message)
        except (TypeError, ValueError) as exc:
            raise DraftProtocolError(f"cannot encode message: {exc}") from exc
        _require(len(header) <= self.max_header_bytes, "message header is too large")
        # Struct constructors are intentionally cheap and do not type-check.
        # Round through the strict decoder before transmitting caller metadata.
        self.decode_header(frame=header)
        if not isinstance(message, Update):
            _require(features is None, "only UPDATE may carry feature data")
            return [header]
        _require(features is not None, "UPDATE requires its feature frame")
        rows, columns, nbytes = self._feature_shape(message)
        _require(features.dtype == torch.bfloat16, "feature dtype must be BF16")
        _require(
            tuple(features.shape) == (rows, columns), "feature tensor shape mismatch"
        )
        _require(features.is_contiguous(), "features must be contiguous")
        _require(
            features.numel() * features.element_size() == nbytes,
            "feature byte mismatch",
        )
        raw = features.detach().view(torch.uint8).numpy().tobytes(order="C")
        return [header, raw]

    def decode(self, frames: list[bytes] | tuple[bytes, ...]) -> DecodedMessage:
        """Validate framing before allocating an independently owned BF16 CPU tensor."""
        _require(type(frames) in (list, tuple), "frames must be a list or tuple")
        _require(1 <= len(frames) <= 2, "invalid multipart frame count")
        for frame in frames:
            _require(type(frame) is bytes, "frames must be copied bytes")
        _require(len(frames[0]) <= self.max_header_bytes, "message header is too large")
        message = self.decode_header(frame=frames[0])
        if not isinstance(message, Update):
            _require(len(frames) == 1, "unexpected feature frame")
            return DecodedMessage(message=message, features=None)
        _require(len(frames) == 2, "UPDATE is missing its feature frame")
        rows, columns, nbytes = self._feature_shape(message)
        _require(
            len(frames[1]) == nbytes, "feature frame length does not match its shape"
        )
        if nbytes == 0:
            features = torch.empty((rows, columns), dtype=torch.bfloat16, device="cpu")
        else:
            features = torch.frombuffer(
                bytearray(frames[1]), dtype=torch.bfloat16
            ).reshape(rows, columns)
        return DecodedMessage(message=message, features=features)

    def decode_header(self, frame: bytes) -> DraftMessage:
        """Validate metadata without tensor allocation for admission preflight.

        ``frame`` is the first copied multipart byte frame. Transport can check
        its session reservation and staging limits before calling ``decode`` on
        the complete message. Payload shape and length still require ``decode``.
        """
        _require(type(frame) is bytes, "header must be copied bytes")
        _require(len(frame) <= self.max_header_bytes, "message header is too large")
        try:
            message = self._decoder.decode(frame)
        except (msgspec.DecodeError, TypeError, ValueError) as exc:
            raise DraftProtocolError(f"invalid draft message: {exc}") from exc
        _validate_message(message, self.contract)
        return message

    def _feature_shape(self, message: Update) -> tuple[int, int, int]:
        rows = message.confirmed_endpoint - message.feature_start
        columns = self.contract.feature_width
        nbytes = rows * columns * 2
        _require(nbytes <= self.max_feature_bytes, "feature transfer is too large")
        return rows, columns, nbytes
