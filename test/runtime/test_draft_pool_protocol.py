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

"""CPU wire tests for bounded draft messages and confirmed-prefix identity."""

import copy
import json
from unittest.mock import patch

import msgspec
import pytest
import torch

from tokenspeed.runtime.draft_pool.protocol import (
    MAX_FEATURE_BYTES,
    MAX_HEADER_BYTES,
    Ack,
    Busy,
    Close,
    DraftMessageCodec,
    DraftProtocolError,
    Failure,
    Hello,
    MissingState,
    Open,
    Opened,
    Proposal,
    Ready,
    Update,
    build_contract,
    matches_confirmed_prefix,
    required_history_start,
    validate_compatibility,
)


@pytest.fixture
def contract():
    return build_contract(
        target_model="target",
        target_revision="a" * 40,
        draft_model="draft",
        draft_revision="b" * 40,
        target_hf_config={"hidden_size": 4, "vocab_size": 128, "num_hidden_layers": 8},
        draft_hf_config={
            "hidden_size": 4,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6,
            "sliding_window": 8,
            "dflash_config": {"block_size": 8, "target_layer_ids": [1, 4, 7]},
        },
        target_verify_tokens=6,
    )


@pytest.fixture
def codec(contract):
    return DraftMessageCodec(
        contract=contract,
        max_header_bytes=MAX_HEADER_BYTES,
        max_feature_bytes=MAX_FEATURE_BYTES,
    )


def test_control_messages_round_trip(codec, contract):
    messages = [
        Hello(contract=contract),
        Ready(
            contract=contract,
            resident_limit=8,
            staging_limit=2,
            max_batch_size=4,
            lease_ms=5000,
        ),
        Open(
            session_id="session",
            confirmed_endpoint=12,
            anchor_token=17,
            history_start=5,
        ),
        Opened(session_id="session", confirmed_endpoint=12, anchor_token=17),
        Ack(session_id="session", confirmed_endpoint=12, anchor_token=17),
        Proposal(
            session_id="session",
            confirmed_endpoint=12,
            anchor_token=17,
            candidate_ids=(1, 2, 3, 4, 5, 6, 7),
        ),
        Close(session_id="session"),
        Busy(session_id="session", reason="staging limit", retry_after_ms=10),
        MissingState(session_id="session", reason="lease expired"),
        Failure(session_id=None, code="incompatible", detail="checkpoint mismatch"),
    ]
    for message in messages:
        frames = codec.encode(message=message, features=None)
        assert len(frames) == 1
        decoded = codec.decode(frames=frames)
        assert decoded.message == message
        assert decoded.features is None


def test_bf16_payload_is_exact_owned_copy(codec):
    message = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    features = torch.arange(28, dtype=torch.float32).to(torch.bfloat16).reshape(7, 4)
    expected = features.clone()
    frames = codec.encode(message=message, features=features)
    features.zero_()
    decoded = codec.decode(frames=frames)
    assert decoded.message == message
    assert decoded.features.device.type == "cpu"
    assert torch.equal(decoded.features.view(torch.uint16), expected.view(torch.uint16))
    decoded.features.zero_()
    assert torch.equal(codec.decode(frames=frames).features, expected)


def test_empty_delta_keeps_explicit_shape(codec):
    message = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=12,
        is_snapshot=False,
    )
    decoded = codec.decode(
        frames=codec.encode(
            message=message, features=torch.empty((0, 4), dtype=torch.bfloat16)
        )
    )
    assert decoded.features.shape == (0, 4)


@pytest.mark.parametrize(
    "features",
    [
        torch.empty((7, 4), dtype=torch.float32),
        torch.empty((7, 3), dtype=torch.bfloat16),
        torch.empty((4, 7), dtype=torch.bfloat16).T,
    ],
)
def test_rejects_bad_feature_tensor(codec, features):
    message = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    with pytest.raises(DraftProtocolError):
        codec.encode(message=message, features=features)


def test_rejects_device_tensor_without_copy_or_data_operations(codec):
    message = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    features = torch.empty((7, 4), dtype=torch.bfloat16, device="meta")
    with patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("must not copy")
    ), patch.object(
        torch.Tensor, "detach", side_effect=AssertionError("must not touch data")
    ):
        with pytest.raises(DraftProtocolError, match="non-CPU"):
            codec.encode(message=message, features=features)


@pytest.mark.parametrize(
    "changes",
    [
        {"confirmed_endpoint": -1},
        {"anchor_token": 128},
        {"feature_start": 4},
        {"feature_start": 13},
        {"feature_start": 6},
        {"session_id": "../../invalid"},
        {"session_id": "x" * 129},
    ],
)
def test_rejects_invalid_snapshot_metadata(codec, changes):
    values = dict(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    values.update(changes)
    with pytest.raises(DraftProtocolError):
        codec.decode(frames=[msgspec.msgpack.encode(Update(**values)), b"\0" * 56])


def test_rejects_malformed_and_unsupported_headers(codec):
    bad_headers = [
        b"not msgpack",
        msgspec.msgpack.encode({"type": "Unrecognized"}),
        msgspec.msgpack.encode(
            {"type": "Close", "session_id": "session", "ignored": True}
        ),
        msgspec.msgpack.encode(
            {
                "type": "Ack",
                "session_id": "session",
                "confirmed_endpoint": True,
                "anchor_token": 1,
            }
        ),
        msgspec.msgpack.encode(msgspec.msgpack.Ext(42, b"untrusted")),
    ]
    for header in bad_headers:
        with pytest.raises(DraftProtocolError):
            codec.decode(frames=[header])


def test_rejects_bad_framing_before_tensor_allocation(codec):
    update = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    header = msgspec.msgpack.encode(update)
    control = msgspec.msgpack.encode(Close(session_id="session"))
    invalid_frames = [
        [],
        [header],
        [header, b"\0" * 55],
        [header, b"\0" * 57],
        [control, b"extra"],
        [header, b"", b"extra"],
        [bytearray(control)],
    ]
    with patch.object(
        torch, "frombuffer", side_effect=AssertionError("must not allocate")
    ):
        for frames in invalid_frames:
            with pytest.raises(DraftProtocolError):
                codec.decode(frames=frames)


def test_header_preflight_does_not_materialize_feature_storage(codec):
    message = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    with patch.object(
        torch, "frombuffer", side_effect=AssertionError("must not allocate")
    ):
        assert codec.decode_header(frame=msgspec.msgpack.encode(message)) == message
    with pytest.raises(DraftProtocolError):
        codec.decode_header(frame=b"\0" * (MAX_HEADER_BYTES + 1))


def test_codec_limits_are_hard_bounded(contract):
    for header_limit, feature_limit in [
        (MAX_HEADER_BYTES + 1, MAX_FEATURE_BYTES),
        (MAX_HEADER_BYTES, MAX_FEATURE_BYTES + 1),
        (0, MAX_FEATURE_BYTES),
    ]:
        with pytest.raises(DraftProtocolError):
            DraftMessageCodec(
                contract=contract,
                max_header_bytes=header_limit,
                max_feature_bytes=feature_limit,
            )
    codec = DraftMessageCodec(
        contract=contract, max_header_bytes=MAX_HEADER_BYTES, max_feature_bytes=48
    )
    update = Update(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        feature_start=5,
        is_snapshot=True,
    )
    with pytest.raises(DraftProtocolError, match="too large"):
        codec.decode(frames=[msgspec.msgpack.encode(update), b"\0" * 56])
    with pytest.raises(DraftProtocolError, match="header is too large"):
        codec.decode(frames=[b"\0" * (MAX_HEADER_BYTES + 1)])


def test_open_reserves_full_snapshot_without_carrying_it(codec):
    valid = Open(
        session_id="session", confirmed_endpoint=12, anchor_token=17, history_start=5
    )
    with pytest.raises(DraftProtocolError, match="only UPDATE"):
        codec.encode(message=valid, features=torch.empty((7, 4), dtype=torch.bfloat16))
    invalid = Open(
        session_id="session", confirmed_endpoint=12, anchor_token=17, history_start=6
    )
    with pytest.raises(DraftProtocolError, match="complete retained"):
        codec.encode(message=invalid, features=None)


def test_native_proposal_width_and_tokens_checked(codec):
    for ids in [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 128), (1, 2, 3, 4, 5, 6, -1)]:
        message = Proposal(
            session_id="session",
            confirmed_endpoint=12,
            anchor_token=17,
            candidate_ids=ids,
        )
        with pytest.raises(DraftProtocolError):
            codec.encode(message=message, features=None)


def test_ack_and_proposal_identity_is_exact(codec):
    ack = Ack(session_id="session", confirmed_endpoint=12, anchor_token=17)
    proposal = Proposal(
        session_id="session",
        confirmed_endpoint=12,
        anchor_token=17,
        candidate_ids=(1, 2, 3, 4, 5, 6, 7),
    )
    for message in [ack, proposal]:
        assert matches_confirmed_prefix(
            message=message,
            session_id="session",
            confirmed_endpoint=12,
            anchor_token=17,
        )
        for session_id, endpoint, anchor in [
            ("reset", 12, 17),
            ("session", 13, 17),
            ("session", 12, 18),
        ]:
            assert not matches_confirmed_prefix(
                message=message,
                session_id=session_id,
                confirmed_endpoint=endpoint,
                anchor_token=anchor,
            )
    # An obsolete proposal does not prevent decoding the independent context ACK.
    assert codec.decode(frames=codec.encode(message=ack, features=None)).message == ack


def test_every_contract_field_participates_in_handshake(codec, contract):
    changes = {
        "target_model": "other",
        "target_revision": "c" * 40,
        "draft_model": "other",
        "draft_revision": "c" * 40,
        "projection_fingerprint": "c" * 64,
        "feature_width": 5,
        "vocab_size": 129,
        "window_tokens": 7,
    }
    for name, value in changes.items():
        altered = msgspec.structs.replace(contract, **{name: value})
        with pytest.raises(DraftProtocolError, match=name):
            validate_compatibility(expected=contract, received=altered)
        with pytest.raises(DraftProtocolError):
            codec.decode(frames=[msgspec.msgpack.encode(Hello(contract=altered))])


@pytest.mark.parametrize(
    "revision", ["main", "master", "HEAD", "latest", "a" * 64, "short"]
)
def test_mutable_or_non_checkpoint_revisions_are_rejected(codec, contract, revision):
    for name in ("target_revision", "draft_revision"):
        altered = msgspec.structs.replace(contract, **{name: revision})
        with pytest.raises(DraftProtocolError, match="immutable"):
            codec.decode(frames=[msgspec.msgpack.encode(Hello(contract=altered))])


def test_contract_builder_fingerprint_is_stable_and_sensitive(contract):
    facts = dict(
        target_model="target",
        target_revision="a" * 40,
        draft_model="draft",
        draft_revision="b" * 40,
        target_hf_config={"num_hidden_layers": 8, "vocab_size": 128, "hidden_size": 4},
        draft_hf_config={
            "dflash_config": {"target_layer_ids": [1, 4, 7], "block_size": 8},
            "sliding_window": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "hidden_size": 4,
        },
        target_verify_tokens=6,
    )
    assert build_contract(**facts) == contract
    changed = dict(facts)
    changed["target_revision"] = "c" * 40
    assert (
        build_contract(**changed).projection_fingerprint
        != contract.projection_fingerprint
    )
    changed = dict(facts)
    changed["draft_hf_config"] = dict(
        facts["draft_hf_config"],
        dflash_config={"target_layer_ids": [7, 4, 1], "block_size": 8},
    )
    assert (
        build_contract(**changed).projection_fingerprint
        != contract.projection_fingerprint
    )


@pytest.fixture
def contract_inputs():
    return dict(
        target_model="publisher/target",
        target_revision="a" * 40,
        draft_model="publisher/draft",
        draft_revision="b" * 40,
        target_hf_config={
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 128,
            "num_hidden_layers": 8,
            "quantization_config": {"quant_method": "fp8", "ignored_layers": ["head"]},
            "rope_parameters": {"rope_theta": 10000.0},
        },
        draft_hf_config={
            "model_type": "qwen3",
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 4,
            "vocab_size": 128,
            "sliding_window": 8,
            "rms_norm_eps": 1e-6,
            "dflash_config": {
                "block_size": 8,
                "target_layer_ids": [1, 4, 7],
                "selector_rank": 256,
            },
            "layer_types": ["sliding_attention"],
        },
        target_verify_tokens=6,
    )


def test_loading_locations_and_loader_metadata_do_not_change_identity(contract_inputs):
    expected = build_contract(**contract_inputs)
    moved = copy.deepcopy(contract_inputs)
    moved["target_model"] = "/local/target-snapshot"
    moved["draft_model"] = "/other/machine/draft-snapshot"
    moved["target_hf_config"].update(
        _name_or_path="/local/target-snapshot",
        _commit_hash="a" * 40,
        transformers_version="different-loader-version",
        _attn_implementation_internal="target-backend",
    )
    moved["draft_hf_config"].update(
        _name_or_path="/other/machine/draft-snapshot",
        _commit_hash="b" * 40,
        _attn_implementation_internal="draft-backend",
    )
    assert build_contract(**moved) == expected
    assert expected.target_model.startswith("glm_moe_dsa:")
    assert expected.draft_model.startswith("qwen3:")


@pytest.mark.parametrize(
    "side,path,value",
    [
        ("target_hf_config", ("rope_parameters", "rope_theta"), 20000.0),
        ("target_hf_config", ("quantization_config", "ignored_layers"), []),
        ("target_hf_config", ("architectures",), ["DifferentModel"]),
        ("draft_hf_config", ("dflash_config", "selector_rank"), 128),
        ("draft_hf_config", ("layer_types",), ["full_attention"]),
        ("draft_hf_config", ("unknown_future_semantic_field",), 123),
    ],
)
def test_complete_config_changes_reject_alias_handshake(
    contract_inputs, side, path, value
):
    expected = build_contract(**contract_inputs)
    changed = copy.deepcopy(contract_inputs)
    node = changed[side]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    received = build_contract(**changed)
    with pytest.raises(DraftProtocolError, match="incompatible"):
        validate_compatibility(expected=expected, received=received)


def test_configuration_commit_must_match_claimed_pin(contract_inputs):
    contract_inputs["target_hf_config"]["_commit_hash"] = "c" * 40
    with pytest.raises(DraftProtocolError, match="commit differs"):
        build_contract(**contract_inputs)


def test_shared_runtime_loader_normalizes_both_staged_copies(tmp_path, contract_inputs):
    loader = pytest.importorskip(
        "tokenspeed.runtime.utils.hf_transformers_utils",
        reason="full runtime dependencies are needed for checkpoint-loader qualification",
        exc_type=ImportError,
    )
    # Exercise the real configuration loader, without model construction or GPU work.
    # Both peers must materialize defaults and restore checkpoint fields through it.
    contracts = []
    for location in ("target-host", "worker-host"):
        kwargs = copy.deepcopy(contract_inputs)
        for side, is_draft in (("target", False), ("draft", True)):
            directory = tmp_path / location / side
            directory.mkdir(parents=True)
            raw = kwargs[f"{side}_hf_config"]
            if is_draft:
                raw["num_hidden_layers"] = 1
            (directory / "config.json").write_text(json.dumps(raw))
            loaded = loader.get_config(
                str(directory),
                trust_remote_code=False,
                revision=kwargs[f"{side}_revision"],
                model_override_args=None,
                is_draft_worker=is_draft,
                speculative_algorithm="DFLASH" if is_draft else None,
            )
            kwargs[f"{side}_model"] = str(directory)
            kwargs[f"{side}_hf_config"] = loaded.to_dict()
        contracts.append(build_contract(**kwargs))
    assert contracts[0] == contracts[1]


def test_history_excludes_anchor_and_keeps_window_left():
    assert required_history_start(confirmed_endpoint=0, window_tokens=2048) == 0
    assert required_history_start(confirmed_endpoint=2047, window_tokens=2048) == 0
    assert required_history_start(confirmed_endpoint=2048, window_tokens=2048) == 1
    assert (
        required_history_start(confirmed_endpoint=10**9, window_tokens=2048)
        == 10**9 - 2047
    )
