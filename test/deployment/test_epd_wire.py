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

import dataclasses

import pytest

from tokenspeed.runtime.epd.entities import (
    REGISTER_ROOM_SENTINEL,
    EmbeddingArgsRegisterInfo,
    EmbeddingTransferInfo,
)


def transfer(**changes):
    return dataclasses.replace(
        EmbeddingTransferInfo(
            room=42,
            endpoint="10.0.0.7",
            dst_port=5123,
            mooncake_session_id="10.0.0.7:41999",
            dst_embedding_ptr=0xFEDCBA9876543210,
            dst_deepstack_ptr=0x123456789ABCDEF0,
            n_tokens=3,
            hidden=4096,
            dtype="torch.bfloat16",
            has_deepstack=True,
            required_dst_info_num=2,
            row_start=5,
            span=8,
        ),
        **changes,
    )


def test_transfer_round_trip_preserves_shard_and_pointer_bytes():
    info = transfer()
    frames = info.to_zmq()
    assert frames[4] == bytes.fromhex("1032547698badcfe")
    assert frames[5] == bytes.fromhex("f0debc9a78563412")
    assert EmbeddingTransferInfo.from_zmq(frames) == info


@pytest.mark.parametrize("has_deepstack", [True, False])
def test_zero_row_shard_heartbeat_needs_no_buffer(has_deepstack):
    info = transfer(
        n_tokens=0,
        row_start=8,
        dst_embedding_ptr=0,
        dst_deepstack_ptr=0,
        has_deepstack=has_deepstack,
    )
    assert EmbeddingTransferInfo.from_zmq(info.to_zmq()) == info


def test_default_span_representation_and_appended_fields_remain_compatible():
    info = transfer(span=0, row_start=0, has_deepstack=False, dst_deepstack_ptr=0)
    assert EmbeddingTransferInfo.from_zmq(info.to_zmq() + [b"extension"]) == info


@pytest.mark.parametrize("frame", [b"", b"true", b"false", b"2", b"01"])
def test_malformed_boolean_is_not_treated_as_absent_deepstack(frame):
    frames = transfer().to_zmq()
    frames[9] = frame
    with pytest.raises(ValueError, match="boolean"):
        EmbeddingTransferInfo.from_zmq(frames)


@pytest.mark.parametrize("fanout", [0, -1])
def test_nonpositive_fanout_cannot_open_transfer_gate(fanout):
    with pytest.raises(ValueError, match="fanout"):
        EmbeddingTransferInfo.from_zmq(transfer(required_dst_info_num=fanout).to_zmq())


@pytest.mark.parametrize(
    "changes", [{"dst_embedding_ptr": 0}, {"dst_deepstack_ptr": 0}]
)
def test_nonempty_transfer_requires_allocated_destination(changes):
    with pytest.raises(ValueError, match="destination"):
        EmbeddingTransferInfo.from_zmq(transfer(**changes).to_zmq())


@pytest.mark.parametrize("length", range(13))
def test_truncated_transfer_is_rejected(length):
    with pytest.raises(ValueError, match="malformed EmbeddingTransferInfo"):
        EmbeddingTransferInfo.from_zmq(transfer().to_zmq()[:length])


def test_registration_round_trip():
    info = EmbeddingArgsRegisterInfo(
        REGISTER_ROOM_SENTINEL, "::1", 5123, "session", 0xFEDCBA9876543210, 0
    )
    assert EmbeddingArgsRegisterInfo.from_zmq(info.to_zmq()) == info


@pytest.mark.parametrize("length", range(6))
def test_truncated_registration_is_rejected(length):
    info = EmbeddingArgsRegisterInfo(
        REGISTER_ROOM_SENTINEL, "::1", 5123, "session", 1, 0
    )
    with pytest.raises(ValueError, match="malformed EmbeddingArgsRegisterInfo"):
        EmbeddingArgsRegisterInfo.from_zmq(info.to_zmq()[:length])
