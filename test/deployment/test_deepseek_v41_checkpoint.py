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

import json
import struct

import pytest

from tokenspeed.deployment.deepseek_v41.checkpoint import audit_checkpoint


def write_shard(path, header, payload):
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def write_index(directory, weight_map, total_size):
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map, "metadata": {"total_size": total_size}})
    )


def checkpoint(directory, tensors):
    header, payload, weight_map = {}, b"", {}
    for name, size in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [len(payload), len(payload) + size],
        }
        payload += bytes(size)
        weight_map[name] = "shard.safetensors"
    write_shard(directory / "shard.safetensors", header, payload)
    write_index(directory, weight_map, len(payload))
    return header


def test_actual_storage_inventory_includes_scales_and_role_namespaces(tmp_path):
    checkpoint(
        tmp_path,
        {
            "layers.1.engram.embed.weight": 256,
            "model.layers.1.engram.embed.scale": 8,
            "layers.1.engram.wkv.scale": 4,
            "layers.0.ffn.experts.0.w1.weight": 100,
            "model.layers.0.ffn.experts.0.w1.scale": 4,
            "layers.0.ffn.shared_experts.w1.weight": 10,
            "vision.weight": 20,
            "model.aligner.weight": 10,
            "model.image_start": 2,
            "image_end": 2,
            "image_newline": 2,
            "mtp.0.ffn.experts.0.w1.weight": 30,
            "model.mtp.2.ffn.experts.0.w1.scale": 2,
            "mtp.0.markov_proj.weight": 8,
            "head.weight": 40,
        },
    )
    audit = audit_checkpoint(tmp_path)
    assert audit["category_bytes"] == {
        "engram": 264,
        "routed_experts": 104,
        "vision": 36,
        "draft": 40,
        "dense_shared_and_other": 54,
    }
    assert audit["total_bytes"] == 498
    assert audit["dtype_bytes"] == {"U8": 498}
    assert audit["tensor_count"] == 15
    assert audit["status"] == "checkpoint_storage_not_runtime_residency"


def test_multiple_shards_metadata_scalar_and_empty_tensors(tmp_path):
    first = {
        "__metadata__": {"format": "pt"},
        "head.weight": {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]},
        "empty.weight": {"dtype": "F32", "shape": [0, 4], "data_offsets": [2, 2]},
    }
    second = {
        "layers.1.engram.embed.scale": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
    }
    write_shard(tmp_path / "a.safetensors", first, bytes(2))
    write_shard(tmp_path / "b.safetensors", second, bytes(4))
    write_index(
        tmp_path,
        {
            "head.weight": "a.safetensors",
            "empty.weight": "a.safetensors",
            "layers.1.engram.embed.scale": "b.safetensors",
        },
        6,
    )
    audit = audit_checkpoint(tmp_path)
    assert audit["total_bytes"] == 6
    assert audit["tensor_count"] == 3
    assert audit["dtype_bytes"] == {"BF16": 2, "F32": 4}


def test_truncated_payload_is_not_a_valid_inventory(tmp_path):
    checkpoint(tmp_path, {"head.weight": 40})
    shard = tmp_path / "shard.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="extent"):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "filename",
    [
        "../other.safetensors",
        "/other.safetensors",
        "sub/shard.safetensors",
        "sub\\shard.safetensors",
        1,
    ],
)
def test_index_traversal_and_invalid_paths_are_rejected(tmp_path, filename):
    write_index(tmp_path, {"x": filename}, 0)
    with pytest.raises(ValueError, match="path"):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", None),
        ("shape", [-1]),
        ("shape", [True]),
        ("shape", [1.5]),
        ("dtype", None),
        ("dtype", []),
        ("dtype", ""),
        ("data_offsets", None),
        ("data_offsets", [0]),
        ("data_offsets", [0, 0, 0]),
        ("data_offsets", [False, 4]),
        ("data_offsets", [0, 4.0]),
        ("data_offsets", [-1, 4]),
        ("data_offsets", [4, 0]),
    ],
)
def test_malformed_tensor_descriptor_is_rejected(tmp_path, field, value):
    header = checkpoint(tmp_path, {"head.weight": 4})
    header["head.weight"][field] = value
    write_shard(tmp_path / "shard.safetensors", header, bytes(4))
    with pytest.raises(ValueError, match="invalid tensor"):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "header",
    [[], {"head.weight": []}, {"__metadata__": []}, {"__metadata__": {"format": 42}}],
)
def test_malformed_header_is_rejected(tmp_path, header):
    checkpoint(tmp_path, {"head.weight": 4})
    write_shard(tmp_path / "shard.safetensors", header, bytes(4))
    with pytest.raises(ValueError):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        struct.pack("<Q", 2**30),
        struct.pack("<Q", 1) + b"{",
        struct.pack("<Q", 1) + b"\xff",
    ],
)
def test_invalid_header_encoding_or_length_is_rejected(tmp_path, data):
    checkpoint(tmp_path, {"head.weight": 4})
    (tmp_path / "shard.safetensors").write_bytes(data)
    with pytest.raises(ValueError, match="header"):
        audit_checkpoint(tmp_path)


def test_duplicate_header_keys_are_not_silently_overwritten(tmp_path):
    checkpoint(tmp_path, {"head.weight": 4})
    encoded = b'{"head.weight": {}, "head.weight": {}}'
    (tmp_path / "shard.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "index",
    [
        [],
        {},
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"x": "shard.safetensors"}, "metadata": []},
        {"weight_map": {"x": "shard.safetensors"}, "metadata": {"total_size": True}},
        {"weight_map": {"x": "shard.safetensors"}, "metadata": {"total_size": -1}},
    ],
)
def test_invalid_index_is_rejected(tmp_path, index):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        audit_checkpoint(tmp_path)


@pytest.mark.parametrize("offsets", [[0, 2], [3, 4]])
def test_overlapping_or_missing_payload_is_rejected(tmp_path, offsets):
    header = checkpoint(tmp_path, {"first": 2, "second": 2})
    header["second"]["data_offsets"] = offsets
    write_shard(tmp_path / "shard.safetensors", header, bytes(4))
    with pytest.raises(ValueError, match="overlapping or missing"):
        audit_checkpoint(tmp_path)


def test_trailing_unindexed_payload_is_rejected(tmp_path):
    header = checkpoint(tmp_path, {"head.weight": 4})
    write_shard(tmp_path / "shard.safetensors", header, bytes(5))
    with pytest.raises(ValueError, match="unindexed payload"):
        audit_checkpoint(tmp_path)


def test_unindexed_tensor_is_rejected(tmp_path):
    checkpoint(tmp_path, {"head.weight": 4})
    write_index(tmp_path, {"different.weight": "shard.safetensors"}, 4)
    with pytest.raises(ValueError, match="tensor/index mismatch"):
        audit_checkpoint(tmp_path)


def test_missing_indexed_tensor_is_rejected(tmp_path):
    checkpoint(tmp_path, {"head.weight": 4})
    write_index(
        tmp_path,
        {"head.weight": "shard.safetensors", "missing.weight": "shard.safetensors"},
        4,
    )
    with pytest.raises(ValueError, match="omit indexed tensors"):
        audit_checkpoint(tmp_path)


def test_index_total_size_must_match_actual_storage(tmp_path):
    checkpoint(tmp_path, {"head.weight": 4})
    write_index(tmp_path, {"head.weight": "shard.safetensors"}, 5)
    with pytest.raises(ValueError, match="total_size disagrees"):
        audit_checkpoint(tmp_path)


def test_tensor_in_wrong_shard_is_rejected(tmp_path):
    header = checkpoint(tmp_path, {"head.weight": 4})
    write_shard(tmp_path / "other.safetensors", header, bytes(4))
    write_index(
        tmp_path,
        {
            "head.weight": "shard.safetensors",
            "other.weight": "other.safetensors",
        },
        8,
    )
    with pytest.raises(ValueError, match="tensor/index mismatch"):
        audit_checkpoint(tmp_path)


def test_missing_shard_reports_the_missing_file(tmp_path):
    write_index(tmp_path, {"head.weight": "missing.safetensors"}, 4)
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        audit_checkpoint(tmp_path)
