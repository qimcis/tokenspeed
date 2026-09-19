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

"""CPU/mocked checks that filtered tensors never reach InstantTensor's GPU loader."""

import json
import os
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadFormat
from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader


class TestInstantTensorFilter(unittest.TestCase):
    def _write_header(self, directory, filename, names):
        # No payload is written, including for the 100 GB tensor: planning must
        # inspect only the bounded JSON header, never materialize any data.
        path = os.path.join(directory, filename)
        header = {"__metadata__": {"format": "pt"}}
        offset = 0
        for name, size, dtype in names:
            header[name] = {
                "dtype": dtype,
                "shape": [size],
                "data_offsets": [offset, offset + size],
            }
            offset += size
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)
        return path

    def _nvidia(self):
        return mock.patch.object(
            weight_utils,
            "current_platform",
            return_value=SimpleNamespace(is_nvidia=True),
        )

    def test_partition_header_only_all_mixed_none_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            full = self._write_header(directory, "full", [("keep.a", 8, "I8")])
            mixed = self._write_header(
                directory,
                "mixed",
                [
                    ("drop.engram.embed.weight", 100_000_000_000, "F8_E4M3"),
                    ("keep.b", 8, "BF16"),
                ],
            )
            excluded = self._write_header(directory, "excluded", [("drop.c", 8, "I8")])
            empty = self._write_header(directory, "empty", [])
            self.assertLess(os.path.getsize(mixed), 1024)
            self.assertEqual(
                weight_utils._partition_safetensors_files_by_weight_names(
                    [mixed, excluded, full, empty],
                    lambda name: name.startswith("keep."),
                ),
                ([full], [mixed]),
            )

    def test_mixed_shard_never_materializes_excluded_tensor_or_prefetches(self):
        with tempfile.TemporaryDirectory() as directory:
            full = self._write_header(directory, "full", [("keep.a", 8, "I8")])
            mixed = self._write_header(
                directory,
                "mixed",
                [
                    ("drop.engram.embed.weight", 100_000_000_000, "F8_E4M3"),
                    ("keep.b", 8, "BF16"),
                ],
            )
            excluded = self._write_header(directory, "excluded", [("drop.c", 8, "I8")])
            handle = mock.MagicMock()
            handle.__enter__.return_value = handle
            handle.keys.return_value = ["drop.engram.embed.weight", "keep.b"]
            handle.get_tensor.side_effect = lambda name: self._allowed_tensor(name)
            with (
                self._nvidia(),
                mock.patch.dict(sys.modules, {"instanttensor": SimpleNamespace()}),
                mock.patch.object(
                    weight_utils,
                    "_instanttensor_tensors",
                    return_value=iter([("keep.a", "gpu")]),
                ) as gpu,
                mock.patch.object(
                    weight_utils, "safe_open", return_value=handle
                ) as cpu,
                mock.patch.object(weight_utils, "CheckpointPrefetcher") as prefetch,
            ):
                result = list(
                    weight_utils.instanttensor_weights_iterator(
                        [full, mixed, excluded],
                        accept=lambda name: name.startswith("keep."),
                    )
                )
            self.assertEqual(result, [("keep.a", "gpu"), ("keep.b", "cpu")])
            self.assertEqual(gpu.call_args.args[1], [full])
            cpu.assert_called_once_with(mixed, framework="pt", device="cpu")
            handle.get_tensor.assert_called_once_with("keep.b")
            prefetch.assert_not_called()

    def _allowed_tensor(self, name):
        self.assertEqual(name, "keep.b", "Excluded tensor was materialized")
        return "cpu"

    def test_no_accepted_weights_does_not_open_either_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write_header(directory, "excluded", [("drop", 8, "I8")])
            with (
                self._nvidia(),
                mock.patch.dict(sys.modules, {"instanttensor": SimpleNamespace()}),
                mock.patch.object(weight_utils, "_instanttensor_tensors") as gpu,
                mock.patch.object(weight_utils, "safe_open") as cpu,
            ):
                self.assertEqual(
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            [shard], accept=lambda name: False
                        )
                    ),
                    [],
                )
            gpu.assert_not_called()
            cpu.assert_not_called()

    def test_no_filter_uses_existing_gpu_iterator(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write_header(directory, "full", [("weight", 8, "I8")])
            with (
                self._nvidia(),
                mock.patch.dict(sys.modules, {"instanttensor": SimpleNamespace()}),
                mock.patch.object(
                    weight_utils,
                    "_instanttensor_tensors",
                    return_value=iter([("weight", "gpu")]),
                ) as gpu,
                mock.patch.object(weight_utils, "safe_open") as cpu,
            ):
                self.assertEqual(
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            [shard], accept=None
                        )
                    ),
                    [("weight", "gpu")],
                )
            self.assertEqual(gpu.call_args.args[1], [shard])
            cpu.assert_not_called()

    def test_f4_guard_still_raises_before_any_loader_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            for accept in (None, lambda name: True, lambda name: name == "keep"):
                shard = self._write_header(
                    directory, "subbyte", [("weight", 1024, "F4"), ("keep", 8, "I8")]
                )
                with (
                    self._nvidia(),
                    mock.patch.dict(sys.modules, {"instanttensor": SimpleNamespace()}),
                    mock.patch.object(weight_utils, "_instanttensor_tensors") as gpu,
                    mock.patch.object(weight_utils, "safe_open") as cpu,
                    self.assertRaisesRegex(ValueError, "F4"),
                ):
                    weight_utils.instanttensor_weights_iterator([shard], accept=accept)
                gpu.assert_not_called()
                cpu.assert_not_called()

    def test_bad_header_raises_before_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad")
            for raw in (b"", struct.pack("<Q", 2**62), struct.pack("<Q", 100) + b"{}"):
                with open(path, "wb") as handle:
                    handle.write(raw)
                with self.assertRaises((ValueError, struct.error)):
                    weight_utils._partition_safetensors_files_by_weight_names(
                        [path], lambda name: True
                    )

    def test_loader_predicate_sees_source_prefix_exactly_once(self):
        from tokenspeed.runtime.model_loader import loader as loader_module

        with tempfile.TemporaryDirectory() as directory:
            shard = self._write_header(
                directory, "mixed", [("keep", 8, "I8"), ("drop", 8, "I8")]
            )
            loader = object.__new__(DefaultModelLoader)
            loader.load_config = SimpleNamespace(load_format=LoadFormat.INSTANTTENSOR)
            source = SimpleNamespace(
                model_or_path=directory,
                revision=None,
                fall_back_to_pt=False,
                prefix="model.",
            )
            seen = []

            def accept(name):
                seen.append(name)
                return name == "model.keep"

            def iterate(files, accept):
                self.assertEqual(files, [shard])
                return ((name, "value") for name in ("keep", "drop") if accept(name))

            with (
                mock.patch.object(
                    loader, "_prepare_weights", return_value=(directory, [shard], True)
                ),
                mock.patch.object(
                    loader_module, "instanttensor_weights_iterator", side_effect=iterate
                ),
            ):
                result = list(loader._get_weights_iterator(source, accept))
            self.assertEqual(result, [("model.keep", "value")])
            self.assertEqual(seen, ["model.keep", "model.drop"])


if __name__ == "__main__":
    unittest.main()
