import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
from tokenspeed.runtime.utils.server_args import ServerArgs


class TestWeightLoaderPrefetch(unittest.TestCase):
    def test_cli_flag_maps_to_server_args(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model",
                "test/model",
                "--weight-loader-prefetch-checkpoints",
                "--weight-loader-prefetch-num-threads",
                "2",
            ]
        )
        with mock.patch.object(ServerArgs, "__post_init__"):
            server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.weight_loader_prefetch_checkpoints)
        self.assertEqual(server_args.weight_loader_prefetch_num_threads, 2)

    def test_load_config_defaults_keep_prefetch_disabled(self):
        load_config = LoadConfig()

        self.assertFalse(load_config.weight_loader_prefetch_checkpoints)
        self.assertEqual(load_config.weight_loader_prefetch_num_threads, 4)

    def test_prefetch_splits_files_by_local_rank(self):
        files = [f"/tmp/model-{idx}.safetensors" for idx in range(6)]
        seen = []

        def record_prefetch(path):
            seen.append(path)
            return 1

        env = {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "3"}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                weight_utils, "_prefetch_checkpoint_file", side_effect=record_prefetch
            ),
        ):
            thread = weight_utils.prefetch_checkpoint_files(files, num_threads=2)
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(seen), [files[1], files[4]])


class TestWeightNameFilter(unittest.TestCase):
    def test_index_filter_selects_matching_and_mixed_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            language = folder / "model-00001.safetensors"
            vision = folder / "model-00002.safetensors"
            mixed = folder / "model-00003.safetensors"
            index = folder / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "language_model.weight": language.name,
                            "vision_tower.weight": vision.name,
                            "language_model.bias": mixed.name,
                            "mm_projector.weight": mixed.name,
                        }
                    }
                )
            )

            selected = weight_utils.filter_duplicate_safetensors_files(
                [str(language), str(vision), str(mixed)],
                str(folder),
                index.name,
                weight_name_filter=lambda name: (
                    "vision_tower" in name or "mm_projector" in name
                ),
            )

        self.assertEqual(selected, [str(vision), str(mixed)])

    def test_tensor_filter_applies_after_source_prefix(self):
        loader = DefaultModelLoader(LoadConfig())
        source = DefaultModelLoader.Source(
            "unused",
            None,
            prefix="base.",
            weight_name_filter=lambda name: name.startswith("base.vision_tower."),
        )
        tensors = [
            ("language_model.weight", object()),
            ("vision_tower.weight", object()),
        ]

        with (
            mock.patch.object(
                loader,
                "_prepare_weights",
                return_value=("unused", ["model.safetensors"], True),
            ) as prepare,
            mock.patch(
                "tokenspeed.runtime.model_loader.loader.safetensors_weights_iterator",
                return_value=iter(tensors),
            ),
        ):
            loaded = list(loader._get_weights_iterator(source))

        self.assertEqual(loaded, [("base.vision_tower.weight", tensors[1][1])])
        passed_filter = prepare.call_args.kwargs["weight_name_filter"]
        self.assertFalse(passed_filter("language_model.weight"))
        self.assertTrue(passed_filter("vision_tower.weight"))

    def test_remote_filter_downloads_only_matching_safetensors_shards(self):
        loader = DefaultModelLoader(LoadConfig())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            language = folder / "model-00001.safetensors"
            vision = folder / "model-00002.safetensors"
            projector = folder / "model-00003.safetensors"
            language.touch()
            vision.touch()
            projector.touch()
            index = folder / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "language_model.weight": language.name,
                            "vision_tower.weight": vision.name,
                            "mm_projector.weight": projector.name,
                        }
                    }
                )
            )
            with (
                mock.patch(
                    "tokenspeed.runtime.model_loader.loader.os.path.isdir",
                    return_value=False,
                ),
                mock.patch(
                    "tokenspeed.runtime.model_loader.loader.download_safetensors_index_file_from_hf",
                    return_value=str(index),
                ),
                mock.patch(
                    "tokenspeed.runtime.model_loader.loader.download_weights_from_hf",
                    return_value=str(folder),
                ) as download,
            ):
                _, files, use_safetensors = loader._prepare_weights(
                    "org/model",
                    None,
                    True,
                    weight_name_filter=lambda name: (
                        "vision_tower" in name or "mm_projector" in name
                    ),
                )

        self.assertTrue(use_safetensors)
        self.assertEqual(files, [str(vision), str(projector)])
        self.assertEqual(download.call_args.args[2], [vision.name, projector.name])

    def test_model_filter_is_forwarded_to_primary_source(self):
        loader = DefaultModelLoader(LoadConfig())
        weight_filter = lambda name: name.startswith("vision_tower.")
        model = SimpleNamespace(checkpoint_weight_name_filter=weight_filter)
        config = SimpleNamespace(model_path="model", revision=None)
        captured = []

        def capture_source(source):
            captured.append(source)
            return iter(())

        with mock.patch.object(loader, "_get_weights_iterator", capture_source):
            list(loader._get_all_weights(config, model))

        self.assertIs(captured[0].weight_name_filter, weight_filter)


if __name__ == "__main__":
    unittest.main()
