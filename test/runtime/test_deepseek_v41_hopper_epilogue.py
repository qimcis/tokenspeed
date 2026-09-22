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

"""CPU contract tests loading the real runtime methods without GPU registries."""

from __future__ import annotations

import ast
import os
import unittest
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = _ROOT / "python/tokenspeed/runtime/models"


def _definitions(path: Path, functions: set[str], classes: dict[str, set[str]]):
    nodes = []
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name in classes:
            node.body = [
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef)
                and child.name in classes[node.name]
            ]
            nodes.append(node)
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


class RuntimeEpilogueContractTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.quantized = []
        events = self.events

        @contextmanager
        def nvtx_range(name):
            events.append(("range", name))
            yield

        class Fork:
            @contextmanager
            def scope(self, *, enable):
                events.append(("scope", enable))
                yield self
                events.append(("join",))

            @contextmanager
            def branch(self):
                events.append(("branch",))
                yield
                events.append(("branch_done",))

        class LinearBase:
            def apply(method, layer, codes, bias, scales, output_dtype):
                self.quantized.append((codes, scales, output_dtype))
                events.append(("shared_projection",))
                return torch.full_like(codes, 3, dtype=output_dtype)

        self.namespace = {
            "torch": torch,
            "nn": SimpleNamespace(Module=object),
            "Fp8LinearMethod": LinearBase,
            "Callable": Callable,
            "nvtx_range": nvtx_range,
            "get_is_capture_mode": lambda: False,
        }
        exec(
            _definitions(
                _MODELS / "deepseek_v4.py",
                set(),
                {
                    "DeepseekV4MoE": {
                        "forward_normal",
                        "_forward_normal_with_shared",
                        "_forward_shared_experts",
                    }
                },
            ),
            self.namespace,
        )
        exec(
            _definitions(
                _MODELS / "deepseek_v41.py",
                {
                    "v41_quantize_fp8",
                    "_v41_shared_quantization_supported",
                    "_v41_hopper_epilogue_supported",
                },
                {
                    "_ReferenceFp8LinearMethod": {"apply", "apply_prequantized"},
                    "DeepseekV41MoE": {"forward_with_shared_quantized"},
                },
            ),
            self.namespace,
        )
        self.ffn = self.namespace["DeepseekV41MoE"]()
        self.ffn.use_mega_moe = False
        self.ffn.stream_fork = Fork()
        self.ffn.routed_scaling_factor = 2.0
        self.ffn.config = SimpleNamespace(norm_topk_prob=False)
        self.bypassed = False
        self.hidden = (
            torch.arange(256, dtype=torch.float32).reshape(4, 64).to(torch.bfloat16)
        )
        self.method = self.namespace["_ReferenceFp8LinearMethod"]()
        gate = SimpleNamespace(
            quant_method=self.method,
            weight=torch.empty((64, 64), dtype=torch.float8_e4m3fn),
            bias=None,
            gather_output=False,
        )

        def down(gate_up, activation):
            events.append(("shared_down",))
            return gate_up, None

        class Shared:
            gate_up_proj = gate
            down_proj = SimpleNamespace(forward_with_activation=down)
            act_fn = object()

            def __call__(shared, hidden):
                value = self.method.apply(gate, hidden, None)
                return down(value, shared.act_fn)[0]

        self.ffn.shared_experts = Shared()

        def select(hidden, input_ids):
            self.assertIs(hidden, self.hidden)
            self.assertEqual(hidden.dtype, torch.bfloat16)
            events.append(("router",))
            weights = torch.tensor([[0.25, 0.5]], dtype=torch.float32).expand(
                hidden.shape[0], -1
            )
            return weights, None, None

        def topk(hidden, weights, ids, scores):
            events.append(("topk",))
            return SimpleNamespace(
                format=SimpleNamespace(is_bypassed=lambda: self.bypassed)
            )

        def experts(**kwargs):
            self.assertIs(kwargs["hidden_states"], self.hidden)
            self.assertEqual(kwargs["num_global_tokens"], self.hidden.shape[0])
            self.assertEqual(kwargs["max_num_tokens_per_gpu"], self.hidden.shape[0])
            events.append(("experts",))
            return torch.full_like(self.hidden, 2)

        self.ffn._select_experts = select
        self.ffn._make_topk_output = topk
        self.ffn.experts = experts

    def test_prequantized_matches_original_routing_scaling_and_stream_order(self):
        quantizer = self.namespace["v41_quantize_fp8"]
        for capture in (False, True):
            for bypassed in (False, True):
                for normalize in (False, True):
                    with self.subTest(
                        capture=capture, bypassed=bypassed, normalize=normalize
                    ):
                        self.namespace["get_is_capture_mode"] = lambda: capture
                        self.bypassed = bypassed
                        self.ffn.config.norm_topk_prob = normalize
                        codes, scales = quantizer(self.hidden)
                        self.events.clear()
                        baseline = self.ffn.forward_normal(self.hidden, None, 4, 4)
                        expected_events = self.events.copy()
                        self.events.clear()
                        with patch.dict(
                            self.namespace,
                            {
                                "v41_quantize_fp8": lambda x: self.fail(
                                    "requantized fused input"
                                )
                            },
                        ):
                            actual = self.ffn.forward_with_shared_quantized(
                                self.hidden, None, 4, 4, codes, scales
                            )
                        self.assertTrue(torch.equal(actual, baseline))
                        expected = 6 if bypassed and not normalize else 7
                        self.assertTrue(
                            torch.equal(actual, torch.full_like(actual, expected))
                        )
                        self.assertEqual(self.events, expected_events)
                        order = [
                            event[0] for event in self.events if event[0] != "range"
                        ]
                        self.assertEqual(
                            order,
                            [
                                "router",
                                "topk",
                                "scope",
                                "experts",
                                "branch",
                                "shared_projection",
                                "shared_down",
                                "branch_done",
                                "join",
                            ],
                        )
                        self.assertIs(self.quantized[-1][0], codes)
                        self.assertIs(self.quantized[-1][1], scales)
                        self.assertEqual(self.quantized[-1][2], torch.bfloat16)

    def test_original_path_without_shared_experts_and_empty_input(self):
        self.ffn.shared_experts = None
        actual = self.ffn.forward_normal(self.hidden, None, 4, 4)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 4)))
        self.hidden = self.hidden[:0]
        self.events.clear()
        self.assertIs(self.ffn.forward_normal(self.hidden, None, 0, 0), self.hidden)
        self.assertEqual(self.events, [])

    def test_shared_quantization_rejects_layout_changes_and_unsupported_linears(self):
        supported = self.namespace["_v41_shared_quantization_supported"]
        comm = SimpleNamespace(
            mapping=SimpleNamespace(moe=SimpleNamespace(has_tp_ep=True)),
            use_all_reduce=lambda **kwargs: True,
        )
        self.assertTrue(supported(self.ffn, comm))
        comm.use_all_reduce = lambda **kwargs: False
        self.assertFalse(supported(self.ffn, comm))
        comm.use_all_reduce = lambda **kwargs: True
        gate = self.ffn.shared_experts.gate_up_proj
        for field, value in (
            ("gather_output", True),
            ("bias", torch.zeros(64)),
            ("quant_method", object()),
        ):
            original = getattr(gate, field)
            setattr(gate, field, value)
            self.assertFalse(supported(self.ffn, comm))
            setattr(gate, field, original)
        self.ffn.use_mega_moe = True
        self.assertFalse(supported(self.ffn, comm))

    def test_prequantized_input_rejects_mismatched_rows_before_routing(self):
        codes, scales = self.namespace["v41_quantize_fp8"](self.hidden)
        for invalid_codes, invalid_scales in (
            (codes[:2], scales),
            (codes, scales[:, :1]),
            (codes.float(), scales),
            (codes, scales.float()),
        ):
            with self.subTest(codes=invalid_codes.shape, scales=invalid_scales.shape):
                self.events.clear()
                with self.assertRaises(ValueError):
                    self.ffn.forward_with_shared_quantized(
                        self.hidden, None, 4, 4, invalid_codes, invalid_scales
                    )
                self.assertEqual(self.events, [])

    def test_hopper_policy_is_h20_only_and_rejects_other_shapes(self):
        supported = self.namespace["_v41_hopper_epilogue_supported"]

        class TensorMetadata:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
                self.ndim = len(shape)
                self.is_cuda, self.device = True, "cuda:0"
                self.contiguous = True

            def is_contiguous(self):
                return self.contiguous

        def arch(major, minor):
            return major, minor

        platform = SimpleNamespace(
            is_nvidia=True, arch_version=(9, 0), device_name="NVIDIA H20"
        )
        module = ModuleType("tokenspeed_kernel.platform")
        module.ArchVersion = arch
        module.Platform = SimpleNamespace(get=lambda: platform)
        x = TensorMetadata((4, 5120), torch.bfloat16)
        residual = TensorMetadata((4, 4, 5120), torch.bfloat16)
        post = TensorMetadata((4, 4), torch.float32)
        comb = TensorMetadata((4, 4, 4), torch.float32)
        pre = TensorMetadata((4, 4), torch.float32)
        norm = SimpleNamespace(weight=TensorMetadata((5120,), torch.float32))
        with patch.dict("sys.modules", {"tokenspeed_kernel.platform": module}):
            self.assertTrue(supported(x, residual, post, comb, pre, norm))
            platform.device_name = "NVIDIA H100"
            self.assertFalse(supported(x, residual, post, comb, pre, norm))
            platform.device_name = "NVIDIA H20"
            post.contiguous = False
            self.assertFalse(supported(x, residual, post, comb, pre, norm))
            post.contiguous = True
            x.shape = (16, 5120)
            self.assertFalse(supported(x, residual, post, comb, pre, norm))
            x.shape = (4, 5120)
            x.is_cuda = False
            self.assertFalse(supported(x, residual, post, comb, pre, norm))

    def test_experimental_flag_defaults_off_and_is_fixed_before_capture(self):
        env_path = _ROOT / "python/tokenspeed/runtime/utils/env.py"
        env_tree = ast.parse(env_path.read_text(), filename=str(env_path))
        env_nodes = [node for node in env_tree.body if isinstance(node, ast.ClassDef)]
        namespace = {"os": os, "warnings": warnings, "contextmanager": contextmanager}
        exec(
            compile(ast.Module(body=env_nodes, type_ignores=[]), str(env_path), "exec"),
            namespace,
        )
        namespace["envs"] = namespace["Envs"]()
        model_path = _MODELS / "deepseek_v41.py"
        model_tree = ast.parse(model_path.read_text(), filename=str(model_path))
        layer = next(
            node
            for node in model_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41DecoderLayer"
        )
        init = next(
            node
            for node in layer.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        flag_assignment = next(
            node
            for node in init.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "_experimental_hopper_ffn_epilogue"
                for target in node.targets
            )
        )
        flag_code = compile(
            ast.Module(body=[flag_assignment], type_ignores=[]), str(model_path), "exec"
        )
        forward = next(
            node
            for node in layer.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        fusion_if = next(
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == "_experimental_hopper_ffn_epilogue"
                for child in ast.walk(node.test)
            )
        )
        dispatch = compile(ast.Expression(body=fusion_if.test), str(model_path), "eval")
        flag = "TOKENSPEED_EXPERIMENTAL_V41_HOPPER_EPILOGUE"
        with patch.dict(os.environ, {}, clear=True):
            for value, expected in ((None, False), ("0", False), ("1", True)):
                with self.subTest(value=value):
                    if value is None:
                        os.environ.pop(flag, None)
                    else:
                        os.environ[flag] = value
                    decoder = SimpleNamespace(ffn_norm=None)
                    namespace["self"] = decoder
                    exec(flag_code, namespace)
                    self.assertIs(decoder._experimental_hopper_ffn_epilogue, expected)
                    # Changing the process environment cannot change an already
                    # constructed layer's eager-vs-graph dispatch.
                    os.environ[flag] = "0" if expected else "1"
                    metadata = {
                        "self": decoder,
                        "ctx": SimpleNamespace(
                            forward_mode=SimpleNamespace(is_decode=lambda: True)
                        ),
                        "rows": SimpleNamespace(keep_rows=None),
                        "x": None,
                        "residual": None,
                        "post": None,
                        "comb": None,
                        "attn_pre": None,
                        "_v41_hopper_epilogue_supported": lambda *args: True,
                    }
                    self.assertIs(eval(dispatch, metadata), expected)


if __name__ == "__main__":
    unittest.main()
