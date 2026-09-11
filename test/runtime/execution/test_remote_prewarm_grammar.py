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
"""Exercise production prewarm orchestration without loading GPU kernel registries.

The methods and mask binder are compiled directly from their production ASTs;
only their GPU/grammar collaborators are fakes. This keeps the CPU scheduler
suite able to test queue ownership and mask geometry on machines without CUDA.
"""

import ast
import queue
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _functions(path, class_name, names):
    module = ast.parse((ROOT / path).read_text())
    nodes = module.body
    if class_name is not None:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    selected = [
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    return ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *selected,
            ],
            type_ignores=[],
        )
    )


@pytest.mark.parametrize("width", (1, 6, 8))
def test_prewarm_queues_each_launch_and_binds_the_actual_width(monkeypatch, width):
    grammar_module = ModuleType("tokenspeed.runtime.grammar.capturable_grammar")
    binder = _functions(
        "python/tokenspeed/runtime/grammar/capturable_grammar.py",
        None,
        {"bind_grammar_mask_buf"},
    )
    exec(compile(binder, "capturable_grammar.py", "exec"), grammar_module.__dict__)
    backend_module = ModuleType("tokenspeed.runtime.grammar.base_grammar_backend")
    backend_module.get_apply_vocab_mask_func = lambda _: object()
    monkeypatch.setitem(sys.modules, grammar_module.__name__, grammar_module)
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)

    scope = {
        "queue": queue,
        "ForwardContext": SimpleNamespace,
        "ForwardMode": SimpleNamespace(DECODE="decode"),
        "CaptureHiddenMode": SimpleNamespace(FULL="full", NULL="null"),
        "SamplingBatchInfo": SimpleNamespace,
        "dist": SimpleNamespace(barrier=lambda: None),
        "CUDA_GRAPH_VARIANT_DEFAULT": "default",
        "_is_cuda_graph_phase": False,
    }
    methods = _functions(
        "python/tokenspeed/runtime/execution/forward_step.py",
        "ForwardStepRunner",
        {"prewarm_comm_states", "_grammar_for_width", "_prepare_sampling_capture"},
    )
    exec(compile(methods, "forward_step.py", "exec"), scope)
    runner_type = type(
        "Runner",
        (),
        {
            name: scope[name]
            for name in (
                "prewarm_comm_states",
                "_grammar_for_width",
                "_prepare_sampling_capture",
            )
        },
    )
    observed = []
    grammar = SimpleNamespace(
        max_tokens_per_req=width,
        bitmask=torch.empty(3 * width, 2, dtype=torch.int32),
        queue=queue.Queue(),
        current_batch=None,
    )
    grammar.add_batch = lambda **batch: grammar.queue.put(batch)

    def reset():
        grammar.current_batch = None

    grammar.reset_state = reset

    def forward(bs, ctx, sampling_info):
        # This is the hostfunc's get_nowait, so a missing add_batch fails.
        batch = grammar.queue.get_nowait()
        grammar.current_batch = batch
        assert batch["bs"] == bs
        assert batch["tokens_per_req"] == width
        assert ctx.decode_input_tokens == width
        assert ctx.input_num_tokens == bs * width
        assert sampling_info.vocab_mask.shape[0] == bs * width
        observed.append(bs)

    runner = runner_type()
    runner._forward_func = forward
    runner.grammar_runtimes = None
    runner.capturable_grammar = grammar
    runner.eager_grammar_buffers = None
    runner.max_tokens_per_req = width
    runner.attn_backend = object()
    runner.token_to_kv_pool = object()
    runner.drafter = None
    runner.prepare_target_capture = object()
    runner.dp_size = 1
    runner.world_size = 1
    runner.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.arange(3), seq_lens_buf=torch.ones(3)
    )
    runner.runtime_states = None
    runner.vocab_size = 64
    runner.device = "cpu"
    runner.grammar_backend = "xgrammar"
    runner.device_module = SimpleNamespace(synchronize=lambda: None)
    runner.sampling_backend = SimpleNamespace(
        prepare_capture_variant=lambda **kwargs: None,
        reset_capture_state=lambda: None,
    )
    runner._init_capture_metadata = lambda bs, ctx: None

    runner.prewarm_comm_states(batch_sizes=(1, 3))

    assert observed == [1, 3]
    assert grammar.queue.empty()
    assert grammar.current_batch is None
    assert not scope["_is_cuda_graph_phase"]
