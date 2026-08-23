"""GLM DSA value projection must dispatch through the kernel registry."""

from __future__ import annotations

import ast
from pathlib import Path


def test_glm_dsa_value_projection_uses_registered_operation() -> None:
    model_path = Path(__file__).parents[2] / "python/tokenspeed/runtime/models/glm5.py"
    source = model_path.read_text()
    tree = ast.parse(source)
    attention_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GlmMoeDsaAttention"
    )
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in attention_class.body
        if isinstance(node, ast.FunctionDef)
    }

    for method_name in (
        "forward_dsa_sparse_prefill",
        "forward_absorb_attn_v_proj",
    ):
        method_source = methods[method_name]
        assert "mla_project_value(" in method_source
        assert "torch.bmm(" not in method_source
