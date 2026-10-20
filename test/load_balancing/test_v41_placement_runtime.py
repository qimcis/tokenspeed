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

"""Opt-in actual runtime checks requiring the vendor dependency closure.

CPU loader/routes: TOKENSPEED_TEST_V41_RUNTIME=1 pytest -q this_file
Two-GPU MegaMoE: TOKENSPEED_TEST_V41_GPU=1 torchrun --standalone \
  --nproc_per_node=2 -m pytest -q this_file -k distributed
These checks are not enabled by CUDA availability alone.
"""

import os

import pytest
import torch

from tokenspeed.runtime.models.v41_expert_placement import pad_expert_tensor

runtime = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_V41_RUNTIME") != "1",
    reason="opt-in complete vendor runtime required",
)
gpu = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_TEST_V41_GPU") != "1",
    reason="opt-in two-GPU B300 qualification",
)


def _placement(rows, logical, rank, device):
    from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

    return ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([rows], device=device),
        logical,
        ep_size=2,
        ep_rank=rank,
        ep_rank_nodes=(0, 0),
    )


def _weights(rows, logical, rank, device):
    from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
    from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
    from tokenspeed.runtime.layers.moe.types import MoELayerSpec
    from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_weight_pair

    placement = _placement(rows, logical, rank, device)
    spec = MoELayerSpec(
        top_k=2,
        num_experts=len(rows),
        num_local_experts=len(rows) // 2,
        hidden_size=512,
        intermediate_size=512,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=rank,
        ep_size=2,
    )

    class Weights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = spec.num_experts
            self.num_local_experts = spec.num_local_experts
            self.hidden_size = spec.hidden_size
            self.intermediate_size = spec.intermediate_size
            self.top_k = spec.top_k
            self.swiglu_arg = None
            self._moe_backend_state = None
            with torch.device(device):
                create_mxfp4_weight_pair(spec, self)

    weights = Weights()
    loader = build_moe_checkpoint_loader(
        params_dict={
            "model.layers.0.ffn.experts." + n: p for n, p in weights.named_parameters()
        },
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="w1", up_proj_name="w3", down_proj_name="w2"
        ),
        num_experts=len(rows),
        ep_rank=rank,
        ep_size=2,
        expert_placement=placement,
    )
    for expert in range(logical):
        for projection in ("w1", "w2", "w3"):
            for scale in (False, True):
                divisor = 32 if scale else 2
                shape = (
                    (512, 128 // divisor)
                    if projection == "w2"
                    else (128, 512 // divisor)
                )
                # Small nonzero E2M1 nibbles; source expert identity affects all projections.
                value = 120 if scale else (0x11 if expert % 2 else 0x22)
                tensor = torch.full(shape, value, dtype=torch.uint8, device=device)
                tensor = pad_expert_tensor(
                    tensor,
                    projection=projection,
                    is_scale=scale,
                    intermediate_size=128,
                    padded_intermediate_size=512,
                )
                name = f"model.layers.0.ffn.experts.{expert}.{projection}." + (
                    "weight_scale" if scale else "weight"
                )
                if loader.matches(name):
                    loader.load(name, tensor)
    return placement, weights


@runtime
def test_real_loader_copies_remote_and_repeated_logical_experts():
    placement, weights = _weights([0, 1, 2, 3, 0, 0], 4, 1, "cpu")
    assert placement.local_slot_logical_experts(0, 1) == [3, 0, 0]
    for tensor in (
        weights.w13_weight,
        weights.w13_weight_scale,
        weights.w2_weight,
        weights.w2_weight_scale,
    ):
        assert torch.equal(tensor[1], tensor[2])
    assert torch.all(weights.w13_weight[0, :128] == 0x11)
    assert torch.all(weights.w13_weight[1, :128] == 0x22)
    assert torch.all(weights.w13_weight[:, 128:512] == 0)
    assert torch.all(weights.w13_weight_scale[:, 128:512] == 127)
    assert torch.all(weights.w2_weight[:, :, 64:] == 0)
    assert torch.all(weights.w2_weight_scale[:, :, 4:] == 127)


@runtime
@pytest.mark.parametrize("algorithm", ["static", "dynamic"])
def test_logical_route_executes_exactly_once_with_replicas_and_empty_rows(algorithm):
    from tokenspeed.runtime.layers.moe.topk import (
        ExpertLocationDispatchInfo,
        topk_ids_logical_to_physical,
    )

    rows = [0, 1, 2, 3, 0, 0]
    for rank in (0, 1):
        placement = _placement(rows, 4, rank, "cpu")
        info = ExpertLocationDispatchInfo.init_new(
            0, algorithm, placement, all_to_all_ep=True
        )
        for logical in (
            torch.tensor([[0, 1], [3, 0], [2, 3]]),
            torch.empty((0, 2), dtype=torch.int64),
        ):
            before = logical.clone()
            physical = topk_ids_logical_to_physical(logical, info, 4)
            assert physical.shape == logical.shape
            assert torch.equal(torch.tensor(rows)[physical], logical)
            assert torch.equal(logical, before)
            for token in physical:
                assert len(token.unique()) == 2


@gpu
def test_distributed_mega_moe_replication_matches_logical_baseline():
    import torch.distributed as dist
    from tokenspeed_kernel.ops.moe import moe_plan
    from tokenspeed_kernel.ops.moe.deep_gemm import (
        deep_gemm_mxfp4_mega_moe_apply,
        deep_gemm_mxfp4_mega_moe_process_weights,
    )
    from tokenspeed_kernel.platform import pdl_enabled

    from tokenspeed.runtime.layers.moe.topk import (
        ExpertLocationDispatchInfo,
        topk_ids_logical_to_physical,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        assert dist.get_world_size() == 2
        assert torch.cuda.get_device_capability(device)[0] == 10
        torch.manual_seed(123 + rank)
        x = torch.randn((8, 512), dtype=torch.bfloat16, device=device)
        logical = torch.tensor([[0, 1], [3, 0], [2, 3], [1, 2]] * 2, device=device)
        probabilities = torch.full((8, 2), 0.5, device=device)
        outputs = []
        for rows in (list(range(32)), list(reversed(range(32))) + list(range(16))):
            placement, weights = _weights(rows, 32, rank, device)
            plan = moe_plan(
                "mxfp4",
                input_dtype=torch.bfloat16,
                activation="swiglu",
                routing_mode="precomputed_topk",
                a2a_backend="none",
                ep_size=2,
                ispp=512,
                hidden=512,
                num_experts=len(rows),
                top_k=2,
                swiglu_form="standard",
                activation_clamped=False,
                expert_id_repeats=False,
                internal_activation_dtype="input",
                process_group=dist.group.WORLD,
                persistent_max_num_tokens_per_gpu=16,
                fast_math=False,
                combine_order="rank",
                features=frozenset(),
                solution="mega_moe",
            )
            # This also controls EPLB SUM versus agreement of rank-local loads.
            assert plan["supports_all_to_all_ep"]
            deep_gemm_mxfp4_mega_moe_process_weights(plan, weights)
            info = ExpertLocationDispatchInfo.init_new(
                0, "static", placement, all_to_all_ep=plan["supports_all_to_all_ep"]
            )
            physical = topk_ids_logical_to_physical(logical, info, 32)
            outputs.append(
                deep_gemm_mxfp4_mega_moe_apply(
                    plan,
                    x,
                    weights,
                    None,
                    probabilities,
                    physical,
                    16,
                    8,
                    True,
                    pdl_enabled(),
                )
            )
            torch.cuda.synchronize()
        assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0.01, atol=0.001)
    finally:
        dist.destroy_process_group()
