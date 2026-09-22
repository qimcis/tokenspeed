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

"""Bounded complete-chain graph comparison; run only under stage4 controller."""

import argparse
import hashlib
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
started = time.monotonic()
record = {
    "status": "running",
    "rows": [],
    "metric": "paired complete-chain hot-input CUDA graph microseconds",
    "calls_per_graph": 16,
    "replays": 4,
    "seeds": [103, 211, 401],
}


def save():
    record["wall_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(record, indent=2) + "\n")


try:
    import torch
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_group32_ue8m0,
    )
    from tokenspeed_kernel.ops.residual.hopper_v41 import v41_post_pre_norm_quant
    from tokenspeed_kernel.ops.residual.triton import (
        mhc_pre_layer_norm_hc4,
        triton_mhc_post,
    )

    torch.set_num_threads(1)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_name() == "NVIDIA H20"
    record["sources"] = {
        str(Path(f.__code__.co_filename)): hashlib.sha256(
            Path(f.__code__.co_filename).read_bytes()
        ).hexdigest()
        for f in (
            v41_post_pre_norm_quant,
            triton_mhc_post,
            mhc_pre_layer_norm_hc4,
            triton_quantize_fp8_group32_ue8m0,
        )
    }
    for m in (1, 2, 4, 8):
        for quantize in (False, True):
            for seed in record["seeds"]:
                torch.manual_seed(seed)
                x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
                r = torch.randn(m, 4, 5120, device="cuda", dtype=torch.bfloat16)
                post = torch.rand(m, 4, device="cuda", dtype=torch.float32)
                comb = torch.rand(m, 4, 4, device="cuda", dtype=torch.float32)
                pre = torch.rand(m, 4, device="cuda", dtype=torch.float32)
                weight = torch.randn(5120, device="cuda", dtype=torch.float32)

                def baseline():
                    hc = triton_mhc_post(x, r, post, comb)
                    norm = torch.empty_like(x)
                    mhc_pre_layer_norm_hc4(pre, hc, weight, norm, eps=1e-6)
                    q, s = (
                        triton_quantize_fp8_group32_ue8m0(
                            norm, "token_group", 32, "ue8m0", False
                        )
                        if quantize
                        else (None, None)
                    )
                    return hc, norm, q, s

                def candidate():
                    return v41_post_pre_norm_quant(
                        x, r, post, comb, pre, weight, 1e-6, quantize
                    )

                expected, actual = baseline(), candidate()
                errors = []
                for e, a in zip(expected[:2], actual[:2]):
                    assert torch.isfinite(a).all()
                    error = (
                        (e.float() - a.float()).norm()
                        / e.float().norm().clamp_min(1e-12)
                    ).item()
                    assert error <= 0.006
                    errors.append(error)
                if quantize:
                    q, s = triton_quantize_fp8_group32_ue8m0(
                        actual[1], "token_group", 32, "ue8m0", False
                    )
                    assert torch.equal(
                        q.view(torch.uint8), actual[2].view(torch.uint8)
                    ) and torch.equal(s, actual[3])
                graphs = []
                for fn in (baseline, candidate):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(16):
                            out = fn()
                    graphs.append(g)
                for _ in range(4):
                    for g in graphs:
                        g.replay()
                torch.cuda.synchronize()
                pairs = []
                for pair in range(4):
                    measurements = {}
                    for arm in ((0, 1) if pair % 2 == 0 else (1, 0)):
                        begin, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(4):
                            graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        measurements[str(arm)] = begin.elapsed_time(end) * 1000 / 64
                    pairs.append(measurements)
                record["rows"].append(
                    {
                        "m": m,
                        "quantize": quantize,
                        "seed": seed,
                        "normalized_l2": errors,
                        "paired_us": pairs,
                    }
                )
                save()
    record["status"] = "complete"
except BaseException as error:
    record["status"] = "failed"
    record["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    save()
