# Hopper MXFP4 W4A16 MoE comparison

These scripts compare the corrected Marlin implementation with the experimental
FlashInfer/CUTLASS implementation on SM90. They exercise one layer's complete
local-expert chain: routing and clearing, both expert GEMMs, clamp-10 SwiGLU,
and weighted finalization. Gate/top-k selection, communication, shared experts,
and the model's routed scaling factor are excluded equally.

The fixed workload is 48 local experts, EP8, hidden size 5120, intermediate size
2304, top-k 6, and token counts 1, 2, 4, 8, 128, and 1024. Both implementations
use the same deterministic packed E2M1 weights, E8M0 group-32 scales, BF16 inputs,
global expert IDs, and globally normalized routing weights. The source weights
are released after preparing both layouts. Outputs are local EP partials.

This is a synthetic single-layer comparison, not an end-to-end serving
benchmark. Decode regressions must not be hidden by prefill improvements.
The experimental adapter is not selected automatically.

## Requirements

The measured stack was an NVIDIA H20 (SM90, 78 SMs), PyTorch
`2.13.0+cu130`, `flashinfer-python==0.6.18`, and
`tokenspeed-triton==3.8.10.post20260906`. The native adapter used heuristic
`profile_ids=[-1, -1]`; no autotuning search was included in the measurement.

Use an SM90 GPU and an installed TokenSpeed kernel environment with BF16 Marlin
and repack libraries, pytest, and a compatible optional FlashInfer package. The
FlashInfer package must expose SM90 mixed-input weight/scale interleaving and
persistent CUTLASS workspace APIs. The adapter checks these interfaces.

Build the FlashInfer `fused_moe_90` module before the bounded runs:

```python
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

gen_cutlass_fused_moe_sm90_module(False).build_and_load()
```

Use the same source, environment, native library, and compiler caches for
validation and timing. Cold compilation may exceed the short worker deadlines;
an incomplete run is a failure, not a performance result. The caller owns GPU
reservation and process cleanup. GNU `timeout` supplies an independent bound in
the examples below.

## Reproduce

Run from the repository root. Set `NATIVE_SO` to the actual prebuilt
`fused_moe_90` shared library and use a fresh output directory:

```bash
REPO="$PWD"
TOOLS="$REPO/tokenspeed-kernel/benchmarks/moe"
RUN=$(mktemp -d)
export PYTHONPATH="$REPO/tokenspeed-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

# CPU-only byte-identity receipt; this does not load the native module.
python "$TOOLS/freeze_native_w4a16.py" \
  --library "$NATIVE_SO" --output "$RUN/native.json"

timeout --signal=TERM --kill-after=2s 30s \
  python "$TOOLS/validate_w4a16.py" \
  --repo "$REPO" --output "$RUN/validation.json" --seconds 25

VALIDATION_SHA=$(sha256sum "$RUN/validation.json" | cut -d' ' -f1)
BUILD_SHA=$(sha256sum "$RUN/native.json" | cut -d' ' -f1)
timeout --signal=TERM --kill-after=2s 28s \
  python "$TOOLS/benchmark_w4a16.py" \
  --source-root "$REPO" \
  --validation "$RUN/validation.json" --validation-sha256 "$VALIDATION_SHA" \
  --native-library "$NATIVE_SO" \
  --build-evidence "$RUN/native.json" --build-evidence-sha256 "$BUILD_SHA" \
  --output "$RUN/benchmark.json" --seed 20260922 --ep-rank 0 --device 0 \
  --wall-seconds 24 --repeats 5 --warmup-replays 2 \
  --target-sample-ms 2 --max-inner 32
```

The benchmark rejects failed validation or changed source/artifact identities.
The native receipt hashes the binary before GPU timing; the benchmark checks
the receipt hash, resolved library path, size, and modification time before and
after the run. Retain the native binary and receipt together. This avoids hashing
a large library inside the 24-second worker limit.

The validator retains source hashes, per-case results, and actual/reference
outputs. Its independent oracle covers small shapes and the actual hidden and
intermediate sizes with two local experts and EP128. It does not claim a full
48-expert oracle. The benchmark additionally requires eager and graph parity
against corrected Marlin at every measured 48-expert shape: per-row relative L2
at most 0.006, maximum absolute error at most `0.002 + 0.02 * reference_row_max`,
and exact zeros for zero reference rows.

## Timing and interpretation

The primary measurement is a cache-cold surrogate. A separate random generator
initializes a persistent 256 MiB int32 eviction buffer. Before every cold sample,
`add_(1)` reads and writes that buffer on the same CUDA stream, followed by a
begin event, one graph replay, an end event, and synchronization. Eviction is
excluded from the event interval but included in worker wall time. Random data
avoids relying on compressible zero fills. This does not guarantee physical L2
state or reproduce all effects of a full model.

Each shape has five balanced, shuffled cold pairs. Schema 2 stores these in
`rows[].cold_cache`; they are the primary comparison. Existing row-level
`pairs`, `median_us`, and `median_paired_speedup` are secondary hot-cache data.
The hot pass rewarms both arms and uses a common replay count targeting 2 ms,
capped at 32 replays. Raw orders and per-pair latencies are retained for both
passes. A row is complete only when both passes finish.

Imports, preparation, correctness checks, warmup, capture, eviction, and timing
all count toward the worker deadline. The result records persistent workspace,
route scratch, total prepared-state storage, and peak allocation. One-layer
memory use does not justify allocating a separate large workspace for every
model layer; a serving integration must budget or safely share scratch.

## Script provenance

These public scripts add full license headers, generic module descriptions,
and repository formatting to the frozen scripts used for the recorded
experiment. Their file hashes therefore differ from the measured originals;
the workload, numerical checks, measurement algorithm, and deadlines are
unchanged. Before formatting, their Python ASTs matched the originals after
removing module docstrings. The frozen original SHA256 values are:

| Script | Frozen original SHA256 |
| --- | --- |
| `benchmark_w4a16.py` | `f2813a0b383535a483914d98bb386b048d8aed472f929bd9d3d5d79804f14407` |
| `validate_w4a16.py` | `5a64691663241ddfc8a51570845553949d548ae9c33da0c6ad5042300e7a8e6a` |
| `freeze_native_w4a16.py` | `1b25b3dd48b957af042d81d5817ba0246cf6e2f2ccc55fa2d51a797aa09107ec` |

Generate fresh validation evidence when running the public scripts: the
validator's own hash is part of the benchmark's source contract. Do not relabel
an old result as a run of the formatted scripts.
