# Dense GEMM backends

## Hopper per-32 FP8 scales

`hopper_block32.py` implements SM90 FP8 E4M3 matrix multiplication with one
scale per row and 32 reduction elements. Each 32-element dot is scaled before
FP32 accumulation. This is distinct from a 128-element scale group and from
converting the weights to BF16. Activation quantization and checkpoint weight
bytes are unchanged. FP8 values are converted exactly to FP16 inside each
32-element tile before tensor-core multiplication with FP32 accumulation. This
avoids the reduced accumulation precision of native FP8 WGMMA; it does not
materialize a converted weight matrix or change the quantization scales.

Explicit launch tactics control output tiling, swapped MMA operands, grouping,
pipeline depth, and split-K. Split-K writes FP32 partials to caller-owned scratch
and reduces them before the final output cast; it changes floating-point
summation order. The direct API accepts independent uint8 E8M0 or FP32 scales
and BF16, FP16, or FP32 output. It handles strided tensors and tails, and rejects
unsupported architectures, overlapping output storage, and incompatible scratch.

The production portable block-scaled GEMM entry point consults
`_hopper_block32_policy.py` for validated exact H20 shapes with `[1, 32]` blocks,
uint8 scales, and BF16 output. Other shapes, formats, and devices retain the
existing backend. Selection is an offline policy: there is no runtime autotuning
or tensor cache. Both regular dispatch and the PDL-disabled prepared-linear path
reach this entry point.

The caller allocates split scratch for each invocation. CUDA graph capture records
the same execution path and retains its allocations; graph replay reads current
activation, weight, and scale values. Scratch must not be reused concurrently.

This Triton implementation is a transitional SM90 backend. The existing optional
CuTe split-K adapter targets SM100 BF16 and cannot implement this FP8 per-32-scale
contract without a different main loop. Architecture guards must not be relaxed
to substitute incompatible scale layouts or arithmetic.

## H20 selection evidence

The selected decode tactics use logical tiles M16/N64, eight contiguous K
partitions, swapped MMA operands, four warps, and three stages. Selection is
limited to M in `{1, 2, 4, 8}` and these exact `(N, K)` pairs:

| N | K | Paired wrapper graph speedup across selected M |
| --- | --- | --- |
| 1792 | 5120 | 3.22–3.29x |
| 4096 | 1280 | 1.91–3.63x |
| 5120 | 1024 | 1.85–4.13x |
| 576 | 5120 | 4.45–4.74x |

These are repeated-input GEMM measurements, not whole-model throughput. Each
selected shape completed five independent input seeds, three alternating timing
pairs, and leaf, wrapper, and quantizer-plus-wrapper measurements. Each timed
CUDA graph contains 16 complete calls. Both arms use supplied output buffers;
wrapper capture includes scratch allocation. The quantizer-plus-wrapper graph
also improves for every selected shape. Eager host-plus-device latency often
regresses, so the graph measurements must not be presented as eager speedups.

The final sweep stopped at its time limit after 22 of 36 shapes completed.
Only 16 fully checked shapes qualified. Tested large-M tactics regressed;
unfinished shapes, including the K288 and Engram pairs, retain the portable
kernel. The implementation supports them, but the policy does not extrapolate
measurements. Production-size correctness uses an independent FP64 oracle on
sampled rows/columns plus full-output finiteness; small-matrix tests compare
complete outputs and exercise tails, scales, strides, split partitions, and
CUDA graph replay.


## Full-model H20 validation

A matched DeepSeek V4.1 Flash comparison on eight H20 GPUs used TP8/EP8,
InstantTensor loading, BF16 activations, Marlin MoE, GPU-resident Engram,
DSpark disabled, and decode CUDA graphs at batch sizes 1, 2, 4, and 8.
The baseline includes the preceding Hopper attention improvements. Both arms
used the same 1024-token prefill chunks, disabled prefix caching, and identical
CPU-precompiled portable indexer variants. Each arm completed a first-use and
warm seven-cell AIPerf matrix on one unchanged server process.

Warm single-request decode latency fell from 24.0–24.5 to 17.5–18.0 ms/token
across 1K/16K inputs and 128–1024 outputs, a 25–27% reduction. For 1K inputs,
measured output throughput increased 33–36%. The 8K-input/512-output,
concurrency-8 case improved output throughput by 5.6%, while median TTFT rose
29%. The 8K-input/128-output, concurrency-4 case had essentially unchanged
throughput (-0.6%) and 22% higher request-level TPOT. Overlapping prefill work
contributes to that TPOT, so it is not a pure decode-kernel measurement.

These are bounded observations: 20 measured requests plus one warmup per
matrix, fixed baseline-then-candidate arm order, and no statistical significance
or steady-state capacity claim. Prefill TTFT changes are mixed. An initial
baseline attempt timed out during cold indexer compilation; its partial results
are excluded from the matched comparison. The replacement comparison used fresh
servers and the same indexer cache preparation for both arms.

All 51 unit cases passed. A separate diagnostic invocation compared 741,376
full-output elements from the four selected M1 geometries on all eight ranks;
maximum normalized L2 error against the unchanged kernel was 0.000897
(limit 0.006). Clean candidate traces confirmed the selected dot and reduction
kernels on all eight ranks. These checks establish scoped GEMM correctness and
production dispatch, not model-quality equivalence.
