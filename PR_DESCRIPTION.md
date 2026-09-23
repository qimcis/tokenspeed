# Add H20 Block32 dense GEMM for V4.1 Flash decode

## Summary

DeepSeek V4.1 Flash uses FP8 weights with a scale for every 32 reduction elements. For the small dense GEMMs in TP8 decode, the portable path leaves substantial latency on H20. This change adds an SM90 tensor-core kernel that applies each Block32 scale before FP32 accumulation and connects it to the production GEMM entry point. The checkpoint format and activation quantization stay the same.

Dispatch is limited to measured `(M, N, K)` shapes with `M ∈ {1, 2, 4, 8}` on NVIDIA H20 and H20-3e. Other devices, shapes, scale layouts, and output types retain the existing kernel. The kernel supports split-K scratch and CUDA graph replay with current input and scale values.

## End-to-end results

AIPerf 0.12.0 replayed identical fixed-length completion requests against the compatible base runtime and this branch on eight H20-3e GPUs, TP8/EP8, InstantTensor, BF16 activations, Marlin MoE, DSpark off, and decode CUDA graphs. Every arm completed 112 measured requests and 14 warmups with matching actual input and output token counts.

| Input/output tokens, concurrency | Mean TPOT, ms base → Block32 | Mean TTFT, ms base → Block32 | Mean request latency, s base → Block32 | Output throughput, tokens/s base → Block32 |
| --- | ---: | ---: | ---: | ---: |
| 1,024/512, C1 | 23.82 → 17.30 (−27.4%) | 278.6 → 267.6 (−3.9%) | 12.45 → 9.11 (−26.9%) | 41.11 → 56.20 (+36.7%) |
| 16,384/512, C1 | 24.46 → 17.93 (−26.7%) | 4,073.8 → 4,188.9 (+2.8%) | 16.58 → 13.35 (−19.5%) | 30.88 → 38.34 (+24.2%) |
| 8,192/128, C4 | 47.30 → 44.10 (−6.8%) | 4,854.2 → 5,030.3 (+3.6%) | 10.86 → 10.63 (−2.1%) | 47.12 → 48.14 (+2.2%) |
| 8,192/512, C8 | 42.09 → 38.45 (−8.7%) | 8,729.6 → 9,046.3 (+3.6%) | 30.24 → 28.69 (−5.1%) | 135.40 → 142.70 (+5.4%) |

TPOT is mean `(request latency − TTFT)/(output tokens − 1)` per request. The improvement is strongest in single-request decode. At C4/C8, throughput gains are smaller and TTFT regresses by about 3.6%. This is one ordered A/B run on synthetic prompts, not a statistically repeated capacity or prefill result. Both arms used the same loader overlay and native libraries because the newer runtime could not boot with the node's installed scheduler extension.

## Test Plan

| Check | Result |
| --- | --- |
| Block32 GPU correctness and graph tests on H20-3e | 42 passed |
| Dispatch-policy tests | 9 passed, including H20-3e selection |
| Production-shape wrapper dispatch | Eight M1/M8 geometries selected Block32; relative L2 error 0.00023–0.00065 versus portable |
| Full-model AIPerf | Four paired workloads completed; exact token lengths and request IDs validated |

The device-name guard for H20-3e is included in this branch; it was also present in the measured candidate. No per-kernel serving profiler trace or model-quality evaluation is claimed.
