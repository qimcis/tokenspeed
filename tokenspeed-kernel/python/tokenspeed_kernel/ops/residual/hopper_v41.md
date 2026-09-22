# Experimental Hopper V4.1 HC4 epilogue

`hopper_v41.py` fuses HC post mixing, collapse using the previous sublayer’s
pre coefficients, RMSNorm, and optional group32 FP8 quantization. The API
requires SM90, contiguous BF16 inputs and FP32 HC4 coefficients. It preserves
BF16 rounding after post mixing, collapse and normalization; FP8 codes and
row-major UE8M0 scales match quantization of the normalized BF16 output.

The H20 model integration is **disabled by default**. Set
`TOKENSPEED_EXPERIMENTAL_V41_HOPPER_EPILOGUE=1` before model construction only
to reproduce the experimental attention-to-FFN boundary. This flag does not
enable the separate collective adapter. Eager and captured forwards share
the same implementation and read the flag once during model construction.
Router and routed-expert inputs remain BF16; only the shared gate/up
projection consumes the optional prequantized codes. Unsupported geometry,
prefill and CED row filtering retain the existing path.

## Measurements

On H20, the standalone candidate passed 20 correctness tests, including
mutated-input graph replay and exact FP8 bytes/scales against the unchanged
quantizer. It nevertheless lost every measured complete-chain comparison.
The baseline is the existing post + collapse/RMSNorm + optional quantizer,
not one isolated kernel. H=5120, FP32 norm weights, three seeds, four
alternating pairs per seed, 16 complete calls per graph and four replays.
These are hot-input graph microbenchmarks, not serving throughput.

| Tokens | Quantize | Baseline median (µs) | Candidate median (µs) | Baseline/candidate |
| --- | --- | ---: | ---: | ---: |
| 1 | False | 3.543 | 4.002 | 0.885× |
| 1 | True | 4.702 | 8.969 | 0.524× |
| 2 | False | 3.591 | 4.043 | 0.888× |
| 2 | True | 4.897 | 9.000 | 0.544× |
| 4 | False | 3.684 | 4.054 | 0.909× |
| 4 | True | 5.160 | 9.002 | 0.573× |
| 8 | False | 3.841 | 4.100 | 0.937× |
| 8 | True | 5.549 | 9.046 | 0.613× |

The quantized H=5120 kernel uses 255 registers per thread with eight warps,
and one CTA per token. Removing launches did not compensate for this
geometry. No model-level improvement or >10% gain has been established.
The original path remains the default.

Run GPU correctness with `pytest tokenspeed-kernel/test/nvidia/ops/residual/test_hopper_v41.py`.
The CPU integration tests are in `test/runtime/test_deepseek_v41_hopper_epilogue.py`.

Raw pairs are retained in `benchmarks/residual/hopper_v41_results.json`;
`benchmarks/residual/bench_hopper_v41.py --output FRESH.json` reproduces the
local complete-chain comparison on an otherwise idle H20.
The microbenchmark records whole-tensor relative errors; the separate unit
tests enforce the per-row error gate.
