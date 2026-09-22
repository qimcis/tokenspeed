# Hopper V4.1 allreduce and HC4 epilogue

`hopper_mhc.py` is an explicit, prepared TP8 adapter. It is not registered as a
general allreduce replacement. It accepts contiguous BF16 `[M,5120]` inputs for
`M=1,2,4,8` on SM90, with one CTA per row and eight warps per CTA. Both eager and
captured execution use the same kernel and persistent workspace.

Call `prepare_v41_allreduce(group, device, max_rows, round_mode, timeout_ns,
collective_timeout_s)` on every rank before capture. Preparation uses existing
TokenSpeed symmetric-memory allocation helpers, verifies a common single-host
TP8 configuration, and allocates dedicated peer staging and signal buffers.
Ranks must share the same visible-device ordinal namespace and have their
current CUDA devices set correctly. Preparation must run under an external
startup watchdog: the final barrier has an explicit timeout, while metadata
agreement and rendezvous also depend on the process-group timeout. A failed
preparation must abort the owned distributed attempt on every rank.

The workspace belongs to one forward executor. All ranks must issue the same
row geometry and collective order. Captures, eager calls and graph replays must
be serialized; changing CUDA streams requires the executor's existing event
ordering. Python cannot inspect a later graph replay's stream or overlapping
use. There is no lazy preparation, implicit fallback, stream synchronization,
or capture-specific state transition in forward.

Each row publishes its local BF16 values to dedicated staging, then exchanges
system-release/acquire binary signals with every peer. The publication and
read-completion phases have distinct signal storage. A signal is consumed with
CAS `1→0` before that sender can publish its next `0→1` signal. The read-completion
phase prevents the next invocation from overwriting values still being read by
a slower peer. Per-thread system fences and explicit Triton CTA barriers
surround the leader's signal exchanges. Device polling has an explicit
`%globaltimer` deadline; a missing peer traps the owned CUDA context instead of
returning a partial reduction. Such failures require context/process cleanup.

`bf16_step` adds rank 0 through rank 7 with a BF16 rounding after every addition.
`fp32` uses the same rank order with FP32 accumulation and one final BF16 cast.
Both expose the BF16-rounded result as FP32 to the HC4 epilogue. The first mode
matches the arithmetic structure of the existing TRT-LLM IPC non-FP32 sum, but
neither mode claims bitwise parity with every NCCL/MNNVL topology. The mode is
explicit and agreed during preparation.

`v41_allreduce(workspace,x)` isolates the collective for validation.
`v41_allreduce_post_pre_norm_quant(workspace,x,residual,post,comb,pre,norm_weight,
eps,quantize)` returns the same four outputs as the local V4.1 epilogue. It calls
the shared `_v41_hc_epilogue` after peer reduction, preserving that helper's HC
coefficient order, intermediate BF16 rounding, normalized BF16 output and
row-major group32 UE8M0 quantization. Only workspace scratch is modified.

CPU protocol tests exercise rank skew, repeated changing geometries, missing
peers, host shape checks and the distinction between reduction modes. They do
not prove GPU memory ordering or code generation. Before runtime enablement,
inspect the SM90 PTX for full-CTA fences/barriers and memory clobbers, then run
bounded eight-rank numerical, mutable-input graph replay, rank-skew and timeout
tests under an external watchdog. Validate the complete epilogue's numerical
tolerance before any timing or production dispatch claim.

## Validation status

The SM90 implementation passed CPU-only compilation of 80 rank, arithmetic,
weight-dtype and quantization specializations. All compiled variants retain
unconditional system fences, CTA barriers outside the leader handshake and
compiler memory clobbers. None spill registers to local memory.

A bounded eight-H20 test passed 112 checks per rank. It covers both reduction
modes; repeated M=1/2/4/8 calls; BF16 and FP32 norm weights; strict per-row
relative error <=0.006; exact codes/scales against quantization of the actual
normalized BF16 output; serial M=1/M=8 graph replay with mutated inputs; and
one delayed rank. Maximum observed relative error was 0.000015019. This is
not exhaustive concurrency validation. The deliberate missing-peer timeout
trap has not been exercised on GPU.

The standalone distributed worker is `test/nvidia/ops/stress_hopper_mhc.py`.
Run it with `torchrun --standalone --nproc-per-node=8` and explicit fresh
`--output-dir`, `--seconds 25`, `--source-sha`, and `--epilogue-sha` arguments.
The hashes must match the installed collective and residual source files.
Use an external process-group watchdog and verify owned GPU cleanup afterward.
The benchmark in `benchmarks/communication/bench_hopper_mhc.py` compares the
complete chain with the actual selected TensorRT-LLM workspace and records
all eight ranks. Neither script enables this adapter in the model.

## Performance result: not promoted

The complete TP8 chain regressed against the actual selected MNNVL one-shot
TensorRT-LLM baseline in every measured case. Both arms include allreduce, HC4
post/collapse, RMSNorm and optional exact group32 quantization. H=5120, FP32
norm weights, eight complete calls per graph, two replays per measurement,
three alternating pairs. The table reports the median of each pair’s slowest
rank. Both eager and graph outputs passed the fixed numerical gates before
timing. These measurements describe GPU kernel chains, not serving throughput.

| Tokens | Quantize | Baseline (µs) | Candidate (µs) | Baseline/candidate |
| --- | --- | ---: | ---: | ---: |
| 1 | False | 8.340 | 41.322 | 0.202× |
| 1 | True | 9.702 | 49.602 | 0.196× |
| 2 | False | 9.458 | 46.294 | 0.204× |
| 2 | True | 12.606 | 59.322 | 0.213× |
| 4 | False | 8.988 | 51.688 | 0.174× |
| 4 | True | 10.862 | 69.612 | 0.156× |
| 8 | False | 10.270 | 64.686 | 0.159× |
| 8 | True | 13.562 | 94.266 | 0.144× |

Raw per-rank pairs and numerical checks are in
`benchmarks/communication/hopper_mhc_results.json`. An earlier benchmark
attempt timed out during startup before any timing cases; it is not included
as a performance sample. The successful rerun kept all measurement cases and
tolerances unchanged and allowed longer startup.

The adapter remains unregistered and is not connected to model dispatch.
No full-model speedup has been established. The local epilogue’s separate
experimental flag does not enable this collective. The existing MNNVL
implementation remains active by default.

The source and instruction audit suggests the two all-peer CAS exchanges
and serialized peer data reads are expensive compared with the native MNNVL
path. This is a diagnosis to investigate, not a measured attribution of each
component. Any future fusion should preserve the efficient collective rather
than assuming fewer launches will outweigh its replacement cost.
