# DSpark compact candidate reduction

`dspark_block_compact_candidates(partials, output)` reduces contiguous
`[rows, n_tiles]` signed int64 tile candidates into a disjoint, caller-owned
`[rows, 1]` buffer on the same GPU. There is one reduction CTA per request.
Vocabulary-derived tile counts are static; request counts are runtime values.
Empty batches launch nothing. The operation allocates no device storage and
performs no host synchronization, so its output can be allocated before capture.

The existing greedy scorer accepts gathered `[tp, rows, n_tiles]` baseline
candidates or `[tp, rows, 1]` compact candidates. It always emits the same full
local tile buffer. Runtime communication must exchange one packed value per
request and rank in compact mode; the final resolver works with either layout.
Input candidates and output tile storage must remain disjoint across steps.

Signed integer maximum is associative, so local reduction followed by rank
reduction preserves the upstream packed winner exactly. Equal score bits favor
the smallest global token ID. Upstream packing distinguishes positive and
negative zero and orders NaN sign/payload bits; the compact path deliberately
preserves that behavior rather than claiming ordinary `torch.argmax` semantics
for those values. Masked tiles use INT64_MIN, which remains below valid negative
infinity. An all-sentinel row resolves to -1. Scoring still uses BF16 Markov
weights and FP32 logits and accumulation.

This portable integer reduction uses the repository Triton wrapper alongside
the existing Triton DSpark scorer. It does not need NVIDIA-specific matrix,
cluster or asynchronous-copy features. A CuTe DSL alternative requires measured
whole-path evidence before adding another implementation. One extra kernel
launch can outweigh reduced communication at small shapes.

CPU reference checks:

```sh
PYTHONPATH=python pytest -q test/performance/test_dspark_candidate_reference.py
```

Opt-in NVIDIA qualification (not run during CPU implementation):

```sh
TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS=1 pytest -q tokenspeed-kernel/test/nvidia/ops/test_dspark_compact_candidates.py
```

The GPU suite covers exact signed-key reductions, special values, vocabulary
tails, multi-step baseline parity, graph replay, storage overlap rejection and
request-count JIT stability. Its rank tests simulate candidate exchange on one
GPU; distributed collective qualification and end-to-end performance remain
separate requirements. CPU algebra tests do not prove GPU compilation or speed.
