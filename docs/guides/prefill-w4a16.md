# Hopper W4A16 prefill

`--moe-backend cutlass_w4a16` selects a FlashInfer/CUTLASS MXFP4 expert
backend for a dedicated DeepSeek V4.1 Flash prefill worker on SM90. It is
explicit-only; use `--moe-backend marlin` on the paired decode worker.

| Worker | Required options |
| --- | --- |
| Prefill | `--disaggregation-mode prefill --moe-backend cutlass_w4a16` |
| Decode | `--disaggregation-mode decode --moe-backend marlin` |

The supported layout is BF16, attention TP8, MoE TP1/EP8, DP1/PP1/CP1/DCP1,
fixed contiguous experts, and `--all2all-backend none`. Set
`0 < --chunked-prefill-size <= --max-prefill-tokens`; the latter sizes one
workspace shared by all serialized MoE layers. The backend rejects other
model geometry, mixed batches, speculative workers, EPLB, redundant experts,
memory saver, and online weight updates.

Weights must come from a canonical checkpoint (`auto`, `safetensors`,
`instanttensor`, or `pt`). Each layer converts its packed MXFP4 weights and
scales once during loading, retains one layout, and uses the same backend for
every prefill chunk, including one-token tails. FlashInfer must provide the
SM90 mixed-input interleaving and caller-owned workspace APIs used by this
backend (validated with FlashInfer 0.6.18).
For checkpoints with excluded Engram tensors, InstantTensor loads fully accepted
shards on GPU and filters mixed shards on CPU before materializing tensors.

Targeted H20 checks passed for numerical parity, graph replay, backend
selection, loader setup, and immutable-weight guards. A TP8/EP8 model reached
engine readiness, but a standalone prefill worker could not complete HTTP
warmup without a paired decode worker. End-to-end serving performance remains
unmeasured.
