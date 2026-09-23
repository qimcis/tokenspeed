# Add opt-in Hopper W4A16 MoE for V4.1 Flash prefill

Large prefill chunks spend substantial time in the Marlin MXFP4 expert path.
This change adds an explicit FlashInfer/CUTLASS W4A16 backend for dedicated
SM90 V4.1 Flash prefill workers. It converts each layer's canonical weights
once, shares one fixed-size workspace across MoE layers, and rejects
unsupported configurations and online weight updates. Decode remains on
Marlin; automatic backend selection is unchanged.
InstantTensor filters mixed shards before GPU materialization so excluded
Engram tensors do not exhaust device memory during checkpoint loading.

| Local expert tokens | Marlin median, µs | W4A16 median, µs | Paired speedup |
| ---: | ---: | ---: | ---: |
| 1 | 67.87 | 61.44 | 1.10× |
| 2 | 72.96 | 79.46 | 0.92× |
| 4 | 81.82 | 96.29 | 0.85× |
| 8 | 119.10 | 149.47 | 0.80× |
| 128 | 1,549.02 | 833.38 | 1.86× |
| 1,024 | 2,903.23 | 950.21 | 3.05× |

These are paired, single-GPU local-expert measurements; they exclude routing,
communication, shared experts, and scheduling. The 2–8-token regressions are
why this backend is opt-in and limited to prefill workers. They do not imply
an end-to-end serving speedup.

| Validation | Result |
| --- | --- |
| Targeted runtime, GPU, numerical, graph-replay, InstantTensor, and FlashInfer boundary tests on H20 | 116 passed; one network-dependent test deselected |
| Kernel registry tests on H20 | 42 passed |
| TP8/EP8 model startup with InstantTensor | Engine ready; shared workspace allocated for 40 MoE layers on each rank |
| Prefill worker without paired decode worker | PD bootstrap failed during HTTP warmup; no serving result |

The model startup used a task-local CUDA linker shim for FlashInfer JIT; the
change does not add that workaround. Full PD handoff, TTFT, throughput, and
quality still need a paired-worker A/B run.
