# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

## Experimental immutable Engram cache

`engram_cached_embedding` gathers global IDs through sorted rank-local hot-row
IDs. Each program handles one row, searches HBM metadata, and loads either cached
codes/scales or the canonical registered host table. The miss loads also carry an
explicit miss mask, so hit rows do not speculatively fetch host bytes. Out-of-shard
IDs write zeros; repeated IDs and arbitrary contiguous index shapes are supported.
The E4M3 decode, E8M0 scale multiply in FP32 and final BF16 conversion follow the
existing MXFP8 numerical contract, including E8M0 exponent 0 and NaN exponent 255.

The caller provides contiguous byte/E4M3 tables, uint8 scales, sorted int64 local
IDs and preallocated BF16 output. Row width is a static model dimension divisible
by 32, normally 256. Row counts, hot-set size and shard bounds remain runtime
scalars. The launcher allocates no output or metadata and does not synchronize.
The caller owns host registration, pointer lifetimes and refresh ordering; an
all-minus-one hot-ID array disables hits without replacing captured storage.
All readers must finish before canonical or hot bytes change. Refresh copies
codes/scales before restoring IDs. The operation performs no collectives.

The portable Triton implementation reuses the established host-gather UVA pointer
contract and MXFP8 conversion path. This is a narrower initial candidate than
introducing a second host-memory bridge in CuteDSL; a native NVIDIA alternative
requires measured evidence before replacing it. The KDA draft and task contract
are in `docs/draft.md` and the performance plan. KernelWiki's memory-bound and
vectorized-load notes informed contiguous row access; no bandwidth improvement
has been measured. The default embedding implementation remains unchanged.

CPU API checks: `PYTHONPATH=python pytest -q test/performance/test_engram_kernel_contract.py`.
GPU qualification (explicitly unrun during CPU-only development):
`TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS=1 pytest -q tokenspeed-kernel/test/nvidia/test_engram_cached_embedding.py`.
This tests all E4M3/E8M0 byte combinations, shard masks, empty/partial/full hot sets,
replay with changing indices, stream ordering, invalidation and scalar JIT reuse.
NaNs compare by classification; every non-NaN BF16 bit, including signed zero and
subnormals, must match the PyTorch reference. Compilation, device visibility and
performance remain unverified until those GPU gates and paired benchmarks run.
