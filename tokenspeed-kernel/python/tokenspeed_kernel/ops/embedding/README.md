# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

## Cached Engram gather

`engram_cached_embedding` looks up global IDs in sorted rank-local hot rows,
loads FP8 codes and E8M0 scales from HBM on a hit or registered host memory on a
miss, and writes caller-owned BF16 output. Out-of-shard IDs produce zeros.
The portable Triton kernel uses explicit miss masks; it allocates no storage,
performs no collective and does not synchronize the host. Row counts and shard
bounds are runtime scalars; row width is static and divisible by 32.

Callers own host registration, stable buffer lifetimes and drained refresh. An
all-minus-one hot-ID array disables hits without replacing captured storage.
Refresh codes/scales before restoring IDs. The conversion follows the existing
MXFP8 contract, including exponent zero and NaNs; GPU numerical and performance
qualification remains outstanding.
