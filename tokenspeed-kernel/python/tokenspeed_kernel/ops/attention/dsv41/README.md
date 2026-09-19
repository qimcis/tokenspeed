# DeepSeek V4.1 selected attention

All implementations share page-planar packed caches: SWA rows use 512 FP8 E4M3
value bytes and 16 E8M0 scales; global rows use 256 packed E2M1 value bytes and
32 E4M3 scales. Each page stores all 64 value rows before all 64 scale rows.
Decoded values round to BF16 before attention. There is one zero-valued sink
per head across both cache segments. Slots, lengths, and cache contents remain
live during CUDA-graph replay.

Query-head padding belongs to the receiving backend. Native Blackwell FlashMLA
keeps its 64/128-head preparation. Portable attention and Hopper's direct reader
use the actual local heads. TP8 has eight local heads.

## Hopper TP8 dispatch

The SM90 implementation currently selects optimized paths only for BF16 H8,
D512 queries. Other head counts retain the portable path.

| Input | Implementation |
|---|---|
| Existing compact BF16 prefill workspace | Native SM90 sparse FlashMLA |
| SWA-only packed cache, up to 64 queries | Direct tiled BF16 attention, K64/eight warps |
| SWA-only packed cache, more queries | Direct tiled BF16 attention, K32/four warps |
| Combined SWA/global cache, at least 16 queries | Bounded BF16 gather plus native sparse FlashMLA |
| Combined cache, fewer queries | Portable FP32 scalar attention |

Native sparse attention pads heads locally and aligns selected width to 128.
It retains invalid index holes, including those before the global segment;
the sum of active segment lengths is not a valid prefix length. Gathered
workspace allocation stays bounded by `query_chunk_size`. The native library
has no output-buffer argument, so the wrapper copies only actual heads into
the caller's output. If that optional API is unavailable, these native routes
fall back to portable attention.

The direct reader removes SWA gather, concatenation, index construction, and
padded output storage. It loads one scale per quantization group and broadcasts
the decoded scale. Arbitrary cache byte strides and padded page strides are
covered by tests. Its BF16 tensor-core QK/PV arithmetic changes reduction order
and rounds probabilities to BF16; outputs are numerically close, not promised
bitwise identical to scalar FP32. The absent-sink/leading-invalid-tile case
uses stable normalization and follows an independent softmax reference.

Five-seed paired CUDA-graph experiments on H20 measured the entire leaf path,
including native padding, gather, and output copies. They supported direct
SWA-only and native prefill routes. The direct combined reader regressed at
small batches, including a separate-loop refinement, so it is not selected.
Kernel timings do not establish whole-model throughput gains. Numerical tests
use `rtol=0.008, atol=0.004`, with tighter relative-L2 checks for the tiled path.
TF32x3 PV is an internal comparison variant; its K64 configuration is explicitly
excluded after exceeding the SM90 shared-memory limit.

These Triton readers reuse the existing portable codec semantics as a transition
within the kernel package. Vendor libraries remain behind `tokenspeed-kernel`;
no runtime dependency or cache representation change is required.
