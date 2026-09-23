# Experimental Hopper MXFP4 W4A16

`cutlass_mxfp4.py` provides an explicit prepared adapter for SM90. It is not
registered or automatically selected. Availability is not a performance claim;
compare the complete call with corrected, clamped Marlin before selecting it.
The optional FlashInfer dependency must expose SM90 mixed-input weight/scale
interleaving, `cutlass_fused_moe_workspace_size`, and `workspace_buffer`.

`prepare_hopper_mxfp4_moe(w13, w2, w13_scales, w2_scales, top_k, ep_size,
ep_rank, max_tokens, swiglu_limit)` accepts contiguous uint8 packed E2M1 weights,
low nibble first: `[E,2I,H/2]` in **gate, up** order and `[E,H,I/2]`.
Raw uint8 E8M0 scales are `[E,2I,H/32]` and `[E,H,I/32]`. H and I must be
positive multiples of 128, and top-k must be at most eight. Preparation swaps
gate/up and their scales together,
interleaves the packed weights once, and allocates maximum routing/native scratch.
It retains no canonical or Marlin layout. Production callers must release the
canonical tensors after selecting this backend; retaining both expert layouts
has not been validated to fit alongside the full model.

`hopper_mxfp4_moe(state, x, topk_weights, topk_ids, out)` consumes contiguous
BF16 `[M,H]` activations and writes caller-owned BF16 output. Routes are global
int32 `[M,K]` IDs and FP32 weights already normalized over global top-k. Invalid
IDs contribute zero. Duplicate IDs are coalesced by summing their weights into
the first occurrence; zero-weight slots receive distinct unused IDs because
native routing requires unique selections. Native EP excludes nonlocal experts.
This returns a local EP partial without renormalizing or applying DeepSeek's
routed factor 1.5.
The model applies that factor once outside this operation. M=0 is supported;
M must not exceed the prepared capacity, and output must not alias inputs.

Clamping requires native **SwigluBias**, with alpha 1, beta 0 and the requested
limit: gate has an upper clamp and up has a symmetric clamp. The native plain
Swiglu adaptor ignores the limit. Default native tactics `[-1,-1]` avoid runtime
autotuning. Fused finalization can change summation order, so validate against
the independent numerical envelope rather than requiring bitwise equality.

Use one state per non-concurrent execution lane. Run the same path to warm it
before capture, then keep weights/workspace alive through graph replay. The
adapter reads live activations/routes every call and has no capture-only path.
The public vendor wrapper still creates small metadata tensors each call; its
large workspace is explicitly reused. No registry or model dispatch is changed.

## H20 evaluation

The adapter passed all 28 independent validation cases, including nine GPU
cases, then passed eager and graph parity with corrected Marlin at all six
measured token counts. The original duplicate-route failure is retained in
the benchmark record; the implementation now coalesces duplicate weights.

Cold-cache-surrogate median MoE-call latencies (microseconds), using H=5120,
I=2304, 48 local experts, EP8 and top-6:

| Tokens | Marlin | CUTLASS W4A16 | Median paired speedup |
| ---: | ---: | ---: | ---: |
| 1 | 67.872 | 61.440 | 1.103× |
| 2 | 72.960 | 79.456 | 0.915× |
| 4 | 81.824 | 96.288 | 0.852× |
| 8 | 119.104 | 149.472 | 0.799× |
| 128 | 1549.024 | 833.376 | 1.860× |
| 1024 | 2903.232 | 950.208 | 3.053× |

**Not promoted:** decode batches 2–8 regress beyond the predefined 2% limit.
These are synthetic single-GPU local-expert calls, not serving measurements.
No full-model AIPerf run was performed for this candidate. Gate/top-k selection,
communication, shared experts and the external routed scaling are excluded.

A prepared 1,024-token layer needs 252,358,528 bytes of native workspace.
Full-model integration would need safe scratch sharing and a single persistent
expert-weight layout. Keeping both Marlin and CUTLASS weights merely to switch
by token count is not a validated memory-feasible solution.

The existing Marlin path now forwards V4.1's SwiGLU clamp instead of silently
dropping it. Both benchmark arms use the corrected clamp semantics.

See [benchmark results](../../../../../benchmarks/moe/hopper_w4a16_results.md)
and the adjacent JSON for raw paired samples and retained validation history.

The two transformed formats are incompatible: the vendor interleaves packed
values and scales differently from Marlin. Forty layers of these local expert
weights and scales occupy about 33.62 GiB per GPU, so a second persistent
format would add about that much storage before scratch. A future hybrid
implementation needs measured small-M execution on one retained format;
per-forward repacking is not included in these results.
