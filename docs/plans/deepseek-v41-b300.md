# DeepSeek V4.1 Flash on B300

## Contract

Build a specialized TokenSpeed engine for `deepseek-ai/DeepSeek-V4.1-Flash-1017`
with encoder/prefill/decode disaggregation. Preserve the upstream scheduler,
execution path, cache ownership, kernel registry and transport abstractions.
The implementation branch is `deepseek-b300`, based on `qimcis/tokenspeed`
main at `4fd85df66b4e3c127a587943c783dbc9d941c87c`. The checkpoint revision is
`1bae0e1088993b9b82de08f59de5be5a34e42170`.

Validation is CPU-only until the user supplies hardware. An implementation or a
CPU test is not evidence of B300 performance or model quality. Keep measured
results separate from analytical estimates and proposed configurations.

The [operator guide](../guides/deepseek-v41-b300.md) describes the implemented
recipe and evidence tooling. The companion
[work log](deepseek-v41-b300-work-log.md) records changes,
verification, decisions and unresolved work. It is the continuation memory.

## Objective and acceptance

Minimize total serving cost divided by successful agent tasks, subject to quality
and service constraints. Include encode, prefill, decode, host, network, storage,
idle allocations, retries and failed attempts. Count rejected tasks in the
attempted-task population. Report committed output throughput, successful tasks
per GPU, TTFT, inter-token latency, task latency, rejection rate and per-rank
memory headroom. Never count draft tokens as completed output.

Quality is a paired non-inferiority test against the pinned reference. Declare a
margin before inspecting results; the lower confidence bound on the success-rate
difference must exceed its negative. Overlapping confidence intervals are not a
parity test. Cluster resampling by conversation/task, retain failed attempts, and
keep optimization traces separate from the held-out acceptance set.

## Corrected source facts

- Upstream already has V4.1 CED, FlatKV, bounded replay, Engram, DSpark, PD and a
  generic encoder service. Extend these rather than replacing them.
- The 512-dimensional attention head includes its 64-dimensional RoPE portion.
- Engram's FP8 codes and scales occupy about 202.8 GB per complete table set.
  Placement can replicate that set; EP does not determine attention replication.
- Raw FP4 routed expert matrices occupy about 271.8 GB before scales and padding.
- Persistent logical global KV plus index state is 890 bytes per input token:
  `(3/2 + 1) * (288 + 68)`. This excludes SWA, compressor tails, speculative
  context, packing, replication and temporary workspaces.
- SWA is live request state. It is omitted from reusable prefix storage and
  rebuilt through bounded replay. PD transfers its retained tail.
- Selected attention is bounded, but index scans and communication may grow with
  context. Neither bandwidth nor memory capacity is assumed to be the bottleneck.
- Current V4.1 PD decode does not support KVP. Current tree speculation does not
  support this model's Engram/SWA/PD combination. Preserve these guards.
- The model's compression ratios vary across layer groups.

Source of truth: checkpoint config and inference code, TokenSpeed's V4.1 model,
cache geometry/recipe, Engram and DSpark implementations, and the existing
context-parallelism and tree-speculation design documents.

## Architecture

E is the multimodal vision encoder and aligner, not the language model's internal
causal encoder. Text bypasses E. E sends image embeddings and span metadata to P.
P executes prompt processing, CED and the required decoder suffix. P sends global
cache and retained request state to D. D executes the autoregressive target and
optional DSpark verification. The gateway presents the existing API.

Do not physically split the two CED layer groups initially: generated tokens still
traverse the encoder portion, so that split adds a per-step activation boundary.
The upstream E transport requires prefill TP to be a multiple of encode TP.
Independent encode servers supply horizontal scaling.

## Workstream A: reproducible serving foundation

Pin model, tokenizer, processor, conversation encoding, source, dependency closure
and container identity. Retain an unchanged upstream baseline. Use strict typed
TOML recipes and reject unknown or incompatible settings before launch. Separate
aggregated, PD and EPD recipes while using the same runtime implementations.
Keep model execution settings explicit and record the effective launch manifest.

Per role and rank, account for weights, scales, replication, live KV, bounded
state, graph storage, speculative scratch, transport buffers and load peaks.
Use measured usable memory rather than nominal hardware capacity. A static
estimate is a lower bound until allocator telemetry establishes peak use.

Gate: valid recipes, reproducible provenance, truthful memory ledger and no
unsupported configuration silently enabled.

## Workstream B: EPD handoff correctness

E-to-P preserves embeddings, image ordering, grid/span metadata, dtype, shape,
processor/model identity and content identity. Image masks must still control
Engram participation and vision-specific routing. Embedding cache keys must
include preprocessing and model identity.

P-to-D includes global KV/index groups and scales, SWA/compressor tails, DSpark
windows and required anchor/proposal state, committed frontier, token history,
positions, sampling state and multimodal identity. Capacity is reserved before
transfer; completion must establish GPU-visible data before execution. Cover
cancellation, duplicates, worker failure, partial-transfer cleanup and ownership.

Keep bounded replay's shared global rows immutable. P reconstructs replayable
state and sends retained tails; D must not reinterpret the transfer as a prefix
hit and replay it again. Begin with completed-prefill transfer, not layerwise
streaming.

Gate: aggregated, PD and EPD pass matched text/image, prefix-hit, odd-chunk,
concurrency, cancellation, recovery and speculative-rollback checks.

## Workstream C: topology and capacity

Baseline families: aggregated eight GPUs; P4/D4; dedicated E2/P2/D4 as a candidate
within eight GPUs. E2/P2/D4 is not a predicted optimum. Shared E pools and
co-resident E are later candidates requiring interference accounting. Never
compare eight aggregated GPUs with eight decode GPUs plus uncounted P/E resources.

Sweep EP and attention TP/DP independently using only supported combinations.
Keep decode-side KVP under PD disabled until separately implemented and verified.
Budget all copies of host and GPU Engram, including TP1 replication. Do not
transfer GB300 host-memory performance claims to B300.

Gate: each rank fits at intended concurrency, including startup and overlap;
compare whole-system cost and tails under equal offered work.

## Workstream D: measurement and calibrated replay

Capture per-stage queues, execution, transfers, memory high-water marks and
committed work. Sample per-layer expert routes, unique experts, token occupancy,
rank skew, padding and dispatch/combine costs. Collect prefix reuse, Engram row
reuse and speculation acceptance without full synchronous per-token logging.

Use a held-out calibrated model of execution and queues. Include batching,
transfers, role capacities, prefix reuse and speculation. Publish residuals and
unsupported workload regimes. Never use a simulator alone as a release gate.
Interleave paired experiments; kernel lower-tail results are diagnostic only.
Require a mechanism hypothesis and evidence above noise before expensive runs.

Gate: accurate held-out predictions within a declared tolerance and an evaluation
report that includes failures, confidence bounds and full resource cost.

## Workstream E: batching and expert placement

Uniform independent routing predicts
`384 * (1 - (1 - 6/384)**B)` distinct experts for B token rows, approximately 244
at B=64. This is an analytical reference, not measured production behavior.

First compute a future-route oracle upper bound on batch-composition savings.
Then evaluate cheap signatures based only on information available at scheduling.
Minimize execution cost with queueing, rank-skew and fairness penalties. Bound
admission delay. Pair composition with exact expert placement/replication through
existing EPLB. Keep router outputs unchanged.

Gate: lower cost or meaningful service improvement without starvation; drop the
unsubstantiated mandatory 1.5x target. Skip policy implementation if oracle
headroom or achievable prediction is inadequate.

## Workstream F: Engram residency

Compare existing GPU, shared-host and sharded-host paths first. Add a hot-row
cache only when measured reuse and exposed gather cost justify it. Preserve
checkpoint bytes and scales, fetch misses exactly, deduplicate gathers and overlap
only independent work. Hash uniformity does not imply access-frequency uniformity.
Measure reuse distance and hit-rate/capacity curves. Free HBM is useful only if it
improves whole-system cost or capacity enough to offset traffic and tail stalls.

Gate: exact lookup semantics and improvement under realistic contention.

## Workstream G: DSpark

Baseline no speculation and upstream checkpoint-native DSpark. Profile the complete
draft/verify/sampling/communication/state-maintenance dependency chain. The
upstream BF16 Markov projection is vocabulary-sharded; projection replication
alone does not solve base-logit ownership. Do not assume an INT8 conversion is
exact or that a particular Hopper kernel implies the drafter must be rewritten.

Optimize committed tokens per complete round cost, not acceptance alone. Require
greedy-equivalence where applicable, numerical tests, and distribution-preserving
stochastic verification. Defer trees until branch-local Engram, SWA, compression,
selection and rollback contracts are implemented. Bound automated search to
correct supported configurations.

Gate: full-round and task-level benefit with quality and service constraints.

## Workstream H: cache economics

Compare remote lookup, physical transfer, restore/replay and queue effects against
local recompute for each topology and workload. No universal prefix-length
threshold follows from halving one compute term. Start with local prefix reuse
and correct PD transfer. Namespace reuse by tenant and model/format identity.

Gate: measured workload-level benefit, including network contention.

## Workstream I: separate model co-design

Router-bias changes and router/expert co-training alter the model. Keep them out
of engine-only acceptance. Any such work gets separately versioned weights,
independent held-out evaluation, explicit quality budgets and rollback.
Architecture feedback can use traces without waiting for model modification.

## Order and completion criteria

Pinned foundation -> correct aggregated baseline -> correct PD/EPD -> measured
capacity/bottlenecks -> justified optimizations -> production-shaped acceptance.
CPU work may implement contracts, tooling and tests but cannot close hardware or
quality gates. Optional experiments remain deferred until their mechanism gates
are met; unsupported research paths must never be advertised as complete.

Release checks cover images and text, tool/reasoning parsing, long-context admission,
prefix replay, speculative rollback, overload, cancellation, recovery and full
resource cost. Qualify a served context limit separately from the architectural
maximum. Every change and result belongs in the companion work log.
