# DeepSeek V4.1 B300 performance implementation

## Purpose and starting point

Implement actual opt-in serving paths for expert-aware batching, exact Engram
hot-row caching, and targeted Blackwell kernels on branch `deepseek-b300`.
The existing upstream execution remains the default and the comparison baseline.
CPU checks are authorized; GPU execution is not part of this implementation run.
A candidate can be implemented without being promoted as a measured improvement.
No speedup, including 3x, is assumed. Never open a pull request.

This extends the [engine plan](deepseek-v41-b300.md). Record implementation,
reviews and remaining evidence in the
[performance implementation record](deepseek-v41-b300-performance-log.md).
The earlier [engine work log](deepseek-v41-b300-work-log.md) remains the record
for deployment and evidence tooling.
The user requested fifteen subagents to deliberate, implement and review. The
workspace concurrency limit requires coordinated waves of at most ten workers.

## Invariants shared by all candidates

- Keep one scheduler and model execution path. Extend existing interfaces; do not
  put a second Python scheduler around the C++ scheduler.
- Preserve checkpoint routers, expert weights, attention/cache layout, sampling
  semantics, request frontiers and ownership. No router-bias retuning.
- Explicitly select experimental behavior. The default deployment remains the
  already inspected upstream implementation; unsupported combinations fail early.
- Bound memory, CPU work, trace queues and scheduling deferrals. No per-token
  device synchronization or unbounded host tracing on the serving path.
- Warm static buffers before CUDA graph capture. Never replace captured storage
  while a replay or transfer may use it. Respect weight-refresh invalidation.
- Keep kernel implementations behind tokenspeed-kernel. Use available upstream
  kernels and numerical references. Prefer CuteDSL for NVIDIA-specific work;
  Triton through the repository wrapper is acceptable for portable candidates.
- Publish precise supported modes and fallback behavior. CPU numerical/reference
  tests cannot establish CUDA compilation, GPU visibility, accuracy or speed.

## A. Online expert-aware batching

### Mechanism

When more ready requests exist than fit in one forward, choose a bounded subset
with a smaller predicted expert union while retaining progress guarantees. Use
only route observations from completed earlier work; never consult future target
routes or run extra model layers to discover them. Missing or stale observations
must fall back to ordinary queue order. Grouping alone does not promise less HBM
traffic: expert occupancy, rank imbalance and dispatch overhead remain measured
outputs.

### Implementation

1. Add a small typed route-signature/feedback contract with request identity,
   generation/frontier and bounded sparse per-layer expert supports. Reject stale
   identity reuse; clear state when a request finishes or retracts as required.
2. Add a bounded capture path using already computed top-k routing results. Keep
   the disabled path inert. Sample at an explicit cadence and drain completed
   observations without waiting for the GPU. Distinguish telemetry from hints.
3. Extend C++ scheduler candidate selection, restricted initially to eligible
   decode requests within a bounded lookahead. Keep existing admission capacity,
   prefill head-of-line, in-flight, preemption and PD readiness rules intact.
4. Greedily score incremental predicted union. Preserve FIFO ties and promote
   eligible hinted requests after a declared number of deferrals. Fair service
   tickets govern backlog progress; the promotion threshold is not an absolute
   elapsed-round bound. A request unable to admit must not consume that budget.
5. Bind feedback and policy through the real scheduler API and runtime event
   path. If a submode cannot supply coherent cross-rank hints, reject it or use
   the documented FIFO fallback; never reorder ranks inconsistently.

### Validation and promotion

CPU tests use the actual policy and scheduler where buildable: equal signatures,
unknown/stale hints, no headroom, oversubscription, fairness, cancellation, ID reuse
and default FIFO equivalence. Add runtime integration tests and explicit GPU
qualification cases for overlap, speculation and supported parallel layouts.
Promote only when end-to-end goodput improves with equal quality and service
constraints; record scheduling overhead and actual versus predicted unions.

## B. Exact Engram hot-row caching

### Mechanism

Keep the canonical FP8 codes and E8M0 scales in the existing host table, while a
bounded HBM hot set serves frequent rows. Misses read the original bytes exactly.
The first candidate uses an immutable hot-set manifest built from observed
frequencies. This removes replacement races from captured forwards and provides
an honest baseline before considering an online replacement policy.

### Implementation

1. Implement a dependency-light cache plan: validate unique global row IDs,
   capacity, layer identity and model revision. Translate to rank-local rows for
   sharded tables; account for copies with shared tables.
2. Build the HBM codes/scales and lookup metadata after both table tensors load.
   No BF16 expansion of the full tables. Use fixed shapes and stable pointers.
   On weight refresh, update or invalidate the hot set before the next forward.
3. Integrate hit/miss selection with ParallelEngramEmbedding. Capacity zero uses
   the existing path. Preserve shard masks, repeated indices and all-reduces.
4. Implement fused lookup/gather/dequantization: read codes and scales from HBM
   on a hit and registered host storage on a miss, then write the same BF16
   values as the reference. Do not fetch host bytes for hits speculatively.
5. Add bounded hit/miss diagnostics and an offline frequency-to-manifest builder.
   Include resident rows, lookup metadata and staging in the memory ledger.

### Validation and promotion

CPU tests cover byte-exact hit/miss behavior, scales, boundaries, repeated IDs,
empty hot sets, shard ownership, manifest mismatch and refresh invalidation.
GPU tests must cover UVA misses, warm/cold graphs, stream ordering and concurrent
requests. Promote only if measured host stalls or cost improve without tail
regression. A static cache is not an adaptive replacement policy; document that
boundary explicitly.

## C. Targeted B300 kernel candidates

### Candidate C1: compact DSpark candidates before communication

The existing greedy block sampler emits one packed (score, token) candidate per
vocabulary tile and gathers them across ranks. Implement a local packed-max
reduction and a compact cross-rank reduction/exchange, preserving global lowest-ID
argmax ties. Keep BF16 Markov weights and FP32 scoring; do not quantize the head
or claim that replicating one projection removes base-logit dependencies.

The candidate reduces communicated elements at the cost of an extra local
reduction. It may lose for small shapes or latency-dominated collectives. Keep
both paths explicit and retain the upstream gather baseline. Check signed zero,
negative scores, padding, empty rows, vocabulary tails and tie behavior against
the existing numerical contract. Reject unsupported sampling/parallel modes.

### Candidate C2: fused Engram cached gather and dequantization

Fuse cache lookup, code/scale gather and dequantization with vectorized row access.
Use static 256-element row and scale geometry, runtime token/row counts, masked
padded lanes and no O(request-count) JIT specialization. Preserve arbitrary
input shape and row-shard zero semantics. Test code/scale bytes independently
from arithmetic. Capture-safe output and metadata storage are part of the API.

### Kernel task contract

Objective: lower complete serving-path cost for the two named mechanisms.
Correctness: match the established numerical and ownership contracts, including
special values and deterministic tie rules. Approaches: existing kernel registry,
NVIDIA-native implementation where justified, portable reference and baseline.
Validation: CPU contract tests under test/performance and AST checks locally;
opt-in tests under tokenspeed-kernel/test/nvidia for compiled execution.
Evaluation: paired microbenchmarks with warmup, synchronized device timing,
identical shapes/dtypes and complete collective or miss-path costs; no benchmark
is run on hardware during this CPU-only implementation.
Promotion: explicit opt-in until numerical GPU tests, graph tests and representative
end-to-end quality/cost/service gates pass. Record compile-unverified separately
from tested. Do not disable baseline fallbacks to force candidate adoption.

## Configuration and evidence

Expose a narrow opt-in configuration for policy selection/bounds, Engram manifest
and capacity, and DSpark candidate communication mode. Fail fast on incompatible
roles, devices and cache layouts. Record resolved settings in launch manifests.
Keep default recipes unchanged; add a separate experimental recipe/example.

Capture mechanism counters without raw prompt content. Distinguish expert load,
unique-expert unions, cache hits, completed output, rejected speculation, and
collective bytes. Add parity/fairness/replay tests and benchmark entrypoints whose
outputs explicitly identify the baseline, candidate, hardware and validation
status. No independent speedup estimates are multiplied together.

## Fifteen-agent ownership and deliberation

| Agent | Responsibility | Required peer discussion |
| --- | --- | --- |
| 1 scheduler_policy | C++ policy, scheduler integration and native tests | route_feedback, scheduler_review |
| 2 route_feedback | Bounded observations and runtime scheduler feedback | scheduler_policy, integration |
| 3 engram_cache | Cache planning, manifest and exact CPU reference | engram_runtime, engram_kernel |
| 4 engram_runtime | Engram model wiring and refresh lifecycle | engram_cache, cache_review |
| 5 dspark_kernel | Compact candidate kernel and numerical contract | dspark_runtime, kernel_review |
| 6 dspark_runtime | DSpark communication/workspace wiring | dspark_kernel, integration |
| 7 engram_kernel | Fused cached row-gather/dequant kernel | engram_cache, engram_runtime |
| 8 benchmarks | CPU mechanism screens and opt-in GPU benchmarks | both kernel owners, performance_review |
| 9 integration | Central arguments, compatibility checks, deployment schema | all runtime owners |
| 10 docs_memory | Operator docs, deliberation record and work memory | all owners |
| 11 scheduler_review | Adversarial scheduler/feedback review and tests | scheduler_policy, route_feedback |
| 12 cache_review | Cache correctness/lifecycle review and tests | engram_cache, engram_runtime |
| 13 kernel_review | Kernel numerical/shape/graph review | both kernel owners, dspark_runtime |
| 14 performance_review | Fairness, measurement and bottleneck review | benchmarks, integration |
| 15 final_review | Independent whole-diff review | all owners |

The coordinator owns final integration, environment changes, full checks, commits
and branch publication. Agents do not install dependencies, launch GPU jobs,
commit, push or open PRs. New test files have explicit owners. Shared interface
changes must be agreed before editing other owners' files.

## Completion checklist

- Document exists before implementation and captures changes from deliberation.
- Three real opt-in serving paths are wired, not merely offline simulators.
- CPU behavior/reference/native tests pass; GPU tests are present and honestly
  unrun. Missing dependencies and unsupported modes are recorded explicitly.
- No baseline regression, unbounded queues or unsafe buffer replacement.
- Fifteen agents have deliberated, implemented or reviewed in coordinated waves.
- Full repository hooks and final review pass; update plan/work log/export and
  push branch updates without opening a PR.

## Resolved implementation scope

The sections above preserve the original implementation objectives. Deliberation
narrows the initial supported surface so that experimental behavior can be
selected explicitly and unsupported combinations fail before execution. See the
[operator guide](../guides/deepseek-v41-b300-performance.md) for current switches
and the [implementation record](deepseek-v41-b300-performance-log.md) for test
and review evidence.

- Route capture and affinity scheduling initially require eager execution with
  prefill graphs, overlap scheduling, mixed batches and speculation disabled.
  Restricting the surface avoids claiming graph-safe asynchronous capture or
  coherent feedback in execution modes that have not been implemented.
- A positive decode-batch cap is an independent setting, available to FIFO as
  well as affinity scheduling. A fair comparison holds this cap constant; a
  baseline with unlimited decode capacity is a different experiment.
- The Engram hot set remains static. Runtime integration must finish loading
  codes and scales before building the cache, preserve storage referenced by
  captured execution, and handle weight refresh under the existing drain rules.
- Compact DSpark uses local packed candidate reduction and the existing TP
  all-gather over one candidate per row and rank. This is a compact exchange,
  not a claim that a new fused or hardware-specialized collective exists.

These restrictions are scope decisions, not hardware qualification results.
Any incomplete original objective remains visible in the implementation record;
CPU tests cannot close a GPU promotion gate.

### Final scope clarifications

All fifteen requested implementation/review assignments ran. The three candidates
are connected to real runtime call sites; component CPU and native evidence is
recorded in the implementation log. Final combined checks are coordinator-owned.
No compiled GPU, CUDA-graph, model-quality or performance result is implied.

The scheduling deferral setting is a promotion threshold for eligible hinted
requests, not a hard bound on global elapsed rounds. Positive-cap FIFO and
affinity share ready-service ticket rotation, with new arrivals joining the tail;
admission order and the cap remain identical between comparison runs. The
original progress objective is evaluated with that actual mechanism and native
adversarial tests rather than a stronger unsupported bound.

The implemented kernels use the repository's portable Triton wrapper and existing
UVA/TP primitives. There is no new B300-specific fused collective or measured
architecture tuning. Engram hit/miss and route-union screens consume supplied
observations; live Engram hit counters and a persistent runtime route trace
exporter remain incomplete relative to the original diagnostics objective.
`hotset` and `performance-screen` are real offline CLI tools, and optional paired
GPU benchmark APIs are present but unrun. Static hot-set membership and eager-only
route capture remain explicit scope limitations.
