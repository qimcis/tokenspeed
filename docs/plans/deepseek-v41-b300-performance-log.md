# DeepSeek V4.1 B300 performance implementation record

This record extends the [engine work log](deepseek-v41-b300-work-log.md).
The [performance plan](deepseek-v41-b300-performance.md) was saved before
implementation. The [operator guide](../guides/deepseek-v41-b300-performance.md)
describes the implemented switches and qualification workflow.

## Evidence boundary

The work is CPU-only. Source integration, CPU reference agreement and tests are
separate from compiled GPU correctness, graph safety, model quality and measured
performance. No B300 result or speedup is claimed. Candidate mechanisms may
reduce a modeled byte count without reducing complete serving latency.

The existing baseline remains available. Router weights and checkpoint semantics
are unchanged. Experimental support is narrower than the engine deployment
recipes; restrictions belong to the candidate configuration, not to claims that
aggregate, PD and EPD are all qualified.

## Deliberation and ownership

All fifteen requested agent assignments ran: ten implementation owners and five
reviewers, in waves constrained by the available slots. The implementation
owners covered scheduler policy, route feedback, Engram planning/runtime/kernel,
DSpark runtime/kernel, benchmark tooling, integration and documentation. Review
assignments covered scheduler, cache, kernels, performance evidence and the
combined change. The coordinator owns final checks, exports and publication.

## Interface agreements and implemented paths

| Path | Serving integration | Scope and invariants |
| --- | --- | --- |
| Expert affinity | Existing MoE top-k observations → bounded pinned-host copy → existing result completion → TP CPU-group broadcast → event-loop tail → C++ scheduler | Actual scheduler candidate ordering; no extra router evaluation; eager/non-overlap/non-speculative only |
| Engram hot rows | V4.1 table loading configures and refreshes `RowShardedEngramEmbedding`; CUDA forward calls fused hit/miss lookup | Static exact code/scale membership; canonical UVA host misses; shared and sharded layouts retain existing collective behavior |
| DSpark compact candidates | Existing scorer → local packed maximum → existing TP all-gather → greedy resolution | Same BF16 Markov weights, FP32 scores and signed packed-key ordering; stable double buffers; TP=1 has no collective |

These are runtime call paths, not claims inferred from offline helper code. The
new kernels remain compile-unverified on the intended GPU environment.

### Scheduler and route feedback

The independent positive decode cap applies to FIFO and affinity, including
first decode and capacity retry. Capped ready-service tickets put newly ready
requests at the tail and move selected requests back to the tail; admission
order stays unchanged. This resolves the starvation hazard of giving every new
arrival a zero/oldest ticket. Native adversarial tests cover continuous arrivals,
unknown/stale/fresh hints, cancellation/identity reuse and D-role readiness.
Remote-prefilling requests remain excluded until their acknowledgement.

`max_deferrals` is a priority-promotion threshold for eligible hinted requests,
not a maximum number of elapsed service rounds. Unknown/stale hints and capacity
boundaries retain service order. The lookahead covers resident slots. Generation
and frontier checks reject late feedback after identity reuse or state changes.
The [scheduler design](../design/scheduler.md) records the policy invariants.

Capture is sampled only for pure single-token decode. Rank zero owns fixed
pinned buffers, bounded to 32 layers and 256 total top-k IDs per request. The
existing result event guards consumption. Runtime observations are broadcast to
all attention-TP ranks before the control thread applies hints. No capture hooks
or buffers are installed when disabled. Eager-only scope avoids pretending that
observer hooks work under captured replay.

### Engram lifecycle and memory

Strict manifests identify model revision, layer geometry and unique global row
IDs. Planning filters rank ownership before checking capacity; shared host
layouts duplicate HBM hot rows on participating ranks. Exact code bytes and
E8M0 scale bytes are copied in chunks of at most 4096 rows. A 256-element hot row
costs 272 resident payload bytes including its int64 search key. Python manifest
and planning objects add host memory beyond the tensor ledger.

Loading invalidates device lookup IDs in place. The drained update boundary
synchronizes that invalidation so a later replay on another stream sees misses,
including when loading fails before the normal success synchronization. Both
canonical tables and shared-host publication must complete before refreshing
stable cache buffers. Invalidation does not make a partially changed canonical
table safe for serving. Membership/revision changes require engine/graph rebuild.

The fused kernel requires caller-owned BF16 output, uses masked canonical loads
only for misses, preserves duplicate indices and zero outputs for non-owned
rows, and has no allocation or synchronization inside the op. It uses the
repository's Triton wrapper and existing UVA mechanism; this is a portable
kernel candidate, not an independently tuned CuteDSL B300 implementation.

### DSpark numerical and buffer contract

The runtime retains the scorer's full tile workspace and stores one candidate
per row/rank in stable double buffers. TP>1 exchanges those candidates through
the existing all-gather. The local reduction is added work; fewer payload bytes
do not prove a latency benefit.

Exact numerical preservation means the existing signed packed-key contract:
identical FP32 score bits tie-break by lowest token ID. Signed zeros stay
separate, and NaNs retain the existing bitwise order. The original plan's argmax
wording must be read with this existing contract, not ordinary numeric NaN
semantics. Actual scorer floating-point behavior and graph replay remain GPU
gates despite CPU integer reduction algebra.

### Configuration and evidence tools

The optional deployment `[performance]` section requires explicit affinity,
execution and DSpark selectors. `execution = "eager_serial"` stays fixed when
affinity is toggled off for comparison. The positive decode cap likewise stays
fixed. Baseline configurations avoid newly added scheduler/kernel methods so
released packages remain usable without experimental feature adoption.

`hotset` builds immutable manifests from strict aggregated frequency records;
`performance-screen` analyzes supplied observations and labels its output
`cpu_mechanism_screen_only`. Neither runs a model or establishes trace provenance.
Independent trace comparability flags, logical Engram accesses and DSpark payload
counts remain mechanism evidence. Deferral-threshold exceedances are not reported
as violations of a nonexistent hard service bound. Optional paired GPU benchmark
functions include the real runtime miss/collective paths, synchronized eager
timing, alternating order and worst-rank timing; none was executed.

## Review decisions and remaining plan differences

- Scheduler review led to strict cap enforcement and ready-ticket rotation under
  continuous arrivals, including mixed hint availability. Promotion-threshold
  wording replaces an unsupported universal service guarantee.
- Cache review exercised manifest/byte boundaries and refresh failure behavior;
  stream-safe invalidation is performed at the drained update boundary.
- Kernel review covered reduction/output contracts and adversarial shapes. CPU
  references and substitutes do not establish compiled GPU behavior.
- Performance review corrected empty-batch collective accounting, separate
  baseline/candidate trace comparability, deferral labeling and replicated
  Engram memory/access reporting.
- Final review and integration corrected baseline compatibility with older
  released scheduler/kernel packages; experimental paths still require updated
  sources. The coordinator records final review closure and full-hook results.
- The original plan sought bounded production diagnostics. The present screen
  consumes supplied observations; it does not provide live Engram hit counters
  or an integrated persistent route trace exporter. Route feedback for selection
  is implemented separately from such telemetry.
- Static hot rows, portable kernels, eager affinity and the existing TP collective
  are deliberate initial boundaries. Adaptive replacement, CUDA-graph affinity,
  a new fused collective and architecture-specific tuning remain outside this
  implementation.

## Verification record

Owner-reported component evidence is preserved below. Counts overlap and must
not be summed. The earlier engine work log's 188 passing tests and 18 skips
belong to the prior engine/tooling work, not this change.

| Component | CPU evidence reported by owner | Explicit remaining boundary |
| --- | --- | --- |
| Scheduler | Full native suite: 610 passed, including 16 affinity tests; Python binding suite: 247 passed | macOS CPU build; GPU/rank serving behavior unverified |
| Route feedback | 17 passed, 2 opt-in CUDA skips, including real two-process CPU gloo and native binding integration | Device-copy ordering and served model execution unrun |
| Engram cache | 34 passed; all 256 FP8 bytes × 256 E8M0 bytes checked with Torch casts | CPU reference does not validate CUDA arithmetic/UVA |
| Engram runtime | 10 passed, 1 opt-in GPU skip; combined cache/runtime 44 passed, 1 skipped | Capture/replay and cross-stream update tests unrun |
| Engram kernel API | 12 passed; Python compilation checks passed | GPU kernel compilation and execution unrun |
| DSpark kernel reference | 11 passed; Python compilation checks passed | GPU special-value/tail/replay tests unrun |
| DSpark runtime | 12 passed, 1 opt-in GPU skip | CPU substitutes cover runtime workspace/sampler; real collectives unrun |
| Configuration integration | 69 passed | Configuration rejection and generated argv, not hardware qualification |
| Benchmark tooling | 14 owner tests passed, plus separate reviewer coverage | No GPU measurements or serving performance claim |

Documentation verification checked `hotset --help`, `performance-screen --help`
and a successful CPU `plan` invocation for `performance-affinity.toml`. The
result remains explicitly analytical/hardware-unvalidated.

Coordinator verification after formatting:

- Combined Python suite: **628 passed, 22 skipped**. This includes the 247 native
  scheduler binding tests; do not count those again. Sixteen skips are endpoint
  smoke cases, two need vendor Triton frontend imports, and four are explicit
  CUDA qualification cases.
- Native C++ suite: **610 passed**, including 16 expert-affinity cases.
- The native library and binding were built from the changed source on CPU.
- A shutdown diagnostic was traced to six native objects held in pre-existing
  module-level pytest parameters. Those parameters now hold factories and create
  objects inside tests. Repeated combined execution shut down cleanly; no
  serving-memory workaround was introduced.
- Independent whole-diff review found no remaining blockers after the baseline
  dependency fixes and fairness corrections. Reviewers were from the available
  GPT family; this is independent review, not cross-family validation.
- `pre-commit run --all-files` passed on the frozen tree. Formatting changes
  were applied before the repeated CPU suites, and both diff checks passed.

Commands for the integrated CPU checks (the native path is a local source build):

```bash
source .venv/bin/activate
PYTHONPATH=work/native-python:python pytest -q -rs \
  test/deployment test/performance tokenspeed-scheduler/python/tests
cmake --build work/build-scheduler -j 4
work/build-scheduler/tokenspeed_scheduler_tests
pre-commit run --all-files
```

The CMake build used the locally available compatible macOS SDK and OpenSSL;
these build paths are workstation artifacts, not deployment configuration.
No GPU code was compiled or executed. Live production trace export and adaptive
cache replacement remain separate follow-ups, as recorded above.

## Hardware acceptance still required

- Compile and compare every new kernel on the intended B300 runtime closure,
  including special values, tails and actual production dimensions.
- Exercise eager execution, capture/replay, warm/cold misses, weight refresh and
  supported distributed modes with real device streams and collectives.
- Confirm route feedback freshness, queue progress, admission behavior and
  coherent rank scheduling under the supported mode constraints.
- Measure real HBM/host allocations, cache hit traffic, collective latency,
  scheduling CPU overhead, tail latency, quality and whole-system cost.
- Use paired workloads with the same offered tasks, checkpoint and sampling;
  retain rejected work, failures, idle allocations and committed token counts.
