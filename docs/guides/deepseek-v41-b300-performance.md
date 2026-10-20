# DeepSeek V4.1 B300 experimental performance paths

The [performance plan](../plans/deepseek-v41-b300-performance.md) defines three
opt-in candidates: expert-aware decode batching, exact Engram hot-row caching,
and compact DSpark candidate communication. The
[implementation record](../plans/deepseek-v41-b300-performance-log.md) separates
source integration and CPU checks from the remaining hardware gates.

These candidates are unqualified on B300. Keep the existing deployment as the
comparison baseline. A smaller expert union, fewer host-row reads or fewer
collective elements is a mechanism measurement; it is not a serving speedup.
See the [deployment guide](deepseek-v41-b300.md) for pinned recipes, request
measurement, independent quality grades and whole-system cost comparisons.

## Select a candidate

The server selectors are explicit and independent. The existing `tiles`
communication and zero-capacity Engram path remain the baseline; omitting the
optional deployment `[performance]` section keeps baseline recipe behavior.

| Candidate | Selector | Initial restrictions |
| --- | --- | --- |
| Expert affinity | `--expert-affinity` with a positive `--max-decode-batch-size` | CUDA DeepSeek V4.1; eager, prefill graphs disabled, overlap disabled, no mixed batches or speculation; aggregate or decode role |
| Engram hot rows | `--engram-hot-cache-manifest`, `--engram-hot-cache-capacity`, `--engram-hot-cache-model-revision` | Host shared/sharded tables; default or safetensors loading; capacity is a per-layer, per-rank bound |
| DSpark compact exchange | `--dspark-candidate-communication compact` | CUDA DSpark, PP=1, no speculative sampling; `tiles` selects the baseline |

Affinity additionally requires DP=PP=QCP=KVP=1, full attention TP over the worker
world, and replicated request inputs. Expert parallelism may partition experts,
but expert load balancing, redundant experts, custom placement and routing
simulation are excluded. These constraints keep the observed logical expert IDs
and scheduler hints coherent across ranks.

Affinity bounds are `--expert-affinity-capture-interval` (16),
`--expert-affinity-max-layers` (4), `--expert-affinity-lookahead` (256),
`--expert-affinity-max-deferrals` (8), and
`--expert-affinity-max-hint-age-tokens` (64). Parenthesized values are current
server defaults when affinity is selected, not measured tuning recommendations.
The lookahead must cover `max_num_seqs` resident slots so every resident request can
participate in selection. `max_deferrals` promotes eligible hinted requests after
the threshold; it is not an absolute bound on elapsed rounds until service.
The independent `--max-decode-batch-size` defaults to 0 (uncapped). Hold the same
positive cap for the FIFO and affinity runs so that batch-size changes do not
masquerade as a locality benefit.

The experimental recipe is `recipes/deepseek-v41-b300/performance-affinity.toml`.
Its `[performance]` section explicitly selects `expert_affinity`, `execution`
(`default` or `eager_serial`) and `dspark_candidate_communication` (`tiles` or
`compact`). `eager_serial` retains the eager, non-overlap execution settings when
`expert_affinity` is changed to `false`; this preserves the FIFO comparator.

Validate locally without starting a model:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m tokenspeed.deployment.deepseek_v41 plan \
  recipes/deepseek-v41-b300/performance-affinity.toml
python -m pytest -q test/performance test/deployment
```

The affinity/capped scheduler needs the updated source-built scheduler. Baseline
settings remain compatible with the released scheduler and kernel packages;
experimental methods are not looked up when their candidates are disabled.
The test command assumes that updated native binding is installed or on
`PYTHONPATH`; native integration coverage can otherwise be unavailable. Building
a source package is separate from the CPU configuration check.

## Route feedback and request progress

Rank zero observes already-computed top-k IDs from a bounded set of actual MoE
layers. The first pure single-token decode and then every configured interval
are sampled; prefill, mixed and idle work do not advance that cadence. Fixed
pinned host buffers receive copies on the execution stream. The existing result
completion event precedes consuming the snapshots, and sampled hints are
broadcast through the attention-TP CPU group before scheduler updates.

Feedback carries immutable dispatch-time request generation and computed-token
frontier. The C++ scheduler rejects stale identity, unsupported states, in-flight
requests, future or nonmonotonic frontiers and overly old observations. No extra
model forward discovers routes. Disabled affinity installs no capture hooks or
buffers. Rank-zero pinned payload is bounded by
`8 * max_num_seqs * sum(selected_topk_widths)` bytes, with at most 32 layers and
256 total expert IDs per request; this excludes Python metadata and transport
bookkeeping.

The independent decode cap uses ready-service tickets: newly ready requests join
the tail, and selected requests rejoin the tail. Admission order remains
unchanged. Affinity starts from that service order, applies locality only to
eligible fresh hinted requests, and treats unknown/stale hints and capacity
boundaries as FIFO barriers. Promotion at the deferral threshold does not
promise service within that many global rounds. Inspect waiting requests and
actual admission/capacity pressure in service measurements.

## Engram manifest and memory

Create a manifest from previously aggregated observations:

```bash
python -m tokenspeed.deployment.deepseek_v41 hotset "$FREQUENCIES_JSON" \
  --capacity-rows "$HOT_ROWS_PER_LAYER" > "$HOTSET_JSON"
```

The frequency JSON requires a pinned 40-character lowercase hexadecimal
`model_revision` and a `layers` array. Each layer has `layer_id`, `total_rows`,
and `counts`, where each count is `{"row_id": 7, "count": 25}`. Aggregate repeats
before calling the builder; duplicate layer or row records are rejected. Supply
all Engram layers required by the deployment. Capacity here selects global rows
per layer; runtime capacity is checked after each rank's ownership filter.

The generated manifest uses schema version 1, an explicit `model_revision`, and a `layers`
array. Every layer contains `layer_id`, `total_rows`, and unique
`global_row_ids`. Unknown fields, duplicate JSON fields, duplicate rows, invalid
row ranges and identity mismatches are rejected. Rows are global table IDs;
sharded tables filter them to rank ownership before checking capacity. Shared
tables allocate a hot copy on every participating rank.

`build_engram_hot_set_manifest` in
`python/tokenspeed/runtime/models/engram_hot_cache.py` builds a static manifest
from pre-aggregated per-layer row frequencies. It selects positive counts in
descending frequency order, breaking frequency ties by smaller row ID. The
builder runs offline; it is not a serving-path trace collector.

For the 256-element rows, each resident row uses 256 code bytes, 8 scale bytes
and an 8-byte search key: 272 bytes before allocator overhead. The cache has no
full-table row-to-slot map. Explicit host staging is bounded to 4096 rows per
copy chunk. Measure loading peaks and allocator overhead separately; the payload
formula is not a measured allocation bound. A shared-table configuration pays
resident payload for each rank, and independent P/D workers keep independent
caches. The deployment ledger reports `engram_hot_cache_per_rank` entries with
`codes_and_scales`, `lookup_metadata` and `host_staging_peak`, plus
`engram_hot_host_staging_peak_per_worker`. Host staging is a separate peak from
resident `known_host_bytes`. Python manifest/planning objects are additional
host memory.

Membership and checkpoint-revision changes require rebuilding the engine and
its graphs. Same-membership refresh copies code and scale bytes into stable
buffers at the existing drained weight-update boundary. Never refresh while a
forward or graph replay is using those buffers. Invalidation fills lookup IDs
with -1 in place and synchronizes at the rare drained update boundary, so later
execution on another stream observes misses. Both canonical tables must finish
loading and shared-host publication before cache refresh. Invalidation does not
make partially updated canonical tables safe for serving.

## Validation boundaries

A successful CPU configuration parse does not prove model fit or distributed
compatibility. A passing numerical reference test does not prove GPU compilation,
UVA visibility or graph replay. Optional hardware tests must be explicitly run
on a separately authorized GPU environment before promotion.

Compare one candidate at a time before studying combinations. Preserve resolved
configuration, source revision, checkpoint revision, runtime dependencies,
hardware topology, workloads, warmup and measured results. Report candidate
setup memory and runtime overhead as well as the work it avoids.

## Mechanisms and tradeoffs

Expert-aware batching predicts locality from past completed route observations.
It changes the eligible request order within bounded constraints; it does not
change the router or know future target routes. Fairness and admission behavior
must be evaluated alongside predicted and actual expert unions.

The Engram cache stores exact FP8 codes and E8M0 scales for a static selected hot
set. Host storage remains authoritative for misses. The initial candidate has no
adaptive eviction or online replacement policy. Include codes, scales, lookup
metadata and staging in memory measurements, and verify refresh behavior before
using changed weights.

DSpark compact selection reduces local packed block candidates before exchanging
rank candidates. It preserves the upstream signed INT64 packed score/token ordering contract.
Identical score bits choose the lowest token ID; signed zeros remain distinct
and NaNs retain the upstream bitwise ordering. This is not a claim of ordinary
`argmax` NaN semantics. The added local
reduction can offset communication savings; benchmark the complete collective
path, including reduction and unpacking, against the baseline.

## Mechanism screens and optional GPU measurements

```bash
python -m tokenspeed.deployment.deepseek_v41 performance-screen \
  "$OBSERVATIONS_JSON" > "$MECHANISM_REPORT_JSON"
```

The input is the strict `MechanismScreen` schema in
`python/tokenspeed/deployment/deepseek_v41/performance_bench.py`; the behavioral
fixture in `test/performance/test_performance_bench.py` illustrates it. It
requires an explicit `evidence` value (`synthetic` or `recorded`), `source`, and
expert, Engram and DSpark observations. The output status is
`cpu_mechanism_screen_only`, with no latency or speedup result. This command does
not execute a scheduler or model, collect a runtime trace, or establish provenance.

Expert output distinguishes known from complete predicted/actual unions and
flags mismatched selected counts, batch counts and eligible populations. A
`deferral_threshold_exceedances` entry exceeds a promotion threshold; it is not
proof of a hard fairness-bound violation. Engram counts logical row accesses,
not PCIe transaction bytes. DSpark counts collective payload elements, not
network wire bytes; TP=1 or empty rows perform no collective. Live per-forward
Engram hit counters and an integrated persistent route trace exporter are not
implemented by this screen.

The same module exposes two explicitly opt-in Python GPU benchmark functions:

- `benchmark_dspark_gpu(logits, bonus_ids, markov_head, lm_head, tp_group, config=...)`
  compares complete tiled and compact sampling, including local reduction and
  the existing collective.
- `benchmark_engram_gpu(baseline, candidate, indices, config=...)` compares actual
  runtime embeddings, including UVA misses and shard all-reduces. Supply
  representative misses; hit-only input is rejected. The two
  embeddings must share canonical host tables; the candidate must already have
  its immutable cache loaded. Retain both objects and avoid further loading
  through aliased parameters.

Both require `GPUExperiment(allow_gpu=True, warmup=..., repetitions=..., source=...,
model_revision=...)`, with at least two repetitions. They perform parity checks
before and after measurements, alternate pair order, synchronize complete eager
operations and report the slowest participating rank. Reports include rank and
group cache payloads, with shared-table replica accesses and allocations counted
separately. Repeated inputs warm caches. These functions do not launch a model,
qualify cold-system behavior, graph execution or concurrent serving. No GPU
benchmark was run for this work.

GPU suites are separately opt-in. On a separately authorized, fully installed
CUDA environment, these are the qualification entrypoints (not commands run here):

```bash
TOKENSPEED_RUN_EXPERIMENTAL_GPU_TESTS=1 python -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/test_dspark_compact_candidates.py \
  tokenspeed-kernel/test/nvidia/test_engram_cached_embedding.py
TOKENSPEED_RUN_DSPARK_COMPACT_GPU_TESTS=1 python -m pytest -q \
  test/performance/test_dspark_runtime.py
TOKENSPEED_RUN_ENGRAM_GPU_TESTS=1 python -m pytest -q \
  test/performance/test_engram_runtime.py
TOKENSPEED_RUN_ROUTE_FEEDBACK_GPU=1 python -m pytest -q \
  test/performance/test_route_feedback_cuda.py
```

The included single-device cases do not replace a multi-rank B300 acceptance run.
Run supported topology and service-level tests after these component checks.

## Qualification sequence

1. Verify explicit configuration and rejection of unsupported combinations.
2. Run CPU behavior/reference tests and inspect every skip reason.
3. On separately authorized hardware, run numerical and stream/graph tests before
   paired microbenchmarks. Include misses, tails and complete collectives.
4. Run equal-workload service checks with quality and tail-latency constraints.
5. Promote only with measured end-to-end benefit under those constraints. Do not
   multiply independent microbenchmark ratios into an overall speedup claim.
