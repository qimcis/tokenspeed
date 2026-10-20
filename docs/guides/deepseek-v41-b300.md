# DeepSeek V4.1 Flash on B300

The B300 work targets `deepseek-ai/DeepSeek-V4.1-Flash-1017` through the existing
TokenSpeed runtime, with complete workers distributed across a cluster.
Encoder (E) means the vision tower and aligner; text requests bypass it.

The pinned recipes in `recipes/deepseek-v41-b300/` are local development
examples; their allocations do not define the cluster size.

| Recipe | Purpose |
| --- | --- |
| `aggregate.toml` | Aggregated baseline |
| `pd.toml` | Prefill/decode separation |
| `epd.toml` | Separate vision encoding |
| `performance-affinity.toml` | Eager expert-affinity comparison |
| `load-balancing.toml` | Expert replicas and a complete-frontend routing proxy |

These are candidate layouts, not measured optima. GPU compilation, full model
quality and end-to-end performance remain unqualified. Keep baseline recipes
unchanged when comparing experimental paths.

## Cluster scope

Request routing accepts remote frontend endpoints and multiple complete workers.
Adding worker replicas is separate from spreading one model-parallel worker
across hosts. EPLB uses configured expert groups and rank topology; Engram
caching is local to each rank, and compact DSpark exchanges candidates within
its TP group. These algorithms have no fixed eight-GPU cluster limit.

For workers spanning hosts, use TokenSpeed's [multi-node launcher](../serving/parallelism.md#multi-node)
with external orchestration. Supported model geometry, backend communication
and fabric still constrain each group. Cross-node MegaMoE, E/P/D transfers and
the combined optimizations require hardware qualification. Affinity currently
requires attention DP1 and TP equal to the worker world size; it does not yet
support attention-DP scale-out within a worker.

## Dependencies and launch

Follow the [installation guide](getting-started.md) and build the runtime,
kernel and scheduler packages from the same checkout. New placement, quota and
compact-sampling options require their corresponding source-built components;
disabled options preserve the baseline dependency path.

### Local recipe helper

The recipe helper emits `--nnodes 1` and loopback addresses, and starts one
worker per configured role. It cannot launch a cluster or a worker across hosts.
From the repository root, inspect a recipe without downloading weights or
starting a server:

```bash
python -m tokenspeed.deployment.deepseek_v41 plan \
  recipes/deepseek-v41-b300/epd.toml
```

The output contains arguments, environment settings and a memory lower bound.
The recipe validator checks GPU ownership, TP compatibility and admission limits.
Attention TP equals each worker's GPU count. Existing restrictions, including
V4.1 PD decode KVP and tree speculation, still apply.

On a configured Linux B300 host:

```bash
python -m tokenspeed.deployment.deepseek_v41 launch \
  recipes/deepseek-v41-b300/epd.toml \
  --output "$RUN_DIRECTORY" \
  --startup-timeout "$STARTUP_BOUND_SECONDS" \
  --shutdown-grace "$SHUTDOWN_BOUND_SECONDS"
```

The output directory must be new. Launch resolves the pinned snapshot, checks
hardware and model geometry, waits for readiness, and records per-role logs,
resolved configuration and dependency versions in `manifest.json`. Shutdown
cleans up owned process groups. Frontends bind to loopback. Version capture does
not replace a pinned runtime/container build.

The memory ledger includes known weight/cache payloads, replicas and selected
buffers; exclusions are reported with it. It does not establish model fit.
`checkpoint "$SNAPSHOT"` audits stored tensor bytes without loading weights.
JSONL reporting uses the existing [metrics plugin interface](../metrics-plugins.md).

## Performance options

The optional `[performance]` recipe section selects experimental paths. Engine
flags can also be used with the normal launcher.

| Option | Contract |
| --- | --- |
| `--max-decode-batch-size` | Independent decode cap; use the same cap for FIFO and affinity comparisons |
| `--expert-affinity` | Past-route locality; eager, no overlap or speculation, replicated attention TP; aggregate/decode roles |
| `--expert-affinity-placement` | Static physical dispatch with coherent placement epochs; combined EPLB applies all layers in one update |
| `--engram-hot-cache-manifest`, `--engram-hot-cache-capacity`, `--engram-hot-cache-model-revision` | Immutable hot rows over canonical shared/sharded host tables |
| `--dspark-candidate-communication compact` | Compact local candidates before TP exchange; requires DSpark, PP1 and no speculative sampling; `tiles` is the baseline |

Affinity also excludes DP, PP, QCP, KVP and routing simulation. Its lookahead
must cover resident slots. The deferral setting is a priority-promotion threshold,
not an absolute latency bound. Retain `execution = "eager_serial"` and the same
batch cap when disabling affinity for comparison. See the
[scheduler design](../design/scheduler.md#6-experimental-expert-affinity-decode-selection).

Create an Engram manifest from aggregated per-layer row counts:

```bash
python -m tokenspeed.deployment.deepseek_v41 hotset "$FREQUENCIES_JSON" \
  --capacity-rows "$HOT_ROWS_PER_LAYER" > "$HOTSET_JSON"
```

The frequency input contains a pinned `model_revision` and `layers`, each with
`layer_id`, `total_rows` and `counts` entries of `row_id`/`count`. Include all
Engram layers. The builder selects global rows; runtime capacity is checked per
rank after ownership filtering. Shared tables duplicate hot rows across ranks.
A 256-element row uses 272 payload bytes including its search key. Membership
changes require rebuilding the engine; refresh occurs at a drained update boundary.

## Expert balancing and prefill quotas

V4.1 placement supports serialized FP4 MegaMoE with EP greater than one and MoE
TP one. It requires eager execution without overlap or prefill graphs. Draft
placement remains trivial. Use the existing placement controls, including
`--init-expert-location`, `--ep-num-redundant-experts` and `--enable-eplb`.
Redundant slots are counted across the complete worker per layer, not per GPU.

`--eplb-algorithm trace` additionally requires explicit trace window, movement
budget, penalty, improvement threshold and objective options. `measured_cost`
requires a coefficient file. The trace objective is a prediction, not measured
latency; keep independent P/D profiles. See
[load-balancing design](../design/load-balancing.md#expert-placement).

Fit prefill work from actual observations with a held-out set:

```bash
python -m tokenspeed.deployment.deepseek_v41 fit-prefill \
  "$OBSERVATIONS_JSON" > "$FIT_JSON"
```

Input has `training` and `heldout` arrays of `uncached_tokens`,
`cached_prefix_tokens` and `elapsed_ms`. Configure the resulting nonnegative
coefficients and budget through `[prefill_quota]` or the four
`--prefill-work-*` flags. Native sizing preserves token caps, replay windows and
cache admission. A tiny budget permits one minimum progress chunk, so this is
an estimated-work target rather than a hard elapsed-time guarantee.

## Request routing and pool control

The optional `[routing]` section starts an HTTP proxy after SMG. Ordinary API
requests balance complete frontend concurrency. Authenticated `/route` accepts
trusted preprocessing facts and selects a configured compatible E/P/D path.
The example has one frontend and demonstrates wiring; balancing requires
multiple verified frontends. It does not configure arbitrary stage overrides
inside external SMG.

The pool library provides bounded allocation and local process actuation;
its device inventory is local to the host, not a cluster-wide inventory.
Automatic fleet operation still needs commands, registration, measured demand,
a controller owner and independent engine-drain acknowledgments. It never
infers that an HTTP disconnect means device work has stopped. API, lease and
shutdown contracts are documented in [load-balancing design](../design/load-balancing.md).

## Measurement

Use the module's `--help` and command-specific help for required arguments.

| Command | Use |
| --- | --- |
| `measure` | Bounded HTTP workload execution; authentication through `TOKENSPEED_BENCH_API_KEY` |
| `grade` | Join every measured task with independent outcomes and complete resource costs |
| `compare` | Paired quality/cost analysis with conversation clusters |
| `calibrate`, `replay` | Held-out step-cost fitting and FCFS queue screens |
| `route-oracle`, `locality`, `cache-economics`, `performance-screen` | Offline mechanism analysis; no measured speedup implied |

HTTP completion is not task success. Include rejected/failed work, retries,
idle allocations and all E/P/D resources; count committed output only. The
workload driver is closed-loop and does not execute an agent harness. Compare
one change at a time, then qualify combinations on hardware, including numerical
parity, graph/stream behavior, transfers, cancellation, quality and tail latency.
