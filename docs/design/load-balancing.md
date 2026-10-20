# Load balancing

Load balancing extends the existing scheduler, expert updater and serving
frontends. It does not create a second inference scheduler. The
[B300 guide](../guides/deepseek-v41-b300.md) covers configuration; the native
[scheduler design](scheduler.md) defines affinity and prefill-quota invariants.

Routing operates across complete workers at local or remote endpoints. Expert
balancing operates within each worker's configured expert group, using rank/node
topology. Neither fixes the fleet to a particular GPU count. Cross-host model
parallelism additionally depends on qualified backend transports; remote HTTP
routing alone does not establish support for it.

## Expert placement

Logical expert selection and router weights remain unchanged. Physical slots
may replicate an expert; the V4.1 loader fills each local copy with its packed
FP4 values, scales and backend padding. Migration uses contiguous views of the
actual processed storage and stages bytes before publishing maps. Refresh
rebuilds processed weights only after a complete validated load. Drafts remain
trivially placed. Placement APIs fail early on unsupported backends or packages.

A routed invocation is `(source rank, token row, top-k slot)`. It contributes
once to that source's output. Baseline attention TP can submit replicated rows
to MegaMoE; this work preserves that execution and does not add the replicated
outputs together. Counters measure physical invocations, not unique user tokens.
The backend's actual communication ownership determines snapshot reduction.

Trace EPLB retains bounded per-forward load vectors and minimizes predicted
busiest-rank work over the window. Stable ties, coverage and slot capacity are
preserved. Movement limits, per-slot penalties and an improvement threshold
avoid unnecessary migration. The objective assumes uniform replica splitting;
`measured_cost` weights routes with supplied coefficients. Neither is a latency
measurement. The existing async worker computes placement off the serving path.

A monotonic epoch identifies each published placement. Validation precedes live
mutation; snapshots cannot attribute old physical slots to new logical experts.
Placement-aware affinity uses the actual static dispatch destinations across
replicated source ranks, retaining replica ordering and topology. Its EPLB
combination applies every layer in one control operation before publication.
Other EPLB operation may use the existing layer-chunk updater.

## Work estimates

Prefill uses `a*q + b*q*c + d*q*q` in milliseconds, with nonnegative calibrated
coefficients. `q` includes replay; `c` is the prefix before replay. Unknown/stale
cache evidence is a miss; remote reuse includes transfer cost. Prefill work is a
budget of predictions, not a wall-clock guarantee. Native admission remains the
hard memory gate.

Decode scoring considers concurrency, context, remaining output, verification
work and KV capacity. Encoder scoring uses postprocessed patch counts. Unpinned
cache hints can improve ranking but cannot reduce hard decode KV or encoder
patch reservations. A worker is an entire TP group, never an independent TP rank.

## Routing service

`python -m tokenspeed.runtime.load_balancing.service --config FILE --host HOST
--port PORT` serves the opt-in HTTP adapter. Configuration requires
`freshness_seconds`, `history_seconds`, `max_workers`, `max_leases`,
`forward_role` and `workers`. Each worker specifies its ID, incarnation, role,
endpoint and capacity. Configured workers begin unready; health probes refresh
complete/pair frontends, while stage readiness/load requires observer reports.

Ordinary `/generate`, `/v1/completions`, `/v1/chat/completions` and `/v1/models`
requests reserve concurrency on a healthy complete/pair frontend and forward
request/response bytes. Streaming has no retries. Disconnect closes upstream
and releases routing reservations; it is not proof that device work has ended.

Optional `paths`, `prefill_policy`, `decode_policy` and `control_token` enable
`/route`. Its body contains `request_id`, `path`, `body`, `prefill`, `decode` and
optional ordered `encoder` items. These are trusted preprocessing facts, not
information the adapter can verify from opaque HTTP bodies. A `CompatiblePath`
names exact E/P/D worker incarnations behind an operator-verified frontend.
Stage scores are normalized by capacity and all reservations are atomic. Paths
sharing a stage share its capacity. Arbitrary external SMG stage composition
is not supported by this interface.

`/route` and all control endpoints require the configured bearer token and are
disabled without it. It authorizes inventory writes and is stripped before
forwarding. Request bodies are limited to 4 MiB.

| POST endpoint | Purpose |
| --- | --- |
| `/control/workers` | Publish a full worker report |
| `/control/reserve` | Reserve explicit work by request ID and role |
| `/control/release` | Release a manual lease, never a live stream's lease |
| `/control/decode-progress` | Replace reservations using monotonic accepted-token progress |

Reports use increasing sequence numbers and controller-monotonic receipt times.
Readiness, health, drain state, freshness and capacity are checked under the same
lock as reservation. Reported load excludes controller-owned leases. Missing or
zero capacity is unavailable. Worker reincarnation fences stale reports, updates
and retries; old leases remain releasable without charging a new incarnation.

Active/completed IDs and retired incarnations have bounded in-process retention.
Active leases do not expire merely because telemetry becomes stale. Restart
requires admission to stop and real engine work to be reconciled: this is not
durable exactly-once accounting. Progress comes from trusted engine observations,
not SSE chunk counts or HTTP status.

## Pool lifecycle

`PoolController` derives role targets from queued/reserved work, measured service
capacity, minima/maxima, spare inventory, hysteresis and a movement budget.
Indivisible TP groups must fit the budget. Stale/regressed observations hold the
current target. Starting and draining workers retain their GPU ownership.

`LocalProcessBackend` manages one host's device inventory and owned process
groups; multi-host actuation requires external orchestration. It requires explicit
`apply=True`. The registry must be the sole ingress and each managed incarnation
must be registered as draining before launch. Admission opens only after fresh
ready/healthy reports. Stopping requires zero leases plus an independent, fresh,
same-incarnation acknowledgment of zero pending requests and transfers after
admission closure. A reversed target can reopen a healthy draining worker.

Timeouts and missing drain proof block instead of killing active work. Failed
starts/stops retain devices until process exit is confirmed. The controller does
not migrate active decoder state. Deployment must provide commands, endpoints,
registration, measurements, tick ownership and qualified drain acknowledgments;
automatic fleet operation is not enabled by the example recipe.

## Embedding handoff failures

Existing encoder-to-prefill frames retain their layout: six registration fields,
thirteen transfer fields, little-endian uint64 pointers and ASCII metadata.
Strict flags, positive fanout and valid destinations are required for nonempty
payloads; empty shards remain valid heartbeats. Sender shape/span checks still
apply. The wire format does not negotiate model or processor revisions.

Every submitted write must settle before terminal status permits ring-slot
reuse. Submission/write errors drain other writes; receiver-notification errors
do not kill the queue worker. Already-terminal queued rooms are discarded.
External cancellation during DMA and GPU-visible completion still require
transport-specific qualification.
