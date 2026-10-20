# Work routing controller

The opt-in `tokenspeed.runtime.load_balancing.service` process is an HTTP routing
and admission controller. It does not schedule engine batches, change TP rank
membership, orchestrate KV transfer, or alter the existing encoder batcher.
`ts serve` still owns its external SMG gateway and engine process.

## Supported forwarding contracts

Ordinary `/generate`, `/v1/completions`, `/v1/chat/completions` and `/v1/models`
requests select a fresh `complete` or `pair` HTTP frontend and reserve concurrency.
The frontend must already implement its entire serving path. The adapter forwards
the original request and streamed response bytes, closes the upstream connection
on disconnect, and releases its reservations exactly once. It never retries.
After a response starts, an upstream failure terminates that response; it does
not produce another response or replay previously delivered tokens.

Authenticated `/route` accepts a generation body plus trusted tokenization,
post-preprocessing encoder patch facts, and decode limits. Configured
`CompatiblePath` records identify the exact stage workers and incarnations behind
each frontend. Actual prefill, decode and encoder policies score these paths;
the registry reserves all stages atomically. Prefill and encoder scores are
normalized by their configured capacity before combining with normalized decode
pressure, so milliseconds and patches are not added as raw units. Several paths sharing a stage use
one worker identity and therefore share one capacity bound. A path has exactly
one complete worker per requested stage. Stage facts must describe the forwarded
body; the controller is not a tokenizer or image preprocessor and cannot verify
that relationship. Callers must provide truthful facts, and cache availability
hints must not substitute for pinned hard-resource capacity.

These mappings are operator-verified deployment contracts. The controller does
not configure or verify the internal routing of an external multi-worker SMG.
The local `serve_smg.py` interface starts an external gateway for one engine;
it exposes no verified arbitrary per-request E/P/D override API. Thus selecting
independent stage endpoints is available as an explicit reservation/control API,
not as an advertised arbitrary EPD serving capability. A frontend must actually
use the fixed stage members named in its path.

## Identity, accounting and lifecycle

`Worker` identifies an entire TP group, not a rank. A new process uses a new
incarnation. Reports carry strictly increasing sequence numbers and controller
monotonic receipt times. Health, readiness, drain state, freshness and every
workload capacity axis are checked in the same lock as admission. Concurrency,
token and patch limits bound explicit reservations. `prefill_work` is an
admission budget over predicted milliseconds, not a guarantee about actual
execution time. Calibration residuals and eviction of an unpinned cache hint
can increase actual compute; unknown or stale cache hints are treated as misses.
The native scheduler remains the hard memory-admission gate. Telemetry
cannot reopen a drained worker. `reported` work excludes all controller-owned
leases, preventing double counting. Missing or zero capacity is unavailable,
not unbounded capacity.

Old-incarnation leases remain explicitly releasable but never charge a new
incarnation. Retired incarnations cannot be resurrected by delayed reports.
Lease updates replace the previous workload and recheck capacity; decode progress
uses an immutable original output budget and a monotonic accepted-token frontier.
SSE byte chunks are not interpreted as token progress. A trusted engine observer
must call `/control/decode-progress` for token-frontier updates.

Worker and lease counts are bounded. Completed request IDs are retained for the
configured `history_seconds`; requests that reuse those IDs are rejected during
that interval. The retention is in-process, not durable exactly-once delivery.
The combined active/completed count is bounded by `max_leases` and fails closed
until old completion records expire. Retired-incarnation history also fails
closed at this bound and requires deliberate controller reconciliation. Active
leases never expire merely because a telemetry report is stale.

Closing an HTTP connection does not prove the engine has stopped generating or
transferring state. Pool stop decisions must also require a fresh, independent
engine-idle/drain observation, as the pool actuator does. Controller restart loses
in-memory reservations and requires stopping admission and reconciling real
engine activity before traffic resumes. No automatic restart reconciliation or
active-decode migration is claimed.

## API and configuration

Start with `python -m tokenspeed.runtime.load_balancing.service --config FILE
--host 127.0.0.1 --port PORT`. Configuration requires `freshness_seconds`,
`history_seconds`, `max_workers`, `max_leases`, `forward_role`, and `workers`.
Worker entries contain `worker_id`, `incarnation`, `role`, `endpoint`, and
`capacity`; optional work counters and context/cache facts use the typed schema.
Configured workers begin unready. The service probes only complete/pair frontend
`/health` endpoints; stage work and readiness require actual observer reports.

Optional `paths`, `prefill_policy`, `decode_policy` and `control_token` enable
stage-aware `/route`. All `/control/*` endpoints and `/route` require the bearer
token and are disabled when it is absent. That token authorizes inventory writes;
it is a trusted control-plane credential, not an end-user inference credential.
The adapter strips it before forwarding `/route` bodies upstream.

Control endpoints are POST `/control/workers` (full worker report),
`/control/reserve` (`request_id`, `role`, `workload`), `/control/release`
(`lease_id` for a manual reservation), and `/control/decode-progress`
(`request_id`, `generated_tokens`, optional `verification_tokens`). Manual release
cannot free an active streaming route's leases.

A `/route` body contains `request_id`, `path`, `body`, `prefill`, `decode`, and
optional ordered `encoder` items. Prefill policy coefficients must be explicitly
provided calibration results, not defaults invented by the controller. Prefill
request queue timestamps and worker observations use the controller's monotonic
clock. Request bodies are limited to 4 MiB; this controller is intended for
ordinary HTTP payloads and trusted metadata, not large embedded tensor transfer.

CPU tests exercise concurrent atomic admission, shared-stage contention, stale
reports/incarnations, real localhost HTTP forwarding, streaming failure,
disconnect cancellation, authenticated path routing and post-completion replay
rejection. External SMG path membership, GPU execution, tokenization alignment,
real cache/transfer behavior and performance remain deployment qualification
requirements.
