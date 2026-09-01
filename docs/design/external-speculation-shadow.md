# External speculation shadow control plane

TokenSpeed exposes a deliberately small, vendor-neutral seam for evaluating an
external speculation coordinator without making that coordinator part of the
correctness or GPU data plane.

## Scope

`--remote-spec-mode shadow` enables three things:

1. an exact semantic prefix stamp for each live request;
2. bounded per-seal readiness, selection, fallback, and commit telemetry; and
3. a one-plan list of preferred decode request IDs.

It does **not** land candidate tensors, alter a verifier, export model features,
or create an external-only execution mode. Every selected row still executes
through TokenSpeed's native autoregressive or local speculative path. The
shadow row records `REMOTE_DISABLED` when an otherwise usable external
candidate was ready.

The native path is a fallback hierarchy, not a synonym for AR. When the loaded
target has a correctness-qualified resident MTP policy, a future remote data
plane must fall back to local MTP first and to AR only when local speculation
is unavailable or unsafe. External candidate value must therefore be measured
over the strongest qualified resident policy.

The mode is off by default. Its HTTP(S) endpoint must be loopback-local. A
deployment that crosses a host boundary puts its authenticated transport,
retries, and policy service behind that local endpoint.

## Semantic identity

A request prefix is identified by all of:

- request ID;
- request incarnation (so ID reuse is not aliasing);
- committed token count;
- SHA-256 over canonical little-endian signed 32-bit token IDs; and
- the final committed token as an anchor; and
- a digest of the normalized sampling contract.

Hints influence ordering only after all fields match. The digest is maintained
incrementally: prompt tokens are hashed once and only newly committed output
tokens extend it. Although TokenSpeed's request state treats prompt IDs as
immutable after construction, the hook also compares its saved prompt snapshot
on every shadow seal. A changed prompt, shrinking output, or replacement
request object starts a new incarnation. Target revision and the configured
maximum candidate depth are checked separately and recorded in the row's
semantic-check vector.

The control-plane hook rejects requests with an uncommitted in-flight target
step. Their host prefix is stale by construction under overlap scheduling, so
using such a hint would make TP ordering depend on timing.

## Seal accounting

One immutable seal binding is created after the plan and retained beside its
pending execution. Each row records:

- native pre-order and selected post-order positions;
- readiness before and after advisory ordering;
- candidate identity when present;
- exactly one primary fallback reason for every selected shadow row; and
- committed token count and digest when the local execution settles.

Each seal also records `native_speculative_algorithm`. A null value identifies
the AR floor; a configured value identifies the resident speculative policy
that actually remained authoritative during shadow execution. This prevents a
future remote-plus-MTP benchmark from being mislabeled as remote-plus-AR.

The serialized fallback histogram is derived from rows and checked to sum to
the number of selected opportunities. Empty or DP-idle plans are published as
unlaunched seals rather than silently disappearing.

## Failure and resource bounds

The event-loop thread never serializes telemetry or performs endpoint I/O. A
daemon worker owns a bounded latest-value mailbox, encodes each seal, applies a
deadline and byte limit to every HTTP exchange, validates the response schema,
and atomically publishes only a strictly newer valid directive. Overflow
evicts the oldest telemetry item and increments
`transport_dropped_seals_total` on the replacement record, so an offline replay
can identify trace gaps without making the event loop wait.
Timeout, malformed response, endpoint failure, and shutdown all leave native
progress unchanged; affected selected rows report `COORDINATOR_UNAVAILABLE`
when that is the primary reason.

Only attention-TP rank zero owns the mailbox. It contains every local hook
failure, then participates in exactly one broadcast over the existing CPU TP
group at the fixed pre-plan hook point. Mirrored C++ schedulers therefore
receive the same preferred-ID vector even when rank zero's local shadow work
fails. Bind, commit, empty-plan, transport, and close hooks are also fail-open;
the native result is authoritative and is never rolled back for telemetry.

Every directive carries the exact engine ID and randomly generated engine
incarnation reported by the target. Empty identity wildcards, duplicate request
hints, and replacement updates at an already observed generation are rejected.

## Future data-plane boundary

Candidate landing, model-specific feature export, and external verification
are intentionally outside this interface. Adding any of them requires a
separate device-side operation with explicit stream ownership and capture
lifetime; it must not be smuggled through these control-plane hooks.
