# V4.1 load-balancing candidates

The [plan](../plans/deepseek-v41-b300-load-balancing.md) and
[implementation record](../plans/deepseek-v41-b300-load-balancing-log.md) describe
this opt-in increment. It adds model expert placement, trace-aware EPLB,
placement-aware native batching, calibrated prefill quotas, request policies and
a routing service, plus a local pool-control library. GPU qualification remains
unrun; no throughput or cost improvement is claimed.

## Starting configuration

```bash
source .venv/bin/activate
PYTHONPATH=python python -m tokenspeed.deployment.deepseek_v41 plan \
  recipes/deepseek-v41-b300/load-balancing.toml
```

This validates and prints an argv/memory manifest without loading the model.
The example uses P4/D4, four additional physical expert slots **per layer across
each complete worker**, and an HTTP proxy over its existing SMG frontend.
Four total redundant slots are not four extra experts on every GPU. The example
has one complete frontend: it exercises integration, not a multi-replica routing
performance comparison. Configure multiple verified compatible frontends for
actual cross-worker balancing.

Baseline recipes remain unchanged. Explicit expert placement requires updated
source-built runtime/kernel packages; a missing processed-slot kernel API fails
before target model allocation. Native quota and placement-aware selection
require the updated scheduler. Disabled paths avoid newly added optional APIs.

## Expert placement and dynamic rebalancing

V4.1 serialized FP4 MegaMoE placement uses the generic logical/physical maps,
checkpoint loader and migration machinery. Every local physical slot is loaded
from its logical expert, including FP4 values, E8M0 scales and backend padding.
Processed tensor views share storage with the actual kernel parameters; migration
does not replace captured pointers. Draft models retain trivial placement.
The initial scope requires eager execution, disabled prefill graphs and disabled
overlap, EP greater than one and MoE TP one.

Static placement uses the existing `--init-expert-location`,
`--ep-num-redundant-experts`, and explicit dispatch controls. Dynamic rebalancing
uses `--enable-eplb`, recording and explicit rebalance bounds. The new
`--eplb-algorithm trace` additionally requires:

- `--eplb-trace-window`
- `--eplb-trace-max-moved-slots`
- `--eplb-trace-movement-penalty`
- `--eplb-trace-min-relative-improvement`
- `--eplb-trace-objective token_count` or `measured_cost`
- `--eplb-trace-expert-costs` only for measured-cost coefficients

Use separate P/D observations. The objective retains per-step co-activation and
minimizes predicted busiest-rank work, with coverage, slot capacity, movement
budget and hysteresis. Its uniform replica-split cost prediction is a heuristic,
not measured latency. See [trace EPLB](../design/trace-eplb.md).

### What the counts mean

A routed invocation is identified by source rank, token row and top-k slot.
Placement preserves one logical expert contribution for each such invocation.
Attention-TP can already submit replicated input rows to MegaMoE in the baseline.
This increment preserves those per-source outputs; it does not sum duplicated
outputs together or claim to eliminate that baseline replicated computation.
Load counters and trace SUM describe actual physical routed invocations, not
unique user output tokens. Source-rank CPU/Gloo reference tests distinguish
replicated and independent inputs; real MegaMoE GPU numerics remain untested.

## Placement-aware batching

`--expert-affinity-placement` opts the existing affinity policy into coherent
placement publication. The existing affinity restrictions remain unless the
specific combination passes capability validation. Static physical dispatch is
required; when combined with dynamic EPLB, the complete target placement is
applied as one all-layer update before the next affinity forward. Do not remove
this guard to allow partial-map estimates.

Logical route hints retain request generation/frontier identity. Placement
snapshots carry a monotonic epoch and immutable rank topology; stale updates are
rejected atomically. The native cost model uses rank-work vectors rather than
only global union size. Unknown costs/hints retain FIFO barriers, hard decode
caps and fair ready tickets. See [scheduler design](../design/scheduler.md).

## Prefill work quotas

Calibrate from actual observations with held-out data:

```bash
PYTHONPATH=python python -m tokenspeed.deployment.deepseek_v41 fit-prefill \
  "$OBSERVATIONS_JSON" > "$FIT_JSON"
```

The input has exactly `training` and `heldout` arrays of `uncached_tokens`,
`cached_prefix_tokens` and `elapsed_ms`. The model is
`a*q + b*q*c + d*q*q` in milliseconds; q includes replay and c precedes that
replay. Coefficients are finite/nonnegative and never invented by the launcher.
The fit reports held-out signed residuals and aggregate errors; it is not an
automatic promotion gate.

Set `--prefill-work-budget-ms` and all three coefficient flags together:
`--prefill-work-linear-ms-per-token`,
`--prefill-work-context-ms-per-token-squared`, and
`--prefill-work-quadratic-ms-per-token-squared`.

Native sizing remains inside the existing token cap, replay/final-window rules,
cache junction and actual admission. The shared quota charges successfully
scheduled work including replay. A tiny positive quota allows one minimum
progress chunk; it is therefore a work target, not a hard wall-clock guarantee.
The optional deployment `[prefill_quota]` forwards only to prefill-capable roles.

## Request routing

The executable service is:

```bash
python -m tokenspeed.runtime.load_balancing.service \
  --config "$ROUTING_CONFIG" --host 127.0.0.1 --port "$ROUTER_PORT"
```

An optional deployment `[routing]` launches it after workers and internal SMG.
Ordinary API requests route among healthy complete/pair frontends by reserved
concurrency. They do not invent token or patch facts from arbitrary HTTP bodies.
Authenticated `/route` accepts trusted preprocessing facts and chooses among
operator-verified fixed E/P/D paths using the actual P/D/E policies. Shared stage
worker IDs share atomic capacity; stale incarnations and live request-ID
collisions cannot free another request's leases.

Prefill scores predicted uncached/replay/transfer work, decode scores concurrency
and context/output/verification burden, and vision scores exact preprocessed
patch work. Unknown cache evidence is a miss. Unpinned cache hints may improve
ranking, but do not reduce hard decode KV or encoder patch reservations.
`prefill_work` is a budget of estimated work, not actual elapsed execution.

The adapter forwards real streamed bytes, never retries after output, and closes
upstream requests on disconnect. Closing HTTP is not evidence that device work
has stopped. Control endpoints require a configured bearer credential and must
receive trusted worker/engine reports. Their in-process history is bounded and
not durable exactly-once delivery. The complete API/security/failure contract is
in [routing design](../design/load-balancing-routing.md).

Arbitrary per-request E/P/D overrides in external SMG are not exposed by this
repo. Independent stage reservations exist as a control API; serving a selected
path requires a frontend that actually uses its configured stage members.
This is an explicit integration boundary, not an ignored option.

## Pool control

`PoolController` and `LocalProcessBackend` implement allocation, actual local
process startup, reversible drain and confirmed stop. Budgets include starting
and draining workers and spares; bounds respect indivisible TP groups.

Application requires explicit `apply=True`, a sole-owner GPU inventory and a
registry that is the sole ingress. Managed incarnations must be registered as
draining before launch. Stopping requires both zero leases and a fresh independent
engine acknowledgment of zero pending requests and transfers after admission
closure. A timeout blocks rather than killing active work.

A deployment must still connect commands/endpoints, registration/retirement,
measured demand/capacity, a controller tick owner and qualified engine drain
acknowledgments. Automatic fleet application is not enabled by the example
recipe. No active decoder-state migration is claimed. See
[pool design](../design/pool-control.md).

## Memory and validation

The manifest includes raw replicated expert values, route counters, trace ring
and previous-counter buffers, and raw one-layer expert migration value staging.
It excludes scale/padding migration staging, placement maps, host snapshots,
logical trace conversion, IPC, allocator overhead and full loading peaks. It
remains a lower bound, not a fit prediction. Inspect actual memory before any
hardware deployment.

CPU tests cover native scheduler/quota semantics, routing/lease concurrency,
source-rank reference parity, byte-wise Gloo movement, actual harmless process
lifecycle and HTTP forwarding. GPU kernel compilation, full checkpoint numerics,
real EP transfer ordering, external service composition, fleet drain evidence,
quality, fairness under real workload and performance remain hardware gates.
