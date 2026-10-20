# Load-balancing implementation record

Read the [plan](deepseek-v41-b300-load-balancing.md), the
[operator guide](../guides/deepseek-v41-b300-load-balancing.md), prior
[performance record](deepseek-v41-b300-performance-log.md), and AGENTS.md first.
Continue on `deepseek-b300`; the user authorizes branch updates and forbids PRs.
CPU-only validation remains in force. GPU compilation and serving qualification
must not be inferred from CPU or Gloo tests.

## Provenance and scope

- Base implementation: `98f71fd7`.
- Prior checks: 610 C++ and 628 Python tests passed with 22 explicit skips.
- Reference: DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`, read-only.
  Its article/Q&A describes production balancing; the fleet services are not
  fully included in the open-source release.
- Existing TokenSpeed placement, byte migration, scheduler, encoder and gateway
  boundaries were reused. At the base revision V4.1/MegaMoE rejected placement;
  this increment implements the required model/backend consumption.
- Extracted attributed dispatch code preserves its existing MIT/Apache notices.
  The new trace algorithm is an original implementation of the described
  objective rather than a silently relicensed vendor copy.

## Fifteen-agent deliberation

The fifteen requested assignments ran in coordinated waves: model placement,
trace EPLB, native scheduler, prefill, decode, encoder, gateway, pool control,
configuration and telemetry, followed by expert, scheduler, routing, pool and
whole-diff reviews. Owners negotiated shared interfaces before changes. Only the
native owner changed C++/bindings; the coordinator owns final integration,
environment changes, documentation, checks, commits and publication.

## Implemented paths

| Area | Actual integration | Qualification boundary |
| --- | --- | --- |
| V4.1 placement | Target capability, logical-to-physical routing, local replica loader, actual processed FP4/scale slot views and reload handling | Eager/no-overlap/no-prefill-graph serialized-FP4 MegaMoE EP; vendor-runtime/GPU tests unrun |
| Trace EPLB | Bounded device load history, existing async worker, movement-limited policy, byte updater and complete-placement callback | Predicted objective assumes uniform replica splits; no latency measurement |
| Placement affinity | Real native rank-cost model, request generation/frontier gates and monotonic placement epoch | Supported static dispatch and coherent all-layer EPLB only |
| Prefill quota | Actual native chunk sizing and shared step budget, replay/junction/final-window/admission preservation | Calibrated estimated-work target, not a wall-clock guarantee |
| P/D/E policies | Actual shared registry reservations and fixed-path selection | Trusted token/cache/postprocessed-patch facts required |
| Routing service | HTTP streaming to complete frontends and verified compatible E/P/D paths; authenticated control/progress APIs | No arbitrary SMG per-request stage override; no durable restart reconciliation |
| Encoder observation | Existing EncodeWorker reports real cache/queue/inflight patch work through leader metrics | Unknown preprocessing geometry remains explicit |
| Pool library | Budgeted allocator and real local process actuator with reversible admission drain | Complete automatic fleet wiring requires external commands, registration, demand and independent engine-idle acknowledgments |
| Config/CLI | Per-worker expert settings, optional quota/proxy, manifest/capability checks and fit-prefill command | Example proxy has one frontend and is wiring evidence, not a routing speedup |

## Important semantic decisions

### Routed identity and baseline TP replication

MegaMoE owns EP communication, which is distinct from a generic DeepEP selector
trait. The actual selected backend capability now reports that ownership so
load snapshots reduce correctly. Existing attention-TP can submit replicated
input rows on multiple source ranks. We preserve this baseline row contract;
we do not add a new scatter/gather or claim to eliminate that duplicate work.

Exactly-once expert contribution applies to each `(source rank, token row,
top-k slot)` output. Six real two-process CPU Gloo cases covered identity,
shuffled and replicated placements with independent and replicated source rows.
Every source output matched its logical reference; replicated outputs are not
added together. This proves the CPU dispatch algebra, not GPU kernel numerics.

Counters and traces count physical routed invocations, including baseline TP
replication. They do not count unique user output tokens. Static placement-aware
batch costs count the production dispatch destinations across every replicated
source rank, preserving initial replica-table order and node topology. Dynamic
trace optimization separately uses a declared uniform-copy prediction; this
approximation must be checked against real dispatch measurements.

### Epoch and movement safety

Prepared placement IDs must be int32/int64 and valid before any mutation.
Validation of rank dispatch materialization precedes publication. Replica
coverage, slot capacity and rank topology remain fixed; stale/nonmonotonic epochs
and topology changes reject atomically. All-layer publication is required for
the placement-aware affinity/EPLB combination. Weight and scale staging retains
actual storage views, and failures before live writes cannot publish new maps.

### Cache evidence and routing

Hard concurrency, encoder patch and decode KV reservations do not discount
unpinned cache hints. Such hints may improve ranking; eviction cannot consume
unreserved hard capacity. Prefill work remains an explicitly predicted-ms budget:
calibration residuals or cache eviction can increase actual compute. Stale or
unknown prefix evidence becomes a miss, and native admission owns actual memory.

Joint E/P/D reservations are atomic across shared stage identities. Live request
ID collisions cannot adopt or release another route's leases. Retired worker
incarnations cannot be reused by idempotent retries or stale reports. Decode
progress is monotonic and replaces prior reserved work rather than double counting.

The service does not infer trustworthy preprocessing facts from opaque API bodies.
Ordinary API forwarding balances complete frontend concurrency; authenticated
`/route` applies stage policies to configured compatible paths. A frontend's
membership must be independently verified. Control credentials are stripped
before forwarding. Streaming failures do not retry already exposed output.

### Pool safety

Starting and draining workers retain their GPU allocations. Indivisible TP groups
must fit both inventory and movement bounds, including fixed minima. Failed starts
retain ownership until child cleanup is confirmed. Scale-in requires zero leases
AND a fresh, same-incarnation, post-admission-close engine report of zero requests
and transfers. HTTP disconnect alone is insufficient. Missing proof, failed stop
or timeout blocks safely; no active decoder migration is invented.

## Review corrections

- Fractional expert IDs could silently truncate; reject before conversion.
- Late placement-dispatch validation could partially mutate maps; validate first.
- Finite trace elements could overflow aggregate scores; validate aggregate range.
- Backend communication ownership was incorrectly inferred from a selector trait.
- Placement epochs could change rank count; reject without partial update.
- Request-ID collision error cleanup could release a live stream's reservations.
- Cache hints could under-reserve decode KV/encoder patches; reserve worst case.
- Sequential health probes could age a healthy frontend while peers stalled;
  use bounded concurrent refresh.
- A fixed minimum TP group larger than the movement bound could never bootstrap;
  reject that configuration.
- Mixing milliseconds and patches directly skewed path scores; normalize by the
  configured capacity for each stage.

## Verification record

Owner checks are component evidence and overlap; do not sum them:

- Native scheduler: 623 C++ tests and 273 binding/feedback checks reported passed.
- Model placement: 19 CPU cases, including six two-process source-rank proofs;
  four full-runtime/GPU cases explicitly skipped.
- Trace EPLB: 85 CPU controller, movement, algorithm and adversarial checks passed.
- Expert reviewer: 14 adversarial cases passed, including real Gloo byte movement.
- Prefill/decode/encoder: 27/33/21 owner policy checks; two full encoder-worker
  tests require unavailable vendor runtime imports.
- Routing/gateway review tested concurrent shared-stage reservations, actual local
  HTTP streaming, disconnects, live-ID collisions, stale generations and probes.
- Pool: 55 combined lifecycle/adversarial checks, including real harmless child
  startup and cleanup; no GPU application occurred.
- Central configuration: 104 owner CPU checks; root added memory-ledger coverage.
- Root added a strict fit-prefill CLI round-trip with held-out algebra fixtures.

Final coordinator checks on the frozen, formatted source:

- **945 Python tests passed, 29 skipped** across deployment, performance,
  load-balancing, native bindings, gateway/pool and selected runtime MoE suites.
  The skips are 16 endpoint checks, seven full-runtime import/opt-in checks and
  six explicit CUDA/GPU checks. These totals include prior component suites;
  do not add overlapping owner counts.
- Native library/bindings rebuilt from source; **623 C++ tests passed**.
- `pre-commit run --all-files`, staged/working diff checks and the final
  independent whole-diff review passed. Review was by available GPT-family
  agents, not a cross-family verification claim.
- The final Python suite exited cleanly. Earlier test-lifetime fixes remain in
  place; no production binding workaround was introduced.

Reproduce the combined CPU selection with the local source-built native import:

```bash
source .venv/bin/activate
PYTHONPATH=work/native-python:python pytest -q -rs \
  test/deployment test/performance test/load_balancing \
  tokenspeed-scheduler/python/tests \
  test/runtime/test_load_balancing_gateway.py \
  test/runtime/test_load_balancing_pool.py test/runtime/load_balancing \
  test/runtime/moe/test_expert_load_trace.py \
  test/runtime/moe/test_expert_rebalance_controller.py \
  test/runtime/moe/test_expert_rebalance_plan.py
cmake --build work/build-scheduler -j 4
work/build-scheduler/tokenspeed_scheduler_tests
pre-commit run --all-files
```

No GPU code was compiled/executed and no throughput, quality or cost gain was
measured. Full runtime imports remain limited by unavailable vendor Triton/kernel
dependencies on this CPU host. The new HTTP service's local tests do not qualify
an external SMG deployment or its E/P/D membership.

## Remaining deployment gates

1. Build the complete target runtime/kernel closure and run the opt-in numerical,
   loader, distributed device, stream and transfer tests on authorized hardware.
2. Supply real calibration/load observations and compare actual rank work,
   movement overhead, queue tails and task success/cost against the baseline.
3. Verify each fixed frontend path's E/P/D membership and tokenization facts;
   integrate a real engine progress observer for token-level lease updates.
4. Connect and qualify the pool's independent drain acknowledgment, exclusive
   inventory/ingress ownership, commands and endpoints, registration lifecycle,
   demand/capacity source and controller tick owner before fleet application.
5. Reconcile real engine work after controller restart; in-memory leases alone
   are not durable exactly-once accounting.
6. Keep default paths and unsupported combinations guarded until these checks
   pass. Arbitrary SMG stage composition and active decode migration remain
   outside the implemented serving contract.
