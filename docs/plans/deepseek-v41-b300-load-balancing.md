# DeepSeek V4.1 B300 load balancing

## Objective and boundaries

Add expert placement/replication, trace-aware dynamic EPLB, placement-aware batch
selection, work-aware prefill/decode routing, patch-aware vision routing, and
bounded pool control to `deepseek-b300`. Existing behavior stays the default;
experimental capabilities are explicit and checked before allocation. CPU/native
checks are authorized; no GPU jobs, purchases or live fleet changes are run.
Push this branch when reviewed and verified. Never create a pull request.

This extends the [performance plan](deepseek-v41-b300-performance.md) and
[engine plan](deepseek-v41-b300.md). Maintain a separate
[implementation record](deepseek-v41-b300-load-balancing-log.md).

Primary references: DeepSeek's [deployment article](https://zhuanlan.zhihu.com/p/2095560295930508953),
[DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md),
and its [placement policy](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/policy.py).
The local reference checkout is read-only. Preserve upstream licensing if code is
adapted; prefer a small original implementation of the specified mechanisms.
Their wide-EP fleet and shared storage are not assumed to exist on eight B300s.

## Required serving contracts

1. Logical experts and router probabilities remain unchanged. Every selected
   logical expert contributes exactly once per source-rank routed invocation,
   even when physically replicated. Existing TP input replication is preserved
   and counted as physical work, not extra user output.
2. Placement consists of coherent maps, actual weight/scale slots and an epoch.
   Stage all new bytes before publishing maps; never use mixed placement epochs.
3. Weight movement is bounded and ordered with existing control operations.
   Preserve captured pointer addresses. Updates must not overlap unsafe forwards.
4. Route observations retain logical expert IDs. Placement-aware costs use the
   current epoch and invalidate stale rank estimates. Rank decisions agree.
5. Request routing reserves capacity atomically and releases exactly once on
   completion, cancellation, rejection or failure. Stale telemetry, retries and
   worker reincarnations must not double-count or silently lose work.
6. TP ranks are one worker, not independent destinations. Per-rank attention-DP
   routing is enabled only where the engine exposes and qualifies that capability.
7. Cache availability is part of predicted work. Keep locality and transfer costs
   until a genuinely shared cache makes prefix affinity unnecessary.
8. Pool changes preserve total GPU budget, min/max capacities, spare capacity and
   drain semantics. Never kill in-flight work merely to satisfy a target ratio.
9. Preserve upstream kernel/runtime boundaries, scheduler FSM/admission invariants
   and disabled-path compatibility with released dependency packages.
10. No control API is assumed from an external gateway. Inspect the actual SMG
    interface; wire supported adapters, reject unsupported integrations and
    distinguish policy/control APIs from full end-to-end serving.

## A. V4.1 expert placement and replication

Adapt the existing generic placement infrastructure to the actual V4.1/MegaMoE
path. Add model capability support only after the constructor, routing, loader,
processed-weight ownership and backend physical expert count all consume it.
The loader must fill each local physical copy from the corresponding logical
FP4 tensor and scales, including padding and all expert projections. Do not merely
remove guards or reinterpret physical IDs as logical checkpoint IDs.

Prefer static recorded placement first, with independent role profiles. Reuse
existing byte-wise expert movement and staging where their real tensor-layout
contract applies. Expose actual processed tensors to the updater; if a backend
cannot safely move them, fail explicitly rather than advertise dynamic support.
Test tiny exact CPU routes/weights, redundant copies, empty slots, remote shards,
ID consistency, scales and refresh paths. Add opt-in GPU qualification separately.

## B. Trace-aware dynamic EPLB

Use bounded per-step load vectors, not only window totals. Allocate redundant
copies and place expert slots to reduce the sum of each step's maximum rank cost.
Keep stable ties, slot capacity and logical coverage. Include movement penalty,
hysteresis and a per-update movement budget. Provide token-count and measured
cost-weighted objectives without conflating them with real latency.

Integrate with the existing async placement worker and layer-chunk updater.
Record before/after predicted peak, actual epoch, moved bytes and reason for
skipping an unprofitable change. Reject invalid/unbounded traces and incomplete
mapping publication. Keep P/D records separate. Configuration chooses the new
algorithm explicitly; the old algorithms remain unchanged.

## C. Placement-aware native batching

Extend the prior route-affinity policy to use logical supports plus an immutable
current expert-to-rank load model. Score incremental maximum rank work instead
of just global union size; incorporate per-expert work when available. Preserve
hard decode cap, fair service tickets, admission ordering, unknown/stale fallbacks
and request generation/frontier checks. Publish placement epochs coherently via
the runtime, invalidate derived costs on changes, and retain logical observations
across pure placement changes only where their identity remains valid.

Native tests cover skewed placement, co-activation, replicas, epoch changes,
continuous arrivals, stale feedback and default FIFO/union compatibility. Relax
EPLB/affinity guards only for the specifically implemented coherent combination.

## D. Work-aware prefill

Implement a strict calibrated cost model of uncached work q and existing prefix c,
with monotonic nonnegative coefficients and explicit units. Fit offline and report
held-out residuals. Runtime uses the model to allocate work, not to claim speed.
Route among ready workers by reserved remaining work plus miss/transfer costs,
with optional context tiers and starvation-safe admission.

Add budget-derived chunk sizing inside the native scheduler. Solve within the
existing token cap, replay-window minimum, final-window retention, cache junction
and actual admission feasibility. Charge replay work and consume the shared step
budget across requests. Tiny budgets need a defined progress fallback or clear
configuration rejection; they must not create a permanent queue stall.

## E. Decode and vision routing

Decode selection balances active request count and predicted attention/indexer
burden from contexts, plus verification work/capacity where exposed. Choose a
complete worker unless true attention DP is supported. A lease reserves both
concurrency and context burden before dispatch. Account unknown output length
conservatively, then update with observed progress. No mid-stream replay/retry of
already exposed output without an explicit resumable transfer contract.

Vision selection uses post-preprocessing patch counts and known embedding-cache
availability rather than image count. Preserve image ordering/identity and bound
queued patch work. Provide deterministic tie behavior and atomic reservations.
The existing encoder batcher remains the execution path.

## F. Routing control plane and pool control

Implement shared typed worker state, incarnation/epoch, health, freshness and
reservation contracts consumed by P/D/E policies. Integrate an executable routing
adapter with the existing gateway surface when supported. Any external-gateway
extension absent from this repo must be named explicitly; no hidden no-op mode.
Local CPU HTTP/fake-transport tests exercise actual adapter behavior without GPUs.

Pool targets derive from queued/reserved work and measured service capacity.
Enforce GPU budget, min/max sizes, hysteresis, bounded changes and spare inventory.
Use a reversible drain/start/stop state machine: stop admission, finish or safely
transfer requests, then release workers. Existing active-decode migration must
be used only if its full state transfer contract exists; otherwise draining waits
or reports blocked instead of pretending migration succeeded. Test stale reports,
worker loss, oscillating demand, failed starts, cancellation and retry idempotency.

## Configuration, observations and rollout

Use narrow explicit opt-ins and immutable identities. Keep baseline recipes as
controls, add experimental examples with no fabricated calibrated coefficients
represented as measurements. Report resource costs for all E/P/D roles and spare
capacity. Emit bounded diagnostics using existing metrics interfaces when possible.
No default promotion follows from lower synthetic load scores.

## Fifteen-agent ownership

| Agent | Scope | Main peers |
| --- | --- | --- |
| lb_moe_runtime | V4.1 physical expert routing/loader/backend integration | lb_eplb, lb_native, lb_expert_review |
| lb_eplb | Trace-aware placement algorithm and dynamic updater integration | lb_moe_runtime, lb_telemetry |
| lb_native | All C++/binding changes: placement-aware affinity and prefill quota | lb_prefill, lb_moe_runtime, lb_scheduler_review |
| lb_prefill | Cost models, prefill policy and native quota interface | lb_native, lb_gateway |
| lb_decode | Decode policy, load updates and lease accounting use | lb_gateway, lb_routing_review |
| lb_encoder | Patch-aware encoder policy and execution/telemetry hooks | lb_gateway, lb_telemetry |
| lb_gateway | Shared worker/lease contracts and executable gateway adapter | all stage-policy owners |
| lb_pool | Bounded pool allocation and drain/actuation state machine | lb_gateway, lb_integration, lb_pool_review |
| lb_integration | Central flags, compatibility and deployment/CLI wiring | all owners |
| lb_telemetry | Bounded work/load observations and evidence tests | lb_eplb, lb_encoder, lb_gateway |
| lb_expert_review | Expert/replica/lifecycle review and adversarial tests | model and EPLB owners |
| lb_scheduler_review | Native fairness/quota/epoch review and tests | native/prefill owners |
| lb_routing_review | Routing/reservation/failure review and tests | gateway/decode/encoder |
| lb_pool_review | Pool lifecycle/resource/measurement review and tests | pool/integration |
| lb_final_review | Independent whole-diff review and documentation audit | all owners |

At most ten workers run together, so the fifteen agents work in coordinated waves.
Agents deliberate directly before changing shared interfaces. Only the native
owner edits C++/bindings. Only integration edits central server/deployment flags.
The coordinator owns final documentation, environment changes, combined checks,
commits, exports and branch push. Agents never install, run GPUs, commit, push or
open PRs independently.

## Acceptance

Deliver implemented paths and executable tests, not just design-only helpers.
Record exact capability gaps rather than removing protective guards blindly.
Run the complete native scheduler suite, Python binding suite, affected CPU
runtime/contract tests and repository hooks. Preserve unrelated behavior and
source-build requirement checks. GPU compilation, real transfers, numerical
parity, model quality, fairness under real load and cost/performance gains remain
separate unrun hardware gates. No exact delivery estimates or dates belong here.
