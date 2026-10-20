# Trace-aware expert placement

Selecting `eplb_algorithm="trace"` keeps the existing EPLB control path: a
same-round snapshot, one spawned CPU placement worker on EP rank zero, a
broadcast commit, and ordered layer chunks that stage weight bytes before
switching routing tables. Existing DeepSeek algorithms retain their behavior.
The trace option requires explicit window, movement budget, movement penalty,
relative-improvement threshold, and objective settings.

The recorder keeps a bounded chronological window of logical expert loads per
forward. Padding is excluded by the existing live-row counters. Device-side
counter differences are recorded outside captured graphs, so graph pointers
remain stable. Snapshot processing agrees on trace dimensions, placement epoch,
and placement maps before reducing all-to-all EP observations. Dedicated prefill
and decode roles have separate histories; aggregate trace EPLB is rejected until
phase-separated recording is implemented.

The CPU objective sums the busiest rank's predicted cost for every step and
layer. Logical expert cost is split uniformly among physical replicas. This is
an approximation of replica routing, especially when static dispatch prefers a
local copy. The `token_count` objective counts routes; `measured_cost` multiplies
routes by explicit finite nonnegative per-layer/per-expert coefficients from the
configured JSON file. Neither objective is a measured serving latency.

Replica allocation considers per-step expert peaks. Greedy rank placement scores
whole step vectors and preserves existing physical slots wherever possible.
When the complete proposal exceeds the whole-update movement budget, a bounded
pass approaches it through legal replica replacements and swaps. Every accepted
intermediate map retains logical coverage and fixed per-rank slot capacity.
Stable expert/rank/slot ordering resolves ties. A per-slot movement penalty and
relative-improvement threshold reject changes that do not justify migration.
This heuristic is deterministic; it does not claim global optimality.

Before and after predicted peak sums, changed slot count, snapshot epoch, and
skip reason are logged at commit. The device updater separately reports actual
local destination bytes overwritten, including processed scales, and the actual
metadata epoch. Native placement-affinity publication occurs only after the last
chunk. Its supported EPLB combination requires a single all-layer chunk so no
forward scores a partially migrated placement through an older rank-cost model.

CPU tests exercise co-activation with identical aggregate counts, replica
allocation, capacity and coverage, stable ties, weighted costs, budget-limited
progress, hysteresis, invalid inputs, source-preserving migration plans, the
spawned worker, and the serving hooks. Multi-rank GPU transfer ordering,
processed-weight numerical parity, captured graph execution, measured dispatch
balance, and latency remain hardware qualification gates.
