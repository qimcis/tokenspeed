# Pool control

`runtime/load_balancing/pool.py` derives complete encoder, prefill and decode
worker targets from queued plus reserved work, measured per-worker service
capacity and an explicit target service interval. Units must match within each
role. It does not infer capacity from GPU count or claim synthetic performance.
Stale or regressed reports hold the current target. Distinct observations,
a deadband and cooldown constrain changes; a per-tick GPU movement bound also
covers indivisible TP groups. Allocation respects per-role minima/maxima and a
shared GPU budget. Minimum spare inventory is a hard constraint; maximum spares
and minimum workers are convergence targets during bootstrap, worker loss and
bounded resizing. Exact reachable GPU counts prevent greedy allocation from
stranding an otherwise satisfiable spare budget.

`PoolController.tick()` connects those targets to actual lifecycle operations.
`reconcile()` closes admission, waits for drain, stops a worker and only then
releases its allocation. Draining workers still own their devices and count
against physical role maxima. A reversed target can reopen a healthy draining
worker. Failed startup, lost workers, failed admission closure or stop failures
retain ownership until cleanup is confirmed. Drain timeout reports blocked;
it never kills in-flight requests or pretends to migrate decoder state.
Repeated reconciliation at the same tick cannot exceed the movement budget.
Movement counts allocation and retirement initiation; reopening admission on
a still-owned worker consumes no GPU movement. Completed controller history is
bounded to 128 workers; exited process groups are forgotten after confirmation.
Failed and active ownership is never evicted.

`GPUInventory` serializes ownership inside one process. `LocalProcessBackend`
uses the deployment's existing `ProcessGroup` to start actual local processes,
sets `CUDA_VISIBLE_DEVICES` to the reserved device IDs, cleans up partial starts
and confirms process-group exit before reuse. Process application requires an
explicit `apply=True`; disabling it rejects application rather than silently
simulating it. These are local ownership contracts, not cross-process locks or
a fleet scheduler. Do not run another allocator against the same devices.

`RegistryAdmissionControl` fences actions by worker incarnation. The shared
registry must be the sole ingress, and managed workers must be registered as
draining before launch. Admission reopens only with fresh ready/healthy reports.
Drain requires zero outstanding registry leases plus a separate fresh engine
acknowledgment, recorded after admission closure, of zero pending requests and
transfers. Zero leases alone is insufficient: cancellation can release a lease
while the engine is still finishing work. Missing observations or reincarnation
mismatches block actuation safely.

The executable local controller is covered with CPU subprocess and fake-network
lifecycle tests. A deployment still has to provide per-incarnation launch
commands and endpoints, registry registration/retirement, measured demand and
service capacity, a tick owner, and a qualified engine drain acknowledgment.
The current engine/external gateway does not supply that complete automatic
pool contract. Hardware auto-application must remain gated until those pieces
are connected and qualified. Active decode migration is not implemented.

CPU verification: `test/runtime/test_load_balancing_pool.py`,
`test/load_balancing/test_pool_adversarial.py` and
`test/deployment/test_deepseek_v41_lifecycle.py` pass together (55 tests).
Coverage includes actual subprocess cleanup, physical GPU overlap during role
swaps, failed stops, worker reincarnation, missing/stale drain acknowledgments,
oscillating demand, bounded history and indivisible allocation constraints.
