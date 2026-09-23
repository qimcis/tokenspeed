# Hopper W4A16 MoE experiment

**Timing gate: REJECT. Full-model trial: REJECT.**
A complete passing timing gate and representative memory proof are required.

No serving speedup is established. The results below describe one local-expert chain on one GPU.

## Correctness and baseline

Marlin uses gate upper clamp10 and up clamp[-10,10]; negative gate is not clamped. Uncorrected historical timings cannot establish a gain for these semantics.
Frozen numerical gate: per-row relative L2≤0.006, absolute error≤0.002+0.02×row reference maximum, finite outputs and exact zero-reference rows.

| Attempt | Worker | Verdict | Passed / total | GPU passed | Skipped |
|---|---|---|---:|---:|---:|
| validation-001 | failed | reject | 7 / 8 | 5 /9 | 0 |
| validation-002 | passed | pass | 28 / 28 | 9 /9 | 0 |

- validation-001: Validation worker did not report passed.

- validation-001: Require all 28 distinct validation cases passed; skipped and missing cases do not pass.

- validation-001: Require the exact nine GPU validation cases.

- validation-001: Require successful pytest exit and unchanged validated source.

## Complete-chain micro results

Promotion requires all six cold-cache shapes to complete, passing independent correctness and arm parity, at least one prefill median paired speedup >= 1.10, and every decode candidate/baseline median ratio <= 1.02. Hot-cache timings are secondary only.

Single-layer, single-GPU EP-partial complete-chain graphs. Primary cold-cache surrogate evicts through a persistent 256 MiB int32 buffer outside each one-call timed interval; physical L2 state is not guaranteed. Hot-cache repeats are secondary. Excludes router-score/top-k selection, communication, shared experts and model factor1.5. Does not establish end-to-end throughput.

### micro-001: REJECT

Worker status: complete. Artifact SHA256: `892fad783ee8c7d25acdb1c6ab977afff6c8c5251a574211992ba1224814af77`.

Cold-cache primary measurements:

| M | Complete | Marlin median µs | Candidate median µs | Median paired speedup | Raw paired speedups |
|---:|---|---:|---:|---:|---|
| 1 | yes | 67.872 | 61.440 | 1.103 | 1.103, 1.084, 1.123, 1.079, 1.119 |
| 2 | yes | 72.960 | 79.456 | 0.915 | 0.912, 0.915, 0.906, 0.932, 0.919 |
| 4 | yes | 81.824 | 96.288 | 0.852 | 0.859, 0.847, 0.850, 0.854, 0.852 |
| 8 | yes | 119.104 | 149.472 | 0.799 | 0.807, 0.799, 0.801, 0.795, 0.798 |
| 128 | yes | 1549.024 | 833.376 | 1.860 | 1.860, 1.859, 1.860, 1.860, 1.854 |
| 1024 | yes | 2903.232 | 950.208 | 3.053 | 3.051, 3.052, 3.054, 3.057, 3.053 |

Hot-cache secondary context (not used by the timing gate):

| M | Complete | Marlin median µs | Candidate median µs | Median paired speedup | Raw paired speedups |
|---:|---|---:|---:|---:|---|
| 1 | yes | 59.410 | 53.142 | 1.118 | 1.114, 1.118, 1.118, 1.119, 1.118 |
| 2 | yes | 64.842 | 70.830 | 0.916 | 0.917, 0.915, 0.917, 0.915, 0.916 |
| 4 | yes | 77.058 | 90.784 | 0.849 | 0.847, 0.848, 0.850, 0.849, 0.850 |
| 8 | yes | 114.395 | 144.091 | 0.794 | 0.795, 0.791, 0.794, 0.793, 0.794 |
| 128 | yes | 1545.296 | 830.128 | 1.862 | 1.862, 1.862, 1.861, 1.862, 1.862 |
| 1024 | yes | 2903.232 | 948.416 | 3.061 | 3.065, 3.056, 3.057, 3.061, 3.062 |

Workspace: 252358528 bytes; distinct persistent storage: 1154838976 bytes; scratch: 252407680 bytes.
One-layer measurement only. Full-model capacity, shared serialized scratch and single retained packed layout are unproven.

- At least one decode shape regresses by more than 2%.

## Budget and failed attempts

Prior charged: 3399.388848s. Global charged: 3439.560780 /3600s; remaining: 160.439220s. Accounting: settled.

| Attempt | Category | Status | Clean | Reserved s | Charged s | Active wall s |
|---|---|---|---|---:|---:|---:|
| attempt-001 | validation | failed | True | 40.000000 | 13.876307 | 13.876307 |
| attempt-002 | validation | complete | True | 35.000000 | 13.974516 | 13.974516 |
| attempt-003 | micro | complete | True | 40.000000 | 12.321109 | 12.321109 |

Original global charges and prior failures are retained in the hashed input ledger; no failed attempt is refunded or reset by this report.
Input ledger SHA256: `9d0487a325118d8cad65b777c9bf3d9562357f8b65a744e9ba5cf7e480cc7b13`.

Original category caps: validation40s / micro40s / serving110s.

One-time allocation seal: VERIFIED. Exactly 10 previously unallocated seconds may raise validation to50s; micro40s, serving110s and the global3600s cap stay fixed. Prior records and charges remain carried forward.

## Profile context

The earlier pre-clamp profile models +6.12% decode and +10.16% prefill throughput for 2× routed GEMMs. Removing the routed branch models +9.44% decode with shared work held fixed. These estimates are not measured candidate speedups.

Build receipt and complete raw paired timings are retained in the accompanying JSON. No incomplete, failed or skipped case is counted as a pass.
