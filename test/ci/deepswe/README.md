# DeepSWE deadline-qualified economics

Reuse `run_deepswe.sh` and the existing Pier grader to measure correct completed
tasks. The runner accepts `DEEPSWE_BASE_URL`, `DEEPSWE_MODEL`,
`DEEPSWE_MAX_CONTEXT_SIZE` and `DEEPSWE_MAX_COMPLETION_TOKENS`; existing Kimi
defaults remain available. The existing agent proxy requires an HTTP endpoint
on port 80 ending in `/v1`. Select the same task count, sample seed, model,
context and completion limits for comparisons. The runner sets `TZ=UTC` so
Pier's naive timestamps have an explicit interpretation.

The offline join needs no serving process, GPU, Pier installation, or new
evaluator. Supply the existing finalized Pier job `result.json` and a workload
and allocation manifest:

```sh
python test/ci/deepswe/economics.py result.json \
  --manifest workload-costs.json --output economics.json
```

Multiple result files may be supplied for distinct trials or retries. Do not
supply successive snapshots of the same job: duplicate `trial_name` values are
rejected. For Pier retries that replace a trial result in place, supply the
final artifact. The allocation ledger still charges all attempts.

A small illustrative manifest is:

```json
{
  "pier_timezone": "UTC",
  "tasks": [
    {
      "task_id": "task-a",
      "trial_names": ["task-a-attempt-1", "task-a-attempt-2"],
      "submitted_at": "2030-01-01T00:00:00Z",
      "deadline_at": "2030-01-01T00:30:00Z"
    },
    {
      "task_id": "task-b-not-admitted",
      "trial_names": [],
      "submitted_at": "2030-01-01T00:00:00Z",
      "deadline_at": "2030-01-01T00:30:00Z"
    }
  ],
  "allocations": [
    {
      "name": "target-allocation",
      "started_at": "2030-01-01T00:00:00Z",
      "finished_at": "2030-01-01T01:00:00Z",
      "hourly_cost_usd": 8.0
    },
    {
      "name": "draft-allocation",
      "started_at": "2030-01-01T00:00:00Z",
      "finished_at": "2030-01-01T01:00:00Z",
      "hourly_cost_usd": 1.0
    }
  ],
  "additional_costs": [{"name": "network", "cost_usd": 1.0}]
}
```

The offered task set and deadlines must come from the workload before examining
outcomes. Include unadmitted, missing, failed, cancelled, expired and unfinished
tasks. Map every result trial exactly once. Multiple attempt names may map to
one logical task; repeated success counts once. Submission times must include
waiting before admission, and retries retain the original deadline.

For a finalized trial without an exception, binary grader reward `1` establishes
correctness. Its `agent_execution.finished_at` records when the agent finished
the work. That timestamp must be on or before the workload deadline. Grading may
finish later; the full trial duration remains charged. Missing timing or
nonbinary reward on finalized results is an error. Unfinished results never
qualify. The report retains each task and attempt outcome.

Each allocation's hourly rate is the **whole allocation** price, including all
its GPUs. Use its actual paid start and end, including warm-up, idle reservation,
fallback, recovery, retries, cancellation and drain. Include worker allocations
even when they produce no useful proposals. Additional costs cover network,
task sandboxes and any other separately billed resources; avoid double-counting
items already in hourly rates. Do not substitute Pier's token/API cost estimate
or multiply cost by utilization. The report checks that recorded execution fits
inside the ledger envelope, but cannot detect omitted resources or verify a
quoted price: the recorded ledger must be complete.

The primary field is `deadline_qualified_tasks_per_usd`. Compare repeated paired
runs against independently tuned autoregressive, native speculative and local
DFlash2 baselines at the same task mix, quality and workload constraints. The
same admission and reordering freedom applies to each configuration. Deadlines
are workload inputs; no artificial token-delay floor is added. Task correctness
and completion remain the release measure; forced-length token throughput is
only a screening proxy. This join reports measurements, not a statistical
significance claim or a substitute for remote deployment qualification.

`test/runtime/test_remote_drafting_acceptance.py` exercises the real localhost
TCP transport and worker service with deterministic CPU inference: independent
cohort batching, cancellation during execution, reconnect and fresh-snapshot
equivalence to incremental history. It does not qualify the model, GPU kernels,
target scheduler integration, target sampling, WAN performance or economics.
