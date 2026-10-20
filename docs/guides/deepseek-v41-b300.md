# DeepSeek V4.1 Flash on B300

This branch specializes the existing TokenSpeed execution path for
`deepseek-ai/DeepSeek-V4.1-Flash-1017` with aggregated, prefill/decode (PD), and
encoder/prefill/decode (EPD) deployment recipes. E is the vision encoder and
aligner; it is not the language model's internal causal encoder. Text requests
bypass E.

The current qualification is CPU-only. Recipe validation, analytical estimates,
and protocol tests do not establish model quality, memory fit, or B300 speed.
The [implementation plan](../plans/deepseek-v41-b300.md) records the full objective
and acceptance gates. The [work log](../plans/deepseek-v41-b300-work-log.md)
records what is implemented, checked, or still deferred.

## Validate a recipe without GPUs

Run from the repository root with the local environment activated:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m tokenspeed.deployment.deepseek_v41 --help
python -m tokenspeed.deployment.deepseek_v41 plan recipes/deepseek-v41-b300/aggregate.toml
python -m tokenspeed.deployment.deepseek_v41 plan recipes/deepseek-v41-b300/pd.toml
python -m tokenspeed.deployment.deepseek_v41 plan recipes/deepseek-v41-b300/epd.toml
python -m pytest -q test/deployment
```

The CPU tooling requires Pydantic and NumPy; the HTTP workload tooling also uses
requests. It does not require importing CUDA, downloading weights, or starting a
server. `plan` emits argv vectors, environment overrides, the effective recipe,
and a memory lower bound. The snapshot placeholder in this output is intentional;
`launch` resolves the pinned checkpoint before constructing executable commands.

| Recipe | Allocation | Meaning |
| --- | --- | --- |
| `aggregate.toml` | Eight GPUs for one model worker | Reference topology |
| `pd.toml` | Four P GPUs and four D GPUs | Isolate P/D disaggregation |
| `epd.toml` | Two E GPUs, two P GPUs, four D GPUs | Dedicated encoder candidate |

These are candidate allocations within eight total GPUs, not a prediction that
all roles fit or that E2/P2/D4 is optimal. Each recipe explicitly sets speculation,
Engram placement, admission limits, transport storage, and metrics reporting.
`[metrics]` explicitly selects `jsonl` or `none` and a bounded queue size. The
transport budget includes encoder GPU/host caches as well as send/receive pools.
Edit a copy when
comparing configurations; retain the original and generated manifest as evidence.

The initial recipe API requires attention TP to equal each worker's GPU count.
It rejects overlapping GPU assignments, unsupported worker sizes, excess total
allocations, incompatible E/P TP, unknown fields, and inconsistent capacities.
It does not expose an unrestricted command-line escape hatch. Existing runtime
constraints still apply: PD decode KVP and V4.1 Engram/SWA tree speculation remain
unsupported. Do not treat a successful recipe parse as validation of those paths.

## Interpret the memory ledger

`memory.status` explicitly identifies an analytical lower bound. It accounts for
known routed-expert value storage, Engram codes and scales, logical global cache,
retained SWA and compressor state, speculative windows, and embedding pools under
the recipe's placement assumptions. Host-resident Engram is counted for each
independent language-model worker; P and D do not automatically share one table.

The ledger excludes expert scales and padding, other model weights, physical
cache packing, allocator slack, workspaces, communication buffers, and loading
peaks. Its exclusions travel with the output. Values below device capacity do not
prove fit. A hardware launch can reject a configuration whose known payload
already exceeds its budget, but only measured per-rank allocation can close the
capacity gate.

For a local checkpoint snapshot, inspect tensor metadata without allocating model
weights:

```bash
python -m tokenspeed.deployment.deepseek_v41 checkpoint "$CHECKPOINT_SNAPSHOT"
```

The checkpoint audit reports stored tensor payloads. It does not calculate the
runtime replication, kernel conversion, or allocator peaks of a deployment.

## Reproducibility and hardware launch

The recipes pin the checkpoint revision. The implementation plan records the fork
base and upstream revision. Launch captures the source commit and dirty state,
installed package versions, GPU inventory, effective arguments, model revision,
and hashes of small checkpoint artifacts in `manifest.json`. It also writes
per-role metrics configuration when JSONL reporting is selected. Preserve these artifacts with each experiment.
This is provenance capture, not a dependency lock or container build: pin the
runtime dependency closure and container digest separately before comparing
hardware results. Do not substitute a moving model name for the pinned snapshot.

Hardware execution is a separate, explicit action and has not been performed for
this work. On a suitably provisioned Linux B300 host, install the supported
TokenSpeed runtime, kernel, scheduler, Mooncake, gateway, and model dependencies
using the repository's [installation guidance](getting-started.md). Set the
startup and shutdown bounds appropriate to that environment, then run:

```bash
python -m tokenspeed.deployment.deepseek_v41 launch \
  recipes/deepseek-v41-b300/epd.toml \
  --output "$RUN_DIRECTORY" \
  --startup-timeout "$STARTUP_BOUND_SECONDS" \
  --shutdown-grace "$SHUTDOWN_BOUND_SECONDS"
```

The output directory must be new. The launcher checks the assigned hardware and
known memory payloads, resolves the checkpoint, checks its geometry, starts owned
worker process groups, waits for workers and gateway readiness, and writes
per-role logs. Shutdown cleans up its owned process groups. The gateway binds to
loopback; make external access an explicit deployment decision. The launcher
removes inherited Mooncake intra-node forcing flags because the generic EPD
shortcut is not a qualified V4.1 PD transport setting.

A successful startup is only a prerequisite for qualification. Compare the same
text, image, prefix-reuse, chunk-boundary, cancellation, recovery, and speculative
workloads across aggregate, PD, and EPD. Qualify the configured context limit;
the architectural context maximum is not an operational acceptance result.

## Opt-in endpoint checks

The endpoint smoke suite skips unless an endpoint is explicitly configured. It
does not launch workers, download weights, or claim access to GPUs. After a
separately authorized hardware deployment is ready:

```bash
DEEPSEEK_V41_EPD_BASE_URL="http://HOST:18000/v1" \
  python -m pytest test/deployment/test_deepseek_v41_epd.py -v
```

`DEEPSEEK_V41_EPD_MODEL` optionally selects a served name; its default is
`deepseek-v41-flash`. `DEEPSEEK_V41_EPD_API_KEY` optionally supplies authentication.
The suite covers text, color/OCR images, an image-last prompt, image replacement
and ordering, repeated/common-prefix requests, concurrent text/image requests,
streaming, and reasoning/tool output. The tests exercise request shapes that
need correct image and Engram handling. CPU wire-validation changes preserve the
existing frame layout and reject malformed flags, fanout, and destination
pointers; see the [embedding transfer contract](../design/epd-wire.md). This
contract also describes a sender barrier that drains submitted synchronous
writes before publishing terminal status, preventing early source-slot reuse
when one fanout write fails. It does not add peer revision negotiation or prove
GPU-visible completion, external cancellation quiescence, or CUDA event recovery. They do not prove actual encoder bypass,
cache hits, quality parity, speculative rollback, or worker-failure recovery.

## Metrics evidence

When the recipe selects the JSONL reporter, each process writes a separate
`metrics-<pid>-<uuid>.jsonl` file in its generated role output directory. Records
preserve metric name, operation (`inc`, `set`, or `observe`), value, labels, and
weight. This is a metrics stream, not task-quality or HTTP workload evidence.
See [metrics plugins](../metrics-plugins.md) for the reporter configuration.

The writer queue is bounded by record count and drops new records when full.
Warnings and writer statistics expose loss and errors. Forced termination can
lose queued records, and invalid/nonfinite records or file failures stop the
affected writer. Inspect these signals before using a trace for comparison;
bounded queueing does not imply bounded disk usage or complete capture.

## Measure requests and join independent task grades

`measure` is an explicit HTTP action against a supplied full chat-completions
URL. It runs one request per workload row with bounded concurrency. The workload
JSON contains `workload_id`, `model`, `sampling_id`, and `requests`; each request
contains `task_id`, `cluster_id`, `messages`, an explicit `sampling` object, and
`stream`. The runner preserves the supplied messages and sampling settings.
It does not supply benchmark prompts or choose generation policy.

```bash
python -m tokenspeed.deployment.deepseek_v41 measure "$REQUESTS_JSON" \
  --endpoint "$CHAT_COMPLETIONS_URL" --concurrency "$CONCURRENCY" \
  --connect-timeout "$CONNECT_BOUND_SECONDS" \
  --request-timeout "$REQUEST_BOUND_SECONDS" \
  --max-response-bytes "$RESPONSE_BYTE_LIMIT" > "$MEASUREMENTS_JSON"
python -m tokenspeed.deployment.deepseek_v41 grade \
  "$MEASUREMENTS_JSON" "$EVALUATIONS_JSON" --costs "$COSTS_JSON" \
  --mode epd --model-revision "$MODEL_REVISION" > "$EXPERIMENT_JSON"
```

Set `TOKENSPEED_BENCH_API_KEY` only if the endpoint requires authentication. The
runner neither follows redirects nor retries requests. Raw output retains every
request, HTTP/transport/protocol status, queue and dispatch latency, response or
SSE events, streaming first-output latency where observable, and completion-token
usage where supplied. Nonstreaming requests have no fabricated TTFT. Missing
admission or token information stays unknown. Socket inactivity is bounded and
elapsed deadlines are checked between reads; OS DNS and blocking network calls
are not a hard real-time cancellation guarantee.

An HTTP success is not a task-quality grade. Provide exactly one independent
`Evaluation` for every measured task, with `task_id`, `success`, `admitted`, and
`committed_tokens`. Supply `COSTS_JSON` as the complete `ResourceCost` array,
including all compute roles, host, network, and storage. The join rejects missing
or contradictory outcomes and produces an `Experiment` for `compare`; caller
supplied revision and costs must describe the actual run. Joined task latency
includes the runner queue. This closed-loop driver does not execute multi-step
agent trajectories or simulate an offered arrival-rate process.

## Offline experiment analysis

The CLI consumes JSON using strict typed schemas in
`python/tokenspeed/deployment/deepseek_v41/analysis.py`. The tests under
`test/deployment/` provide small behavioral fixtures. Keep measured data separate
from synthetic test fixtures.

| Command | Input | Interpretation |
| --- | --- | --- |
| `compare` | Baseline and candidate `Experiment` objects | Paired quality and whole-system cost evidence |
| `calibrate` | Object with `training` and `held_out` arrays of `Step` objects | Identifiable per-phase linear cost screen |
| `replay` | Array of `PipelineRequest` objects | FCFS queue screen with explicit E/P/D capacities |
| `route-oracle` | Token rows, each containing layers of six distinct expert IDs | Exact small-window future-route union bound |
| `locality` | Ordered nonnegative row IDs | LRU hit curves at requested capacities |
| `cache-economics` | One `CacheReuseEstimate` object | Serial remote cache-hit cost versus recomputation |

```bash
python -m tokenspeed.deployment.deepseek_v41 compare \
  "$BASELINE_JSON" "$CANDIDATE_JSON" \
  --margin "$QUALITY_MARGIN" --confidence "$CONFIDENCE" \
  --resamples "$RESAMPLES" --seed "$SEED"
python -m tokenspeed.deployment.deepseek_v41 calibrate \
  "$OBSERVATIONS_JSON" --tolerance "$RELATIVE_ERROR_TOLERANCE"
python -m tokenspeed.deployment.deepseek_v41 replay \
  "$WORKLOAD_JSON" --encode-workers 1 --prefill-workers 1 --decode-workers 1
python -m tokenspeed.deployment.deepseek_v41 route-oracle \
  "$ROUTES_JSON" --batch-size 2
python -m tokenspeed.deployment.deepseek_v41 locality \
  "$ROW_ACCESSES_JSON" --capacities 0 1024 4096
python -m tokenspeed.deployment.deepseek_v41 cache-economics "$CACHE_ESTIMATE_JSON"
```

An `Experiment` must contain every offered task, including rejections and failures,
with retries aggregated into one outcome. Itemize every compute role plus host,
network, and storage, including zero-cost entries where appropriate. Include idle
allocations and failed work in total cost. Count only committed output tokens.
Paired experiments must match task population, workload, checkpoint, sampling,
and conversation cluster assignments.

Quality comparison combines paired cluster bootstrap diagnostics with a
conservative finite-sample bound. A small sample with identical observed outcomes
does not establish non-inferiority. Cost uncertainty is conditional on supplied
total costs; it does not estimate variation in resource cost across runs. The confidence
level applies per gate, not to a joint release decision. Neither quality nor cost
gates grant release approval or replace service constraints.

Calibration rejects unidentified or negative-cost fits and flags feature values
outside observed coordinate ranges. This check does not detect every distribution
shift; callers must supply genuinely held-out observations.
FCFS replay has no batching or admission model. The route oracle uses future
information and reports union reduction, not measured speedup; it is not an
online scheduling policy. LRU curves do not establish the cost or benefit of an
Engram GPU cache. Each screen exists to decide whether a hardware experiment is
worth performing.

`CacheReuseEstimate` requires physical transfer bytes, effective bandwidth, remote
lookup/restore/replay/queue costs, local recomputation/queue costs, and independent
remote/local monetary costs. Its serial cache-hit screen excludes misses, overlap,
and fleet allocation effects; it establishes no universal token-length threshold.

## Model integration correction

The decode frontend preserves the same physical image-padding token positions as
prefill when language-model-only processing is selected. Authored token IDs remain
separate, and decode does not retain vision features. This aligns prefix and
retained-state positions across the P/D boundary. The
[model audit](../plans/deepseek-v41-b300-model-audit.md) records source evidence,
regression coverage, and remaining restrictions.

Prefill still allocates the vision module even when consuming remote encoder
output. Avoid using `language_model_only` on EPD prefill to remove that allocation:
it would also remove required multimodal participation. A consumer-only vision
capability is a separate follow-up requiring runtime validation.

## Work remaining after CPU checks

The CPU work supplies deployment and evidence tooling plus targeted runtime
correctness hardening. It does not implement every conditional optimization in
the plan. B300 numerical correctness, real transfer completion, cancellation and
recovery, per-rank peak memory, model quality, performance, and cost acceptance
remain hardware gates. Expert-aware online scheduling, Engram hot-row caching,
new B300 DSpark kernels, remote cache policy, and tree speculation require their
own evidence and correctness gates before implementation or promotion.
