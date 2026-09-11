# Remote DFlash2 for GLM-5.3

This recipe describes the remote-drafting implementation and its qualification
configuration. **GPU correctness, recovery and goodput-per-dollar qualification
are pending.** Building an image, loading weights or passing CPU tests does not
establish a deployment result.

The target serves full `zai-org/GLM-5.3` FP8 through the existing `tokenspeed
serve` API. An independent TP1 worker runs `incoai/GLM-5.3-DFlash2` in BF16 and
batches proposals from the target's attention cohorts. The worker proposes from
confirmed prefixes while the target services other requests. All sampling,
acceptance and visible output remain on the target.

## Configuration

| Component | Configuration |
| --- | --- |
| Target | One NVLink-connected node with eight B200 GPUs |
| Attention and dense layers | TP1 / DPA8 and TP1 / DP8 |
| Experts | EP8 / TP1 on the same eight GPUs |
| Pipeline and context parallelism | PP1 / CP1 |
| Draft worker | One independent TP1 GPU; an SM86 build for A10 |
| Checkpoints | Full GLM-5.3 FP8 plus matching DFlash2 BF16, both pinned to immutable revisions |
| Draft/target geometry | Native draft block 8; target verification 6; fallback 1 |

DPA8 is not eight independent GLM replicas: expert execution is shared. Every
cohort participates in the post-plan width exchange. Uniform active widths can
use the matching graph variant; mixed width-one/width-six cohorts use eager
execution on all ranks. Idle cohorts join the same decision.

DFlash2 runs its full eight-position block and selector. The worker returns
seven proposals and the target consumes the first five plus the explicit
anchor. Do not edit the checkpoint's block size to six. Missing proposals use
real width-one execution; the target retains a projector and feature cache,
not a complete local drafter for fallback.

## Build both processes from the same source

From one pinned TokenSpeed checkout, build separate NVIDIA images for the two
architectures:

```bash
docker build --file docker/Dockerfile.nvidia \
  --build-arg CUDA_ARCH_LIST='10.0a' \
  --tag tokenspeed-remote:target .

docker build --file docker/Dockerfile.nvidia \
  --build-arg CUDA_ARCH_LIST='8.6' \
  --tag tokenspeed-remote:worker .
```

The build argument selects the Torch and FlashInfer architecture lists. Use the
same source revision for both images, and expose the appropriate NVIDIA GPUs,
checkpoint cache and private network when running them. The worker's portable
BF16 GQA, convolution and selector paths must pass a native-eight batched
forward and context/KV comparison on SM86. Do not copy the target's Blackwell
attention or FP8 overrides into the worker environment.

Set `TARGET_REVISION` and `DRAFT_REVISION` to the selected 40-character
checkpoint commit IDs in both processes. Branch names such as `main` are
rejected. Local snapshot paths may replace the model IDs, while retaining the
matching immutable revision arguments. The worker loads only the target's
matching embedding/head tensors; it does not construct the full target model.

Run the worker on an existing authenticated private network. The TCP address
validator is not authentication. The worker endpoint is a private protocol,
not an OpenAI-compatible API and not a target NCCL/Gloo rank.

## Launch the worker

Set `DRAFT_LISTEN_ADDRESS` to its private bind address, for example
`tcp://10.0.0.20:5557` on a private network where that address belongs to the
worker. Set `DRAFT_MAX_RESIDENT_SESSIONS` and `DRAFT_MAX_BATCH_SIZE` from the
available worker memory and measured service rate. Batch size cannot exceed
resident-session capacity.

Inside the SM86 worker image:

```bash
tokenspeed draft-worker \
  --target-model-path zai-org/GLM-5.3 \
  --target-revision "${TARGET_REVISION:?Set the target checkpoint commit}" \
  --draft-model-path incoai/GLM-5.3-DFlash2 \
  --draft-revision "${DRAFT_REVISION:?Set the draft checkpoint commit}" \
  --listen-address "${DRAFT_LISTEN_ADDRESS:?Set the private bind endpoint}" \
  --device-id 0 \
  --max-resident-sessions "${DRAFT_MAX_RESIDENT_SESSIONS:?Set the resident capacity}" \
  --max-batch-size "${DRAFT_MAX_BATCH_SIZE:?Set the worker batch limit}" \
  --staging-slots "${DRAFT_MAX_BATCH_SIZE:?Set the worker batch limit}" \
  --max-queued-jobs 64 \
  --max-peers 8 \
  --max-header-bytes 65536 \
  --max-feature-bytes 25165824 \
  --max-host-memory-bytes 1073741824 \
  --lease-ms 30000 \
  --heartbeat-ms 1000 \
  --poll-ms 10 \
  --linger-ms 1000
```

Eight peer slots serve the eight cohort leaders in this recipe. Staging slots
must be at least the batch limit and at most the resident-session limit. The
remaining limits are explicit starting values, not a worker-capacity
measurement. Keep heartbeat shorter than the finite session lease. The
configured feature-frame limit cannot exceed 24 MiB, and the service validates
its conservative host memory budget before accepting work; larger resident or
staging limits may require a larger budget. Worker KV is bounded independently of
target KV; weight fit alone does not establish that the selected resident and
batch limits fit alongside workspace.

## Launch the target

Set `REMOTE_DRAFT_ENDPOINT` to the worker's reachable `tcp://host:port` address.
Set `REMOTE_DRAFT_MIN_READY` and `REMOTE_DRAFT_MAX_DEFER_MS` explicitly from the
workload being evaluated. The minimum ready count must be positive; maximum
deferral must be finite and nonnegative. Neither setting imposes a mandatory
service cadence, and the target continues width-one work when a remote worker
is unavailable.

Inside the target image:

```bash
tokenspeed serve zai-org/GLM-5.3 \
  --revision "${TARGET_REVISION:?Set the target checkpoint commit}" \
  --served-model-name glm-5.3 \
  --trust-remote-code \
  --world-size 8 \
  --nprocs-per-node 8 \
  --attn-tp-size 1 \
  --dense-tp-size 1 \
  --moe-tp-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --moe-backend flashinfer_trtllm \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/GLM-5.3-DFlash2 \
  --speculative-draft-model-revision "${DRAFT_REVISION:?Set the draft checkpoint commit}" \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --speculative-num-steps 7 \
  --speculative-verify-tokens 6 \
  --remote-draft-endpoint "${REMOTE_DRAFT_ENDPOINT:?Set the private worker endpoint}" \
  --remote-draft-min-ready "${REMOTE_DRAFT_MIN_READY:?Set the calibrated minimum ready count}" \
  --remote-draft-max-defer-ms "${REMOTE_DRAFT_MAX_DEFER_MS:?Set the workload deferral bound}" \
  --max-model-len 1048576 \
  --chunked-prefill-size 4096 \
  --max-num-seqs 128 \
  --host 0.0.0.0 \
  --port 8000
```

The context limit is 1,048,576 total prompt and generated tokens, matching the
checkpoint's advertised 1M context and `max_position_embeddings`. Prefill chunk
size and admission count above are qualification starting points, not measured
capacity recommendations. Actual admission remains subject to the cache budget.
Keep the existing
frontend's health/readiness, request, cancellation and streaming interfaces.
Do not send application requests directly to the draft worker.

For an attention-layout comparison on the same eight-GPU node, change attention
and dense TP to four and data parallelism to two; retain EP8/TP1, PP1 and CP1.
Report the same global admitted workload, not an unchanged per-cohort batch
that silently changes total concurrency. Count the TP4 projector/feature
replication in the comparison.

## Limits and recovery

Remote mode is limited to this full GLM/DFlash2 pair and CUDA execution. It
rejects PP greater than one, attention CP, PD/EPD disaggregation and
`--dp-sampling`. Attention data parallelism is supported. Hybrid expert TP/EP,
ragged per-request widths within one cohort batch and target width eight are
outside this configuration. The initial service uses one static worker
endpoint; it does not provide automatic fleet routing or migration.

Projected BF16 features use 12 KiB per context token; a bounded initial window
is approximately 24 MiB. Only the cohort leader exports. Worker KV, projected
feature storage, resident sessions, queued frames and staging all consume
memory in addition to the weights. Session and staging admission precede
snapshot allocation. Tune limits from measured memory and on-time supply,
not from checkpoint file size alone. Features use separate cache parents over
aliased target storage planes; this preserves parent size but still consumes
cache capacity. Include packed-page rounding and gather/scatter address scratch
in the target memory budget, rather than charging only the BF16 wire payload.

A missing, busy, late or disconnected worker cannot commit tokens or stall
target progress. A matching reply is usable only before planning its forward.
Healthy fallback does not automatically discard installed worker context:
valid context acknowledgements still permit contiguous updates when the
associated proposal is obsolete. Lost state, retraction or reset requires a
fresh session and a retained-history snapshot. Cancellation invalidates work
without prematurely reusing export buffers; finite session leases reclaim
orphaned worker state.

## Qualification

Before claiming support, exercise native-eight drafting with target-six
verification and alternating width-one fallback, rejection, stopping, prefix
reuse, page/window boundaries, cache restore and request-slot reuse. Run actual
eight-cohort GLM EP8 forwards for all-six, all-one, mixed one/six, one active
cohort and idle cohorts, with eager and graph decisions checked. Interrupt the
worker during concurrent multi-turn requests and verify bounded recovery.
Another accelerator can establish functional behavior only for the hardware
actually tested; it cannot establish B200/A10 performance or economics.

The worker's opt-in checkpoint test covers native-eight batched inference and
cold-snapshot versus incremental context KV. From the source checkout on the
intended worker GPU, with test dependencies installed:

```bash
TOKENSPEED_TEST_DRAFT_WORKER_CUDA=1 \
TOKENSPEED_TEST_TARGET_CHECKPOINT=zai-org/GLM-5.3 \
TOKENSPEED_TEST_TARGET_REVISION="${TARGET_REVISION:?Set the target checkpoint commit}" \
TOKENSPEED_TEST_DRAFT_CHECKPOINT=incoai/GLM-5.3-DFlash2 \
TOKENSPEED_TEST_DRAFT_REVISION="${DRAFT_REVISION:?Set the draft checkpoint commit}" \
PYTHONPATH=python \
python -m pytest -q test/runtime/draft_pool/test_worker_cuda.py
```

This test is skipped unless explicitly enabled. A pass covers the worker test
cases only; target DPA/EP execution, transport recovery and economics still
need the deployment qualification above.

Compare against the best valid local policy, including AR, native MTP and local
DFlash2. Let each system tune batching and concurrency under the same request
arrivals, required work, task mix and completion deadlines. Charge target and
worker allocations, idle time, projection, feature memory/export, network,
warm-up, drain and fallback. Include the feature consumer's pre-forward L2
restore fence and any overlap it removes. Uniform-width graph savings and
mixed-width eager costs are part of the whole target batch, not per-row credits.

The acceptance metric excludes the bonus token. Token throughput is a systems
proxy; economic qualification requires more correct, deadline-qualified work
per total dollar in repeated paired trials beyond measurement uncertainty.
No throughput, recovery or cost advantage is claimed by this recipe.

See the design contracts for [scheduling](../design/scheduler.md),
[event-loop ownership](../design/event-loop.md),
[feature history](../design/cache-concepts.md#projected-feature-history-is-a-cache-group)
and [execution geometry](../design/unified_path.md#native-draft-width-and-active-target-width).
