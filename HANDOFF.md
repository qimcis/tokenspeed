# Handoff: cross-request encoded feature cache for VLMs

## What was implemented

- New cache class in `python/tokenspeed/runtime/multimodal/feature_cache.py`
  - `EncodedFeatureCache`: thread-safe, byte-bounded LRU cache keyed by `(modality, content_hash)`.
  - Stores `encoded` and optional `encoded_deepstack` tensors.

- New env var in `python/tokenspeed/runtime/utils/env.py`
  - `TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES` (default `0` = disabled).

- Integration in `python/tokenspeed/runtime/multimodal/embedder.py`
  - `_plan`: cross-request cache lookup before within-batch dedup; on hit, sets `item.encoded`/`encoded_deepstack` and drops the raw feature.
  - `_encode`: after encoding, detaches tensors from shared/reusable buffers via `_detach_from_shared_storage`, then puts them in the cache.
  - `apply`: logs per-call `cache_hits`/`cache_misses`/`cache_bytes` when `TOKENSPEED_LOG_MM_TIMING=1`.

## Branch

- `origin/cross-request-encoded-feature-cache`
- Worktree: `/Users/chimcisaac/tokenspeed-worktrees/deep-dive-encoder-vlm`

## Benchmarks added

- `test/runtime/benchmark/bench_encoded_feature_cache.py` — isolated cache microbenchmark.
- `test/runtime/benchmark/bench_vision_embedder_cache.py` — end-to-end `VisionEmbedder` benchmark that works on both `main` and this branch.

## How to enable

```bash
export TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES=1073741824
```

## Verification done

- `python3 -m py_compile` passed for all changed/new files.
- `EncodedFeatureCache` isolated unit test passed.
- `VisionEmbedder` end-to-end run was **not** completed on the local mac because `triton`/CUDA are unavailable; the benchmark script is ready for a proper TokenSpeed GPU environment.

## Suggested end-to-end benchmark

### Kimi-K2.5 (known supported VLM)

```bash
# Baseline
python3 -m smg_grpc_servicer.tokenspeed \
  --model /models/Kimi-K2.5-NVFP4 \
  --served-model-name kimi-k25-nvfp4 \
  --trust-remote-code \
  --tp 8 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-prefill-tokens 32768 \
  --chunked-prefill-size 32768 \
  --gpu-memory-utilization 0.85 \
  --quantization nvfp4 \
  --kv-cache-dtype fp8 \
  --enable-mixed-batch

# With cache
TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES=1073741824 \
TOKENSPEED_LOG_MM_TIMING=1 \
python3 -m smg_grpc_servicer.tokenspeed \
  --model /models/Kimi-K2.5-NVFP4 \
  --served-model-name kimi-k25-nvfp4 \
  --trust-remote-code \
  --tp 8 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-prefill-tokens 32768 \
  --chunked-prefill-size 32768 \
  --gpu-memory-utilization 0.85 \
  --quantization nvfp4 \
  --kv-cache-dtype fp8 \
  --enable-mixed-batch
```

### Kimi 2.7

Kimi 2.7 is **not** in this branch's model registry yet. Try the command below on the remote machine; if it loads, the cache path will be exercised as long as the model uses the standard multimodal `image_encoder` + `VisionEmbedder` flow. If it fails with "model architecture not supported", a new model adapter is needed.

```bash
TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES=1073741824 \
TOKENSPEED_LOG_MM_TIMING=1 \
python3 -m smg_grpc_servicer.tokenspeed \
  --model <remote_path_to_Kimi_2.7> \
  --served-model-name kimi-27 \
  --trust-remote-code \
  --tp 8 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-prefill-tokens 32768 \
  --chunked-prefill-size 32768 \
  --gpu-memory-utilization 0.85 \
  --quantization nvfp4 \
  --kv-cache-dtype fp8 \
  --enable-mixed-batch
```

### Client

`tokenspeed bench serve` is text-only. For a repeated-image workload, use a small OpenAI-client script that posts the same image URL repeatedly to `/v1/chat/completions`.

### Headline metrics

- Median / p99 TTFT and throughput.
- `mm_timing` log lines: confirm `cache_hits > 0` for repeated images.

## Next steps for the next agent

1. Run `bench_vision_embedder_cache.py` on a GPU cluster to collect PR numbers (baseline `--cache-bytes 0` vs `--cache-bytes <budget>`).
2. Optionally add `VisionEmbedder` unit tests with a dummy encoder in `test/runtime/`.
3. Review the `_detach_from_shared_storage` heuristic for CUDA-graph output edge cases.
4. Confirm Kimi 2.7 model-class mapping if that checkpoint is the target.
