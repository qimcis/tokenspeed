"""Benchmark VisionEmbedder cross-request encoded feature cache.

This benchmark is intentionally self-contained and does not require a real
vision tower or GPU.  It measures the end-to-end `VisionEmbedder.apply` path
with a synthetic encoder so the only variable is the cross-request cache.

Apples-to-apples usage:
  # Baseline (no cache) -- equivalent to main when the env var is unset
  PYTHONPATH=python python3 test/runtime/benchmark/bench_vision_embedder_cache.py \
      --cache-bytes 0 --output-json baseline.jsonl

  # Optimized (with cache)
  PYTHONPATH=python python3 test/runtime/benchmark/bench_vision_embedder_cache.py \
      --cache-bytes 1073741824 --output-json cached.jsonl

Then diff the JSON lines; `total_ms`, `encoder_calls`, and `cache_hits`
are the headline numbers for a PR.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
from torch import nn


def _set_cache_env(value: int) -> None:
    """Set the cache size env var for the upcoming VisionEmbedder instance."""
    os.environ["TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES"] = str(value)


def _make_item(
    content_hash: int,
    n_tokens: int,
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Build a synthetic MultimodalDataItem for the benchmark."""
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
    )

    # Deterministic feature tensor for a given hash so encode time is stable.
    g = torch.Generator().manual_seed(content_hash % (2**32))
    feature = torch.randn(n_tokens, hidden, dtype=dtype, generator=g, device="cpu")

    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=content_hash,
        pad_value=1_000_000 + (content_hash & ((1 << 30) - 1)),
        offsets=[(0, n_tokens - 1)],
        feature=feature,
    )


def _build_request_distribution(
    n_requests: int,
    n_unique: int,
    repetition_ratio: float,
    seed: int,
) -> list[int]:
    """Return a list of content hashes with the requested repeat/unique mix."""
    torch.manual_seed(seed)
    # The first ``repeated`` positions are drawn from a small working set;
    # the rest are unique.
    repeated = int(n_requests * repetition_ratio)
    hashes: list[int] = []
    for i in range(repeated):
        hashes.append(i % n_unique)
    for i in range(n_requests - repeated):
        hashes.append(n_unique + i)
    return hashes


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one configuration and return the result dict."""
    # Imports are local so the env var is read fresh by VisionEmbedder.__init__.
    from tokenspeed.runtime.multimodal.embedder import EncoderSpec, VisionEmbedder
    from tokenspeed.runtime.multimodal.inputs import (
        MultimodalForwardContext,
        MultimodalInputs,
        Modality,
    )

    _set_cache_env(args.cache_bytes)
    embedder = VisionEmbedder()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    text_embedding = nn.Embedding(args.vocab_size, args.hidden).to(device)

    encoder_call_count = [0]
    encoder_time_s = [0.0]

    def dummy_encoder(items: list[Any]) -> torch.Tensor:
        encoder_call_count[0] += 1
        t0 = time.perf_counter()
        # Simulate a modest amount of compute proportional to token count.
        out = torch.cat([it.feature.to(device, dtype=dtype) for it in items], dim=0)
        # A small matmul makes the dummy encoder non-trivial.
        weight = torch.randn(out.shape[-1], out.shape[-1], dtype=dtype, device=device)
        out = out @ weight
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_time_s[0] += time.perf_counter() - t0
        return out

    def dummy_deepstack(emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = emb.shape[-1]
        return emb[..., : h // 2], emb[..., h // 2 :]

    class FakeMultimodalModel(nn.Module):
        deepstack_visual_indexes = [0]

        @staticmethod
        def separate_deepstack_embeds(emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dummy_deepstack(emb)

    encoders = {Modality.IMAGE: EncoderSpec(dummy_encoder, deepstack=args.deepstack)}
    multimodal_model = FakeMultimodalModel()

    hashes = _build_request_distribution(
        args.n_requests,
        args.n_unique,
        args.repetition_ratio,
        args.seed,
    )

    # Warm-up: one representative request to amortize any one-time setup.
    warmup_item = _make_item(0, args.n_tokens, args.hidden, device, dtype)
    warmup_ctx = MultimodalForwardContext(
        mm_inputs=[MultimodalInputs([warmup_item])],
        extend_prefix_lens=[0],
        extend_seq_lens=[args.n_tokens],
    )
    warmup_ids = torch.full((args.n_tokens,), 0, dtype=torch.long, device=device)
    embedder.apply(warmup_ids, text_embedding, warmup_ctx, encoders, multimodal_model)

    # Main benchmark run.
    per_request_times_ms: list[float] = []
    total_t0 = time.perf_counter()
    for h in hashes:
        item = _make_item(int(h), args.n_tokens, args.hidden, device, dtype)
        ctx = MultimodalForwardContext(
            mm_inputs=[MultimodalInputs([item])],
            extend_prefix_lens=[0],
            extend_seq_lens=[args.n_tokens],
        )
        input_ids = torch.full((args.n_tokens,), 0, dtype=torch.long, device=device)

        t0 = time.perf_counter()
        embedder.apply(input_ids, text_embedding, ctx, encoders, multimodal_model)
        per_request_times_ms.append((time.perf_counter() - t0) * 1e3)

    total_elapsed_ms = (time.perf_counter() - total_t0) * 1e3

    feature_cache = getattr(embedder, "_encoded_feature_cache", None)
    cache_stats = (
        feature_cache.stats()
        if feature_cache is not None
        else {"hits": 0, "misses": 0, "current_bytes": 0, "entries": 0}
    )

    per_request_times_ms.sort()
    return {
        "system": "tokenspeed",
        "benchmark": "vision_embedder_cache",
        "cache_bytes": args.cache_bytes,
        "n_requests": args.n_requests,
        "n_unique": args.n_unique,
        "repetition_ratio": args.repetition_ratio,
        "n_tokens": args.n_tokens,
        "hidden": args.hidden,
        "deepstack": args.deepstack,
        "device": args.device,
        "dtype": args.dtype,
        "total_ms": round(total_elapsed_ms, 4),
        "median_request_ms": round(per_request_times_ms[len(per_request_times_ms) // 2], 4),
        "p99_request_ms": round(per_request_times_ms[int(len(per_request_times_ms) * 0.99)], 4),
        "encoder_calls": encoder_call_count[0],
        "encoder_ms": round(encoder_time_s[0] * 1e3, 4),
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_entries": cache_stats["entries"],
        "cache_bytes_used": cache_stats["current_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VisionEmbedder encoded feature cache."
    )
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=0,
        help="TOKENSPEED_MM_ENCODER_FEATURE_CACHE_MAX_BYTES value (0 disables cache).",
    )
    parser.add_argument(
        "--n-requests", type=int, default=100, help="Number of simulated requests."
    )
    parser.add_argument(
        "--n-unique",
        type=int,
        default=10,
        help="Number of unique images in the working set (the remainder are unique misses).",
    )
    parser.add_argument(
        "--repetition-ratio",
        type=float,
        default=0.8,
        help="Fraction of requests that reuse the working-set images.",
    )
    parser.add_argument(
        "--n-tokens", type=int, default=256, help="Encoded tokens per image."
    )
    parser.add_argument(
        "--hidden", type=int, default=4096, help="Hidden dimension of encoded tensors."
    )
    parser.add_argument(
        "--deepstack",
        action="store_true",
        help="Exercise the deepstack (main + deep) code path.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=128_000, help="Text embedding vocab size."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device to run on (cpu or cuda)."
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="torch dtype for encoded tensors."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write the result JSON object to this file instead of stdout.",
    )
    args = parser.parse_args()

    # Be deterministic.
    torch.manual_seed(args.seed)

    result = _run_benchmark(args)

    line = json.dumps(result, sort_keys=True, indent=2 if args.output_json is None else None)
    if args.output_json is None:
        print(line)
    else:
        with open(args.output_json, "w") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
