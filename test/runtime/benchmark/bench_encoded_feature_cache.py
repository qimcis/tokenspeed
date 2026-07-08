"""Microbenchmark EncodedFeatureCache hit/miss/evict throughput.

Run with:
  PYTHONPATH=python python3 test/runtime/benchmark/bench_encoded_feature_cache.py

Output is JSON lines so the results can be parsed, diffed, or plotted.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

import torch

from tokenspeed.runtime.multimodal.feature_cache import EncodedFeatureCache


def _tensor(n_tokens: int, hidden: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a deterministic tensor given shape parameters."""
    return torch.randn(n_tokens, hidden, dtype=dtype)


def _run_once(
    cache: EncodedFeatureCache,
    keys: list[tuple[str, int]],
    encoded_tensors: dict[int, torch.Tensor],
    ops: list[str],
) -> None:
    """Replay one pre-generated operation sequence."""
    for key, op in zip(keys, ops):
        if op == "put":
            # Content is keyed by hash; the tensor data is irrelevant for the
            # cache key, so we reuse the pre-generated pool.
            cache.put(key[0], key[1], encoded_tensors[key[1] % len(encoded_tensors)])
        else:
            cache.get(key[0], key[1])


def _thread_worker(
    cache: EncodedFeatureCache,
    keys: list[tuple[str, int]],
    encoded_tensors: dict[int, torch.Tensor],
    ops: list[str],
    results: list[float],
    index: int,
) -> None:
    """Worker for the concurrent scenario."""
    t0 = time.perf_counter()
    _run_once(cache, keys, encoded_tensors, ops)
    results[index] = time.perf_counter() - t0


def _bench_scenario(
    label: str,
    max_bytes: int,
    n_ops: int,
    n_unique: int,
    n_tokens: int,
    hidden: int,
    hit_rate: float | None,
    threads: int = 1,
    repeats: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one benchmark scenario and return a JSON-serializable result dict."""
    torch.manual_seed(seed)

    # Pre-generate a pool of encoded tensors.  Each tensor represents one
    # unique multimodal item's encoded output.
    encoded_tensors = {
        i: _tensor(n_tokens, hidden) for i in range(max(n_unique, n_ops // 2 + 1))
    }

    # Build the operation sequence.
    keys: list[tuple[str, int]] = []
    ops: list[str] = []
    if hit_rate is None:
        # Pure insertion benchmark: cycle through unique content hashes.
        for i in range(n_ops):
            keys.append(("image", i % n_unique))
            ops.append("put")
        working_set: list[int] | None = None
    else:
        # Warm the cache with a working set, then replay gets/puts where
        # ``hit_rate`` fraction of the operations target the working set.
        working_set = list(range(min(n_unique, max(1, n_unique))))
        for i in range(n_ops):
            if torch.rand(1).item() < hit_rate:
                k = working_set[torch.randint(0, len(working_set), (1,)).item()]
                keys.append(("image", k))
                ops.append("get")
            else:
                # Miss target: a key outside the working set.
                k = n_unique + i
                if torch.rand(1).item() < 0.5:
                    keys.append(("image", k))
                    ops.append("put")
                else:
                    keys.append(("image", k))
                    ops.append("get")

    times: list[float] = []
    final_stats: dict[str, int] | None = None
    for _ in range(repeats):
        cache = EncodedFeatureCache(max_bytes=max_bytes)

        if working_set is not None:
            for i in working_set:
                cache.put("image", i, encoded_tensors[i])

        if threads == 1:
            t0 = time.perf_counter()
            _run_once(cache, keys, encoded_tensors, ops)
            times.append(time.perf_counter() - t0)
        else:
            # Partition the operation sequence across threads.
            chunk_size = (len(keys) + threads - 1) // threads
            results: list[float] = [0.0] * threads
            thread_list: list[threading.Thread] = []
            for t in range(threads):
                start = t * chunk_size
                end = min(start + chunk_size, len(keys))
                thread_list.append(
                    threading.Thread(
                        target=_thread_worker,
                        args=(
                            cache,
                            keys[start:end],
                            encoded_tensors,
                            ops[start:end],
                            results,
                            t,
                        ),
                    )
                )
            t0 = time.perf_counter()
            for th in thread_list:
                th.start()
            for th in thread_list:
                th.join()
            # Report wall-clock time for the parallel batch.
            times.append(time.perf_counter() - t0)

        final_stats = cache.stats()

    times.sort()
    median = times[len(times) // 2]
    return {
        "system": "tokenspeed",
        "benchmark": "encoded_feature_cache",
        "label": label,
        "max_bytes": max_bytes,
        "n_ops": n_ops,
        "n_unique": n_unique,
        "n_tokens": n_tokens,
        "hidden": hidden,
        "hit_rate": hit_rate,
        "threads": threads,
        "median_ms": round(median * 1e3, 4),
        "min_ms": round(times[0] * 1e3, 4),
        "max_ms": round(times[-1] * 1e3, 4),
        "ops_per_sec": round(n_ops / median, 1),
        "cache_entries": final_stats["entries"],
        "cache_bytes": final_stats["current_bytes"],
    }


def _run_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_config = {
        "n_ops": args.n_ops,
        "n_unique": args.n_unique,
        "n_tokens": args.n_tokens,
        "hidden": args.hidden,
        "repeats": args.repeats,
        "seed": args.seed,
    }

    # Scenario 1: pure insertion throughput (no contention, grows the cache).
    results.append(
        _bench_scenario(
            label="put_unique",
            max_bytes=args.max_bytes,
            hit_rate=None,
            **base_config,
        )
    )

    # Scenario 2-5: read-heavy workloads with increasing hit rates.
    for hit_rate in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0):
        results.append(
            _bench_scenario(
                label=f"mixed_hit_{int(hit_rate * 100):03d}",
                max_bytes=args.max_bytes,
                hit_rate=hit_rate,
                **base_config,
            )
        )

    # Scenario 6: eviction stress test (insert far more unique content than
    # the budget allows).
    eviction_config = dict(base_config)
    eviction_config["n_ops"] = args.n_ops * 4
    results.append(
        _bench_scenario(
            label="eviction_stress",
            max_bytes=args.max_bytes,
            hit_rate=None,
            **eviction_config,
        )
    )

    # Scenario 7: concurrent readers hitting a warm cache.
    if args.threads > 1:
        results.append(
            _bench_scenario(
                label=f"concurrent_{args.threads}t_hit_100",
                max_bytes=args.max_bytes,
                hit_rate=1.0,
                threads=args.threads,
                **base_config,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark EncodedFeatureCache throughput."
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Cache byte budget (default 1 GiB).",
    )
    parser.add_argument(
        "--n-ops", type=int, default=10_000, help="Number of cache operations."
    )
    parser.add_argument(
        "--n-unique", type=int, default=1_000, help="Number of unique content hashes."
    )
    parser.add_argument(
        "--n-tokens", type=int, default=256, help="Encoded token count per item."
    )
    parser.add_argument(
        "--hidden", type=int, default=4096, help="Hidden dimension of encoded tensors."
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Thread count for the concurrent scenario."
    )
    parser.add_argument(
        "--repeats", type=int, default=7, help="Repeat each scenario and take the median."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write JSON lines to this file instead of stdout.",
    )
    args = parser.parse_args()

    results = _run_suite(args)

    out = []
    for r in results:
        line = json.dumps(r, sort_keys=True)
        out.append(line)
        if args.output_json is None:
            print(line)

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            f.write("\n".join(out))
            f.write("\n")


if __name__ == "__main__":
    main()
