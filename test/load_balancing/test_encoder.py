# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Patch routing and actual encoder execution observations (CPU only)."""

import importlib.util
from dataclasses import replace

import pytest
import torch

from tokenspeed.runtime.cache.embedding_cache import EmbeddingCache
from tokenspeed.runtime.epd.encode_scheduler import EncodeScheduler
from tokenspeed.runtime.load_balancing.encoder import (
    EncoderItemWork,
    EncoderPolicy,
    EncoderRequestWork,
    encoder_request_work,
    preprocessed_patch_count,
)
from tokenspeed.runtime.load_balancing.registry import NoCapacity, WorkerRegistry
from tokenspeed.runtime.load_balancing.types import Worker, Workload


def registry():
    return WorkerRegistry(
        freshness_seconds=10, max_workers=8, max_leases=100, history_seconds=60
    )


def worker(name, *, patches=1000, reported=0, cached=frozenset()):
    return Worker(
        worker_id=name,
        incarnation="first",
        role="encoder",
        endpoint=f"http://{name}:8000",
        capacity=Workload(requests=10, encoder_patches=patches),
        observed_at=100,
        reported=Workload(encoder_patches=reported),
        cache_keys=cached,
    )


def request(name, counts=(10,)):
    return EncoderRequestWork(
        name, tuple(EncoderItemWork(i, str(i), count) for i, count in enumerate(counts))
    )


def test_routes_by_postprocessed_patches_and_reserved_work():
    reg = registry()
    reg.upsert(worker("a", reported=90))
    reg.upsert(worker("b", reported=20))
    policy = EncoderPolicy(reg)
    big = policy.reserve(request("big", (100,)), now=100)
    small = policy.reserve(request("small", (2, 3, 4)), now=100)
    assert (big.worker_id, big.workload.encoder_patches) == ("b", 100)
    assert (small.worker_id, small.workload.encoder_patches) == ("a", 9)
    assert reg.get("b").reserved.requests == 1


def test_known_cache_hits_preserve_indices_and_repeated_misses():
    reg = registry()
    reg.upsert(worker("a", cached=frozenset({"cached"})))
    req = EncoderRequestWork(
        "r",
        (
            EncoderItemWork(0, "cached", 400),
            EncoderItemWork(1, "miss", 20),
            EncoderItemWork(2, "miss", 20),
        ),
    )
    lease = EncoderPolicy(reg).reserve(req, now=100)
    assert lease.workload == Workload(requests=1, encoder_patches=440)
    assert req.uncached_patches(reg.get("a")) == 40
    assert tuple(item.cache_key for item in req.items) == ("cached", "miss", "miss")


def test_cache_affinity_overrides_lighter_uncached_worker():
    reg = registry()
    reg.upsert(worker("a", reported=15, cached=frozenset({"0"})))
    reg.upsert(worker("b"))
    assert EncoderPolicy(reg).reserve(request("r", (100,)), now=100).worker_id == "a"


def test_capacity_reservation_idempotency_and_release():
    reg = registry()
    reg.upsert(worker("a", patches=10))
    policy = EncoderPolicy(reg)
    lease = policy.reserve(request("first", (10,)), now=100)
    assert policy.reserve(request("first", (10,)), now=100) == lease
    with pytest.raises(NoCapacity):
        policy.reserve(request("second", (1,)), now=100)
    assert reg.release(lease.lease_id, now=100)
    assert not reg.release(lease.lease_id, now=100)
    assert (
        policy.reserve(request("second", (1,)), now=100).workload.encoder_patches == 1
    )


def test_cached_request_still_reserves_patch_capacity_and_concurrency():
    reg = registry()
    reg.upsert(
        replace(
            worker("a", cached=frozenset({"0"})),
            capacity=Workload(requests=1, encoder_patches=10),
        )
    )
    policy = EncoderPolicy(reg)
    assert policy.reserve(request("r"), now=100).workload == Workload(
        requests=1, encoder_patches=10
    )
    with pytest.raises(NoCapacity):
        policy.reserve(request("next"), now=100)


def test_cache_eviction_cannot_exceed_reserved_patch_capacity():
    reg = registry()
    old = worker("a", patches=10, cached=frozenset({"0"}))
    reg.upsert(old)
    policy = EncoderPolicy(reg)
    lease = policy.reserve(request("r"), now=100)
    reg.upsert(replace(old, sequence=1, cache_keys=frozenset()))
    assert (
        reg.get("a").reserved
        == lease.workload
        == Workload(requests=1, encoder_patches=10)
    )
    with pytest.raises(NoCapacity):
        policy.reserve(request("next", (1,)), now=100)


def test_stale_cache_reports_and_incarnation_isolation():
    reg = registry()
    old = worker("a", patches=10, cached=frozenset({"0"}))
    reg.upsert(old)
    policy = EncoderPolicy(reg)
    lease = policy.reserve(request("old", (10,)), now=100)
    with pytest.raises(NoCapacity):
        policy.reserve(request("stale"), now=111)
    reg.upsert(
        replace(old, incarnation="second", cache_keys=frozenset(), observed_at=111)
    )
    with pytest.raises(NoCapacity):
        policy.reserve(request("new", (100,)), now=111)
    new_lease = policy.reserve(request("new", (10,)), now=111)
    assert reg.release(lease.lease_id, now=111)
    assert reg.get("a").reserved == new_lease.workload


def test_equal_scores_stably_choose_worker_id_not_registration_order():
    reg = registry()
    reg.upsert(worker("z"))
    reg.upsert(worker("a"))
    assert EncoderPolicy(reg).reserve(request("r"), now=100).worker_id == "a"


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rejects_invalid_patch_count(value):
    with pytest.raises(ValueError):
        EncoderItemWork(0, "content", value)


def test_rejects_reordered_original_indices():
    with pytest.raises(ValueError):
        EncoderRequestWork(
            "r", (EncoderItemWork(1, "x", 1), EncoderItemWork(0, "y", 1))
        )


def item(key, grid=None, *, geometry="vit_grid"):
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=key,
        offsets=[(0, 1)],
        model_specific_data={} if grid is None else {geometry: torch.tensor(grid)},
    )


@pytest.mark.parametrize(
    ("grid", "geometry", "count"),
    [
        ([3, 7], "vit_grid", 21),
        ([[1, 6, 8], [1, 2, 4]], "image_grid_thw", 56),
        ([[3, 6, 8]], "video_grid_thw", 144),
    ],
)
def test_exact_preprocessed_grids_are_independent_of_output_tokens(
    grid, geometry, count
):
    assert preprocessed_patch_count({geometry: torch.tensor(grid)}) == count


@pytest.mark.parametrize("grid", [None, [0, 8], [-1, 8], [1.0, 8.0], [1, 2, 3]])
def test_unknown_or_invalid_geometry_cannot_become_a_routing_estimate(grid):
    data = {} if grid is None else {"vit_grid": torch.tensor(grid)}
    assert preprocessed_patch_count(data) is None


class CpuExecutor:
    """CPU substitute for unavailable vision GPU + transport only."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.executed = []
        self.sent = []
        self.failed = []

    def register(self, *args):
        pass

    def execute(self, request_items):
        if self.fail:
            raise ValueError("tower failure")
        self.executed.append([(rid, image.hash) for rid, image in request_items])
        for _, image in request_items:
            image.encoded = torch.zeros(2, 4)

    def send_item(self, rid, image):
        self.sent.append((rid, image.hash))

    def reap_concluded_senders(self, requests):
        pass

    def drain_deferred(self):
        pass

    def has_deferred(self):
        return False

    def fail_rooms(self, requests, error):
        self.failed.extend(requests)
        return len(self.failed)


def encode_request(rid, images):
    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    return EncodeRequest(
        request_id=rid,
        bootstrap_host="host",
        bootstrap_port=1,
        bootstrap_room=1,
        items=images,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("tokenspeed_triton") is None,
    reason="full encoder import requires unavailable vendor Triton package",
)
def test_real_worker_observes_cache_misses_queue_and_forward_once():
    from tokenspeed.runtime.epd.encode_worker import EncodeWorker

    executor = CpuExecutor()
    cache = EmbeddingCache(4096)
    cache.put(1, torch.zeros(2, 4), 32)
    observations = []
    encoder = EncodeWorker(
        executor, EncodeScheduler(100, 10), cache, observer=observations.append
    )
    images = [item(1, [10, 10]), item(2, [3, 7]), item(3, [2, 4])]
    assert encoder_request_work("r", images).items == (
        EncoderItemWork(0, "1", 100),
        EncoderItemWork(1, "2", 21),
        EncoderItemWork(2, "3", 8),
    )
    encoder.submit(encode_request("r", images))
    assert observations[-1].queued_patch_work == 29
    assert observations[-1].cache_hits == 1
    assert encoder.step() == 2
    assert observations[-2].queued_patch_work == 0
    assert observations[-2].in_flight_patch_work == 29
    assert observations[-1].completed_patch_work == 29
    assert observations[-1].in_flight_patch_work == 0
    assert executor.executed == [[("r", 2), ("r", 3)]]
    assert executor.sent == [("r", 1)]
    encoder.submit(encode_request("next", [item(2, [3, 7])]))
    assert encoder.load_snapshot().cache_hits == 2
    assert encoder.step() == 0


@pytest.mark.skipif(
    importlib.util.find_spec("tokenspeed_triton") is None,
    reason="full encoder import requires unavailable vendor Triton package",
)
def test_real_worker_unknown_geometry_is_visible_and_failure_releases_work():
    from tokenspeed.runtime.epd.encode_worker import EncodeWorker

    executor = CpuExecutor(fail=True)
    observations = []
    encoder = EncodeWorker(
        executor,
        EncodeScheduler(100, 10),
        EmbeddingCache(0),
        observer=observations.append,
    )
    encoder.submit(encode_request("r", [item(1, [2, 3]), item(2)]))
    assert encoder.load_snapshot().unknown_patch_items == 1
    assert encoder.step() == 0
    assert encoder.load_snapshot().queued_patch_work == 0
    assert encoder.load_snapshot().failed_patch_work == 6
    assert encoder.load_snapshot().in_flight_patch_work == 0
    assert encoder.load_snapshot().completed_patch_work == 0
