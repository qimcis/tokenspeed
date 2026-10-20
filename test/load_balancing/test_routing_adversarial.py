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

"""Adversarial contracts for shared E/P/D resources and streaming ownership."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

from tokenspeed.runtime.load_balancing.decode import DecodePolicy, DecodeRequest
from tokenspeed.runtime.load_balancing.encoder import (
    EncoderItemWork,
    EncoderRequestWork,
)
from tokenspeed.runtime.load_balancing.registry import NoCapacity, WorkerRegistry
from tokenspeed.runtime.load_balancing.service import RoutingService
from tokenspeed.runtime.load_balancing.types import CompatiblePath, Worker, Workload


def registry():
    return WorkerRegistry(
        freshness_seconds=30, history_seconds=60, max_workers=16, max_leases=256
    )


def stage(name, role, capacity, *, now=100, cache_keys=frozenset()):
    return Worker(
        name,
        "boot-1",
        role,
        "http://127.0.0.1:9000",
        capacity,
        now,
        cache_keys=cache_keys,
    )


def test_simultaneous_paths_share_encoder_and_decode_limits_without_partial_leases():
    reg = registry()
    reg.upsert(stage("e", "encoder", Workload(requests=20, encoder_patches=24)))
    reg.upsert(
        stage(
            "d", "decode", Workload(requests=20, decode_context_tokens=50, kv_tokens=50)
        )
    )
    for name in ("p1", "p2"):
        reg.upsert(stage(name, "prefill", Workload(requests=20, prefill_work=100)))
    paths = tuple(
        CompatiblePath(
            name,
            "http://127.0.0.1:9001",
            (("e", "boot-1"), (name, "boot-1"), ("d", "boot-1")),
        )
        for name in ("p1", "p2")
    )
    work = {
        "encoder": Workload(requests=1, encoder_patches=6),
        "prefill": Workload(requests=1, prefill_work=10),
        "decode": Workload(requests=1, decode_context_tokens=10, kv_tokens=10),
    }
    demands = {
        role: (
            lambda snapshot: snapshot.total.requests,
            lambda snapshot, demand=demand: demand,
        )
        for role, demand in work.items()
    }
    barrier = threading.Barrier(12)

    def compete(index):
        barrier.wait()
        try:
            return reg.reserve_path(str(index), paths, demands, now=100)
        except NoCapacity:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        admitted = [
            result for result in pool.map(compete, range(12)) if result is not None
        ]
    assert len(admitted) == 4
    assert reg.get("e").reserved == Workload(requests=4, encoder_patches=24)
    assert reg.get("d").reserved == Workload(
        requests=4, decode_context_tokens=40, kv_tokens=40
    )
    assert reg.get("p1").reserved.requests + reg.get("p2").reserved.requests == 4
    for _, leases in admitted:
        for lease in leases:
            assert reg.release(lease.lease_id, now=101)
    assert all(snapshot.reserved == Workload() for snapshot in reg.snapshot())


def test_joint_reservation_policy_exception_does_not_leave_an_earlier_stage_reserved():
    reg = registry()
    for name, role in (("p", "prefill"), ("d", "decode")):
        reg.upsert(stage(name, role, Workload(requests=1)))
    path = CompatiblePath(
        "pd", "http://127.0.0.1:9001", (("p", "boot-1"), ("d", "boot-1"))
    )

    def invalid(snapshot):
        raise ValueError("unusable decode evidence")

    demands = {
        "prefill": (lambda snapshot: 0, lambda snapshot: Workload(requests=1)),
        "decode": (invalid, lambda snapshot: Workload(requests=1)),
    }
    with pytest.raises(ValueError, match="unusable"):
        reg.reserve_path("failed", (path,), demands, now=100)
    assert all(snapshot.reserved == Workload() for snapshot in reg.snapshot())


def test_old_path_membership_cannot_reserve_restarted_worker():
    reg = registry()
    original = stage("d", "decode", Workload(requests=1))
    reg.upsert(original)
    path = CompatiblePath("d", "http://127.0.0.1:9001", (("d", "boot-1"),))
    reg.upsert(replace(original, incarnation="boot-2", observed_at=101))
    demands = {"decode": (lambda snapshot: 0, lambda snapshot: Workload(requests=1))}
    with pytest.raises(NoCapacity):
        reg.reserve_path("request", (path,), demands, now=101)
    assert reg.get("d").reserved == Workload()


def test_decode_cache_hint_cannot_admit_work_that_needs_more_kv_than_available():
    reg = registry()
    reg.upsert(
        stage(
            "d",
            "decode",
            Workload(
                requests=2,
                decode_context_tokens=1000,
                decode_output_tokens=1000,
                kv_tokens=50,
            ),
            cache_keys=frozenset({"cached"}),
        )
    )
    policy = DecodePolicy(unknown_output_tokens=10)
    request = DecodeRequest(
        context_tokens=100,
        max_new_tokens=10,
        cache_key="cached",
        cached_context_tokens=100,
    )
    with pytest.raises(NoCapacity):
        reg.reserve(
            "request",
            "decode",
            lambda snapshot: policy.score(request, snapshot),
            lambda snapshot: policy.workload(request, snapshot),
            now=100,
        )
    assert reg.get("d").reserved == Workload()


def test_encoder_cache_hint_cannot_bypass_hard_patch_capacity():
    reg = registry()
    reg.upsert(
        stage(
            "e",
            "encoder",
            Workload(requests=2, encoder_patches=4),
            cache_keys=frozenset({"cached"}),
        )
    )
    request = EncoderRequestWork("request", (EncoderItemWork(0, "cached", 8),))
    with pytest.raises(NoCapacity):
        reg.reserve(
            "request", "encoder", request.uncached_patches, request.workload, now=100
        )
    assert reg.get("e").reserved == Workload()


def service_config():
    return {
        "freshness_seconds": 30,
        "history_seconds": 60,
        "max_workers": 16,
        "max_leases": 256,
        "forward_role": "complete",
        "workers": [],
        "control_token": "controller-secret",
        "prefill_policy": {
            "cost_model": {
                "linear_ms_per_token": 1.0,
                "context_ms_per_token_squared": 0.0,
                "quadratic_ms_per_token_squared": 0.0,
            },
            "cache_max_age_seconds": 10.0,
            "context_tiers": [],
            "max_tier_wait_seconds": 1.0,
        },
        "decode_policy": {"unknown_output_tokens": 10},
        "paths": [
            {
                "path_id": "pd",
                "endpoint": "http://127.0.0.1:9001",
                "workers": [["p", "boot-1"], ["d", "boot-1"]],
            }
        ],
    }


def ready_service(client):
    service = RoutingService(service_config(), client=client)
    now = time.monotonic()
    service.registry.upsert(
        stage("p", "prefill", Workload(requests=1, prefill_work=100), now=now)
    )
    service.registry.upsert(
        stage(
            "d",
            "decode",
            Workload(
                requests=1,
                decode_context_tokens=100,
                decode_output_tokens=100,
                kv_tokens=100,
            ),
            now=now,
        )
    )
    service.registry.upsert(
        stage("complete", "complete", Workload(requests=1), now=now)
    )
    return service


def route_payload(request_id="live"):
    return {
        "request_id": request_id,
        "path": "/generate",
        "body": {"text": "hello", "stream": True},
        "prefill": {
            "prompt_tokens": 10,
            "replay_tokens": 0,
            "queued_at": time.monotonic(),
        },
        "decode": {"context_tokens": 10, "max_new_tokens": 4},
    }


async def asgi_call(service, path, payload, *, headers=(), send=None, disconnect=None):
    delivered = False
    events = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        if disconnect is None:
            await asyncio.Future()
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def collect(event):
        events.append(event)
        if send is not None:
            await send(event)

    await service(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": list(headers),
            "query_string": b"",
        },
        receive,
        collect,
    )
    return events


class HeldStream(httpx.AsyncByteStream):
    def __init__(self):
        self.first = asyncio.Event()
        self.finish = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b"data: first\n\n"
        self.first.set()
        await self.finish.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    "collision", ["duplicate", "invalid-target", "invalid-metadata"]
)
def test_colliding_control_request_cannot_release_another_live_stream(collision):
    async def run():
        stream = HeldStream()
        calls = []

        async def transport(request):
            calls.append(request)
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            service = ready_service(client)
            headers = [(b"authorization", b"Bearer controller-secret")]
            payload = route_payload()
            live = asyncio.create_task(
                asgi_call(service, "/route", payload, headers=headers)
            )
            try:
                await asyncio.wait_for(stream.first.wait(), 2)
                attack = dict(payload)
                if collision == "invalid-target":
                    attack["path"] = "http://169.254.169.254/latest/meta-data"
                elif collision == "invalid-metadata":
                    attack["decode"] = {"context_tokens": -1, "max_new_tokens": 1}
                rejected = await asgi_call(service, "/route", attack, headers=headers)
                assert rejected[0]["status"] in {400, 409}
                assert service.registry.get("p").reserved.requests == 1
                assert service.registry.get("d").reserved.requests == 1
                assert len(calls) == 1
            finally:
                stream.finish.set()
                await live
            assert stream.closed
            assert service.registry.get("p").reserved == Workload()
            assert service.registry.get("d").reserved == Workload()

    asyncio.run(run())


def test_untrusted_route_cannot_reserve_or_forward_and_ordinary_body_is_opaque():
    async def run():
        calls = []

        async def transport(request):
            calls.append(request)
            stream = HeldStream()
            stream.finish.set()
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            service = ready_service(client)
            payload = route_payload()
            rejected = await asgi_call(service, "/route", payload)
            assert rejected[0]["status"] == 403
            assert calls == []
            payload.update(
                endpoint="http://169.254.169.254/latest/meta-data", worker_id="d"
            )
            response = await asgi_call(service, "/v1/chat/completions", payload)
            assert response[0]["status"] == 200
            assert str(calls[0].url) == "http://127.0.0.1:9000/v1/chat/completions"
            assert json.loads(calls[0].content) == payload
            assert service.registry.get("p").reserved == Workload()
            assert service.registry.get("d").reserved == Workload()

    asyncio.run(run())


def test_client_disconnect_after_first_chunk_closes_stream_and_all_stage_leases():
    async def run():
        stream = HeldStream()
        disconnected = asyncio.Event()
        calls = []

        async def transport(request):
            calls.append(request)
            return httpx.Response(200, stream=stream)

        async def send(event):
            if event["type"] == "http.response.body" and event.get("body"):
                disconnected.set()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            service = ready_service(client)
            events = await asyncio.wait_for(
                asgi_call(
                    service,
                    "/route",
                    route_payload(),
                    headers=[(b"authorization", b"Bearer controller-secret")],
                    send=send,
                    disconnect=disconnected,
                ),
                2,
            )
            assert events[1]["body"] == b"data: first\n\n"
            assert len(calls) == 1
            assert stream.closed
            assert service.registry.get("p").reserved == Workload()
            assert service.registry.get("d").reserved == Workload()

    asyncio.run(run())


def test_failed_progress_keeps_accepted_frontier_and_reservation_unchanged():
    async def run():
        async with httpx.AsyncClient() as client:
            service = ready_service(client)
            from tokenspeed.runtime.load_balancing.prefill import PrefillRequest

            service.adapter.reserve_stages(
                "progress",
                prefill_request=PrefillRequest(
                    prompt_tokens=10, replay_tokens=0, queued_at=time.monotonic()
                ),
                prefill_policy=service.prefill_policy,
                decode_request=DecodeRequest(10, None),
                decode_policy=service.decode_policy,
            )
            before = service.registry.get("d").reserved
            try:
                with pytest.raises(NoCapacity):
                    service.adapter.decode_progress("progress", generated_tokens=100)
                assert service.registry.get("d").reserved == before
                assert (
                    service.adapter.active["progress"].decode_request.generated_tokens
                    == 0
                )
                service.adapter.decode_progress("progress", generated_tokens=3)
                assert service.registry.get("d").reserved.kv_tokens == 23
                with pytest.raises(ValueError, match="regress"):
                    service.adapter.decode_progress("progress", generated_tokens=2)
                assert service.registry.get("d").reserved.kv_tokens == 23
            finally:
                service.adapter.release("progress")

    asyncio.run(run())


def test_idempotent_retry_cannot_return_a_retired_incarnation_lease():
    reg = registry()
    original = stage("d", "decode", Workload(requests=2))
    reg.upsert(original)
    lease = reg.reserve(
        "old", "decode", lambda snapshot: 0, Workload(requests=1), now=100
    )
    assert (
        reg.reserve("old", "decode", lambda snapshot: 0, Workload(requests=1), now=100)
        == lease
    )
    reg.upsert(replace(original, incarnation="boot-2", observed_at=101))
    with pytest.raises(ValueError, match="incarnation"):
        reg.reserve("old", "decode", lambda snapshot: 0, Workload(requests=1), now=101)
    current = reg.reserve(
        "new", "decode", lambda snapshot: 0, Workload(requests=1), now=101
    )
    with pytest.raises(ValueError, match="incarnation"):
        reg.update_lease(lease.lease_id, Workload(requests=2))
    assert reg.release(lease.lease_id, now=101)
    assert reg.get("d").reserved.requests == 1
    assert reg.release(current.lease_id, now=101)


class FailedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b"data: visible-token\n\n"
        raise httpx.ReadError("lost upstream after committed output")

    async def aclose(self):
        self.closed = True


def test_committed_stage_output_failure_never_retries_or_emits_second_status():
    async def run():
        stream = FailedStream()
        calls, sent = [], []

        async def transport(request):
            calls.append(request)
            return httpx.Response(200, stream=stream)

        async def send(event):
            sent.append(event)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            service = ready_service(client)
            with pytest.raises(httpx.ReadError, match="committed"):
                await asgi_call(
                    service,
                    "/route",
                    route_payload(),
                    headers=[(b"authorization", b"Bearer controller-secret")],
                    send=send,
                )
            assert [
                event["status"]
                for event in sent
                if event["type"] == "http.response.start"
            ] == [200]
            assert sent[1]["body"] == b"data: visible-token\n\n"
            assert len(calls) == 1
            assert stream.closed
            assert service.registry.get("p").reserved == Workload()
            assert service.registry.get("d").reserved == Workload()

    asyncio.run(run())
