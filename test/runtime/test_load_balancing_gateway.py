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

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from tokenspeed.runtime.load_balancing.adapter import RoutingAdapter
from tokenspeed.runtime.load_balancing.registry import (
    NoCapacity,
    RequestFinished,
    WorkerRegistry,
)
from tokenspeed.runtime.load_balancing.service import RoutingService
from tokenspeed.runtime.load_balancing.types import CompatiblePath, Worker, Workload


def registry():
    return WorkerRegistry(
        freshness_seconds=10, history_seconds=60, max_workers=16, max_leases=256
    )


def worker(
    name="a",
    *,
    role="complete",
    capacity=None,
    now=1,
    incarnation="one",
    endpoint="http://localhost:9999",
):
    return Worker(
        name, incarnation, role, endpoint, capacity or Workload(requests=1), now
    )


def reserve(reg, name, *, now=1):
    return reg.reserve(
        name, "complete", lambda s: s.total.requests, Workload(requests=1), now=now
    )


def test_atomic_racing_reservations_and_idempotent_release():
    reg = registry()
    reg.upsert(worker())

    def attempt(i):
        try:
            return reserve(reg, str(i))
        except NoCapacity:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        leases = list(pool.map(attempt, range(32)))
    assert len([item for item in leases if item]) == 1
    lease = next(item for item in leases if item)
    assert reserve(reg, lease.request_id) == lease
    assert reg.release(lease.lease_id, now=2)
    assert not reg.release(lease.lease_id, now=2)
    assert reg.get("a").reserved.requests == 0
    with pytest.raises(RequestFinished):
        reserve(reg, lease.request_id, now=2)


def test_stale_future_unhealthy_draining_and_reincarnated_reports():
    reg = registry()
    original = worker()
    reg.upsert(original)
    for now in (0, 12):
        with pytest.raises(NoCapacity):
            reserve(reg, str(now), now=now)
    lease = reserve(reg, "old")
    reg.upsert(replace(original, incarnation="two", observed_at=2))
    assert reg.get("a").reserved.requests == 0
    new = reserve(reg, "new", now=2)
    reg.release(lease.lease_id, now=2)
    assert reg.get("a").reserved.requests == 1
    with pytest.raises(ValueError, match="retired"):
        reg.upsert(replace(original, sequence=5))
    with pytest.raises(ValueError, match="stale"):
        reg.upsert(replace(original, incarnation="two", observed_at=2))
    reg.release(new.lease_id, now=2)
    reg.set_draining("a", "two", True)
    reg.upsert(replace(original, incarnation="two", sequence=1, observed_at=2))
    with pytest.raises(NoCapacity):
        reserve(reg, "drain", now=2)
    reg.set_draining("a", "two", False)
    reg.upsert(
        replace(original, incarnation="two", sequence=2, observed_at=2, healthy=False)
    )
    with pytest.raises(NoCapacity):
        reserve(reg, "health", now=2)


def test_path_reservation_is_atomic_and_shared_worker_capacity_is_global():
    reg = registry()
    for name, role in (("p1", "prefill"), ("p2", "prefill"), ("d", "decode")):
        reg.upsert(worker(name, role=role))
    paths = tuple(
        CompatiblePath(p, f"http://{p}", ((p, "one"), ("d", "one")))
        for p in ("p1", "p2")
    )
    demands = {
        role: (lambda s: s.total.requests, lambda s: Workload(requests=1))
        for role in ("prefill", "decode")
    }
    path, leases = reg.reserve_path("request", paths, demands, now=1)
    assert path.path_id == "p1"
    with pytest.raises(NoCapacity):
        reg.reserve_path("second", paths, demands, now=1)
    assert reg.get("p2").reserved.requests == 0
    assert reg.get("d").reserved.requests == 1
    for lease in leases:
        reg.release(lease.lease_id, now=2)
    assert all(s.reserved.requests == 0 for s in reg.snapshot())


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, *, fail=False):
        self.closed = False
        self.fail = fail

    async def __aiter__(self):
        yield b"data: first\n\n"
        if self.fail:
            raise httpx.ReadError("upstream died")
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def test_streaming_failure_never_retries_and_releases_after_committed_output():
    async def run():
        reg = registry()
        reg.upsert(worker(now=time.monotonic()))
        stream = BytesStream(fail=True)
        requests = []

        async def transport(req):
            requests.append(req)
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            adapter = RoutingAdapter(reg, client, forward_role="complete")
            endpoint, _ = adapter.reserve_frontend("r")
            events = []

            async def receive():
                await asyncio.Future()

            async def send(event):
                events.append(event)
                assert reg.get("a").reserved.requests == 1

            with pytest.raises(httpx.ReadError):
                await adapter.forward(
                    request_id="r",
                    endpoint=endpoint,
                    method="POST",
                    path="/generate",
                    query=b"",
                    headers=[],
                    body=b"{}",
                    receive=receive,
                    send=send,
                )
            assert events[1]["body"] == b"data: first\n\n"
            assert len(requests) == 1
            assert stream.closed
            assert reg.get("a").reserved.requests == 0

    asyncio.run(run())


def test_disconnect_cancels_upstream_and_releases():
    async def run():
        reg = registry()
        reg.upsert(worker(now=time.monotonic()))
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def transport(req):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            adapter = RoutingAdapter(reg, client, forward_role="complete")
            endpoint, _ = adapter.reserve_frontend("r")

            async def receive():
                await entered.wait()
                return {"type": "http.disconnect"}

            async def send(event):
                raise AssertionError("no response expected")

            await adapter.forward(
                request_id="r",
                endpoint=endpoint,
                method="POST",
                path="/generate",
                query=b"",
                headers=[],
                body=b"{}",
                receive=receive,
                send=send,
            )
            assert cancelled.is_set()
            assert reg.get("a").reserved.requests == 0

    asyncio.run(run())


def config(endpoint):
    return {
        "freshness_seconds": 10,
        "history_seconds": 60,
        "max_workers": 8,
        "max_leases": 128,
        "forward_role": "complete",
        "workers": [
            {
                "worker_id": "a",
                "incarnation": "one",
                "role": "complete",
                "endpoint": endpoint,
                "capacity": {"requests": 2},
            }
        ],
        "paths": [],
    }


def test_actual_localhost_http_forward_and_control_auth():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b'{"data": []}'
            self.send_response(200)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            seen.append((self.path, json.loads(body)))
            payload = b"data: first\n\ndata: [DONE]\n\n"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def run():
        async with httpx.AsyncClient() as upstream:
            service = RoutingService(
                config(f"http://127.0.0.1:{server.server_port}"), client=upstream
            )
            await service.probe_frontends()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service), base_url="http://router"
            ) as client:
                response = await client.post(
                    "/generate", json={"text": "hello", "stream": True}
                )
                assert response.status_code == 200
                assert response.content == b"data: first\n\ndata: [DONE]\n\n"
                assert (await client.get("/v1/models")).status_code == 200
                assert (
                    await client.post("/control/reserve", json={})
                ).status_code == 403
            assert service.registry.get("a").reserved.requests == 0

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert seen == [("/generate", {"text": "hello", "stream": True})]


def test_stage_route_invokes_policies_and_forwards_selected_frontend():
    async def run():
        cfg = config("http://complete")
        cfg.update(
            control_token="test-secret",
            prefill_policy={
                "cost_model": {
                    "linear_ms_per_token": 1.0,
                    "context_ms_per_token_squared": 0.0,
                    "quadratic_ms_per_token_squared": 0.0,
                },
                "cache_max_age_seconds": 10.0,
                "context_tiers": [],
                "max_tier_wait_seconds": 1.0,
            },
            decode_policy={"unknown_output_tokens": 8},
        )
        cfg["workers"] = []
        cfg["paths"] = [
            {
                "path_id": "one",
                "endpoint": "http://pair-one",
                "workers": [["p1", "one"], ["d1", "one"]],
            },
            {
                "path_id": "two",
                "endpoint": "http://pair-two",
                "workers": [["p2", "one"], ["d2", "one"]],
            },
        ]
        seen = []

        async def transport(request):
            seen.append(str(request.url))
            assert request.headers.get("authorization") is None
            return httpx.Response(200, stream=BytesStream())

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as upstream:
            service = RoutingService(cfg, client=upstream)
            now = time.monotonic()
            for number in (1, 2):
                service.registry.upsert(
                    worker(
                        f"p{number}",
                        role="prefill",
                        now=now,
                        capacity=Workload(requests=5, prefill_work=100),
                        endpoint=f"http://p{number}",
                    )
                )
                service.registry.upsert(
                    worker(
                        f"d{number}",
                        role="decode",
                        now=now,
                        capacity=Workload(
                            requests=5,
                            decode_context_tokens=100,
                            decode_output_tokens=100,
                            kv_tokens=100,
                        ),
                        endpoint=f"http://d{number}",
                    )
                )
            occupied = service.registry.reserve(
                "prior",
                "prefill",
                lambda s: 0 if s.worker.worker_id == "p1" else float("inf"),
                Workload(requests=1, prefill_work=40),
                now=now,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service), base_url="http://router"
            ) as client:
                payload = {
                    "request_id": "stage",
                    "path": "/generate",
                    "body": {"text": "hello"},
                    "prefill": {
                        "prompt_tokens": 10,
                        "replay_tokens": 0,
                        "queued_at": now,
                    },
                    "decode": {"context_tokens": 10, "max_new_tokens": 4},
                }
                response = await client.post(
                    "/route",
                    json=payload,
                    headers={"authorization": "Bearer test-secret"},
                )
                assert response.status_code == 200, response.text
                assert seen == ["http://pair-two/generate"]
                assert (
                    await client.post(
                        "/route",
                        json=payload,
                        headers={"authorization": "Bearer test-secret"},
                    )
                ).status_code == 409
            assert service.registry.get("p2").reserved.requests == 0
            assert service.registry.get("d2").reserved.requests == 0
            service.registry.release(occupied.lease_id)

    asyncio.run(run())


def test_manual_reservation_cannot_adopt_live_forwarding_leases():
    async def run():
        async with httpx.AsyncClient() as upstream:
            cfg = config("http://127.0.0.1:9999")
            cfg["control_token"] = "secret"
            service = RoutingService(cfg, client=upstream)
            original = service.registry.get("a").worker
            service.registry.upsert(replace(original, ready=True, sequence=1))
            service.adapter.reserve_frontend("active")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service), base_url="http://router"
            ) as client:
                response = await client.post(
                    "/control/reserve",
                    json={
                        "request_id": "active",
                        "role": "complete",
                        "workload": {"requests": 1},
                    },
                    headers={"authorization": "Bearer secret"},
                )
                assert response.status_code == 400
                assert service.registry.get("a").reserved.requests == 1
            service.adapter.release("active")

    asyncio.run(run())


def test_stalled_frontends_do_not_age_healthy_worker_reports():
    async def run():
        async def transport(request):
            if request.url.host != "healthy":
                await asyncio.sleep(1)
            return httpx.Response(200, content=b"ok")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            cfg = config("http://healthy")
            cfg["freshness_seconds"] = 0.2
            for index in range(3):
                cfg["workers"].append(
                    {
                        "worker_id": f"dead{index}",
                        "incarnation": "one",
                        "role": "complete",
                        "endpoint": f"http://dead{index}",
                        "capacity": {"requests": 2},
                    }
                )
            service = RoutingService(cfg, client=client)
            await service.probe_frontends()
            # Sequential probes age the first report beyond freshness here.
            endpoint, _ = service.adapter.reserve_frontend("still-healthy")
            assert endpoint == "http://healthy"
            assert all(
                not service.registry.get(f"dead{i}").worker.healthy for i in range(3)
            )
            service.adapter.release("still-healthy")

    asyncio.run(run())
