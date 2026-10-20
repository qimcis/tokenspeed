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

"""Executable opt-in HTTP router and authenticated routing control API.

Run with ``python -m tokenspeed.runtime.load_balancing.service --config file.json``.
Control endpoints require a configured bearer token. Ordinary /v1 requests route
whole frontends; /route consumes trusted tokenized facts and fixed stage paths.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import httpx

from tokenspeed.runtime.load_balancing.adapter import RoutingAdapter
from tokenspeed.runtime.load_balancing.registry import (
    NoCapacity,
    RequestFinished,
    WorkerRegistry,
)
from tokenspeed.runtime.load_balancing.types import CompatiblePath, Worker, Workload


class RoutingService:
    def __init__(self, config: dict, *, client: httpx.AsyncClient):
        allowed = {
            "freshness_seconds",
            "history_seconds",
            "max_workers",
            "max_leases",
            "forward_role",
            "workers",
            "paths",
            "control_token",
            "prefill_policy",
            "decode_policy",
        }
        if set(config) - allowed:
            raise ValueError(f"unknown router config keys: {set(config) - allowed}")
        self.registry = WorkerRegistry(
            **{
                k: config[k]
                for k in (
                    "freshness_seconds",
                    "history_seconds",
                    "max_workers",
                    "max_leases",
                )
            }
        )
        self.client = client
        self.control_token = config.get("control_token")
        if self.control_token is not None and (
            not isinstance(self.control_token, str) or not self.control_token
        ):
            raise ValueError("control_token must be a nonempty string")
        self.prefill_policy = None
        self.decode_policy = None
        if "prefill_policy" in config:
            from tokenspeed.runtime.load_balancing.prefill import PrefillPolicy

            self.prefill_policy = PrefillPolicy.model_validate_json(
                json.dumps(config["prefill_policy"])
            )
        if "decode_policy" in config:
            from tokenspeed.runtime.load_balancing.decode import DecodePolicy

            self.decode_policy = DecodePolicy(**config["decode_policy"])
        paths = tuple(
            CompatiblePath(
                path_id=p["path_id"],
                endpoint=p["endpoint"],
                workers=tuple(tuple(w) for w in p["workers"]),
            )
            for p in config.get("paths", [])
        )
        if len(paths) > self.registry.max_workers or len(
            {p.path_id for p in paths}
        ) != len(paths):
            raise ValueError("path count must be bounded with unique IDs")
        if paths and (
            self.prefill_policy is None
            or self.decode_policy is None
            or self.control_token is None
        ):
            raise ValueError(
                "stage paths require explicit prefill/decode policies and control_token"
            )
        self.adapter = RoutingAdapter(
            self.registry, client, forward_role=config["forward_role"], paths=paths
        )
        self._probe_task: asyncio.Task | None = None
        self._manual: set[str] = set()
        for data in config["workers"]:
            worker = self._worker(data)
            # A configured address is not evidence of readiness.
            self.registry.upsert(replace(worker, ready=False))

    @staticmethod
    def _worker(data: dict) -> Worker:
        values = dict(data)
        values["capacity"] = Workload(**values["capacity"])
        values["reported"] = Workload(**values.get("reported", {}))
        values["cache_keys"] = frozenset(values.get("cache_keys", []))
        values["observed_at"] = time.monotonic()
        return Worker(**values)

    async def probe_frontends(self):
        """Refresh only frontend liveness; never fabricate stage work reports."""

        async def probe(worker: Worker):
            timeout = min(5.0, self.registry.freshness_seconds / 2)
            try:
                response = await asyncio.wait_for(
                    self.client.get(
                        worker.endpoint.rstrip("/") + "/health", timeout=timeout
                    ),
                    timeout=timeout,
                )
                healthy = response.is_success
            except (httpx.HTTPError, TimeoutError):
                healthy = False
            try:
                current = self.registry.get(worker.worker_id).worker
            except KeyError:
                return
            if current.incarnation == worker.incarnation:
                self.registry.upsert(
                    replace(
                        current,
                        observed_at=time.monotonic(),
                        sequence=current.sequence + 1,
                        healthy=healthy,
                        ready=healthy,
                    )
                )

        # The registry's explicit max_workers bounds fanout. A dead endpoint
        # cannot delay other probes, and the sweep has one shared time bound.
        await asyncio.gather(
            *(
                probe(snapshot.worker)
                for snapshot in self.registry.snapshot()
                if snapshot.worker.role in {"complete", "pair"}
            )
        )

    async def _probe_loop(self):
        while True:
            await self.probe_frontends()
            await asyncio.sleep(self.registry.freshness_seconds / 3)

    @staticmethod
    async def _json(send, status: int, payload):
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await self.probe_frontends()
                    self._probe_task = asyncio.create_task(self._probe_loop())
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    if self._probe_task is not None:
                        self._probe_task.cancel()
                        await asyncio.gather(self._probe_task, return_exceptions=True)
                    await self.client.aclose()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        headers = list(scope.get("headers", []))
        if path == "/health" and method == "GET":
            await self._json(
                send,
                200,
                {"status": "running", "stage_composition": "fixed_paths_only"},
            )
            return
        control = path.startswith("/control/") or path == "/route"
        if control:
            auth = dict(headers).get(b"authorization", b"")
            expected = (
                ("Bearer " + self.control_token).encode() if self.control_token else b""
            )
            if not expected or not hmac.compare_digest(auth, expected):
                await self._json(
                    send,
                    403,
                    {"error": "routing control requires a configured bearer token"},
                )
                return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > 4 * 1024 * 1024:
                await self._json(send, 413, {"error": "request body exceeds 4 MiB"})
                return
            if not event.get("more_body", False):
                break
        committed = False
        owns_route = False
        request_id = uuid.uuid4().hex

        async def tracked_send(event):
            nonlocal committed
            if event["type"] == "http.response.start":
                committed = True
            await send(event)

        try:
            if control:
                data = json.loads(body or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                if path == "/control/workers" and method == "POST":
                    worker = self._worker(data)
                    self.registry.upsert(worker)
                    await self._json(
                        send,
                        200,
                        {
                            "worker_id": worker.worker_id,
                            "incarnation": worker.incarnation,
                        },
                    )
                    return
                if path == "/control/reserve" and method == "POST":
                    if data["request_id"] in self.adapter.active:
                        raise ValueError(
                            "request is owned by an active forwarding route"
                        )
                    lease = self.registry.reserve(
                        data["request_id"],
                        data["role"],
                        lambda s: s.total.requests,
                        Workload(**data["workload"]),
                    )
                    self._manual.add(lease.lease_id)
                    await self._json(send, 200, asdict(lease))
                    return
                if path == "/control/release" and method == "POST":
                    lease_id = data["lease_id"]
                    if lease_id in self._manual:
                        self.registry.release(lease_id)
                        self._manual.remove(lease_id)
                    await self._json(send, 200, {"released": lease_id})
                    return
                if path == "/control/decode-progress" and method == "POST":
                    lease = self.adapter.decode_progress(**data)
                    await self._json(send, 200, asdict(lease))
                    return
                if path == "/route" and method == "POST":
                    from tokenspeed.runtime.load_balancing.decode import DecodeRequest
                    from tokenspeed.runtime.load_balancing.encoder import (
                        EncoderItemWork,
                        EncoderRequestWork,
                    )
                    from tokenspeed.runtime.load_balancing.prefill import PrefillRequest

                    if self.prefill_policy is None or self.decode_policy is None:
                        raise ValueError("stage routing policies are not configured")
                    if set(data) - {
                        "request_id",
                        "path",
                        "body",
                        "prefill",
                        "decode",
                        "encoder",
                    }:
                        raise ValueError("unknown route fields")
                    request_id = data["request_id"]
                    if (
                        not isinstance(request_id, str)
                        or not request_id.isascii()
                        or not request_id
                        or len(request_id) > 256
                    ):
                        raise ValueError("request_id must be 1..256 ASCII characters")
                    target = data["path"]
                    if target not in {
                        "/v1/chat/completions",
                        "/v1/completions",
                        "/generate",
                    }:
                        raise ValueError("unsupported generation path")
                    encoder = (
                        EncoderRequestWork(
                            request_id,
                            tuple(EncoderItemWork(**i) for i in data["encoder"]),
                        )
                        if "encoder" in data
                        else None
                    )
                    endpoint, _ = self.adapter.reserve_stages(
                        request_id,
                        prefill_request=PrefillRequest.model_validate_json(
                            json.dumps(data["prefill"])
                        ),
                        prefill_policy=self.prefill_policy,
                        decode_request=DecodeRequest(**data["decode"]),
                        decode_policy=self.decode_policy,
                        encoder_request=encoder,
                    )
                    owns_route = True
                    # Credentials for the routing controller are not upstream credentials.
                    headers = [
                        (k, v) for k, v in headers if k.lower() != b"authorization"
                    ]
                    headers = [
                        (k, v) for k, v in headers if k.lower() != b"content-type"
                    ] + [(b"content-type", b"application/json")]
                    await self.adapter.forward(
                        request_id=request_id,
                        endpoint=endpoint,
                        method="POST",
                        path=target,
                        query=b"",
                        headers=headers,
                        body=json.dumps(data["body"]).encode(),
                        receive=receive,
                        send=tracked_send,
                    )
                    return
                await self._json(send, 404, {"error": "unknown control endpoint"})
                return
            allowed = (method == "GET" and path == "/v1/models") or (
                method == "POST"
                and path in {"/v1/chat/completions", "/v1/completions", "/generate"}
            )
            if not allowed:
                await self._json(
                    send, 404, {"error": "unsupported forwarding endpoint"}
                )
                return
            endpoint, _ = self.adapter.reserve_frontend(request_id)
            owns_route = True
            await self.adapter.forward(
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                path=path,
                query=scope.get("query_string", b""),
                headers=headers,
                body=bytes(body),
                receive=receive,
                send=tracked_send,
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            NoCapacity,
            RequestFinished,
            httpx.HTTPError,
        ) as exc:
            if owns_route:
                self.adapter.release(request_id)
            if committed:
                raise
            status = (
                503
                if isinstance(exc, NoCapacity)
                else (
                    409
                    if isinstance(exc, RequestFinished)
                    else 502 if isinstance(exc, httpx.HTTPError) else 400
                )
            )
            await self._json(send, status, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import uvicorn

    config = json.loads(args.config.read_text())
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=600, write=30, pool=30),
        follow_redirects=False,
    )
    app = RoutingService(config, client=client)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
