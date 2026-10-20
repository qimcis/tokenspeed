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

"""HTTP forwarding to complete gateways, with no output replay or stage fiction."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from tokenspeed.runtime.load_balancing.registry import WorkerRegistry
from tokenspeed.runtime.load_balancing.types import CompatiblePath, Lease, Workload

if TYPE_CHECKING:
    from tokenspeed.runtime.load_balancing.decode import DecodePolicy, DecodeRequest
    from tokenspeed.runtime.load_balancing.encoder import EncoderRequestWork
    from tokenspeed.runtime.load_balancing.prefill import PrefillPolicy, PrefillRequest


_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


def end_to_end_headers(
    headers: list[tuple[bytes, bytes]], *, request: bool
) -> list[tuple[bytes, bytes]]:
    blocked = set(_HOP_HEADERS)
    for key, value in headers:
        if key.lower() == b"connection":
            blocked.update(part.strip().lower() for part in value.split(b","))
    if request:
        blocked.update({b"host", b"content-length", b"x-request-id"})
    return [(k, v) for k, v in headers if k.lower() not in blocked]


@dataclass
class ActiveRoute:
    leases: tuple[Lease, ...]
    decode_request: DecodeRequest | None
    decode_policy: DecodePolicy | None


class RoutingAdapter:
    def __init__(
        self,
        registry: WorkerRegistry,
        client: httpx.AsyncClient,
        *,
        forward_role: str,
        paths: tuple[CompatiblePath, ...] = (),
    ):
        if forward_role not in {"complete", "pair"}:
            raise ValueError(
                "HTTP forwarding requires complete or configured pair frontends"
            )
        self.registry = registry
        self.client = client
        self.forward_role = forward_role
        self.paths = paths
        self.active: dict[str, ActiveRoute] = {}

    def reserve_frontend(self, request_id: str) -> tuple[str, tuple[Lease, ...]]:
        if request_id in self.active:
            raise ValueError("request already forwarding")
        lease = self.registry.reserve(
            request_id,
            self.forward_role,
            lambda s: (
                (s.total.requests + 1) / s.worker.capacity.requests
                if s.worker.capacity.requests
                else float("inf")
            ),
            Workload(requests=1),
        )
        worker = self.registry.get(lease.worker_id).worker
        if worker.incarnation != lease.incarnation:
            self.registry.release(lease.lease_id)
            raise ValueError("worker changed during admission")
        self.active[request_id] = ActiveRoute((lease,), None, None)
        return worker.endpoint, (lease,)

    def reserve_stages(
        self,
        request_id: str,
        *,
        prefill_request: PrefillRequest,
        prefill_policy: PrefillPolicy,
        decode_request: DecodeRequest,
        decode_policy: DecodePolicy,
        encoder_request: EncoderRequestWork | None = None,
    ) -> tuple[str, tuple[Lease, ...]]:
        """Route trusted tokenized/preprocessed facts through fixed compatible paths."""
        if request_id in self.active:
            raise ValueError("request already forwarding")
        now = time.monotonic()
        demands = {
            "prefill": (
                lambda s: (
                    prefill_policy.score(prefill_request, s, now=now)
                    / s.worker.capacity.prefill_work
                    if s.worker.capacity.prefill_work > 0
                    else float("inf")
                ),
                lambda s: prefill_policy.workload(prefill_request, s, now=now),
            ),
            "decode": (
                lambda s: decode_policy.score(decode_request, s),
                lambda s: decode_policy.workload(decode_request, s),
            ),
        }
        if encoder_request is not None:
            demands["encoder"] = (
                lambda s: (
                    (s.total.encoder_patches + encoder_request.uncached_patches(s))
                    / s.worker.capacity.encoder_patches
                    if s.worker.capacity.encoder_patches > 0
                    else float("inf")
                ),
                encoder_request.workload,
            )
        path, leases = self.registry.reserve_path(
            request_id, self.paths, demands, now=now
        )
        self.active[request_id] = ActiveRoute(leases, decode_request, decode_policy)
        return path.endpoint, leases

    def decode_progress(
        self,
        request_id: str,
        *,
        generated_tokens: int,
        verification_tokens: int | None = None,
    ) -> Lease:
        route = self.active[request_id]
        if route.decode_request is None or route.decode_policy is None:
            raise ValueError("route has no typed decode reservation")
        request = route.decode_policy.advance(
            route.decode_request,
            generated_tokens=generated_tokens,
            verification_tokens=verification_tokens,
        )
        for lease in route.leases:
            snapshot = self.registry.get(lease.worker_id)
            if snapshot.worker.role == "decode":
                if snapshot.worker.incarnation != lease.incarnation:
                    raise ValueError("decode worker incarnation changed")
                updated = self.registry.update_lease(
                    lease.lease_id, route.decode_policy.workload(request, snapshot)
                )
                route.decode_request = request
                return updated
        raise ValueError("route has no decode lease")

    def release(self, request_id: str) -> None:
        route = self.active.pop(request_id, None)
        if route is not None:
            for lease in route.leases:
                self.registry.release(lease.lease_id)

    async def forward(
        self,
        *,
        request_id: str,
        endpoint: str,
        method: str,
        path: str,
        query: bytes,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
        receive,
        send,
    ) -> None:
        """Stream unchanged bytes once; close upstream and release on disconnect.

        Successful writes, errors, and cancellation all terminate the reservation.
        There is deliberately no retry, including failures before response bytes.
        """

        async def disconnect():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return

        async def stream():
            url = endpoint.rstrip("/") + path
            if query:
                url += "?" + query.decode("ascii")
            forwarded = end_to_end_headers(headers, request=True)
            forwarded.append((b"x-request-id", request_id.encode("ascii")))
            response = await self.client.send(
                self.client.build_request(method, url, headers=forwarded, content=body),
                stream=True,
            )
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status_code,
                        "headers": end_to_end_headers(
                            list(response.headers.raw), request=False
                        ),
                    }
                )
                async for chunk in response.aiter_raw():
                    await send(
                        {"type": "http.response.body", "body": chunk, "more_body": True}
                    )
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
            finally:
                await response.aclose()

        stream_task = asyncio.create_task(stream())
        disconnect_task = asyncio.create_task(disconnect())
        try:
            done, _ = await asyncio.wait(
                (stream_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED
            )
            if stream_task in done:
                await stream_task
        finally:
            stream_task.cancel()
            disconnect_task.cancel()
            await asyncio.gather(stream_task, disconnect_task, return_exceptions=True)
            self.release(request_id)
