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

"""Execution-stream copies of existing top-k routes, with no device wait."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.moe.route_feedback import (
    MAX_EXPERT_KEYS,
    RouteCaptureCadence,
    RouteObservation,
    observations_from_rows,
)


@dataclass(frozen=True)
class CapturedRoutes:
    """Borrowed pinned buffers; read only after the forward result copy event.

    Non-overlap execution commits this result before the next begin can reuse
    storage. No request identity is read from mutable device slots.
    """

    request_ids: tuple[str, ...]
    generations: tuple[int, ...]
    computed_frontiers: tuple[int, ...]
    layers: tuple[tuple[int, torch.Tensor], ...]

    def observations(self) -> tuple[RouteObservation, ...]:
        return observations_from_rows(
            request_ids=self.request_ids,
            generations=self.generations,
            computed_frontiers=self.computed_frontiers,
            layer_rows=tuple(
                (layer, tuple(tuple(row) for row in tensor.tolist()))
                for layer, tensor in self.layers
            ),
        )


class RouteCapture:
    """One bounded set of pinned buffers, owned exclusively by the forward thread.

    ``layer_widths`` names actual observed router modules. CPU tensors are allowed
    for unit tests; production always supplies pin_memory=True. Hooks copy on the
    router's execution stream and the existing forward copy event lands them.
    """

    def __init__(
        self,
        *,
        layer_widths: tuple[tuple[int, int], ...],
        interval: int,
        max_requests: int,
        pin_memory: bool,
    ) -> None:
        if not layer_widths or len(layer_widths) > 32:
            raise ValueError("route capture requires 1..32 layers")
        if len({layer for layer, _ in layer_widths}) != len(layer_widths):
            raise ValueError("route capture layers must be unique")
        if any(
            layer < 0 or layer >= 2**32 or width <= 0 for layer, width in layer_widths
        ):
            raise ValueError("invalid route layer or top-k width")
        if sum(width for _, width in layer_widths) > MAX_EXPERT_KEYS:
            raise ValueError("captured top-k widths exceed 256 expert keys")
        self.cadence = RouteCaptureCadence(interval=interval, max_requests=max_requests)
        self.buffers = {
            layer: torch.empty(
                (max_requests, width),
                dtype=torch.int64,
                device="cpu",
                pin_memory=pin_memory,
            )
            for layer, width in layer_widths
        }
        self._active = False
        self._rows = 0
        self._seen: set[int] = set()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def begin(self, *, num_extends: int, input_lengths: tuple[int, ...]) -> None:
        self._active = self.cadence.begin(
            num_extends=num_extends, input_lengths=input_lengths
        )
        self._rows = len(input_lengths)
        self._seen.clear()

    def observe(self, layer: int, topk_ids: torch.Tensor) -> None:
        if not self._active:
            return
        target = self.buffers[layer][: self._rows]
        if topk_ids.shape != target.shape:
            raise RuntimeError(
                "router rows differ from the replicated decode request layout"
            )
        if layer in self._seen:
            raise RuntimeError(
                "a router was executed twice in a captured decode forward"
            )
        target.copy_(topk_ids, non_blocking=True)
        self._seen.add(layer)

    def attach(self, layer: int, router: torch.nn.Module) -> None:
        def observe(_module, _inputs, output):
            if self._active:
                if output.format.is_bypassed():
                    raise RuntimeError(
                        "expert affinity requires precomputed top-k routes"
                    )
                self.observe(layer, output.topk_ids)

        self._hooks.append(router.register_forward_hook(observe))

    def finish(
        self,
        *,
        request_ids: tuple[str, ...],
        generations: tuple[int, ...],
        computed_frontiers: tuple[int, ...],
    ) -> CapturedRoutes | None:
        if not self._active:
            return None
        self._active = False
        if self._seen != set(self.buffers):
            raise RuntimeError("expert affinity did not observe all selected routers")
        if (
            not len(request_ids)
            == len(generations)
            == len(computed_frontiers)
            == self._rows
        ):
            raise RuntimeError("route capture identity differs from the forward rows")
        return CapturedRoutes(
            request_ids,
            generations,
            computed_frontiers,
            tuple(
                (layer, buffer[: self._rows]) for layer, buffer in self.buffers.items()
            ),
        )


def create_route_capture(
    model: torch.nn.Module,
    *,
    interval: int,
    max_layers: int,
    max_requests: int,
    leader: bool,
) -> RouteCapture | None:
    """Validate actual V4/V4.1 routers on every rank; capture only the TP leader."""
    if not 1 <= max_layers <= 32 or interval <= 0 or max_requests <= 0:
        raise ValueError("invalid expert affinity capture bounds")

    from tokenspeed.runtime.models.deepseek_v41_moe import DeepseekV4MoE

    layers = [module for module in model.modules() if isinstance(module, DeepseekV4MoE)]
    if not layers:
        raise ValueError("expert affinity currently requires DeepSeek V4/V4.1 routers")
    layers = sorted(layers, key=lambda module: module.layer_index)[:max_layers]
    widths = tuple(
        (module.layer_index, module.topk.topk_config.top_k) for module in layers
    )
    if sum(width for _, width in widths) > MAX_EXPERT_KEYS:
        raise ValueError("expert affinity selected layers exceed the 256-key bound")
    if any(module.topk.simulate_routing for module in layers):
        raise ValueError("expert affinity does not support simulated routing")
    if not leader:
        return None
    capture = RouteCapture(
        layer_widths=widths,
        interval=interval,
        max_requests=max_requests,
        pin_memory=True,
    )
    for module in layers:
        capture.attach(module.layer_index, module.topk)
    return capture
