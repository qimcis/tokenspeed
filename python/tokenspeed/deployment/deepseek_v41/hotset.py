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

"""Build reproducible immutable Engram hot sets from bounded frequency records."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from tokenspeed.deployment.deepseek_v41.config import StrictConfig
from tokenspeed.runtime.models.engram_hot_cache import build_engram_hot_set_manifest


class RowFrequency(StrictConfig):
    row_id: Annotated[int, Field(ge=0)]
    count: Annotated[int, Field(ge=0)]


class LayerFrequencies(StrictConfig):
    layer_id: Annotated[int, Field(ge=0)]
    total_rows: Annotated[int, Field(gt=0)]
    counts: list[RowFrequency]

    @model_validator(mode="after")
    def validate_rows(self):
        ids = [row.row_id for row in self.counts]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "aggregate repeated row observations before building a hot set"
            )
        if any(row >= self.total_rows for row in ids):
            raise ValueError("observed row is outside its Engram table")
        return self


class HotSetFrequencies(StrictConfig):
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    layers: list[LayerFrequencies] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_layers(self):
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("frequency records must have unique layer IDs")
        return self


def build_hotset(raw: dict, *, capacity_rows: int) -> dict:
    """Select frequent rows per layer; capacity is explicit, never an inferred win."""
    records = HotSetFrequencies.model_validate(raw)
    return build_engram_hot_set_manifest(
        model_revision=records.model_revision,
        layer_rows={layer.layer_id: layer.total_rows for layer in records.layers},
        frequencies={
            layer.layer_id: {row.row_id: row.count for row in layer.counts}
            for layer in records.layers
        },
        capacity_rows=capacity_rows,
    ).to_dict()
