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

"""CPU protocol checks; these do not prove CUDA memory ordering or compilation."""

import ast
import random
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def _workspace_class():
    source = (
        Path(__file__).parents[3]
        / "python/tokenspeed_kernel/ops/communication/hopper_mhc.py"
    ).read_text()
    tree = ast.parse(source)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "V41AllReduceWorkspace"
    )
    namespace = {
        "dataclass": dataclass,
        "Any": Any,
        "torch": torch,
        "dist": SimpleNamespace(ProcessGroup=object),
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "workspace", "exec"),
        namespace,
    )
    return namespace["V41AllReduceWorkspace"]


class _Protocol:
    """Interleave individual publication/consumption operations across CTAs."""

    def __init__(self, rows, world):
        self.rows = rows
        self.world = world
        self.call = [0] * world
        self.phase = [[0] * 8 for _ in range(world)]
        self.cursor = [[0] * 8 for _ in range(world)]
        self.signals = [
            [[[0] * world for _ in range(world)] for _ in range(8)] for _ in range(2)
        ]
        self.data = [[None] * 8 for _ in range(world)]
        self.reads = 0

    def active(self):
        return [
            (rank, row)
            for rank in range(self.world)
            if self.call[rank] < len(self.rows)
            for row in range(self.rows[self.call[rank]])
            if self.phase[rank][row] != 6
        ]

    def step(self, rank, row):
        call = self.call[rank]
        phase = self.phase[rank][row]
        peer = self.cursor[rank][row]
        if phase == 0:
            self.data[rank][row] = (call, row, rank)
            self.phase[rank][row] = 1
            return True
        if phase in (1, 4):
            signals = self.signals[0 if phase == 1 else 1][row]
            if signals[peer][rank]:
                return False
            signals[peer][rank] = 1
        elif phase in (2, 5):
            signals = self.signals[0 if phase == 2 else 1][row]
            if not signals[rank][peer]:
                return False
            signals[rank][peer] = 0
        else:
            assert phase == 3
            assert self.data[peer][row] == (call, row, peer), (
                call,
                rank,
                row,
                peer,
                self.data[peer][row],
            )
            self.reads += 1
        peer += 1
        if peer == self.world:
            self.phase[rank][row] += 1
            peer = 0
        self.cursor[rank][row] = peer
        if all(self.phase[rank][r] == 6 for r in range(self.rows[call])):
            self.call[rank] += 1
            self.phase[rank] = [0] * 8
            self.cursor[rank] = [0] * 8
        return True


class TestHopperMhcProtocol(unittest.TestCase):
    def test_random_rank_skew_and_replayed_geometry(self):
        rows = (8, 1, 4, 2, 1, 8, 2, 4) * 3
        for seed in range(16):
            rng = random.Random(seed)
            protocol = _Protocol(rows, 8)
            active = protocol.active()
            while active:
                rng.shuffle(active)
                self.assertTrue(
                    any(protocol.step(rank, row) for rank, row in active),
                    "the valid serial collective sequence must make progress",
                )
                active = protocol.active()
            self.assertEqual(protocol.reads, sum(rows) * 8 * 8)
            self.assertFalse(
                any(
                    signal
                    for phase in protocol.signals
                    for row in phase
                    for rank in row
                    for signal in rank
                )
            )

    def test_missing_peer_cannot_produce_complete_output(self):
        protocol = _Protocol((1,), 8)
        while True:
            active = [(rank, row) for rank, row in protocol.active() if rank != 7]
            if not any(protocol.step(rank, row) for rank, row in active):
                break
        self.assertEqual(protocol.reads, 0)
        self.assertEqual(protocol.call, [0] * 8)

    def test_round_modes_are_intentionally_different(self):
        values = torch.tensor([256, 1, -256, 0, 0, 0, 0, 0], dtype=torch.bfloat16)
        bf16_step = values[0]
        fp32 = values[0].float()
        for value in values[1:]:
            bf16_step = (bf16_step.float() + value.float()).to(torch.bfloat16)
            fp32 = fp32 + value.float()
        self.assertEqual(bf16_step.item(), 0)
        self.assertEqual(fp32.to(torch.bfloat16).item(), 1)

    def test_host_geometry_gate(self):
        cls = _workspace_class()
        workspace = cls(
            object(),
            torch.device("cpu"),
            0,
            4,
            "bf16_step",
            1_000_000,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        workspace.validate_input(torch.empty((4, 5120), dtype=torch.bfloat16))
        for shape, dtype in (
            ((3, 5120), torch.bfloat16),
            ((8, 5120), torch.bfloat16),
            ((1, 4096), torch.bfloat16),
            ((1, 5120), torch.float32),
        ):
            with self.assertRaises(ValueError):
                workspace.validate_input(torch.empty(shape, dtype=dtype))
        with self.assertRaises(ValueError):
            workspace.validate_input(torch.empty((5120, 2), dtype=torch.bfloat16).T)


if __name__ == "__main__":
    unittest.main()
