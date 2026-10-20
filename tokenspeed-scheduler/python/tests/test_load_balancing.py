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

"""Actual native binding contracts for placement and shared prefill work."""

import math

import pytest
from test_fsm_and_scheduling import make_config, submit
from tokenspeed_scheduler import Scheduler


def test_shared_budget_and_milliseconds_diagnostic():
    scheduler = Scheduler(make_config())
    scheduler.configure_prefill_work_budget(3, 1, 0, 0)
    submit(scheduler, "a", [1, 2])
    submit(scheduler, "b", [3, 4, 5])
    batch = scheduler.next_execution_plan().forward[0]
    assert batch.input_lengths == [2, 1]
    assert scheduler.last_prefill_work_ms() == 3
    assert scheduler.prefill_work_overruns() == 0


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (math.inf, 1, 0, 0),
        (1, -1, 0, 0),
        (1, 1, math.nan, 0),
        (1, 1e308, 0, 0),
    ],
)
def test_invalid_cost_is_rejected(values):
    scheduler = Scheduler(make_config())
    with pytest.raises(ValueError):
        scheduler.configure_prefill_work_budget(*values)


def test_epoch_replacement_is_atomic_and_validated():
    scheduler = Scheduler(make_config(max_batch_size=4))
    scheduler.configure_expert_affinity(4, 2, 8)
    scheduler.set_expert_placement(1, [1, 2], [[0.5, 0.5], [1, 0]])
    assert scheduler.expert_placement_epoch() == 1
    for keys, rows in [
        ([1], [[1, 0]]),
        ([1, 2], [[1, 0], [0, 0]]),
        ([1, 2], [[1, 0], [math.nan, 1]]),
        ([1, 1], [[1, 0], [0, 1]]),
    ]:
        with pytest.raises(ValueError):
            scheduler.set_expert_placement(2, keys, rows)
        assert scheduler.expert_placement_epoch() == 1
    scheduler.set_expert_placement(2, [1, 2], [[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        scheduler.set_expert_placement(1, [1, 2], [[1, 0], [0, 1]])
    assert scheduler.expert_placement_epoch() == 2
