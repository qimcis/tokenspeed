# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Request-level acceptance reflects actual verify work, not fallback or native width."""

from types import SimpleNamespace

from tokenspeed.runtime.engine.request_stats import RequestStats, RequestStatsTracker
from tokenspeed.runtime.engine.request_types import FINISH_LENGTH


def state(rounds: int, verified: int, spec_output: int, fallback_output: int):
    return SimpleNamespace(
        stats=RequestStatsTracker(),
        input_length=10,
        output_length=1 + spec_output + fallback_output,
        spec_verify_ct=rounds,
        spec_verify_tokens=verified,
        spec_output_tokens=spec_output,
        # A cached terminal summary must not replace the actual round counters.
        accept_draft_tokens=None,
        finished_reason=FINISH_LENGTH(length=1 + spec_output + fallback_output),
        cached_tokens=0,
        created_time=0.0,
    )


def test_fixed_width_preserves_legacy_anchor_inclusive_rate():
    # Ten width-four rounds committed three tokens each: two draft tokens and
    # one target token per round. The legacy rate is 20 / 40, not 20 / 30.
    stats = RequestStats.from_state(
        state(10, 40, 30, 0), spec_algorithm="EAGLE", spec_num_tokens=4
    )
    assert stats.acc_len == 3.0
    assert stats.acc_rate == 0.5


def test_fallback_does_not_dilute_acceptance_or_add_draft_slots():
    # Two width-six rounds commit six then three tokens; three additional
    # ordinary target rounds do not alter the speculative acceptance measures.
    speculative = RequestStats.from_state(
        state(2, 12, 9, 0), spec_algorithm="DFLASH", spec_num_tokens=8
    )
    with_fallback = RequestStats.from_state(
        state(2, 12, 9, 3), spec_algorithm="DFLASH", spec_num_tokens=8
    )
    assert with_fallback.acc_len == speculative.acc_len == 4.5
    assert with_fallback.acc_rate == speculative.acc_rate == 0.5833
    assert with_fallback.output_tokens == speculative.output_tokens + 3


def test_actual_verify_widths_are_summed_without_rounding_the_average_first():
    # A width-six round commits six and a width-four round commits two.
    stats = RequestStats.from_state(
        state(2, 10, 8, 5), spec_algorithm="DFLASH", spec_num_tokens=8
    )
    assert stats.acc_len == 4.0
    assert stats.acc_rate == 0.6


def test_fallback_only_has_no_speculative_acceptance():
    stats = RequestStats.from_state(
        state(0, 0, 0, 10), spec_algorithm="DFLASH", spec_num_tokens=8
    )
    assert stats.acc_len is None
    assert stats.acc_rate is None
