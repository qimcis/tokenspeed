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

"""CPU token-span contracts and a real frontend regression when runtime is installed."""

import asyncio
import importlib.util
import logging
from dataclasses import dataclass

import pytest

from tokenspeed.runtime.multimodal.token_padding import pad_input_tokens


@dataclass
class SpanItem:
    pad_value: int | None
    offsets: list[tuple[int, int]] | None


@dataclass
class SpanInputs:
    mm_items: list[SpanItem]


def test_image_sentinels_and_adjacent_history_keep_physical_ids():
    # V4.1's image block includes start, patch, newline and end rows.
    authored = [12, 13, 5, 5, 5, 5]
    image_id = 1_000_123
    padded = pad_input_tokens(authored, SpanInputs([SpanItem(image_id, [(2, 5)])]))
    assert padded == [12, 13, image_id, image_id, image_id, image_id]
    assert authored == [12, 13, 5, 5, 5, 5]
    # A bootstrap token immediately after an image sees out-of-vocabulary IDs.
    assert (padded + [44])[-4:] == [image_id, image_id, image_id, 44]


def test_multiple_images_and_repeated_spans_preserve_text_and_order():
    ids = [10, 5, 5, 20, 5, 5, 30, 5, 5, 40]
    first, second = 1_000_123, 1_000_456
    padded = pad_input_tokens(
        ids, SpanInputs([SpanItem(first, [(1, 2), (7, 8)]), SpanItem(second, [(4, 5)])])
    )
    assert padded == [10, first, first, 20, second, second, 30, first, first, 40]
    assert len(padded) == len(ids)


@pytest.mark.parametrize(
    "items", [[], [SpanItem(None, [(1, 2)])], [SpanItem(1_000_123, None)]]
)
def test_absent_spans_leave_text_unchanged(items):
    ids = [10, 11, 12]
    assert pad_input_tokens(ids, SpanInputs(items)) == ids


def test_empty_prompt_needs_no_padding():
    assert pad_input_tokens([], SpanInputs([SpanItem(1_000_123, [(0, 3)])])) == []


@pytest.mark.parametrize("role,language_only", [("prefill", False), ("decode", True)])
def test_frontend_preserves_image_tail_without_decode_features(role, language_only):
    # Importing the real frontend loads GPU kernel registrations. Keep this
    # regression runnable in runtime CI without replacing the frontend or ops.
    if importlib.util.find_spec("tokenspeed_triton") is None:
        pytest.skip(
            "Real frontend requires tokenspeed-triton runtime imports; pure padding checks run independently"
        )
    from tokenspeed.runtime.engine.input_processor import InputProcessor
    from tokenspeed.runtime.engine.io_struct import GenerateReqInput
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    @dataclass
    class Server:
        disaggregation_mode: str
        language_model_only: bool
        reasoning_parser: str | None = None
        enable_prefix_caching: bool = True
        enable_output_logprobs: bool = False

    @dataclass
    class Config:
        is_multimodal_active: bool
        is_multimodal: bool = True
        vocab_size: int = 128
        hf_config: object = None

    @dataclass
    class Engine:
        server_args: Server
        model_config: Config
        context_len: int = 128
        max_req_input_len: int = 127
        is_generation: bool = True
        tokenizer: object = None
        logger: object = logging.getLogger(__name__)

    authored = [12, 13, 5, 5, 5, 5]
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=123, offsets=[(2, 5)])
    mm = MultimodalInputs(mm_items=[item])
    request = GenerateReqInput(
        input_ids=authored,
        precomputed_multimodal_inputs=mm,
        sampling_params={"max_new_tokens": 4},
    )
    request.normalize_batch_and_arguments()
    processor = InputProcessor(
        Engine(Server(role, language_only), Config(not language_only))
    )
    result = asyncio.run(processor.tokenize_one_request(request))
    assert result.input_ids == [
        12,
        13,
        item.pad_value,
        item.pad_value,
        item.pad_value,
        item.pad_value,
    ]
    assert result.input_ids_unpadded == authored
    assert result.multimodal_inputs is (None if language_only else mm)
