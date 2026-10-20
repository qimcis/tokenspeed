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

"""Opt-in functional smoke checks against an already provisioned V4.1 EPD gateway.

Set DEEPSEEK_V41_EPD_BASE_URL to the gateway's full /v1 URL to enable these
checks. DEEPSEEK_V41_EPD_MODEL selects the served model alias; the default
matches the B300 recipe. DEEPSEEK_V41_EPD_API_KEY is optional.

This file never launches workers or downloads weights. Without the explicit
endpoint all checks skip, including on GPU hosts. Requests exercise observable
API behavior; successful responses do not establish actual encoder bypass,
cache-hit rates, rollback correctness, quality parity, or reliability under
worker failure. Those require instrumentation and separate acceptance runs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pytest


def _png_uri(rows: list[bytes], width: int) -> str:
    """Encode small RGB fixtures without Pillow, fonts, or remote resources."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    raw = b"".join(b"\0" + row for row in rows)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, len(rows), 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _color_image(color: str) -> str:
    rgb = {"red": b"\xff\0\0", "blue": b"\0\0\xff", "green": b"\0\xff\0"}[color]
    return _png_uri([rgb * 256] * 256, 256)


def _number_image() -> str:
    # Large block digits avoid font availability and OCR fixture downloads.
    four = ("10001", "10001", "10001", "11111", "00001", "00001", "00001")
    two = ("11111", "00001", "00001", "11111", "10000", "10000", "11111")
    rows = [b"\xff" * (320 * 3)] * 48
    for left, right in zip(four, two, strict=True):
        pattern = "00" + left + "00" + right + "00"
        row = b"".join(
            (b"\0\0\0" if bit == "1" else b"\xff\xff\xff") * 20 for bit in pattern
        )
        rows.extend([row] * 20)
    rows.extend([b"\xff" * (320 * 3)] * 48)
    return _png_uri(rows, 320)


def _messages(prompt: str, images: tuple[str, ...] = ()) -> list[dict]:
    if not images:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in images
                ),
                {"type": "text", "text": prompt},
            ],
        }
    ]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A configured inference endpoint must not forward fixtures or secrets.
        return None


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    api_key: str | None

    def request(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        thinking: bool = False,
        **options,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1024 if thinking else 128,
            "chat_template_kwargs": {"thinking": thinking},
            "stream": stream,
            **options,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        # Timeout bounds I/O; this harness does not retry inference requests.
        with build_opener(_NoRedirect()).open(request, timeout=300) as response:
            limit = 4 * 1024 * 1024
            if not stream:
                raw = response.read(limit + 1)
                assert len(raw) <= limit, "Response exceeded smoke-test byte limit"
                data = json.loads(raw)
                assert "error" not in data, data
                return data
            from tokenspeed.deployment.deepseek_v41.workload import SSEParser

            assert response.headers.get_content_type() == "text/event-stream"
            parser = SSEParser()
            events = []
            size = 0
            while raw := response.read1(65536):
                size += len(raw)
                assert size <= limit, "Response exceeded smoke-test byte limit"
                for text in parser.feed(raw):
                    if text == "[DONE]":
                        return events
                    event = json.loads(text)
                    assert "error" not in event, event
                    events.append(event)
            pytest.fail("Streaming response closed without [DONE]")


@pytest.fixture(scope="module")
def endpoint():
    url = os.environ.get("DEEPSEEK_V41_EPD_BASE_URL")
    if not url:
        pytest.skip(
            "Set DEEPSEEK_V41_EPD_BASE_URL to opt in to hardware endpoint checks"
        )
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/v1")
    ):
        pytest.fail(
            "DEEPSEEK_V41_EPD_BASE_URL must be an http(s) /v1 URL without credentials/query/fragment"
        )
    return Endpoint(
        url.rstrip("/"),
        os.environ.get("DEEPSEEK_V41_EPD_MODEL", "deepseek-v41-flash"),
        os.environ.get("DEEPSEEK_V41_EPD_API_KEY"),
    )


def _message(data: dict) -> dict:
    choices = data["choices"]
    assert len(choices) == 1
    assert choices[0]["finish_reason"] in {"stop", "tool_calls"}, choices[0]
    message = choices[0]["message"]
    assert message["role"] == "assistant"
    return message


def _content(data: dict) -> str:
    content = _message(data)["content"]
    assert isinstance(content, str) and content.strip()
    assert not any(
        marker in content
        for marker in ("<think>", "</think>", "<｜DSML｜", "<｜Assistant｜>")
    )
    return content


def _has_word(content: str, expected: str):
    assert re.search(
        r"\b" + re.escape(expected) + r"\b", content, re.IGNORECASE
    ), content


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("What is the capital of France? Reply with one word.", "Paris"),
        ("What is 17 times 23? Reply with the number only.", "391"),
    ],
)
def test_text_only_smoke(endpoint, prompt, expected):
    # API coverage of the intended text bypass. Bypass itself needs E telemetry.
    _has_word(_content(endpoint.request(_messages(prompt))), expected)


@pytest.mark.parametrize("color", ["red", "blue", "green"])
def test_image_color_smoke(endpoint, color):
    message = _messages(
        "What color is this image? Reply with one word.", (_color_image(color),)
    )
    _has_word(_content(endpoint.request(message)), color)


def test_image_at_end_of_user_content(endpoint):
    # Preserve the image-last ordering: appending prompt text after the image
    # can mask image-placeholder handling in the retained Engram token history.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this image? Reply with one word.",
                },
                {"type": "image_url", "image_url": {"url": _color_image("red")}},
            ],
        }
    ]
    _has_word(_content(endpoint.request(messages)), "red")


def test_image_content_smoke(endpoint):
    message = _messages(
        "What number is in this image? Reply with digits only.", (_number_image(),)
    )
    _has_word(_content(endpoint.request(message)), "42")


def test_repeated_and_replaced_image_isolation(endpoint):
    # Identical text and image dimensions; only content changes, then repeats.
    for color in ("red", "red", "blue", "red"):
        message = _messages(
            "What color is this image? Reply with one word.", (_color_image(color),)
        )
        _has_word(_content(endpoint.request(message)), color)


@pytest.mark.parametrize("colors", [("red", "blue"), ("blue", "red")])
def test_image_order(endpoint, colors):
    message = _messages(
        "Name the color of the FIRST image, then the SECOND image. Reply with just those two colors, in order.",
        tuple(_color_image(color) for color in colors),
    )
    content = _content(endpoint.request(message))
    observed = re.findall(r"\b(red|blue|green)\b", content.lower())
    assert observed == list(colors), content


def test_shared_prefix_and_image_followup(endpoint):
    # Repeat a long common prefix with distinct final questions. This exercises
    # reuse eligibility, but does not assert a hit without cache telemetry.
    prefix = "This is background context for the image question. " * 512
    first = _messages(
        prefix + "What color is the image? Reply with one word.", (_color_image("red"),)
    )
    answer = _content(endpoint.request(first))
    _has_word(answer, "red")
    conversation = (
        first
        + [{"role": "assistant", "content": answer}]
        + _messages(
            "Now what color is this new image? Reply with one word.",
            (_color_image("blue"),),
        )
    )
    _has_word(_content(endpoint.request(conversation)), "blue")
    _has_word(_content(endpoint.request(first)), "red")


def test_concurrent_text_and_images(endpoint):
    checks = [
        (_messages("What is 17 times 23? Reply with the number only."), "391"),
        (
            _messages(
                "What color is this image? Reply with one word.",
                (_color_image("blue"),),
            ),
            "blue",
        ),
        (_messages("What is the capital of France? Reply with one word."), "Paris"),
        (
            _messages(
                "What color is this image? Reply with one word.", (_color_image("red"),)
            ),
            "red",
        ),
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(endpoint.request, messages) for messages, _ in checks
        ]
        for future, (_, expected) in zip(futures, checks, strict=True):
            _has_word(_content(future.result()), expected)


@pytest.mark.parametrize("with_image", [False, True])
def test_streaming_content(endpoint, with_image):
    messages = (
        _messages(
            "What color is this image? Reply with one word.", (_color_image("blue"),)
        )
        if with_image
        else _messages("What is the capital of France? Reply with one word.")
    )
    events = endpoint.request(messages, stream=True)
    content = []
    finish_reasons = []
    for event in events:
        for choice in event.get("choices", []):
            assert choice["index"] == 0
            fragment = choice.get("delta", {}).get("content")
            if fragment is not None:
                assert isinstance(fragment, str)
                content.append(fragment)
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])
    assert finish_reasons == ["stop"], finish_reasons
    _has_word("".join(content), "blue" if with_image else "Paris")


def test_reasoning_parser(endpoint):
    data = endpoint.request(
        _messages("Compute 17 times 23 and give the result."),
        thinking=True,
        reasoning_effort="low",
    )
    message = _message(data)
    assert isinstance(message.get("reasoning_content"), str)
    assert message[
        "reasoning_content"
    ].strip(), "Reasoning parser returned no reasoning channel"
    _has_word(_content(data), "391")


def test_tool_call_parser(endpoint):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_color",
                "description": "Look up the named color.",
                "parameters": {
                    "type": "object",
                    "properties": {"color": {"type": "string", "enum": ["red"]}},
                    "required": ["color"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    data = endpoint.request(
        _messages("Call lookup_color with color red. Do not answer in prose."),
        tools=tools,
        tool_choice="auto",
        max_tokens=256,
    )
    message = _message(data)
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    calls = message.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 1, message
    call = calls[0]
    assert isinstance(call["id"], str) and call["id"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup_color"
    assert json.loads(call["function"]["arguments"]) == {"color": "red"}
