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

"""Bounded HTTP measurements; task quality requires an external evaluator.

This is a closed-loop request driver, not an agent runner or arrival-rate load
model. Latency begins when a worker dispatches a request; queue time is reported
separately. A workload row must represent one independently graded task.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import requests
from pydantic import Field, JsonValue, model_validator
from urllib3.exceptions import HTTPError as TransportError

from tokenspeed.deployment.deepseek_v41.analysis import Outcome
from tokenspeed.deployment.deepseek_v41.config import StrictConfig

NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def strict_json(value: str | bytes) -> JsonValue:
    """Read JSON without nonfinite numbers or ambiguous duplicate object keys."""

    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    result = json.loads(
        value, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )
    # JSON numeric overflow (e.g. 1e999) does not use parse_constant.
    json.dumps(result, allow_nan=False)
    return result


class WorkloadRequest(StrictConfig):
    task_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    messages: list[dict[str, JsonValue]] = Field(min_length=1)
    sampling: dict[str, JsonValue]
    stream: bool

    @model_validator(mode="after")
    def validate_request(self):
        reserved = {"model", "messages", "stream", "stream_options"}
        if reserved.intersection(self.sampling):
            raise ValueError("sampling cannot override model, messages or streaming")
        if "n" in self.sampling and (
            type(self.sampling["n"]) is not int or self.sampling["n"] != 1
        ):
            raise ValueError("measurements require one response choice (n=1)")
        roles = {"system", "developer", "user", "assistant", "tool", "function"}
        if any(
            not isinstance(message.get("role"), str) or message["role"] not in roles
            for message in self.messages
        ):
            raise ValueError("each message needs a supported role")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class Workload(StrictConfig):
    workload_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    sampling_id: str = Field(min_length=1)
    requests: list[WorkloadRequest] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def unique_tasks(self):
        if len({row.task_id for row in self.requests}) != len(self.requests):
            raise ValueError(
                "task IDs must be unique; retries require external aggregation"
            )
        return self


def load_workload(path: str | Path) -> Workload:
    """Read a strict JSON workload. No default prompts or sampling are inserted."""
    return Workload.model_validate(strict_json(Path(path).read_bytes()))


class ReplayConfig(StrictConfig):
    endpoint: str
    concurrency: Annotated[int, Field(ge=1, le=256)]
    connect_timeout_seconds: PositiveFloat
    request_timeout_seconds: PositiveFloat
    max_response_bytes: Annotated[int, Field(ge=1, le=64 * 1024 * 1024)]

    @model_validator(mode="after")
    def validate_endpoint(self):
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "endpoint must be an HTTP(S) URL without credentials, query or fragment"
            )
        if not parsed.path.endswith("/chat/completions"):
            raise ValueError("endpoint must include the full /chat/completions path")
        if self.connect_timeout_seconds > self.request_timeout_seconds:
            raise ValueError("connect timeout cannot exceed request timeout")
        return self


class Measurement(StrictConfig):
    task_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    status: Literal["completed", "http_error", "transport_error", "protocol_error"]
    http_status: int | None
    admitted: bool | None
    success: None = None
    queue_ms: NonnegativeFloat
    latency_ms: NonnegativeFloat
    ttft_ms: NonnegativeFloat | None
    completion_tokens: NonnegativeInt | None
    response: dict[str, JsonValue] | None
    events: list[dict[str, JsonValue]]
    error: str | None

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.ttft_ms is not None and self.ttft_ms > self.latency_ms:
            raise ValueError("TTFT cannot exceed request latency")
        if self.admitted is False and (
            self.ttft_ms is not None or self.completion_tokens
        ):
            raise ValueError("rejected requests cannot produce output")
        if self.status == "completed" and (
            self.http_status is None
            or not 200 <= self.http_status < 300
            or self.admitted is not True
            or self.error is not None
        ):
            raise ValueError("completed measurements need a successful HTTP response")
        if self.status == "http_error" and (
            self.http_status is None or 200 <= self.http_status < 300
        ):
            raise ValueError("HTTP errors need an unsuccessful HTTP status")
        return self


class MeasurementRun(StrictConfig):
    workload_id: str
    model: str
    sampling_id: str
    duration_seconds: PositiveFloat
    requests: list[Measurement] = Field(min_length=1, max_length=10000)
    quality: Literal["unknown"] = "unknown"
    release_approved: Literal[False] = False


class SSEParser:
    """Parse UTF-8 SSE events across arbitrary byte and CR/LF boundaries.

    Only dispatched data fields produce events. Comments, id, event and retry
    fields are ignored; multiline data is joined using the SSE newline rule.
    An incomplete event at EOF is not a dispatched event.
    """

    def __init__(self):
        self.line = bytearray()
        self.data: list[str] = []
        self.skip_lf = False
        self.first_line = True

    def _line(self) -> str | None:
        line = self.line.decode("utf-8", errors="strict")
        self.line.clear()
        if self.first_line:
            line = line.removeprefix("\ufeff")
            self.first_line = False
        if not line:
            if not self.data:
                return None
            event = "\n".join(self.data)
            self.data.clear()
            return event
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.data.append(value)
        return None

    def feed(self, chunk: bytes) -> list[str]:
        events = []
        for byte in chunk:
            if self.skip_lf:
                self.skip_lf = False
                if byte == 10:
                    continue
            if byte in (10, 13):
                event = self._line()
                if event is not None:
                    events.append(event)
                self.skip_lf = byte == 13
            else:
                self.line.append(byte)
        return events


def _completion_tokens(payload: dict) -> int | None:
    usage = payload.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise ValueError("invalid usage object")
    tokens = usage.get("completion_tokens")
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError("invalid completion token count")
    return tokens


def _has_output(delta: dict) -> bool:
    # Chunk counts are not token counts. Tool names alone and assistant-role
    # announcements do not establish first generated output.
    if any(
        isinstance(delta.get(key), str) and delta[key]
        for key in ("content", "reasoning_content", "reasoning")
    ):
        return True
    calls = delta.get("tool_calls", [])
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict) and function.get("arguments"):
                    return True
    function = delta.get("function_call")
    return isinstance(function, dict) and bool(function.get("arguments"))


def _measure(
    row: WorkloadRequest,
    workload: Workload,
    config: ReplayConfig,
    api_key: str | None,
    run_started: float,
) -> Measurement:
    started = time.monotonic()
    deadline = started + config.request_timeout_seconds
    http_status = None
    admitted = None
    status = "transport_error"
    error = None
    ttft = None
    tokens = None
    response_payload = None
    events = []
    body = {
        "model": workload.model,
        "messages": row.messages,
        **row.sampling,
        "stream": row.stream,
    }
    if row.stream:
        body["stream_options"] = {"include_usage": True}
    headers = {
        "Accept": "text/event-stream" if row.stream else "application/json",
        "Accept-Encoding": "identity",
    }
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        # Disable redirects to avoid replaying private prompts to a different URL.
        # Requests does not retry POST by default, so attempts stays exactly one.
        with requests.post(
            config.endpoint,
            json=body,
            headers=headers,
            stream=True,
            timeout=(config.connect_timeout_seconds, config.request_timeout_seconds),
            allow_redirects=False,
        ) as response:
            http_status = response.status_code
            if not 200 <= http_status < 300:
                status = "http_error"
                admitted = False if http_status == 429 else None
                error = f"HTTP {http_status}"
            else:
                admitted = True
                status = "protocol_error"
                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                expected = "text/event-stream" if row.stream else "application/json"
                if content_type != expected:
                    raise ValueError(f"expected {expected} response")
                if (
                    response.headers.get("Content-Encoding", "identity").lower()
                    != "identity"
                ):
                    raise ValueError("server ignored identity content encoding")
                parser = SSEParser()
                raw = bytearray()
                received = 0
                done = False
                finished = False
                # read1 returns available socket data without waiting to fill a
                # large buffer, preserving SSE timing without per-byte I/O calls.
                # Socket inactivity is separately bounded by the requests timeout.
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("request elapsed deadline exceeded")
                    chunk = response.raw.read1(65536, decode_content=False)
                    if time.monotonic() > deadline:
                        raise TimeoutError("request elapsed deadline exceeded")
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > config.max_response_bytes:
                        raise ValueError("response exceeds configured byte limit")
                    if not row.stream:
                        raw.extend(chunk)
                        continue
                    for event in parser.feed(chunk):
                        if event == "[DONE]":
                            done = True
                            break
                        payload = strict_json(event)
                        if not isinstance(payload, dict) or "error" in payload:
                            raise ValueError("invalid SSE response or server error")
                        choices = payload.get("choices")
                        if not isinstance(choices, list) or len(choices) > 1:
                            raise ValueError("invalid SSE choices")
                        if choices:
                            choice = choices[0]
                            if not isinstance(choice, dict) or not isinstance(
                                choice.get("delta"), dict
                            ):
                                raise ValueError("invalid SSE choice delta")
                            if choice.get("index") != 0:
                                raise ValueError("unexpected response choice index")
                            if ttft is None and _has_output(choice["delta"]):
                                ttft = (time.monotonic() - started) * 1000
                            if choice.get("finish_reason") is not None:
                                finished = True
                        observed_tokens = _completion_tokens(payload)
                        if observed_tokens is not None:
                            tokens = observed_tokens
                        events.append(payload)
                    if done:
                        break
                if row.stream:
                    if not done or not finished:
                        raise ValueError(
                            "stream ended without finish reason and [DONE]"
                        )
                else:
                    payload = strict_json(bytes(raw))
                    if not isinstance(payload, dict) or "error" in payload:
                        raise ValueError("invalid completion response or server error")
                    choices = payload.get("choices")
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 1
                        or not isinstance(choices[0], dict)
                        or not isinstance(choices[0].get("message"), dict)
                        or choices[0].get("finish_reason") is None
                    ):
                        raise ValueError("response needs one finished message")
                    response_payload = payload
                    tokens = _completion_tokens(payload)
                status = "completed"
    except (requests.RequestException, TransportError, TimeoutError) as exc:
        status = "transport_error"
        error = type(exc).__name__
    except (ValueError, UnicodeError) as exc:
        status = "protocol_error"
        error = str(exc)
    return Measurement(
        task_id=row.task_id,
        cluster_id=row.cluster_id,
        status=status,
        http_status=http_status,
        admitted=admitted,
        queue_ms=(started - run_started) * 1000,
        latency_ms=(time.monotonic() - started) * 1000,
        ttft_ms=ttft,
        completion_tokens=tokens,
        response=response_payload,
        events=events,
        error=error,
    )


def run_workload(
    workload: Workload, config: ReplayConfig, *, api_key: str | None = None
) -> MeasurementRun:
    """POST one request per task with bounded concurrency, bytes and timeouts.

    Endpoint must be the full chat-completions URL. No retries, automatic task
    grading, costs or model-revision claims are inferred. The caller owns
    endpoint selection and credential retrieval; credentials are not serialized.
    Socket inactivity is bounded by request_timeout_seconds, while the elapsed
    deadline is checked on every available socket read. DNS is controlled by the OS.
    """
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        rows = list(
            executor.map(
                lambda row: _measure(row, workload, config, api_key, started),
                workload.requests,
            )
        )
    return MeasurementRun(
        workload_id=workload.workload_id,
        model=workload.model,
        sampling_id=workload.sampling_id,
        duration_seconds=max(time.monotonic() - started, 1e-9),
        requests=rows,
    )


class Evaluation(StrictConfig):
    task_id: str = Field(min_length=1)
    success: bool
    admitted: bool
    committed_tokens: NonnegativeInt


def join_evaluations(
    run: MeasurementRun, evaluations: list[Evaluation]
) -> list[Outcome]:
    """Join independently graded tasks without dropping transport failures.

    Evaluators must explicitly resolve unknown admission and token counts.
    The returned rows can populate Experiment.outcomes once actual total resource
    costs, checkpoint revision and execution mode are supplied separately.
    """
    graded = {row.task_id: row for row in evaluations}
    ids = [row.task_id for row in run.requests]
    if (
        len(graded) != len(evaluations)
        or len(set(ids)) != len(ids)
        or set(graded) != set(ids)
    ):
        raise ValueError("provide exactly one evaluation for every measured task")
    outcomes = []
    for row in run.requests:
        evaluation = graded[row.task_id]
        if row.admitted is not None and row.admitted != evaluation.admitted:
            raise ValueError(f"admission disagrees for task {row.task_id}")
        if row.status != "completed" and evaluation.success:
            raise ValueError("incomplete HTTP requests cannot be graded successful")
        if (
            row.completion_tokens is not None
            and row.completion_tokens != evaluation.committed_tokens
        ):
            raise ValueError(f"committed token count disagrees for task {row.task_id}")
        outcomes.append(
            Outcome(
                task_id=row.task_id,
                cluster_id=row.cluster_id,
                admitted=evaluation.admitted,
                success=evaluation.success,
                attempts=1,
                committed_tokens=evaluation.committed_tokens,
                latency_ms=row.queue_ms + row.latency_ms,
                ttft_ms=None if row.ttft_ms is None else row.queue_ms + row.ttft_ms,
            )
        )
    return outcomes
