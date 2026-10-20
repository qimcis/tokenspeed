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

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from tokenspeed.deployment.deepseek_v41.workload import (
    Evaluation,
    ReplayConfig,
    SSEParser,
    Workload,
    join_evaluations,
    load_workload,
    run_workload,
    strict_json,
)


def workload(*, stream=False, count=1):
    return Workload.model_validate(
        {
            "workload_id": "held-out",
            "model": "test-model",
            "sampling_id": "greedy",
            "requests": [
                {
                    "task_id": str(i),
                    "cluster_id": "conversation",
                    "messages": [{"role": "user", "content": f"test {i}"}],
                    "sampling": {"temperature": 0, "max_tokens": 10, "seed": 42},
                    "stream": stream,
                }
                for i in range(count)
            ],
        }
    )


@contextmanager
def endpoint(responder):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append((self.path, payload))
            responder(self, payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def send(handler, content, *, status=200, content_type="application/json"):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    try:
        handler.wfile.write(content)
    except (BrokenPipeError, ConnectionResetError):
        pass


def config(url, **changes):
    return ReplayConfig.model_validate(
        {
            "endpoint": url,
            "concurrency": 2,
            "connect_timeout_seconds": 1.0,
            "request_timeout_seconds": 2.0,
            "max_response_bytes": 16384,
            **changes,
        }
    )


def completion(tokens=3):
    return json.dumps(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": tokens},
        }
    ).encode()


def sse_payload():
    parts = [
        {
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "think"},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"content": "✓"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]
    return (
        ": keepalive\r\n\r\n"
        + "".join(
            f"data: {json.dumps(part, ensure_ascii=False)}\r\n\r\n" for part in parts
        )
        + "data: [DONE]\r\n\r\n"
    ).encode()


def test_sse_handles_fragmentation_utf8_multiline_comments_and_crlf():
    data = '\ufeff: hi\r\ndata: {"text":\r\ndata: "✓"}\r\n\r\nevent: ignored\ndata:[DONE]\n\n'.encode()
    parser = SSEParser()
    events = []
    for byte in data:
        events.extend(parser.feed(bytes([byte])))
    assert events == ['{"text":\n"✓"}', "[DONE]"]
    assert parser.feed(b"data: unfinished") == []


def test_completed_http_has_unknown_quality_and_preserves_sampling():
    with endpoint(lambda handler, _: send(handler, completion())) as (url, captured):
        run = run_workload(workload(), config(url))
    assert run.quality == "unknown"
    row = run.requests[0]
    assert row.status == "completed" and row.success is None and row.admitted is True
    assert row.completion_tokens == 3 and row.ttft_ms is None
    assert row.response["choices"][0]["message"]["content"] == "test"
    assert captured == [
        (
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test 0"}],
                "temperature": 0,
                "max_tokens": 10,
                "seed": 42,
                "stream": False,
            },
        )
    ]
    outcomes = join_evaluations(
        run, [Evaluation(task_id="0", success=False, admitted=True, committed_tokens=3)]
    )
    assert outcomes[0].success is False
    assert outcomes[0].committed_tokens == 3
    assert outcomes[0].latency_ms == row.queue_ms + row.latency_ms


def test_stream_collects_usage_and_first_generated_output():
    with endpoint(
        lambda handler, _: send(
            handler, sse_payload(), content_type="text/event-stream"
        )
    ) as (url, captured):
        run = run_workload(workload(stream=True), config(url))
    row = run.requests[0]
    assert row.status == "completed"
    assert row.completion_tokens == 3
    assert row.ttft_ms is not None and 0 <= row.ttft_ms <= row.latency_ms
    assert len(row.events) == 5 and row.response is None
    assert captured[0][1]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    "content",
    [
        sse_payload().replace(b"data: [DONE]\r\n\r\n", b""),
        b'data: {"choices": []}\n\ndata: [DONE]\n\n',
        b'data: {"error": {"message": "failed"}}\n\n',
        b"data: not-json\n\n",
    ],
)
def test_stream_truncation_and_server_errors_are_failures(content):
    with endpoint(
        lambda handler, _: send(handler, content, content_type="text/event-stream")
    ) as (url, _):
        run = run_workload(workload(stream=True), config(url))
    assert run.requests[0].status == "protocol_error"
    with pytest.raises(ValueError, match="incomplete HTTP"):
        join_evaluations(
            run,
            [Evaluation(task_id="0", success=True, admitted=True, committed_tokens=3)],
        )


@pytest.mark.parametrize("status,admitted", [(429, False), (503, None), (302, None)])
def test_http_failures_are_retained_and_not_retried(status, admitted):
    with endpoint(lambda handler, _: send(handler, b"{}", status=status)) as (
        url,
        captured,
    ):
        run = run_workload(workload(), config(url))
    row = run.requests[0]
    assert row.status == "http_error" and row.http_status == status
    assert row.admitted is admitted and len(captured) == 1
    evaluation = Evaluation(
        task_id="0", success=False, admitted=False, committed_tokens=0
    )
    assert join_evaluations(run, [evaluation])[0].success is False


def test_timeout_and_response_limit():
    def stalled(handler, _):
        time.sleep(0.15)
        send(handler, completion())

    with endpoint(stalled) as (url, _):
        run = run_workload(
            workload(),
            config(url, connect_timeout_seconds=0.04, request_timeout_seconds=0.04),
        )
    assert run.requests[0].status == "transport_error"
    with endpoint(lambda handler, _: send(handler, completion())) as (url, _):
        run = run_workload(workload(), config(url, max_response_bytes=10))
    assert run.requests[0].status == "protocol_error"
    assert "byte limit" in run.requests[0].error


def test_actual_http_concurrency_is_bounded_and_output_order_preserved():
    lock = threading.Lock()
    active = 0
    peak = 0

    def respond(handler, _):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        send(handler, completion())
        with lock:
            active -= 1

    with endpoint(respond) as (url, captured):
        run = run_workload(workload(count=6), config(url, concurrency=2))
    assert 1 <= peak <= 2
    assert len(captured) == 6
    assert [row.task_id for row in run.requests] == [str(i) for i in range(6)]
    assert all(row.status == "completed" for row in run.requests)


def test_evaluations_cannot_drop_failures_or_override_measured_counts():
    with endpoint(lambda handler, _: send(handler, completion())) as (url, _):
        run = run_workload(workload(), config(url))
    with pytest.raises(ValueError, match="every measured task"):
        join_evaluations(run, [])
    for bad, match in [
        (dict(admitted=False, committed_tokens=3), "admission"),
        (dict(admitted=True, committed_tokens=4), "token count"),
    ]:
        with pytest.raises(ValueError, match=match):
            join_evaluations(run, [Evaluation(task_id="0", success=False, **bad)])


@pytest.mark.parametrize("value", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}'])
def test_nonfinite_or_duplicate_json_is_rejected(value):
    with pytest.raises(ValueError):
        strict_json(value)


def test_strict_workload_rejects_unknown_fields_and_reserved_overrides(tmp_path):
    data = workload().model_dump()
    data["requests"][0]["sampling"]["stream"] = True
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="cannot override"):
        load_workload(path)
    data = workload().model_dump()
    data["extra"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_workload(path)
    data = workload().model_dump()
    data["requests"][0]["stream"] = "false"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_workload(path)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/chat/completions",
        "https://secret@example.com/chat/completions",
        "https://example.com/chat/completions?key=secret",
        "https://example.com/v1",
    ],
)
def test_endpoint_requires_explicit_completion_url_without_credentials(url):
    with pytest.raises(ValidationError):
        config(url)


def test_stream_first_output_is_measured_before_delayed_completion():
    def streaming(handler, _):
        first = b'data: {"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}\n\n'
        last = b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(first) + len(last)))
        handler.end_headers()
        handler.wfile.write(first)
        handler.wfile.flush()
        time.sleep(0.15)
        handler.wfile.write(last)
        handler.wfile.flush()

    with endpoint(streaming) as (url, _):
        run = run_workload(workload(stream=True), config(url))
    row = run.requests[0]
    assert row.status == "completed"
    assert row.ttft_ms is not None
    assert row.latency_ms - row.ttft_ms >= 80
    assert row.completion_tokens is None


def test_chunked_stream_and_tool_arguments_are_measured():
    def streaming(handler, _):
        first = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"function": {"arguments": '{"value":1}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ).encode()
        data = (
            b"data: "
            + first
            + b'\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
        )
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for part in (data[:7], data[7:25], data[25:]):
            handler.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            handler.wfile.flush()
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()

    with endpoint(streaming) as (url, _):
        run = run_workload(workload(stream=True), config(url))
    row = run.requests[0]
    assert row.status == "completed" and row.ttft_ms is not None
    assert len(row.events) == 2


def test_malformed_role_is_a_validation_error():
    data = workload().model_dump()
    data["requests"][0]["messages"][0]["role"] = []
    with pytest.raises(ValidationError, match="supported role"):
        Workload.model_validate(data)
