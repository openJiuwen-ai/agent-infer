# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Server-side JSON escaping must preserve SSE content at every UTF-8 split."""

import asyncio
import json

import pytest
from httpx._decoders import LineDecoder

from agentinfer.agentcache.core.sse_framing import SSEJsonFramingMiddleware

pytestmark = pytest.mark.cpu_test


def send_chunks(chunks, *, headers=None, scope_type="http"):
    messages = []
    headers = headers if headers is not None else [(b"content-type", b"text/event-stream")]

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        for i, chunk in enumerate(chunks):
            await send({"type": "http.response.body", "body": chunk, "more_body": i < len(chunks) - 1})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def capture(message):
        messages.append(message)

    asyncio.run(SSEJsonFramingMiddleware(app)({"type": scope_type}, receive, capture))
    return messages


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_each_asgi_utf8_boundary_preserves_exact_json_and_done(separator):
    payload = {
        "choices": [{"text": "你好 😀 prefix" + separator + "suffix\\u2028", "finish_reason": None}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 7},
    }
    wire = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
    for boundary in range(len(wire) + 1):
        messages = send_chunks([wire[:boundary], wire[boundary:]])
        output = b"".join(m.get("body", b"") for m in messages)
        lines = LineDecoder().decode(output.decode())
        assert json.loads(lines[0][5:]) == payload
        assert [line for line in lines if line.startswith("data:")][1] == "data: [DONE]"
        assert separator.encode() not in output
        assert messages[-1]["more_body"] is False


def test_one_byte_chunks_preserve_unrelated_unicode_and_multiple_separators():
    value = {"text": "x\u0085\u2028\u2029😀\r\n\t汉字"}
    wire = ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()
    messages = send_chunks([bytes([byte]) for byte in wire])
    output = b"".join(m.get("body", b"") for m in messages)
    assert json.loads(LineDecoder().decode(output.decode())[0][5:]) == value


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-type", b"application/json")],
        [(b"content-type", b"text/event-stream"), (b"content-encoding", b"gzip")],
    ],
)
def test_unrelated_or_encoded_bodies_are_untouched(headers):
    body = b"raw" + b"\xe2\x80\xa8"
    messages = send_chunks([body], headers=headers)
    assert messages[-1]["body"] == body
    assert messages[0]["headers"] == headers


def test_content_length_is_removed_only_for_rewritten_stream_protocol():
    messages = send_chunks(
        [b'data: {"text":"\xc2\x85"}\n\n'],
        headers=[(b"Content-Type", b"text/event-stream; charset=utf-8"), (b"content-length", b"21"), (b"x-id", b"123")],
    )
    assert messages[0]["headers"] == [(b"Content-Type", b"text/event-stream; charset=utf-8"), (b"x-id", b"123")]


def test_websocket_is_untouched():
    messages = send_chunks([b"\xe2\x80\xa8"], scope_type="websocket")
    assert messages[-1]["body"] == b"\xe2\x80\xa8"


def test_plain_stream_bytes_and_incomplete_terminal_utf8_are_preserved():
    for chunks in [[b'data: {"text":"abc"}\n\n'], [b"data: ", b"\xe2", b"\x80"]]:
        messages = send_chunks(chunks)
        assert b"".join(m.get("body", b"") for m in messages) == b"".join(chunks)
