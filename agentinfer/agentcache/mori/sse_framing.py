# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Escape Unicode line separators in SSE JSON without changing parsed values."""

from __future__ import annotations

import logging

from agentinfer.agentcache.core.api_adapter import AsgiMessage, AsgiSend

logger = logging.getLogger(__name__)
_SEPARATORS = {b"\xc2\x85": b"\\u0085", b"\xe2\x80\xa8": b"\\u2028", b"\xe2\x80\xa9": b"\\u2029"}
_PREFIXES = frozenset(pattern[:end] for pattern in _SEPARATORS for end in range(1, len(pattern)))


def sse_framing_send(send: AsgiSend) -> AsgiSend:
    """Keep JSON events intact for clients that split all Unicode line breaks.

    Pydantic emits these valid JSON string characters literally. Some streaming
    line decoders split them as line endings even though SSE uses CR/LF. JSON
    escapes preserve the exact decoded content, generation, usage and finish
    events. Buffer at most two bytes when an ASGI chunk splits a UTF-8 separator.
    """

    streaming = False
    pending = b""

    async def forward(message: AsgiMessage) -> None:
        nonlocal streaming, pending
        if message.get("type") == "http.response.start":
            headers = message.get("headers", [])
            if isinstance(headers, list):
                streaming = any(
                    key.lower() == b"content-type" and b"text/event-stream" in value.lower() for key, value in headers
                ) and not any(
                    key.lower() == b"content-encoding" and value.lower() not in (b"identity", b"")
                    for key, value in headers
                )
                if streaming:
                    message = {
                        **message,
                        "headers": [(key, value) for key, value in headers if key.lower() != b"content-length"],
                    }
        elif streaming and message.get("type") == "http.response.body":
            body = message.get("body", b"")
            if isinstance(body, bytes):
                data = pending + body
                keep = (
                    max((len(prefix) for prefix in _PREFIXES if data.endswith(prefix)), default=0)
                    if message.get("more_body", False)
                    else 0
                )
                pending = data[-keep:] if keep else b""
                data = data[:-keep] if keep else data
                escaped = 0
                for pattern, replacement in _SEPARATORS.items():
                    escaped += data.count(pattern)
                    data = data.replace(pattern, replacement)
                if escaped:
                    logger.info("AgentInfer SSE JSON line separators escaped count=%d", escaped)
                message = {**message, "body": data}
        await send(message)

    return forward
