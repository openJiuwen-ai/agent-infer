# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Preserve native Chat JSON and SGLang text and streaming behavior."""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import wraps

from starlette.requests import Request

from agentinfer.agentcache.core.api_adapter import _replay_receive, _scope_with_content_length

_PARSED_BODY = "agentinfer.mori.parsed_body"
_MEDIA_PARTS = frozenset(
    (
        "input_image",
        "image_url",
        "image_pil",
        "image_embeds",
        "audio_url",
        "input_audio",
        "audio_embeds",
        "video_url",
        "video_embeds",
        "vision_chunk",
    )
)


@contextmanager
def mori_parsed_request_body():
    """Share decoded Chat JSON with the native FastAPI handler."""
    native = Request.json

    @wraps(native)
    async def read_json(request):
        if _PARSED_BODY in request.scope:
            return request.scope[_PARSED_BODY]
        return await native(request)

    vars(read_json)["_agentinfer_mori_body"] = True
    Request.json = read_json
    try:
        yield
    finally:
        Request.json = native


async def forward_parsed_json(app, request, payload, receive, send) -> None:
    """Replay the wire bytes while sharing the adapted object with FastAPI."""
    scope = request.scope
    if getattr(Request.json, "_agentinfer_mori_body", False):
        scope = {**scope, _PARSED_BODY: payload}
        body = await request.body()
    else:
        body = json.dumps(payload, ensure_ascii=True).encode()
        scope = _scope_with_content_length(scope, len(body))
    message = {"type": "http.request", "body": body, "more_body": False}
    await app(scope, _replay_receive([message], receive), send)


def normalize_sglang_text_chat(payload: dict) -> None:
    """Match SGLang's string-template formatting before vLLM renders Chat."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list) and all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        ):
            message["content"] = " ".join(part["text"] for part in content)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    if payload.get("tool_choice") == "none":
        payload.pop("tools")
        return
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type", "function") == "function":
            function = tool.get("function")
            if isinstance(function, dict):
                # Qwen's template emits function metadata such as <strict>.
                function.setdefault("strict", False)


def shape_chat_frames(send, payload):
    """Mirror the original content/finish framing counted by ThunderAgent."""
    options = payload.get("stream_options")
    if (
        not payload.get("stream")
        or payload.get("logprobs")
        or payload.get("return_token_ids")
        or (options is not None and (not isinstance(options, dict) or options.get("continuous_usage_stats")))
    ):
        return send

    async def forward(message):
        if (
            message["type"] == "http.response.body"
            and message.get("more_body")
            and (body := message.get("body", b"")).startswith(b"data: ")
        ):
            if not body.endswith(b"\n\n") or body.count(b"\n\n") != 1:
                await send(message)
                return
            if b'"finish_reason":"' in body:
                frame = json.loads(body[6:-2])
                choices = frame.get("choices", ())
                if len(choices) == 1:
                    choice = choices[0]
                    if choice.get("finish_reason") and any(choice["delta"].values()):
                        content = {**choice, "finish_reason": None}
                        content.pop("stop_reason", None)
                        data_frame = {**frame, "choices": [content]}
                        data_frame.pop("system_fingerprint", None)
                        finish = {**frame, "choices": [{**choice, "delta": {}}]}
                        message["body"] = (
                            b"data: " + json.dumps(data_frame, ensure_ascii=False).encode() + b"\n\n"
                            b"data: " + json.dumps(finish, ensure_ascii=False).encode() + b"\n\n"
                        )
            elif b'"delta":{"content":""}' in body:
                return
        await send(message)

    return forward


def chat_input_error(payload: dict) -> str | None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            inferred = kind is None or "uuid" in part
            if (isinstance(kind, str) and kind in _MEDIA_PARTS) or (inferred and not _MEDIA_PARTS.isdisjoint(part)):
                return "MORI vLLM supports text inputs only"
    return None
