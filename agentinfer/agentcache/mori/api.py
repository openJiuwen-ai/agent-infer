# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Expose MORI scheduling and preserve the native vLLM Chat protocol."""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import wraps

from starlette._utils import get_route_path
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from agentinfer.agentcache.core.api_adapter import _replay_receive, _scope_with_content_length
from agentinfer.agentcache.mori.admission_api import MoriSchedulingMiddleware

_PARSED_BODY = "agentinfer.mori.parsed_body"
_DIAGNOSTICS = {
    "/programs": "programs",
    "/agentinfer/mori/metrics": "metrics",
    "/agentinfer/mori/cache": "cache",
}
_LEGACY_CONTROL = frozenset(("/mark_program_type", "/release_program", "/batch_program_ops"))
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


async def _forward_parsed_json(app, request, payload, receive, send) -> None:
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


def _mori_chat_frames(send, payload):
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


class MORIDirectMiddleware(MoriSchedulingMiddleware):
    """Prepare native Chat for MORI ADD and adapt its wire format."""

    sglang_text_chat = False
    unseeded_chat = False

    def _chat_input_error(self, payload: dict) -> str | None:
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

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return
        path, method = get_route_path(scope), scope["method"]
        if path.rstrip("/") in _LEGACY_CONTROL:
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        action = _DIAGNOSTICS.get(path) if method == "GET" else None
        if action and not self._authorized(scope):
            await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
            return
        if action is not None:
            core = scope["app"].state.engine_client.engine_core
            result = await core.call_utility_async("agentinfer_mori_diagnostics", action)
            await JSONResponse(result)(scope, receive, send)
            return
        if method == "POST" and self.sglang_text_chat and path == "/tokenize":
            request = Request(scope, receive)
            try:
                payload = await request.json()
            except ClientDisconnect:
                return
            except (ValueError, RecursionError):
                payload = None
            if not isinstance(payload, dict):
                await JSONResponse({"detail": "Expected a JSON object"}, status_code=400)(scope, receive, send)
                return
            normalize_sglang_text_chat(payload)
            await _forward_parsed_json(self.app, request, payload, receive, send)
            return
        await super().__call__(scope, receive, send)

    async def _forward_chat(self, request: Request, payload: dict, receive, send) -> None:
        # Admission facts describe the original request, before SGLang's text template normalization.
        if self.sglang_text_chat:
            normalize_sglang_text_chat(payload)
        if self.unseeded_chat:
            payload.pop("seed", None)
        await _forward_parsed_json(self.app, request, payload, receive, _mori_chat_frames(send, payload))


class MORIDirectTextChatMiddleware(MORIDirectMiddleware):
    """Normalize the Chat and tokenize bodies for SGLang's string templates."""

    sglang_text_chat = True


class MORIDirectUnseededTextChatMiddleware(MORIDirectTextChatMiddleware):
    """Use SGLang's default unseeded Chat sampling."""

    unseeded_chat = True
