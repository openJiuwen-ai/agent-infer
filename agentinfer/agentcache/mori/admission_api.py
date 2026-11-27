# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Translate vLLM Chat ingress and response boundaries into Program events."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from contextvars import ContextVar
from functools import partial, wraps
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_core import to_json
from starlette._utils import get_route_path
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from agentinfer.agentcache.core.api_adapter import _replay_receive, _scope_with_content_length
from agentinfer.agentcache.mori.ipc import (
    ATTEMPT_ARG,
    CHILD_COUNT_ARG,
    CONTEXT_CHARS_ARG,
    PREFIX_ARG,
    PROGRAM_ARG,
    RESERVED_ARGS,
    UTILITY_METHOD,
    Event,
    chat_facts,
)

logger = logging.getLogger(__name__)
_CHAT_CHOICES = TypeAdapter(int | None)
_CHAT_STREAM = TypeAdapter(bool | None)


def program_id(payload: dict) -> str | None:
    """Resolve an explicit Program ID from the body or its OpenAI-compatible extension."""
    extra = payload.get("extra_body")
    for value in (payload.get("program_id"), extra.get("program_id") if isinstance(extra, dict) else None):
        if value is not None and str(value).strip():
            return str(value)
    return None


def extract_usage(payload: object) -> tuple[int | None, int, int]:
    """Port ThunderAgent vllm_request_processor.extract_usage_info for the IPC tuple."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None, 0, 0

    def integer(value: object) -> int | None:
        return int(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    details = usage.get("prompt_tokens_details")
    cached = integer(details.get("cached_tokens")) if isinstance(details, dict) else None
    return integer(usage.get("total_tokens")), integer(usage.get("prompt_tokens")) or 0, cached or 0


class _ChatAdd:
    """One HTTP attempt shared by all of vLLM's native sampling children."""

    def __init__(self, pid: str, chars: int, prefix: str, count: int) -> None:
        self.attempt = uuid4().hex
        self.facts = {
            ATTEMPT_ARG: self.attempt,
            PROGRAM_ARG: pid,
            CONTEXT_CHARS_ARG: chars,
            PREFIX_ARG: prefix,
            CHILD_COUNT_ARG: count,
        }
        self.sending = asyncio.Lock()
        self.closed = False
        self.submitted = False


_current_chat: ContextVar[_ChatAdd | None] = ContextVar("mori_chat_add", default=None)


def _bind_add(core) -> None:
    """Attach trusted Chat facts at the common API-to-Core boundary."""
    native_add = core.add_request_async
    if getattr(native_add, "_mori_chat_add", False):
        return

    @wraps(native_add)
    async def add_request(request):
        params = request.sampling_params
        extra = params.extra_args if params is not None else None
        chat = _current_chat.get()
        if chat is None or params is None:
            # Other HTTP protocols can forward arbitrary user-supplied extra_args.
            if extra and not RESERVED_ARGS.isdisjoint(extra):
                params.extra_args = {k: v for k, v in extra.items() if k not in RESERVED_ARGS}
            return await native_add(request)

        async with chat.sending:
            if chat.closed:
                raise asyncio.CancelledError("MORI Chat already ended")
            params.extra_args = {**(extra or {}), **chat.facts}
            chat.submitted = True
            return await native_add(request)

    vars(add_request)["_mori_chat_add"] = True
    core.add_request_async = add_request


class MoriResponseObserver:
    """Observe completed SSE frames and end-of-response usage."""

    def __init__(self, send, emit, chat: _ChatAdd, streaming: bool) -> None:
        self.send = send
        self.emit = emit
        self.chat = chat
        self.streaming = streaming
        self.finished = False
        self._buffer = bytearray()
        self._usage: tuple[int | None, int, int] = (None, 0, 0)
        self._frames = 0

    async def __call__(self, message) -> None:
        if message["type"] == "http.response.body":
            self._buffer.extend(message.get("body", b""))
            if self.streaming:
                self._read_frames()
            if not message.get("more_body"):
                await self.finish()
        await self.send(message)

    def _read_frames(self) -> None:
        """Port ThunderAgent's SSE progress rule to ASGI fragments."""
        while (boundary := self._buffer.find(b"\n\n")) != -1:
            frame = self._buffer[:boundary]
            del self._buffer[: boundary + 2]
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                self._frames += 1
                if self._frames == 20:
                    if self.chat.submitted:
                        self.emit(Event.PROGRESS, self.chat.attempt, self._frames)
                    self._frames = 0
                if self._usage[0] is None and b'"usage"' in data:
                    self._read_usage(data)

    def _read_usage(self, data) -> None:
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError):
            return
        usage = extract_usage(payload)
        if usage[0] is not None:
            self._usage = usage

    async def finish(self, cancelled: bool = False) -> None:
        if self.finished:
            return
        self.chat.closed = True
        # A disconnected streaming response can end before n native children were sent.
        async with self.chat.sending:
            if self.finished:
                return
            self.finished = True
            if not self.chat.submitted:
                return
            if cancelled:
                delivery = self.emit(Event.CANCEL, self.chat.attempt)
            else:
                if not self.streaming:
                    self._read_usage(self._buffer)
                delivery = self.emit(Event.FINISH, self.chat.attempt, *self._usage)
        # Match ThunderAgent's usage callback before closing a response, so the
        # next request from this Program observes its completed predecessor.
        await asyncio.shield(asyncio.gather(delivery, return_exceptions=True))


class MoriSchedulingMiddleware:
    """Use the standard vLLM API as the source of MORI request facts."""

    def __init__(self, app) -> None:
        self.app = app
        self._delivery: set[asyncio.Task] = set()
        self._bound_core = None
        self._auth_initialized = False
        self._auth = None

    def _authorized(self, scope) -> bool:
        if not self._auth_initialized:
            args = getattr(scope["app"].state, "args", None)
            configured = getattr(args, "api_key", None) or [os.environ.get("VLLM_API_KEY")]
            if tokens := [key for key in configured if key]:
                try:
                    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
                except ModuleNotFoundError as exc:
                    if exc.name not in (
                        "vllm.entrypoints.serve.middleware",
                        "vllm.entrypoints.serve.middleware.authenticate",
                    ):
                        raise
                    from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

                self._auth = AuthenticationMiddleware(self.app, tokens=tokens)
            self._auth_initialized = True
        return self._auth is None or self._auth.verify_token(Headers(scope=scope))

    def _send(self, core, event: Event, *args) -> asyncio.Task:
        task = asyncio.create_task(core.call_utility_async(UTILITY_METHOD, event.value, list(args)))
        self._delivery.add(task)
        task.add_done_callback(self._delivered)
        return task

    def _delivered(self, task: asyncio.Task) -> None:
        self._delivery.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("MORI event delivery failed: %s", error)

    async def _forward_chat(self, request: Request, payload: dict, receive, send) -> None:
        """Deliver Chat JSON to the native server as UTF-8 bytes."""
        body = to_json(payload)
        message = {"type": "http.request", "body": body, "more_body": False}
        scope = _scope_with_content_length(request.scope, len(body))
        await self.app(scope, _replay_receive([message], receive), send)

    def _chat_input_error(self, payload: dict) -> str | None:
        """Let the engine reject unsupported Chat inputs before Program admission."""
        return None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            client = getattr(scope["app"].state, "engine_client", None)
            core = getattr(client, "engine_core", None)
            if core is not None and self._bound_core is not core:
                _bind_add(core)
                self._bound_core = core

        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = get_route_path(scope)
        if path not in ("/v1/chat/completions", "/programs/release"):
            await self.app(scope, receive, send)
            return

        if not self._authorized(scope):
            await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
            return

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

        core = scope["app"].state.engine_client.engine_core
        emit = partial(self._send, core)
        pid = program_id(payload)
        if path == "/programs/release":
            if pid is None:
                await JSONResponse({"detail": "Missing program_id"}, status_code=400)(scope, receive, send)
                return
            released = await emit(Event.RELEASE, pid)
            await JSONResponse({"program_id": pid, "released": released})(scope, receive, send)
            return

        pid = pid or "default"
        if input_error := self._chat_input_error(payload):
            await JSONResponse({"detail": input_error}, status_code=400)(scope, receive, send)
            return
        xargs = payload.get("vllm_xargs")
        if xargs is not None and not isinstance(xargs, dict):
            await JSONResponse({"detail": "vllm_xargs must be a JSON object"}, status_code=400)(scope, receive, send)
            return
        try:
            native_count = max(_CHAT_CHOICES.validate_python(payload.get("n")) or 1, 1)
            streaming = bool(_CHAT_STREAM.validate_python(payload.get("stream")))
            chars, prefix = chat_facts(payload)
        except (TypeError, ValueError, RecursionError):
            await JSONResponse({"detail": "Invalid MORI Chat request"}, status_code=400)(scope, receive, send)
            return

        chat = _ChatAdd(pid, chars, prefix, native_count)
        observer = MoriResponseObserver(send, emit, chat, streaming)
        token = _current_chat.set(chat)
        try:
            payload.pop("program_id", None)
            for key in RESERVED_ARGS:
                payload.pop(key, None)
            if xargs:
                payload["vllm_xargs"] = {k: v for k, v in xargs.items() if k != "program_id" and k not in RESERVED_ARGS}
            if streaming:
                options = payload.get("stream_options")
                if options is None:
                    payload["stream_options"] = {"include_usage": True}
                elif isinstance(options, dict):
                    options.setdefault("include_usage", True)
            await self._forward_chat(request, payload, receive, observer)
        finally:
            _current_chat.reset(token)
            if not observer.finished:
                await asyncio.shield(observer.finish(cancelled=True))
