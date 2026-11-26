# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Translate vLLM Chat ingress and response boundaries into Program events."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from functools import partial
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_core import to_json
from starlette._utils import get_route_path
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from agentinfer.agentcache.core.api_adapter import _replay_receive, _scope_with_content_length
from agentinfer.agentcache.core.program_scheduling import ATTEMPT_ARG
from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD, Event, chat_facts
from agentinfer.agentcache.mori.programs import PROGRAM_ARG

logger = logging.getLogger(__name__)
_CHAT_CHOICES = TypeAdapter(int | None)
_CHAT_STREAM = TypeAdapter(bool | None)


def program_id(payload: dict) -> str:
    """Port ThunderAgent app.get_program_id, preserving body-field precedence."""
    if "program_id" in payload:
        return str(payload["program_id"])
    extra = payload.get("extra_body")
    return str(extra["program_id"]) if isinstance(extra, dict) and "program_id" in extra else "default"


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


class MoriResponseObserver:
    """Observe completed SSE frames and end-of-response usage."""

    def __init__(self, send, emit, attempt: str, streaming: bool) -> None:
        self.send = send
        self.emit = emit
        self.attempt = attempt
        self.streaming = streaming
        self.started = False
        self._finished = False
        self._buffer = bytearray()
        self._usage: tuple[int | None, int, int] = (None, 0, 0)
        self._frames = 0

    async def __call__(self, message) -> None:
        if message["type"] == "http.response.start":
            self.started = True
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
                    self.emit(Event.PROGRESS, self.attempt, self._frames)
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

    async def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self.streaming:
            self._read_usage(self._buffer)
        delivery = self.emit(Event.FINISH, self.attempt, *self._usage)
        # Match ThunderAgent's usage callback before closing a response, so the
        # next request from this Program observes its completed predecessor.
        await asyncio.shield(asyncio.gather(delivery, return_exceptions=True))


class MoriSchedulingMiddleware:
    """Use the standard vLLM API as the source of MORI request facts."""

    def __init__(self, app) -> None:
        self.app = app
        self._delivery: set[asyncio.Task] = set()

    def _send(self, core, event: Event, *args) -> asyncio.Task:
        task = asyncio.create_task(core.call_utility_async(UTILITY_METHOD, event.value, list(args)))
        self._delivery.add(task)
        task.add_done_callback(self._delivered)
        return task

    def _delivered(self, task: asyncio.Task) -> None:
        self._delivery.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("MORI event delivery failed: %s", error)

    @staticmethod
    async def _wait_for_disconnect(receive) -> None:
        while (await receive())["type"] != "http.disconnect":
            pass

    async def _await_admission(self, reply: asyncio.Task, receive) -> bool | None:
        disconnected = asyncio.create_task(self._wait_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait((reply, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                disconnected.result()
                return None
            return reply.result()
        finally:
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)

    async def _forward_chat(self, request: Request, payload: dict, receive, send) -> None:
        """Deliver admitted Chat JSON to the native server as UTF-8 bytes."""
        body = to_json(payload)
        message = {"type": "http.request", "body": body, "more_body": False}
        scope = _scope_with_content_length(request.scope, len(body))
        await self.app(scope, _replay_receive([message], receive), send)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = get_route_path(scope)
        if path not in ("/v1/chat/completions", "/programs/release"):
            await self.app(scope, receive, send)
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
            if not payload.get("program_id"):
                await JSONResponse({"detail": "Missing program_id"}, status_code=400)(scope, receive, send)
                return
            released = await emit(Event.RELEASE, pid)
            await JSONResponse({"program_id": pid, "released": released})(scope, receive, send)
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

        attempt = uuid4().hex
        observer = MoriResponseObserver(send, emit, attempt, streaming)
        try:
            admitted = await self._await_admission(
                emit(Event.BEGIN, attempt, pid, chars, prefix, native_count), receive
            )
            if admitted is None:
                return
            if not admitted:
                await Response(status_code=409)(scope, receive, send)
                return
            for key in ("program_id", PROGRAM_ARG, ATTEMPT_ARG):
                payload.pop(key, None)
            native_xargs = dict(xargs or {})
            native_xargs.pop("program_id", None)
            native_xargs.update({PROGRAM_ARG: pid, ATTEMPT_ARG: attempt})
            payload["vllm_xargs"] = native_xargs
            if streaming:
                options = payload.get("stream_options")
                if options is None:
                    payload["stream_options"] = {"include_usage": True}
                elif isinstance(options, dict):
                    options.setdefault("include_usage", True)
            await self._forward_chat(request, payload, receive, observer)
        finally:
            if observer.started:
                await observer.finish()
            else:
                await asyncio.shield(emit(Event.CANCEL, attempt))
