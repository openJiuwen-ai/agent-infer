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

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentinfer.agentcache.core.api_adapter import _replay_receive, _scope_with_content_length
from agentinfer.agentcache.core.program_scheduling import ATTEMPT_ARG
from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD, Event, chat_facts

logger = logging.getLogger(__name__)


def program_id(payload: dict) -> str:
    if "program_id" in payload:
        return str(payload["program_id"])
    extra = payload.get("extra_body")
    return str(extra["program_id"]) if isinstance(extra, dict) and "program_id" in extra else "default"


def extract_usage(payload: object) -> tuple[int | None, int, int]:
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
                self.finish()
        await self.send(message)

    def _read_frames(self) -> None:
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
                if self._usage[0] is None:
                    self._read_usage(data)

    def _read_usage(self, data) -> None:
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError):
            return
        usage = extract_usage(payload)
        if usage[0] is not None:
            self._usage = usage

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self.streaming:
            self._read_usage(self._buffer)
        self.emit(Event.FINISH, self.attempt, *self._usage)


async def run_ticks(send, interval: float, *, clock=None) -> None:
    """Preserve separate observation and tick inputs and the v3 deadline rule."""
    clock = clock or asyncio.get_running_loop().time
    deadline = clock() + interval
    while True:
        await asyncio.sleep(max(0.0, deadline - clock()))
        observation = send(Event.OBSERVE)
        observed = clock()
        applied = await send(Event.TICK)
        await observation
        deadline = (clock() if applied else observed) + interval


class MoriSchedulingMiddleware:
    """Use the standard vLLM API as the source of MORI request facts."""

    def __init__(self, app) -> None:
        self.app = app
        self._tick_task: asyncio.Task | None = None
        self._observation_task: asyncio.Task | None = None
        self._startup = asyncio.Lock()
        self._delivery: set[asyncio.Task] = set()

    async def _start(self, send) -> None:
        async with self._startup:
            if self._tick_task is None:
                interval = await send(Event.CONFIG)
                await send(Event.OBSERVE)
                self._tick_task = asyncio.create_task(run_ticks(send, interval))
                self._observation_task = asyncio.create_task(self._observe(send))
            elif self._tick_task.done():
                self._tick_task.result()
            if self._observation_task.done():
                self._observation_task.result()

    @staticmethod
    async def _observe(send) -> None:
        while True:
            await asyncio.sleep(5)
            await send(Event.OBSERVE)

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

    async def _await_admission(self, reply: asyncio.Task, receive) -> bool:
        disconnected = asyncio.create_task(self._wait_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait((reply, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                disconnected.result()
                return False
            return reply.result()
        finally:
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            try:
                await self.app(scope, receive, send)
            finally:
                tasks = [task for task in (self._tick_task, self._observation_task) if task is not None]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            return
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path not in ("/v1/chat/completions", "/programs/release"):
            await self.app(scope, receive, send)
            return

        try:
            payload = await Request(scope, receive).json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            await JSONResponse({"detail": "Expected a JSON object"}, status_code=400)(scope, receive, send)
            return

        core = scope["app"].state.engine_client.engine_core
        emit = partial(self._send, core)
        await self._start(emit)
        pid = program_id(payload)
        if path == "/programs/release":
            if not payload.get("program_id"):
                await JSONResponse({"detail": "Missing program_id"}, status_code=400)(scope, receive, send)
                return
            released = await emit(Event.RELEASE, pid)
            await JSONResponse({"program_id": pid, "released": released})(scope, receive, send)
            return

        attempt = uuid4().hex
        chars, prefix = chat_facts(payload)
        observer = MoriResponseObserver(send, emit, attempt, bool(payload.get("stream")))
        xargs = payload.get("vllm_xargs") or {}
        if not isinstance(xargs, dict):
            await JSONResponse({"detail": "vllm_xargs must be a JSON object"}, status_code=400)(scope, receive, send)
            return
        count = payload.get("n", 1)
        native_count = count if isinstance(count, int) and count > 0 else 1
        try:
            if not await self._await_admission(emit(Event.BEGIN, attempt, pid, chars, prefix, native_count), receive):
                await Response(status_code=409)(scope, receive, send)
                return
            payload.pop("program_id", None)
            payload["vllm_xargs"] = {**xargs, ATTEMPT_ARG: attempt, "program_id": pid}
            if observer.streaming:
                options = payload.get("stream_options")
                if options is None:
                    payload["stream_options"] = {"include_usage": True}
                elif isinstance(options, dict):
                    options.setdefault("include_usage", True)
            body = json.dumps(payload, ensure_ascii=False).encode()
            message = {"type": "http.request", "body": body, "more_body": False}
            scope = _scope_with_content_length(scope, len(body))
            await self.app(scope, _replay_receive([message], receive), observer)
        finally:
            if observer.started:
                observer.finish()
            else:
                await emit(Event.CANCEL, attempt)
