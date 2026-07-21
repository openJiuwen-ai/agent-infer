"""Transparent Anthropic Messages request proxy."""

import asyncio
import contextlib
import json
import time
import uuid
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from agentcache.utils.headers import AgentRequestIdentity, parse_agent_identity, select_router_headers

from .request_trace import RequestFact, RequestTraceWriter, TraceHealth

_PROTOCOL_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "anthropic-beta",
        "anthropic-version",
        "authorization",
        "content-type",
        "user-agent",
        "x-api-key",
    }
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass(frozen=True)
class RequestProxyLaunchConfig:
    """Carry pickleable per-run state from Benchkit into the spawned proxy process."""

    host: str
    port: int
    upstream_url: str
    run_id: str
    trace_path: Path
    request_timeout_seconds: float
    shutdown_timeout_seconds: float
    shutdown_token: str
    stdout_path: Path
    stderr_path: Path


class RequestProxyServer:
    """Forward Anthropic requests transparently while recording request facts."""

    def __init__(
        self,
        upstream_url: str,
        run_id: str,
        writer: RequestTraceWriter,
        request_timeout_seconds: float,
        shutdown_token: str,
        shutdown_timeout_seconds: float = 30,
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.run_id = run_id
        self.writer = writer
        self.shutdown_token = shutdown_token
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.shutdown_callback = shutdown_callback
        self.client = httpx.AsyncClient(timeout=request_timeout_seconds, trust_env=False)
        self.accepting = True
        self._active_requests = 0
        self._active_condition = asyncio.Condition()
        self._close_lock = asyncio.Lock()
        self._close_health: TraceHealth | None = None
        self.app = FastAPI()
        self.app.add_api_route("/health", self.handle_health, methods=["GET"])
        self.app.add_api_route("/v1/messages", self.handle_messages, methods=["POST"])
        self.app.add_api_route("/shutdown", self.handle_shutdown, methods=["POST"])

    async def handle_health(self) -> dict[str, object]:
        return {"status": "ok" if self.accepting else "stopping", "active_requests": self._active_requests}

    async def _admit(self) -> None:
        async with self._active_condition:
            if not self.accepting:
                raise HTTPException(503, "proxy is shutting down")
            self._active_requests += 1

    async def _release(self) -> None:
        async with self._active_condition:
            self._active_requests -= 1
            if self._active_requests == 0:
                self._active_condition.notify_all()

    async def handle_messages(self, request: Request) -> Response:
        await self._admit()
        release_here = True
        response: httpx.Response | None = None
        try:
            body = await request.body()
            identity = parse_agent_identity(request.headers)
            request_id = str(uuid.uuid4())
            started = datetime.now(timezone.utc)
            started_clock = time.monotonic()
            try:
                upstream_request = self.client.build_request(
                    "POST",
                    f"{self.upstream_url}/v1/messages",
                    content=body,
                    headers=_upstream_headers(request.headers),
                )
                response = await self.client.send(upstream_request, stream=True)
            except asyncio.CancelledError:
                self._submit_safely(identity, request_id, started, started_clock, None, None, {}, "request cancelled")
                raise
            except Exception as exc:
                self._submit_safely(identity, request_id, started, started_clock, None, None, {}, str(exc))
                raise

            content_type = response.headers.get("content-type", "").lower()
            content_encoding = response.headers.get("content-encoding", "").lower()
            response_headers = _response_headers(response.headers)
            if "text/event-stream" not in content_type:
                try:
                    payload = (
                        response.content
                        if response.is_stream_consumed
                        else b"".join([chunk async for chunk in response.aiter_raw()])
                    )
                    observed_payload = _decode_payload(payload, content_encoding)
                    self._submit_safely(
                        identity,
                        request_id,
                        started,
                        started_clock,
                        response.status_code,
                        None,
                        _usage(observed_payload) if observed_payload is not None else {},
                        None,
                    )
                    return Response(content=payload, status_code=response.status_code, headers=response_headers)
                except asyncio.CancelledError:
                    self._submit_safely(
                        identity,
                        request_id,
                        started,
                        started_clock,
                        response.status_code,
                        None,
                        {},
                        "request cancelled",
                    )
                    raise
                except Exception as exc:
                    self._submit_safely(
                        identity, request_id, started, started_clock, response.status_code, None, {}, str(exc)
                    )
                    raise

            first: float | None = None
            usage_parser = _AnthropicSSEUsageObserver()
            decoder = _ContentDecoder(content_encoding)
            error: str | None = None
            finalized = False

            async def finalize() -> None:
                nonlocal finalized
                if finalized:
                    return
                finalized = True
                try:
                    await response.aclose()
                finally:
                    try:
                        self._submit_safely(
                            identity,
                            request_id,
                            started,
                            started_clock,
                            response.status_code,
                            first,
                            usage_parser.usage,
                            error,
                        )
                    finally:
                        await self._release()

            async def stream() -> AsyncIterator[bytes]:
                nonlocal first, error
                try:
                    async for chunk in response.aiter_raw():
                        observed = decoder.decode(chunk)
                        if observed is not None and usage_parser.observe(observed) and first is None:
                            first = time.monotonic() - started_clock
                        yield chunk
                    observed = decoder.flush()
                    if observed is not None and usage_parser.observe(observed) and first is None:
                        first = time.monotonic() - started_clock
                except asyncio.CancelledError:
                    error = "request cancelled"
                    raise
                except Exception as exc:
                    error = str(exc)
                    raise

            release_here = False
            return _ObservedStreamingResponse(
                stream(), finalize, status_code=response.status_code, headers=response_headers
            )
        finally:
            if release_here:
                try:
                    if response is not None:
                        await response.aclose()
                finally:
                    await self._release()

    def _submit_safely(
        self,
        identity: AgentRequestIdentity,
        request_id: str,
        started: datetime,
        started_clock: float,
        status_code: int | None,
        ttft: float | None,
        usage: Mapping[str, object],
        error: str | None,
    ) -> None:
        try:
            self._submit(identity, request_id, started, started_clock, status_code, ttft, usage, error)
        except RuntimeError:
            pass

    def _submit(
        self,
        identity: AgentRequestIdentity,
        request_id: str,
        started: datetime,
        started_clock: float,
        status_code: int | None,
        ttft: float | None,
        usage: Mapping[str, object],
        error: str | None,
    ) -> None:
        self.writer.submit(
            RequestFact(
                schema_version="1",
                run_id=self.run_id,
                request_id=request_id,
                session_id=identity.session_id,
                actor_id=identity.actor_id,
                actor_role=identity.actor_role,
                started_at=started.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                status="success" if status_code is not None and status_code < 400 and error is None else "error",
                status_code=status_code,
                latency_seconds=time.monotonic() - started_clock,
                ttft_seconds=ttft,
                input_tokens=_int_or_none(usage.get("input_tokens")),
                output_tokens=_int_or_none(usage.get("output_tokens")),
                cache_creation_tokens=_int_or_none(usage.get("cache_creation_input_tokens")),
                cached_tokens=_int_or_none(usage.get("cache_read_input_tokens")),
                upstream=self.upstream_url,
                error=error,
            )
        )

    async def handle_shutdown(self, request: Request) -> TraceHealth:
        if request.headers.get("authorization") != f"Bearer {self.shutdown_token}":
            raise HTTPException(403, "invalid shutdown token")
        health = await self.close()
        if self.shutdown_callback:
            self.shutdown_callback()
        return health

    async def close(self) -> TraceHealth:
        async with self._close_lock:
            if self._close_health is not None:
                return self._close_health
            async with self._active_condition:
                self.accepting = False
                await asyncio.wait_for(
                    self._active_condition.wait_for(lambda: self._active_requests == 0), self.shutdown_timeout_seconds
                )
            await self.client.aclose()
            self._close_health = await self.writer.close()
            return self._close_health


async def _serve_request_proxy(config: RequestProxyLaunchConfig) -> None:
    writer = RequestTraceWriter(config.trace_path)
    uvicorn_server: uvicorn.Server
    proxy = RequestProxyServer(
        config.upstream_url,
        config.run_id,
        writer,
        config.request_timeout_seconds,
        config.shutdown_token,
        shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        shutdown_callback=lambda: setattr(uvicorn_server, "should_exit", True),
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(proxy.app, host=config.host, port=config.port, log_level="warning", access_log=False)
    )
    await writer.start()
    try:
        await uvicorn_server.serve()
    except BaseException:
        with contextlib.suppress(Exception):
            await proxy.close()
        with contextlib.suppress(Exception):
            await writer.close()
        raise
    cleanup_errors = []
    try:
        await proxy.close()
    except Exception as exc:
        cleanup_errors.append(exc)
    try:
        await writer.close()
    except Exception as exc:
        cleanup_errors.append(exc)
    if cleanup_errors:
        details = "; ".join(str(error) for error in cleanup_errors)
        raise RuntimeError(f"request proxy cleanup failed: {details}")


def run_request_proxy(config: RequestProxyLaunchConfig) -> None:
    """Run one proxy child process and redirect its console output to evidence files."""

    config.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        config.stdout_path.open("a", encoding="utf-8") as stdout,
        config.stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            asyncio.run(_serve_request_proxy(config))


class _ObservedStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[bytes],
        finalize: Callable[[], Awaitable[None]],
        *,
        status_code: int,
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers)
        self._finalize = finalize

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._finalize()


class _IncrementalDecoder(Protocol):
    def decompress(self, data: bytes) -> bytes: ...

    def flush(self) -> bytes: ...


class _ContentDecoder:
    def __init__(self, encoding: str) -> None:
        self._decoder = _zlib_decoder(encoding)
        self._supported = not encoding or self._decoder is not None

    def decode(self, chunk: bytes) -> bytes | None:
        if not self._supported:
            return None
        try:
            return self._decoder.decompress(chunk) if self._decoder is not None else chunk
        except zlib.error:
            self._supported = False
            return None

    def flush(self) -> bytes | None:
        if not self._supported:
            return None
        try:
            return self._decoder.flush() if self._decoder is not None else b""
        except zlib.error:
            self._supported = False
            return None


class _AnthropicSSEUsageObserver:
    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, int] = {}

    def observe(self, chunk: bytes) -> bool:
        self._buffer += chunk
        observed_event = False
        while True:
            lf_boundary = self._buffer.find(b"\n\n")
            crlf_boundary = self._buffer.find(b"\r\n\r\n")
            boundaries = [(index, size) for index, size in ((lf_boundary, 2), (crlf_boundary, 4)) if index >= 0]
            if not boundaries:
                break
            index, size = min(boundaries)
            event, self._buffer = self._buffer[:index], self._buffer[index + size :]
            data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")]
            if not data_lines:
                continue
            try:
                payload = json.loads(b"\n".join(data_lines))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("type") == "content_block_delta":
                observed_event = True
            usage = payload.get("usage") if isinstance(payload, dict) else None
            if not isinstance(usage, dict) and isinstance(payload, dict):
                message = payload.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.usage[key] = value
        return observed_event


def _upstream_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forwarded = {key: value for key, value in headers.items() if key.lower() in _PROTOCOL_REQUEST_HEADERS}
    forwarded.update(select_router_headers(headers))
    return forwarded


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}}


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _usage(payload: bytes) -> dict[str, object]:
    try:
        row = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    usage = row.get("usage") if isinstance(row, dict) else None
    return usage if isinstance(usage, dict) else {}


def _decode_payload(payload: bytes, encoding: str) -> bytes | None:
    decoder = _ContentDecoder(encoding)
    decoded = decoder.decode(payload)
    tail = decoder.flush()
    return None if decoded is None or tail is None else decoded + tail


def _zlib_decoder(encoding: str) -> _IncrementalDecoder | None:
    if not encoding:
        return None
    if encoding == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if encoding == "deflate":
        return zlib.decompressobj()
    return None
