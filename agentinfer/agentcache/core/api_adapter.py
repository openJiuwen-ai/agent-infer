# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Optional vLLM API identity/lifecycle adapters and local EngineCore signal channel.

``AgentCacheLifecycleMiddleware`` uses vLLM's public ``--middleware`` hook to observe final OpenAI or Anthropic
response semantics after tool parsing. It forwards only a protocol-neutral Program lifecycle fact over a local Unix
datagram socket. The EngineCore receiver is non-blocking and is polled by the Scheduler bridge; this module never
changes response bytes, buffers a streaming response, or imports scheduling policy code.

``AgentCacheIdentityMiddleware`` resolves framework metadata through the shared identity parser and serializes only
the canonical result into vLLM's existing scalar ``vllm_xargs`` transport. OpenAI Chat accepts that field directly;
Anthropic Messages carries it through its existing ``metadata`` field until a narrow conversion hook copies it into
the internal Chat request. The Adapter does not own framework field semantics.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from agentinfer.agentcache.core.vllm_logging import attach_agentinfer_to_vllm_logging
from agentinfer.scheduling.identity import (
    JsonObject,
    MetadataError,
    encode_agent_identity,
    parse_agent_identity,
)
from agentinfer.scheduling.lifecycle import ProgramLifecycle

logger = logging.getLogger(__name__)

LIFECYCLE_SOCKET_ENV = "AGENTCACHE_VLLM_LIFECYCLE_SOCKET"
_MAX_SIGNAL_BYTES = 4096
_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_MAX_NON_STREAM_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_STREAM_EVENT_BYTES = 1024 * 1024
_ANTHROPIC_IDENTITY_METADATA_KEY = "_agentinfer_agentic_context"
_ANTHROPIC_BRIDGE_MARKER = "_agentinfer_anthropic_identity_bridge"

AsgiMessage: TypeAlias = dict[str, object]
AsgiReceive: TypeAlias = Callable[[], Awaitable[AsgiMessage]]
AsgiSend: TypeAlias = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp: TypeAlias = Callable[[dict[str, object], AsgiReceive, AsgiSend], Awaitable[None]]


class _AnthropicBridgeRequest(Protocol):
    """Existing vLLM Anthropic request surface consumed by the bridge."""

    metadata: dict[str, object] | None


class _AnthropicBridgeResult(Protocol):
    """Internal vLLM Chat request surface produced by Anthropic conversion."""

    vllm_xargs: dict[str, object] | None


class ApiEndpoint(str, Enum):
    """Wire protocols whose response lifecycle semantics differ at the API boundary."""

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"


@dataclass(frozen=True)
class LifecycleSignal:
    """One API-derived lifecycle observation addressed by stable Program id."""

    program_id: str
    lifecycle: ProgramLifecycle

    def __post_init__(self) -> None:
        if not self.program_id:
            raise ValueError("lifecycle signal program_id must not be empty")


class UnixLifecycleReceiver:
    """Non-blocking EngineCore endpoint for lifecycle signals from API workers."""

    def __init__(self, socket_path: str, dp_rank: int) -> None:
        if not socket_path:
            raise ValueError("lifecycle socket path must not be empty")
        self.path = _rank_socket_path(socket_path, dp_rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.bind(str(self.path))
        self._socket.setblocking(False)
        atexit.register(self.close)

    def receive(self) -> tuple[LifecycleSignal, ...]:
        """Drain currently available validated signals without blocking EngineCore."""
        signals: list[LifecycleSignal] = []
        while True:
            try:
                payload = self._socket.recv(_MAX_SIGNAL_BYTES)
            except BlockingIOError:
                break
            try:
                decoded = json.loads(payload)
                if not isinstance(decoded, Mapping):
                    raise ValueError("lifecycle signal must be an object")
                program_id = decoded.get("program_id")
                lifecycle = decoded.get("lifecycle")
                if not isinstance(program_id, str) or not isinstance(lifecycle, str):
                    raise ValueError("lifecycle signal fields must be strings")
                signals.append(LifecycleSignal(program_id, ProgramLifecycle(lifecycle)))
            except (ValueError, RecursionError, UnicodeDecodeError) as exc:
                logger.warning("Ignoring invalid AgentCache lifecycle signal: %s", exc)
        return tuple(signals)

    def close(self) -> None:
        """Close the socket and remove only this receiver's filesystem endpoint."""
        sock = getattr(self, "_socket", None)
        if sock is None:
            return
        self._socket = None  # type: ignore[assignment]
        sock.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class UnixLifecycleSender:
    """Short-lived API-side sender that targets one rank-specific EngineCore socket."""

    def __init__(self, socket_path: str) -> None:
        if not socket_path:
            raise ValueError("lifecycle socket path must not be empty")
        self.socket_path = socket_path

    def send(self, signal: LifecycleSignal, dp_rank: int) -> bool:
        """Best-effort delivery of one bounded datagram to the selected DP rank."""
        payload = json.dumps(
            {"program_id": signal.program_id, "lifecycle": signal.lifecycle.value},
            separators=(",", ":"),
        ).encode()
        if len(payload) > _MAX_SIGNAL_BYTES:
            raise ValueError("lifecycle signal exceeds datagram limit")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.setblocking(False)
            sock.sendto(payload, str(_rank_socket_path(self.socket_path, dp_rank)))
        except (FileNotFoundError, BlockingIOError, ConnectionRefusedError, OSError) as exc:
            logger.warning("AgentCache lifecycle signal delivery failed: %s", exc)
            return False
        finally:
            sock.close()
        return True


def _rank_socket_path(socket_path: str, dp_rank: int) -> Path:
    """Derive one filesystem endpoint from a shared lifecycle socket base path."""
    if not socket_path:
        raise ValueError("lifecycle socket path must not be empty")
    if dp_rank < 0:
        raise ValueError("dp_rank must be non-negative")
    return Path(f"{socket_path}.dp{dp_rank}")


def _adapt_identity_request_body(
    scope: dict[str, object],
    request_body: bytes,
) -> bytes:
    """Resolve raw API metadata once and inject its canonical transport representation."""
    try:
        decoded = json.loads(request_body)
    except (ValueError, RecursionError, UnicodeDecodeError):
        return request_body
    if not isinstance(decoded, dict):
        return request_body
    payload = cast(JsonObject, decoded)
    identity = parse_agent_identity(
        vllm_xargs=payload.get("vllm_xargs"),
        agent_hint=payload.get("agent_hint"),
        headers=_request_headers(scope),
    )
    if identity is None:
        return request_body
    encoded_identity = encode_agent_identity(identity)
    if _endpoint_for_scope(scope) is ApiEndpoint.ANTHROPIC_MESSAGES:
        raw_metadata = payload.get("metadata")
        if raw_metadata is None:
            metadata: dict[str, object] = {}
        elif isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            return request_body
        metadata[_ANTHROPIC_IDENTITY_METADATA_KEY] = encoded_identity
        payload["metadata"] = cast(JsonObject, metadata)
        return json.dumps(payload, separators=(",", ":")).encode()
    raw_xargs = payload.get("vllm_xargs")
    if raw_xargs is None:
        xargs: dict[str, object] = {}
    elif isinstance(raw_xargs, dict):
        xargs = dict(raw_xargs)
    else:
        return request_body
    xargs["agentic_context"] = encoded_identity
    payload["vllm_xargs"] = cast(JsonObject, xargs)
    return json.dumps(payload, separators=(",", ":")).encode()


def _install_anthropic_identity_bridge() -> None:
    """Copy middleware-owned Anthropic metadata into vLLM's internal Chat request.

    vLLM's public Anthropic request schema has no ``vllm_xargs`` field, but its
    implementation converts every Messages request into a ``ChatCompletionRequest``
    before creating ``SamplingParams``. The explicit API middleware installs this
    process-local hook only when Anthropic identity adaptation is requested.
    """
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    descriptor = AnthropicServingMessages.__dict__.get("_build_base_request")
    original = getattr(descriptor, "__func__", None)
    if original is None:
        raise RuntimeError("vLLM Anthropic identity integration requires _build_base_request")
    if getattr(original, _ANTHROPIC_BRIDGE_MARKER, False):
        return

    original_builder = cast(
        Callable[[type[object], _AnthropicBridgeRequest, list[dict[str, object]]], _AnthropicBridgeResult],
        original,
    )

    def build_base_request(
        cls: type[object],
        anthropic_request: _AnthropicBridgeRequest,
        openai_messages: list[dict[str, object]],
    ) -> _AnthropicBridgeResult:
        request = original_builder(cls, anthropic_request, openai_messages)
        metadata = getattr(anthropic_request, "metadata", None)
        encoded_identity = metadata.get(_ANTHROPIC_IDENTITY_METADATA_KEY) if isinstance(metadata, dict) else None
        if isinstance(encoded_identity, str):
            xargs = dict(request.vllm_xargs or {})
            xargs["agentic_context"] = encoded_identity
            request.vllm_xargs = xargs
        return request

    setattr(build_base_request, _ANTHROPIC_BRIDGE_MARKER, True)
    AnthropicServingMessages._build_base_request = classmethod(build_base_request)


def _replay_receive(messages: list[AsgiMessage], receive: AsgiReceive) -> AsgiReceive:
    """Replay buffered ASGI messages before reading from the original receive channel."""
    index = 0

    async def replay() -> AsgiMessage:
        nonlocal index
        if index < len(messages):
            message = messages[index]
            index += 1
            return message
        return await receive()

    return replay


def _scope_with_content_length(scope: dict[str, object], content_length: int) -> dict[str, object]:
    """Copy an ASGI scope and update Content-Length after canonical metadata injection."""
    adapted = dict(scope)
    headers = scope.get("headers")
    if not isinstance(headers, list):
        return adapted
    normalized = [
        item
        for item in headers
        if not (isinstance(item, tuple) and len(item) == 2 and item[0].lower() == b"content-length")
    ]
    normalized.append((b"content-length", str(content_length).encode()))
    adapted["headers"] = normalized
    return adapted


class AgentCacheIdentityMiddleware:
    """Transport resolved OpenAI Chat or Anthropic identity to EngineCore."""

    def __init__(self, app: AsgiApp) -> None:
        attach_agentinfer_to_vllm_logging()
        self.app = app

    async def __call__(self, scope: dict[str, object], receive: AsgiReceive, send: AsgiSend) -> None:
        endpoint = _endpoint_for_scope(scope)
        if endpoint not in (ApiEndpoint.OPENAI_CHAT_COMPLETIONS, ApiEndpoint.ANTHROPIC_MESSAGES):
            await self.app(scope, receive, send)
            return
        if endpoint is ApiEndpoint.ANTHROPIC_MESSAGES:
            _install_anthropic_identity_bridge()
        messages: list[AsgiMessage] = []
        request_body = bytearray()
        complete = False
        while not complete:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body = message.get("body", b"")
            if isinstance(body, bytes):
                request_body.extend(body)
                if len(request_body) > _MAX_REQUEST_BODY_BYTES:
                    await self.app(scope, _replay_receive(messages, receive), send)
                    return
            complete = message.get("more_body", False) is False
        if not complete:
            await self.app(scope, _replay_receive(messages, receive), send)
            return
        original_body = bytes(request_body)
        try:
            adapted_body = _adapt_identity_request_body(scope, original_body)
        except MetadataError as exc:
            body = json.dumps({"error": {"message": str(exc), "type": "invalid_request_error"}}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": body, "more_body": False})
            return
        if adapted_body == original_body:
            await self.app(scope, _replay_receive(messages, receive), send)
            return
        adapted_scope = _scope_with_content_length(scope, len(adapted_body))
        adapted_message: AsgiMessage = {"type": "http.request", "body": adapted_body, "more_body": False}
        await self.app(adapted_scope, _replay_receive([adapted_message], receive), send)


class AgentCacheLifecycleMiddleware:
    """Observe response lifecycle after vLLM tool parsing without rewriting ASGI messages."""

    def __init__(self, app: AsgiApp) -> None:
        attach_agentinfer_to_vllm_logging()
        self.app = app
        socket_path = os.environ.get(LIFECYCLE_SOCKET_ENV)
        if not socket_path:
            raise RuntimeError(f"{LIFECYCLE_SOCKET_ENV} is required when AgentCache lifecycle middleware is enabled")
        self.sender = UnixLifecycleSender(socket_path)

    async def __call__(self, scope: dict[str, object], receive: AsgiReceive, send: AsgiSend) -> None:
        endpoint = _endpoint_for_scope(scope)
        if endpoint is None:
            await self.app(scope, receive, send)
            return
        request_body = bytearray()
        request_body_overflow = False
        observer = _LifecycleResponseObserver(endpoint)

        async def observed_receive() -> AsgiMessage:
            nonlocal request_body_overflow
            message = await receive()
            if not request_body_overflow and message.get("type") == "http.request":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    request_body.extend(body)
                    if len(request_body) > _MAX_REQUEST_BODY_BYTES:
                        request_body.clear()
                        request_body_overflow = True
            return message

        async def observed_send(message: AsgiMessage) -> None:
            message_type = message.get("type")
            if message_type == "http.response.start":
                observer.start(message)
            elif message_type == "http.response.body":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    observer.feed(body)
                if message.get("more_body", False) is False:
                    try:
                        self._finish(scope, None if request_body_overflow else bytes(request_body), observer)
                    except Exception:
                        logger.exception("AgentCache lifecycle observation failed")
            await send(message)

        await self.app(scope, observed_receive, observed_send)

    def _finish(
        self,
        scope: dict[str, object],
        request_body: bytes | None,
        observer: _LifecycleResponseObserver,
    ) -> None:
        lifecycle = observer.finish()
        if request_body is None or lifecycle is ProgramLifecycle.UNKNOWN or not 200 <= observer.status_code < 300:
            return
        try:
            payload = json.loads(request_body)
        except (ValueError, RecursionError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        request_payload = cast(JsonObject, payload)
        headers = _request_headers(scope)
        metadata = parse_agent_identity(
            vllm_xargs=request_payload.get("vllm_xargs"),
            agent_hint=request_payload.get("agent_hint"),
            headers=headers,
        )
        if metadata is None:
            return
        dp_rank = _request_dp_rank(headers)
        if dp_rank is None:
            logger.warning("AgentCache lifecycle signal has an invalid X-data-parallel-rank header")
            return
        self.sender.send(LifecycleSignal(metadata.program_id, lifecycle), dp_rank)


class _LifecycleResponseObserver:
    """Incrementally extract the final semantic outcome from JSON or SSE output."""

    def __init__(self, endpoint: ApiEndpoint) -> None:
        self.endpoint = endpoint
        self.status_code = 0
        self.streaming = False
        self._buffer = bytearray()
        self._lifecycle = ProgramLifecycle.UNKNOWN
        self._overflow = False

    def start(self, message: AsgiMessage) -> None:
        status = message.get("status", 0)
        self.status_code = status if isinstance(status, int) else 0
        headers = message.get("headers", [])
        if isinstance(headers, list):
            self.streaming = any(
                isinstance(item, tuple)
                and len(item) == 2
                and item[0].lower() == b"content-type"
                and b"text/event-stream" in item[1].lower()
                for item in headers
            )

    def feed(self, body: bytes) -> None:
        if self._overflow:
            return
        self._buffer.extend(body)
        if self.streaming:
            while (line_end := self._buffer.find(b"\n")) >= 0:
                if line_end > _MAX_STREAM_EVENT_BYTES:
                    self._overflow = True
                    self._buffer.clear()
                    return
                line = bytes(self._buffer[:line_end]).rstrip(b"\r")
                del self._buffer[: line_end + 1]
                self._observe_sse_line(line)
            if len(self._buffer) > _MAX_STREAM_EVENT_BYTES:
                self._overflow = True
                self._buffer.clear()
        elif len(self._buffer) > _MAX_NON_STREAM_RESPONSE_BYTES:
            self._overflow = True
            self._buffer.clear()

    def finish(self) -> ProgramLifecycle:
        if self._overflow:
            return ProgramLifecycle.UNKNOWN
        if self.streaming:
            if self._buffer:
                self._observe_sse_line(bytes(self._buffer).rstrip(b"\r"))
        elif self._buffer:
            self._observe_json(bytes(self._buffer))
        self._buffer.clear()
        return self._lifecycle

    def _observe_sse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data and data != b"[DONE]":
            self._observe_json(data)

    def _observe_json(self, data: bytes) -> None:
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError, UnicodeDecodeError):
            return
        lifecycle = _response_lifecycle(self.endpoint, payload)
        if lifecycle is ProgramLifecycle.CONTINUE:
            self._lifecycle = lifecycle
        elif lifecycle is ProgramLifecycle.TERMINAL and self._lifecycle is not ProgramLifecycle.CONTINUE:
            self._lifecycle = lifecycle


def _endpoint_for_scope(scope: Mapping[str, object]) -> ApiEndpoint | None:
    if scope.get("type") != "http" or scope.get("method") != "POST":
        return None
    path = scope.get("path")
    if path in {"/v1/chat/completions", "/chat/completions"}:
        return ApiEndpoint.OPENAI_CHAT_COMPLETIONS
    if path in {"/v1/messages", "/messages"}:
        return ApiEndpoint.ANTHROPIC_MESSAGES
    return None


def _request_headers(scope: Mapping[str, object]) -> dict[str, str]:
    headers = scope.get("headers", [])
    if not isinstance(headers, list):
        return {}
    result: dict[str, str] = {}
    for item in headers:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        try:
            result[item[0].decode("latin-1")] = item[1].decode("latin-1")
        except (AttributeError, UnicodeDecodeError):
            continue
    return result


def _request_dp_rank(headers: Mapping[str, str]) -> int | None:
    """Return the Router-selected DP rank, defaulting headerless DP=1 traffic to rank zero."""
    normalized = {key.lower(): value for key, value in headers.items()}
    raw_rank = normalized.get("x-data-parallel-rank")
    if raw_rank is None:
        return 0
    try:
        dp_rank = int(raw_rank)
    except ValueError:
        return None
    return dp_rank if dp_rank >= 0 else None


def _response_lifecycle(endpoint: ApiEndpoint, payload: object) -> ProgramLifecycle:
    if not isinstance(payload, Mapping):
        return ProgramLifecycle.UNKNOWN
    if endpoint is ApiEndpoint.ANTHROPIC_MESSAGES:
        candidates = (payload, payload.get("message"), payload.get("delta"))
        reasons = [item.get("stop_reason") for item in candidates if isinstance(item, Mapping)]
        if "tool_use" in reasons:
            return ProgramLifecycle.CONTINUE
        if "end_turn" in reasons:
            return ProgramLifecycle.TERMINAL
        content = payload.get("content")
        if isinstance(content, list) and any(
            isinstance(block, Mapping) and block.get("type") == "tool_use" for block in content
        ):
            return ProgramLifecycle.CONTINUE
        return ProgramLifecycle.UNKNOWN
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ProgramLifecycle.UNKNOWN
    reasons: list[object] = []
    tool_call_seen = False
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        reasons.append(choice.get("finish_reason"))
        for candidate in (choice.get("message"), choice.get("delta")):
            if isinstance(candidate, Mapping) and candidate.get("tool_calls"):
                tool_call_seen = True
    if tool_call_seen or "tool_calls" in reasons:
        return ProgramLifecycle.CONTINUE
    if "stop" in reasons:
        return ProgramLifecycle.TERMINAL
    return ProgramLifecycle.UNKNOWN
