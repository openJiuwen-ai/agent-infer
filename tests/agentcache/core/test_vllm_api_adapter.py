# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Tests for optional vLLM API lifecycle parsing and local signal delivery."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentinfer.agentcache.core import api_adapter
from agentinfer.agentcache.core.api_adapter import (
    LIFECYCLE_SOCKET_ENV,
    AgentCacheIdentityMiddleware,
    AgentCacheLifecycleMiddleware,
    ApiEndpoint,
    LifecycleSignal,
    UnixLifecycleReceiver,
    UnixLifecycleSender,
    _adapt_identity_request_body,
    _LifecycleResponseObserver,
    _parse_anthropic_replay_sampling,
    _response_lifecycle,
)
from agentinfer.scheduling.identity import parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle

pytestmark = pytest.mark.cpu_test


def test_anthropic_replay_sampling_metadata_is_strictly_validated() -> None:
    key = "_agentinfer_replay_sampling"

    assert _parse_anthropic_replay_sampling(None) is None
    assert _parse_anthropic_replay_sampling({}) is None
    assert _parse_anthropic_replay_sampling({key: {"seed": 7, "min_tokens": 11, "ignore_eos": True}}) == (7, 11, True)


@pytest.mark.parametrize(
    ("sampling", "match"),
    [
        (None, "must be an object"),
        ({}, "missing fields"),
        ({"seed": True, "min_tokens": 1, "ignore_eos": True}, "seed must be"),
        ({"seed": -1, "min_tokens": 1, "ignore_eos": True}, "seed must be"),
        ({"seed": 1, "min_tokens": True, "ignore_eos": True}, "min_tokens must be"),
        ({"seed": 1, "min_tokens": -1, "ignore_eos": True}, "min_tokens must be"),
        ({"seed": 1, "min_tokens": 1, "ignore_eos": 1}, "ignore_eos must be"),
    ],
)
def test_anthropic_replay_sampling_metadata_rejects_malformed_values(
    sampling: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _parse_anthropic_replay_sampling({"_agentinfer_replay_sampling": sampling})


def test_anthropic_bridge_without_replay_sampling_keeps_vllm_defaults() -> None:
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    api_adapter._install_anthropic_identity_bridge()
    anthropic_request = AnthropicMessagesRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "metadata": {"caller": "benchkit"},
        }
    )

    chat_request = AnthropicServingMessages._convert_anthropic_to_openai_request(anthropic_request)

    assert chat_request.seed is None
    assert chat_request.min_tokens == 0
    assert chat_request.ignore_eos is False


def test_openai_lifecycle_uses_post_tool_parser_semantics() -> None:
    assert (
        _response_lifecycle(
            ApiEndpoint.OPENAI_CHAT_COMPLETIONS,
            {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]},
        )
        is ProgramLifecycle.TERMINAL
    )
    assert (
        _response_lifecycle(
            ApiEndpoint.OPENAI_CHAT_COMPLETIONS,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"tool_calls": [{"function": {"name": "search"}}]},
                    }
                ]
            },
        )
        is ProgramLifecycle.CONTINUE
    )
    assert (
        _response_lifecycle(
            ApiEndpoint.OPENAI_CHAT_COMPLETIONS,
            {"choices": [{"finish_reason": "tool_calls"}]},
        )
        is ProgramLifecycle.CONTINUE
    )


def test_anthropic_lifecycle_distinguishes_end_turn_and_tool_use() -> None:
    assert (
        _response_lifecycle(
            ApiEndpoint.ANTHROPIC_MESSAGES,
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )
        is ProgramLifecycle.TERMINAL
    )
    assert (
        _response_lifecycle(
            ApiEndpoint.ANTHROPIC_MESSAGES,
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        )
        is ProgramLifecycle.CONTINUE
    )


def test_streamed_tool_call_is_not_overwritten_by_later_stop_reason() -> None:
    observer = _LifecycleResponseObserver(ApiEndpoint.OPENAI_CHAT_COMPLETIONS)
    observer.start(
        {
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    observer.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}\n')
    observer.feed(b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n')

    assert observer.finish() is ProgramLifecycle.CONTINUE


def test_streaming_observer_rejects_an_unbounded_sse_event(monkeypatch) -> None:
    monkeypatch.setattr(api_adapter, "_MAX_STREAM_EVENT_BYTES", 8)
    observer = _LifecycleResponseObserver(ApiEndpoint.OPENAI_CHAT_COMPLETIONS)
    observer.start({"status": 200, "headers": [(b"content-type", b"text/event-stream")]})

    observer.feed(b"data: " + b"x" * 9)

    assert observer.finish() is ProgramLifecycle.UNKNOWN


@pytest.mark.parametrize("decoder_error", [ValueError("integer limit"), RecursionError("nested input")])
def test_api_json_observers_do_not_leak_decoder_errors(monkeypatch, decoder_error: Exception) -> None:
    monkeypatch.setattr(api_adapter.json, "loads", lambda payload: (_ for _ in ()).throw(decoder_error))
    request_body = b'{"model":"test"}'
    observer = _LifecycleResponseObserver(ApiEndpoint.OPENAI_CHAT_COMPLETIONS)
    observer.start({"status": 200, "headers": [(b"content-type", b"text/event-stream")]})

    assert _adapt_identity_request_body({"headers": []}, request_body) == request_body
    observer.feed(b"data: {}\n")
    assert observer.finish() is ProgramLifecycle.UNKNOWN


def test_identity_middleware_writes_canonical_agent_hint_to_vllm_xargs() -> None:
    captured_scope: dict[str, object] = {}
    captured_body = b""

    async def app(scope, receive, send) -> None:
        nonlocal captured_scope, captured_body
        captured_scope = scope
        message = await receive()
        body = message.get("body")
        assert isinstance(body, bytes)
        captured_body = body

    middleware = AgentCacheIdentityMiddleware(app)
    request_payload = {
        "model": "deepseek-v4",
        "messages": [],
        "agent_hint": {
            "session_id": "agent-sess-abc123",
            "parent_session_id": "parent-sess-001",
            "expected_resume": True,
            "cache_control": {"type": "ephemeral"},
        },
    }
    inbound = [
        {
            "type": "http.request",
            "body": json.dumps(request_payload).encode(),
            "more_body": False,
        }
    ]

    async def receive() -> dict[str, object]:
        return inbound.pop(0)

    async def send(message: dict[str, object]) -> None:
        return None

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"content-length", b"1")],
            },
            receive,
            send,
        )
    )

    adapted_payload = json.loads(captured_body)
    canonical = parse_agent_identity(vllm_xargs=adapted_payload["vllm_xargs"], headers={})
    assert canonical is not None
    assert canonical.program_id == canonical.session_id == "agent-sess-abc123"
    assert canonical.task_id is None
    assert canonical.agent_id is None
    assert canonical.parent_program_id == "parent-sess-001"
    assert canonical.blocks_parent is True
    assert canonical.expected_resume is True
    assert dict(captured_scope["headers"])[b"content-length"] == str(len(captured_body)).encode()


def test_middleware_initialization_explicitly_attaches_vllm_logging(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(api_adapter, "attach_agentinfer_to_vllm_logging", lambda: calls.append("attach"))
    monkeypatch.setenv(LIFECYCLE_SOCKET_ENV, str(tmp_path / "lifecycle.sock"))

    async def app(scope, receive, send) -> None:
        return None

    AgentCacheIdentityMiddleware(app)
    AgentCacheLifecycleMiddleware(app)

    assert calls == ["attach", "attach"]


def test_identity_middleware_carries_anthropic_headers_to_internal_vllm_xargs() -> None:
    """Anthropic Messages must preserve canonical identity through vLLM's Chat conversion."""
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    captured_body = b""

    async def app(scope, receive, send) -> None:
        nonlocal captured_body
        message = await receive()
        body = message.get("body")
        assert isinstance(body, bytes)
        captured_body = body

    middleware = AgentCacheIdentityMiddleware(app)
    request_payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "metadata": {
            "caller": "benchkit",
            "_agentinfer_replay_sampling": {
                "seed": 7,
                "min_tokens": 8,
                "ignore_eos": True,
            },
        },
    }
    inbound = [
        {
            "type": "http.request",
            "body": json.dumps(request_payload).encode(),
            "more_body": False,
        }
    ]

    async def receive() -> dict[str, object]:
        return inbound.pop(0)

    async def send(message: dict[str, object]) -> None:
        return None

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/messages",
                "headers": [
                    (b"x-claude-code-session-id", b"session-a"),
                    (b"x-claude-code-agent-id", b"child-a"),
                ],
            },
            receive,
            send,
        )
    )

    adapted_payload = json.loads(captured_body)
    assert adapted_payload["metadata"]["caller"] == "benchkit"
    assert "vllm_xargs" not in adapted_payload
    anthropic_request = AnthropicMessagesRequest.model_validate(adapted_payload)
    chat_request = AnthropicServingMessages._convert_anthropic_to_openai_request(anthropic_request)
    assert chat_request.seed == 7
    assert chat_request.min_tokens == 8
    assert chat_request.ignore_eos is True
    canonical = parse_agent_identity(vllm_xargs=chat_request.vllm_xargs, headers={})
    assert canonical is not None
    assert canonical.program_id == "session-a:child-a"
    assert canonical.parent_program_id == "session-a:lead"
    assert canonical.blocks_parent is True
    assert canonical.expected_resume is False


def test_identity_middleware_passes_oversized_request_through_without_unbounded_buffering(monkeypatch) -> None:
    monkeypatch.setattr(api_adapter, "_MAX_REQUEST_BODY_BYTES", 4)
    captured_body = bytearray()

    async def app(scope, receive, send) -> None:
        more_body = True
        while more_body:
            message = await receive()
            captured_body.extend(message.get("body", b""))
            more_body = message.get("more_body", False)

    middleware = AgentCacheIdentityMiddleware(app)
    inbound = [
        {"type": "http.request", "body": b"12345", "more_body": True},
        {"type": "http.request", "body": b"678", "more_body": False},
    ]

    async def receive() -> dict[str, object]:
        return inbound.pop(0)

    async def send(message: dict[str, object]) -> None:
        return None

    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
            receive,
            send,
        )
    )

    assert bytes(captured_body) == b"12345678"


def test_identity_middleware_ignores_provider_private_dsh_headers() -> None:
    request_body = json.dumps({"model": "deepseek-v4", "messages": []}).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [
            (b"x-deepseek-harness-user-id", b"benchmark-root"),
            (b"x-deepseek-harness-session-id", b"dsh-child"),
        ],
    }

    assert _adapt_identity_request_body(scope, request_body) == request_body


def test_identity_adapter_ignores_unsupported_body_locations() -> None:
    original = json.dumps(
        {
            "model": "test",
            "program_id": "unsupported-top-level",
            "agentic_context": {"program_id": "unsupported-context"},
            "extra_body": {"agentic_context": {"program_id": "unsupported-extra-body"}},
        }
    ).encode()

    assert _adapt_identity_request_body({"headers": []}, original) == original


def test_middleware_delivers_terminal_program_signal_without_rewriting_response(tmp_path, monkeypatch) -> None:
    socket_path = tmp_path / "lifecycle.sock"
    receiver = UnixLifecycleReceiver(str(socket_path), 2)
    monkeypatch.setenv(LIFECYCLE_SOCKET_ENV, str(socket_path))
    response_payload = {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}

    async def app(scope, receive, send) -> None:
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(response_payload).encode(),
                "more_body": False,
            }
        )

    middleware = AgentCacheLifecycleMiddleware(app)
    request_payload = {
        "model": "test",
        "messages": [],
        "vllm_xargs": {
            "agentic_context": json.dumps({"program_id": "program-1", "task_id": "task", "agent_id": "lead"}),
        },
    }
    inbound = [
        {
            "type": "http.request",
            "body": json.dumps(request_payload).encode(),
            "more_body": False,
        }
    ]
    outbound: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return inbound.pop(0)

    async def send(message: dict[str, object]) -> None:
        outbound.append(message)

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"x-data-parallel-rank", b"2")],
            },
            receive,
            send,
        )
    )

    assert outbound[-1]["body"] == json.dumps(response_payload).encode()
    assert receiver.receive() == (LifecycleSignal("program-1", ProgramLifecycle.TERMINAL),)
    receiver.close()


def test_lifecycle_channel_isolated_by_dp_rank(tmp_path) -> None:
    socket_path = str(tmp_path / "lifecycle.sock")
    receiver_zero = UnixLifecycleReceiver(socket_path, 0)
    receiver_one = UnixLifecycleReceiver(socket_path, 1)
    sender = UnixLifecycleSender(socket_path)
    signal = LifecycleSignal("program-1", ProgramLifecycle.TERMINAL)

    assert sender.send(signal, 1) is True
    assert receiver_zero.receive() == ()
    assert receiver_one.receive() == (signal,)
    receiver_zero.close()
    receiver_one.close()
