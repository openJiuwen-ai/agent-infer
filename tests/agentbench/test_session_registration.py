import asyncio

import httpx

from agentinfer.agentbench.benchkit import session_registration

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def test_register_sends_only_session_id(monkeypatch) -> None:
    request = None

    def handler(value: httpx.Request) -> httpx.Response:
        nonlocal request
        request = value
        return httpx.Response(200, request=value)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        session_registration.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(transport=transport, **kwargs),
    )

    result = asyncio.run(session_registration.register_session("http://router/", "session-1"))

    assert result.success
    assert request is not None
    assert request.url.path == "/v1/sessions/register"
    assert request.content == b'{"session_id":"session-1"}'


def test_cleanup_returns_non_success_response(monkeypatch) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="unavailable", request=request))
    monkeypatch.setattr(
        session_registration.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(transport=transport, **kwargs),
    )

    result = asyncio.run(session_registration.cleanup_session("http://router", "session-1"))

    assert not result.success
    assert result.status_code == 503
    assert result.error == "unavailable"


def test_registration_captures_http_error(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        session_registration.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(transport=transport, **kwargs),
    )

    result = asyncio.run(session_registration.register_session("http://router", "session-1"))

    assert not result.success
    assert result.status_code is None
    assert result.error == "offline"
