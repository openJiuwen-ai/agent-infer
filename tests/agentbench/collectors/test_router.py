import asyncio

import httpx

from agentinfer.agentbench.benchkit.collectors.router import capture_router_snapshot

from .http_fakes import FakeClient, FakeResponse


def test_capture_router_snapshot_returns_raw_payload(monkeypatch) -> None:
    payload = {"events": [{"event": "route_selected"}]}
    response = FakeResponse(payload=payload)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient(response, **kwargs))

    capture = asyncio.run(capture_router_snapshot("http://router/"))

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {"raw": payload}


def test_capture_router_snapshot_reports_invalid_payload(monkeypatch) -> None:
    response = FakeResponse(payload="invalid")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient(response, **kwargs))

    capture = asyncio.run(capture_router_snapshot("http://router"))

    assert capture.available is False
    assert capture.reason == "metrics must contain a JSON object or array"


def test_capture_router_snapshot_reports_http_failure(monkeypatch) -> None:
    error = httpx.ConnectError("connection failed", request=httpx.Request("GET", "http://router/metrics"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient(error=error, **kwargs))

    capture = asyncio.run(capture_router_snapshot("http://router"))

    assert capture.available is False
    assert capture.path is None
    assert "connection failed" in capture.reason
