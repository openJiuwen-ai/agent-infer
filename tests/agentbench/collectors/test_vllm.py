# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio

import httpx

from agentinfer.agentbench.benchkit.collectors.vllm import capture_vllm_metrics

from .http_fakes import FakeClient, FakeResponse


def test_capture_vllm_metrics_returns_raw_text(monkeypatch) -> None:
    response = FakeResponse(text="vllm:prefix_cache_hits_total 4\n")
    urls = []

    class Client(FakeClient):
        async def get(self, url: str):
            urls.append(url)
            return await super().get(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client(response, **kwargs))

    capture = asyncio.run(capture_vllm_metrics("http://vllm:8000/metrics"))

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {"text": "vllm:prefix_cache_hits_total 4\n"}
    assert urls == ["http://vllm:8000/metrics"]


def test_capture_vllm_metrics_reports_http_failure(monkeypatch) -> None:
    error = httpx.ConnectError("connection failed", request=httpx.Request("GET", "http://vllm:8000/metrics"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient(error=error, **kwargs))

    capture = asyncio.run(capture_vllm_metrics("http://vllm:8000/metrics"))

    assert capture.available is False
    assert capture.path is None
    assert "connection failed" in capture.reason
