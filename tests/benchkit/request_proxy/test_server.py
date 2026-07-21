import asyncio
import gzip
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from agentcache.benchmarks.request_proxy.request_trace import RequestTraceWriter
from agentcache.benchmarks.request_proxy.server import (
    RequestProxyLaunchConfig,
    RequestProxyServer,
    _serve_request_proxy,
)
from agentcache.utils.headers import CLAUDE_AGENT_HEADER, CLAUDE_SESSION_HEADER


def _runner_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serve: Callable[[], Awaitable[None]],
    *,
    proxy_error: Exception | None = None,
    writer_error: Exception | None = None,
):
    instances = {}

    class FakeWriter:
        def __init__(self, path):
            instances["writer"] = self
            self.path = path
            self.start = AsyncMock()
            self.close = AsyncMock(side_effect=writer_error)

    class FakeProxy:
        def __init__(self, *_args, **_kwargs):
            instances["proxy"] = self
            self.app = object()
            self.close = AsyncMock(side_effect=proxy_error)

    class FakeUvicornServer:
        def __init__(self, _config):
            pass

        async def serve(self):
            await run_serve()

    run_serve = serve

    monkeypatch.setattr("agentcache.benchmarks.request_proxy.server.RequestTraceWriter", FakeWriter)
    monkeypatch.setattr("agentcache.benchmarks.request_proxy.server.RequestProxyServer", FakeProxy)
    monkeypatch.setattr("agentcache.benchmarks.request_proxy.server.uvicorn.Config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("agentcache.benchmarks.request_proxy.server.uvicorn.Server", FakeUvicornServer)
    config = RequestProxyLaunchConfig(
        "127.0.0.1",
        8123,
        "http://upstream",
        "run",
        tmp_path / "trace.jsonl",
        10,
        2,
        "token",
        tmp_path / "out",
        tmp_path / "err",
    )
    return config, instances


def test_server_runner_closes_proxy_and_writer_when_uvicorn_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def serve():
        raise RuntimeError("serve failed")

    config, instances = _runner_fakes(
        tmp_path,
        monkeypatch,
        serve,
        proxy_error=RuntimeError("proxy cleanup failed"),
    )

    with pytest.raises(RuntimeError, match="serve failed"):
        asyncio.run(_serve_request_proxy(config))
    instances["writer"].start.assert_awaited_once()
    instances["proxy"].close.assert_awaited_once()
    instances["writer"].close.assert_awaited_once()


def test_server_runner_reports_cleanup_failure_after_normal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def serve():
        return None

    config, instances = _runner_fakes(
        tmp_path,
        monkeypatch,
        serve,
        proxy_error=RuntimeError("proxy cleanup failed"),
        writer_error=RuntimeError("writer cleanup failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="request proxy cleanup failed: proxy cleanup failed; writer cleanup failed",
    ):
        asyncio.run(_serve_request_proxy(config))
    instances["proxy"].close.assert_awaited_once()
    instances["writer"].close.assert_awaited_once()


def test_transparent_nonstream_body_protocol_headers_status_and_response(tmp_path: Path):
    async def run():
        seen = {}
        raw_response = b'{"content": [ ] , "usage":{"input_tokens":4,"output_tokens":2}}'

        def upstream(request: httpx.Request):
            seen["body"] = request.content
            seen["headers"] = dict(request.headers)
            return httpx.Response(
                202, content=raw_response, headers={"content-type": "application/json", "x-upstream": "yes"}
            )

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        await writer.start()
        body = b'{"model":"m","messages":[{"role":"user","content":"hello"}]}'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            response = await client.post(
                "/v1/messages",
                content=body,
                headers={
                    CLAUDE_SESSION_HEADER: "s1",
                    CLAUDE_AGENT_HEADER: "a1",
                    "authorization": "Bearer secret",
                    "x-api-key": "api-secret",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "feature",
                    "content-type": "application/json",
                    "x-unrelated": "drop",
                },
            )
        first, second = await asyncio.gather(server.close(), server.close())
        assert first == second
        assert response.status_code == 202
        assert response.content == raw_response
        assert response.headers["x-upstream"] == "yes"
        assert seen["body"] == body
        assert seen["headers"][CLAUDE_SESSION_HEADER.lower()] == "s1"
        assert seen["headers"][CLAUDE_AGENT_HEADER.lower()] == "a1"
        assert seen["headers"]["authorization"] == "Bearer secret"
        assert seen["headers"]["x-api-key"] == "api-secret"
        assert seen["headers"]["anthropic-version"] == "2023-06-01"
        assert seen["headers"]["anthropic-beta"] == "feature"
        assert "x-unrelated" not in seen["headers"]
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert (row["actor_id"], row["actor_role"], row["input_tokens"], row["status_code"]) == (
            "a1",
            "subagent",
            4,
            202,
        )

    asyncio.run(run())


def test_nonstream_compressed_response_preserves_raw_body_and_observes_usage(tmp_path: Path):
    async def run():
        raw_response = gzip.compress(b'{"content":[],"usage":{"input_tokens":4}}')

        class RawStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield raw_response

        def upstream(_request: httpx.Request):
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                stream=RawStream(),
            )

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            async with client.stream("POST", "/v1/messages", content=b"{}") as response:
                received = b"".join([chunk async for chunk in response.aiter_raw()])
                assert response.headers["content-encoding"] == "gzip"
        await server.close()

        assert received == raw_response
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert row["input_tokens"] == 4

    asyncio.run(run())


def test_invalid_compressed_payload_remains_transparent(tmp_path: Path):
    async def run():
        raw_response = b"not-gzip"

        class RawStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield raw_response

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"content-type": "application/json", "content-encoding": "gzip"},
                    stream=RawStream(),
                )
            )
        )
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            async with client.stream("POST", "/v1/messages", content=b"{}") as response:
                received = b"".join([chunk async for chunk in response.aiter_raw()])
        await server.close()

        assert received == raw_response
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert row["input_tokens"] is None

    asyncio.run(run())


def test_ttft_waits_for_content_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def run():
        observed_times = iter((10.0, 10.1))
        clock = type("Clock", (), {"monotonic": staticmethod(lambda: next(observed_times, 10.2))})()
        monkeypatch.setattr("agentcache.benchmarks.request_proxy.server.time", clock)
        chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n',
            b'event: ping\ndata: {"type":"ping"}\n\n',
            b"data: not-json\n\n",
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"x"}}\n\n',
        ]

        class ChunkStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in chunks:
                    yield chunk

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=ChunkStream()
                )
            )
        )
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            await client.post("/v1/messages", content=b"{}")
        await server.close()

        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert row["ttft_seconds"] == pytest.approx(0.1)

    asyncio.run(run())


def test_usage_null_does_not_change_upstream_response(tmp_path: Path):
    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"usage": None}))
        )
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            response = await client.post("/v1/messages", content=b"{}")
        await server.close()

        assert response.status_code == 200
        assert response.json() == {"usage": None}
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert row["input_tokens"] is None

    asyncio.run(run())


def test_streaming_bytes_headers_status_and_sse_usage_are_preserved(tmp_path: Path):
    async def run():
        chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7,',
            b'"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}\r\n\r\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":5}}\n\n',
        ]

        class ChunkStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in chunks:
                    yield chunk

        def stream(_request: httpx.Request):
            return httpx.Response(
                206, headers={"content-type": "Text/Event-Stream", "x-stream": "yes"}, stream=ChunkStream()
            )

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(stream))
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            response = await client.post("/v1/messages", content=b"{}", headers={CLAUDE_SESSION_HEADER: "s1"})
        await server.close()
        assert response.status_code == 206
        assert response.headers["x-stream"] == "yes"
        assert response.content == b"".join(chunks)
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert (
            row["input_tokens"],
            row["output_tokens"],
            row["cache_creation_tokens"],
            row["cached_tokens"],
        ) == (7, 5, 2, 3)
        assert row["ttft_seconds"] is not None

    asyncio.run(run())


def test_compressed_sse_preserves_raw_bytes_and_observes_metrics(tmp_path: Path):
    async def run():
        decoded = (
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"x"}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":5}}\n\n'
        )
        encoded = gzip.compress(decoded)

        class RawStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield encoded[:17]
                yield encoded[17:]

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
                    stream=RawStream(),
                )
            )
        )
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            async with client.stream("POST", "/v1/messages", content=b"{}") as response:
                received = b"".join([chunk async for chunk in response.aiter_raw()])
        await server.close()

        assert received == encoded
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert (row["input_tokens"], row["output_tokens"]) == (7, 5)
        assert row["ttft_seconds"] is not None

    asyncio.run(run())


def test_response_start_disconnect_releases_stream(tmp_path: Path):
    class FakeRequest:
        headers = {}

        async def body(self):
            return b"{}"

    async def run():
        upstream_closed = False

        class RawStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"content_block_delta"}\n\n'

            async def aclose(self):
                nonlocal upstream_closed
                upstream_closed = True

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=RawStream())
            )
        )
        await writer.start()
        response = await server.handle_messages(FakeRequest())

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                raise ConnectionError("client disconnected")

        with pytest.raises(ConnectionError, match="client disconnected"):
            await response({"type": "http", "method": "POST", "path": "/v1/messages"}, receive, send)

        assert upstream_closed is True
        assert server._active_requests == 0
        await server.close()

    asyncio.run(run())


def test_trace_writer_failure_does_not_change_upstream_response(tmp_path: Path):
    async def run():
        writer = RequestTraceWriter(tmp_path)
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"ok"))
        )
        await writer.start()
        await writer.drain()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            response = await client.post("/v1/messages", content=b"{}")
        health = await server.close()

        assert response.status_code == 200
        assert response.content == b"ok"
        assert health.writer_error is not None

    asyncio.run(run())


def test_cancelled_upstream_send_releases_admission(tmp_path: Path):
    class FakeRequest:
        headers = {}

        async def body(self):
            return b"{}"

    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        await writer.start()
        server.client.send = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await server.handle_messages(FakeRequest())

        assert server._active_requests == 0
        await server.close()

    asyncio.run(run())


def test_upstream_send_failure_records_request_fact(tmp_path: Path):
    async def run():
        def fail(_request: httpx.Request):
            raise httpx.ConnectError("connect failed")

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token")
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://proxy") as client:
            with pytest.raises(httpx.ConnectError):
                await client.post("/v1/messages", content=b"{}")
        await server.close()
        row = json.loads((tmp_path / "requests.jsonl").read_text())
        assert row["status"] == "error"
        assert row["status_code"] is None
        assert "connect failed" in row["error"]

    asyncio.run(run())


def test_authenticated_shutdown_waits_for_inflight_and_stops_admission(tmp_path: Path):
    async def run():
        gate = asyncio.Event()

        class BlockedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                await gate.wait()
                yield b"data: done\n\n"

        def stream(_request: httpx.Request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BlockedStream())

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token", shutdown_timeout_seconds=1)
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(stream))
        await writer.start()
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            request_task = asyncio.create_task(client.post("/v1/messages", content=b"{}"))
            while server._active_requests != 1:
                await asyncio.sleep(0)
            assert (await client.post("/shutdown", headers={"authorization": "Bearer wrong"})).status_code == 403
            shutdown_task = asyncio.create_task(client.post("/shutdown", headers={"authorization": "Bearer token"}))
            while server.accepting:
                await asyncio.sleep(0)
            assert (await client.post("/v1/messages", content=b"{}")).status_code == 503
            assert not shutdown_task.done()
            gate.set()
            await request_task
            shutdown = await shutdown_task
            assert shutdown.status_code == 200
            assert shutdown.json()["pending"] == 0

    asyncio.run(run())


def test_stream_read_failure_records_error_and_releases_shutdown(tmp_path: Path):
    async def run():
        stream_entered = asyncio.Event()
        fail_stream = asyncio.Event()

        class FailingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                stream_entered.set()
                await fail_stream.wait()
                raise httpx.ReadError("stream exploded")
                yield b""  # pragma: no cover

        def stream(_request: httpx.Request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=FailingStream())

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://router", "run", writer, 10, "token", shutdown_timeout_seconds=1)
        server.client = httpx.AsyncClient(transport=httpx.MockTransport(stream))
        await writer.start()
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            request_task = asyncio.create_task(client.post("/v1/messages", content=b"{}"))
            await stream_entered.wait()
            shutdown_task = asyncio.create_task(client.post("/shutdown", headers={"authorization": "Bearer token"}))
            while server.accepting:
                await asyncio.sleep(0)
            assert not shutdown_task.done()
            fail_stream.set()
            with pytest.raises(httpx.ReadError, match="stream exploded"):
                await request_task
            shutdown = await shutdown_task

        assert shutdown.status_code == 200
        assert shutdown.json()["pending"] == 0
        assert server._active_requests == 0
        row = json.loads((tmp_path / "requests.jsonl").read_text(encoding="utf-8"))
        assert row["status"] == "error"
        assert row["status_code"] == 200
        assert "stream exploded" in row["error"]

    asyncio.run(run())
