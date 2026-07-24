import asyncio
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.benchkit.config import RequestProxyConfig
from agentinfer.agentbench.request_proxy.lifecycle import (
    RequestProxyHandle,
    RequestProxyLifecycle,
    _available_port,
)
from agentinfer.agentbench.request_proxy.process import ProcessExit, RequestProxyProcess
from agentinfer.agentbench.request_proxy.server import RequestProxyLaunchConfig, run_request_proxy


def launch_config(tmp_path: Path) -> RequestProxyLaunchConfig:
    return RequestProxyLaunchConfig(
        host="127.0.0.1",
        port=8123,
        upstream_url="http://upstream",
        run_id="run",
        trace_path=tmp_path / "requests.jsonl",
        request_timeout_seconds=10,
        shutdown_timeout_seconds=2,
        shutdown_token="secret",
        stdout_path=tmp_path / "proxy.out",
        stderr_path=tmp_path / "proxy.err",
    )


def test_launch_config_is_frozen_and_spawn_serializable(tmp_path: Path) -> None:
    config = launch_config(tmp_path)
    assert pickle.loads(pickle.dumps(config)) == config
    with pytest.raises(FrozenInstanceError):
        config.port = 9000  # type: ignore[misc]


def test_process_uses_spawn_context_target_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = launch_config(tmp_path)
    calls = {}

    class SpawnedProcess:
        exitcode = None

        def start(self):
            calls["started"] = True

    class SpawnContext:
        def Process(self, *, target, args):
            calls.update(target=target, args=args)
            return SpawnedProcess()

    def get_context(method):
        calls["method"] = method
        return SpawnContext()

    monkeypatch.setattr("agentinfer.agentbench.request_proxy.process.multiprocessing.get_context", get_context)

    async def run():
        process = RequestProxyProcess(config.stdout_path, config.stderr_path)
        await process.start(config)

    asyncio.run(run())
    assert calls == {"method": "spawn", "target": run_request_proxy, "args": (config,), "started": True}


def test_process_wait_joins_without_termination(tmp_path: Path) -> None:
    calls = []

    class JoinedProcess:
        exitcode = 0

        def join(self, timeout):
            calls.append(("join", timeout))

        def close(self):
            calls.append("close")

        def is_alive(self):
            return False

    async def run():
        process = RequestProxyProcess(tmp_path / "out", tmp_path / "err")
        process._process = JoinedProcess()
        result = await process.wait(3)
        assert result == ProcessExit(0, tmp_path / "out", tmp_path / "err")
        assert await process.wait(3) == result

    asyncio.run(run())
    assert calls == [("join", 3), "close"]


def test_process_terminate_joins_then_kills_and_joins(tmp_path: Path) -> None:
    calls = []

    class StubbornProcess:
        exitcode = -9
        alive = True

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.alive = False

        def close(self):
            calls.append("close")

        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return self.alive

    async def run():
        process = RequestProxyProcess(tmp_path / "out", tmp_path / "err")
        process._process = StubbornProcess()
        result = await process.terminate(2)
        assert result.return_code == -9

    asyncio.run(run())
    assert calls == ["terminate", ("join", 2), "kill", ("join", 2), "close"]


class FakeProcess:
    def __init__(self, tmp_path: Path, polls: list[int | None]):
        self.polls = polls
        self.exit = ProcessExit(0, tmp_path / "out", tmp_path / "err")
        self.stdout_path = self.exit.stdout_path
        self.stderr_path = self.exit.stderr_path
        self.started_with = None
        self.waited = False
        self.terminated = False

    async def start(self, config):
        self.started_with = config

    async def poll(self):
        return self.polls.pop(0) if self.polls else None

    async def wait(self, _timeout):
        self.waited = True
        return self.exit

    async def terminate(self, _timeout):
        self.terminated = True
        return self.exit


class FakeClient:
    def __init__(self, responses: list[httpx.Response]):
        self.responses = responses
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _url):
        return self.responses.pop(0)

    async def post(self, url, headers=None):
        self.posts.append((url, headers))
        return self.responses.pop(0)


def test_lifecycle_start_readiness_authenticated_shutdown_join_and_idempotent_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run():
        lifecycle = RequestProxyLifecycle(RequestProxyConfig(), "http://upstream", "run", tmp_path)
        process = FakeProcess(tmp_path, [None])
        lifecycle.process = process
        handle = await lifecycle.start()
        assert process.started_with == lifecycle.launch_config
        assert process.started_with.shutdown_token == lifecycle.token
        assert handle.trace_path == tmp_path / "requests.jsonl"

        responses = [
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(
                200,
                json={"submitted": 1, "written": 1, "pending": 0, "writer_error": None},
                request=httpx.Request("POST", f"{handle.base_url}/shutdown"),
            ),
        ]
        client = FakeClient(responses)
        monkeypatch.setattr("agentinfer.agentbench.request_proxy.lifecycle.httpx.AsyncClient", lambda **_kwargs: client)
        await lifecycle.wait_ready(1)
        first, second = await asyncio.gather(lifecycle.close(), lifecycle.close())
        close = first
        assert second is first
        assert close.graceful is True
        assert close.trace_health is not None
        assert close.process_exit is not None
        assert process.waited is True
        assert process.terminated is False
        assert client.posts == [(f"{handle.base_url}/shutdown", {"authorization": f"Bearer {lifecycle.token}"})]
        assert await lifecycle.close() is close
        assert len(client.posts) == 1

    asyncio.run(run())


def test_lifecycle_formats_ipv6_auto_and_fixed_ports(tmp_path: Path) -> None:
    async def run():
        auto = RequestProxyLifecycle(
            RequestProxyConfig(listen_url="http://[::1]:0"), "http://upstream", "run", tmp_path
        )
        auto.process = FakeProcess(tmp_path, [])
        auto_handle = await auto.start()
        assert auto_handle.base_url.startswith("http://[::1]:")
        assert auto.launch_config.host == "::1"
        assert auto.launch_config.port > 0

        fixed = RequestProxyLifecycle(
            RequestProxyConfig(listen_url="http://[::1]:8123"), "http://upstream", "run", tmp_path
        )
        fixed.process = FakeProcess(tmp_path, [])
        fixed_handle = await fixed.start()
        assert fixed_handle.base_url == "http://[::1]:8123"

    try:
        _available_port("::1")
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    asyncio.run(run())


def test_lifecycle_cancellation_terminates_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run():
        lifecycle = RequestProxyLifecycle(RequestProxyConfig(), "http://upstream", "run", tmp_path)
        process = FakeProcess(tmp_path, [])
        lifecycle.process = process
        lifecycle.handle = RequestProxyHandle("http://proxy", tmp_path / "requests.jsonl")

        class CancelledClient(FakeClient):
            async def post(self, url, headers=None):
                raise asyncio.CancelledError

        monkeypatch.setattr(
            "agentinfer.agentbench.request_proxy.lifecycle.httpx.AsyncClient",
            lambda **_kwargs: CancelledClient([]),
        )

        with pytest.raises(asyncio.CancelledError):
            await lifecycle.close()
        assert process.terminated is True

    asyncio.run(run())


def test_lifecycle_reports_early_exit_and_uses_failure_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run():
        lifecycle = RequestProxyLifecycle(RequestProxyConfig(), "http://upstream", "run", tmp_path)
        lifecycle.handle = RequestProxyHandle("http://proxy", tmp_path / "missing")
        lifecycle.process = FakeProcess(tmp_path, [7])
        monkeypatch.setattr(
            "agentinfer.agentbench.request_proxy.lifecycle.httpx.AsyncClient", lambda **_kwargs: FakeClient([])
        )
        with pytest.raises(RuntimeError, match="exited before readiness"):
            await lifecycle.wait_ready(1)
        close = await lifecycle.close()
        assert close.trace_health is None
        assert close.graceful is False
        assert close.process_exit is not None
        assert lifecycle.process.terminated is True

    asyncio.run(run())


def test_lifecycle_reports_nonzero_exit_after_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run():
        lifecycle = RequestProxyLifecycle(RequestProxyConfig(), "http://upstream", "run", tmp_path)
        process = FakeProcess(tmp_path, [])
        process.exit = ProcessExit(7, tmp_path / "out", tmp_path / "err")
        lifecycle.process = process
        lifecycle.handle = RequestProxyHandle("http://proxy", tmp_path / "requests.jsonl")
        response = httpx.Response(
            200,
            json={"submitted": 1, "written": 1, "pending": 0, "writer_error": None},
            request=httpx.Request("POST", "http://proxy/shutdown"),
        )
        monkeypatch.setattr(
            "agentinfer.agentbench.request_proxy.lifecycle.httpx.AsyncClient",
            lambda **_kwargs: FakeClient([response]),
        )

        close = await lifecycle.close()

        assert close.graceful is False
        assert close.process_exit == process.exit
        assert close.error == "request proxy exited with code 7"

    asyncio.run(run())
