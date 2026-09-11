# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Request Proxy startup, readiness, bounded shutdown, and evidence recovery."""

import asyncio
import secrets
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from ..benchkit.config import RequestProxyConfig
from .process import ProcessExit, RequestProxyProcess
from .request_trace import TraceHealth
from .server import RequestProxyLaunchConfig

_TRACE_HEALTH_ADAPTER = TypeAdapter(TraceHealth)


@dataclass(frozen=True)
class RequestProxyHandle:
    """Expose the ready-check endpoint and request-trace path to Benchkit."""

    base_url: str
    trace_path: Path


@dataclass(frozen=True)
class RequestProxyCloseResult:
    """Combine graceful-shutdown trace health with child-process evidence."""

    trace_health: TraceHealth | None
    process_exit: ProcessExit | None
    graceful: bool
    error: str | None


class RequestProxyLifecycle:
    """Coordinate proxy spawn, readiness, authenticated shutdown, and evidence recovery."""

    def __init__(
        self,
        config: RequestProxyConfig,
        upstream_url: str,
        run_id: str,
        output_dir: Path,
        endpoint: str,
    ) -> None:
        self.config = config
        self.upstream_url = upstream_url
        self.run_id = run_id
        self.output_dir = output_dir
        self.endpoint = endpoint
        self.trace_path = output_dir / "requests.jsonl"
        self.token = secrets.token_urlsafe(32)
        stdout_path = self.output_dir / "proxy.stdout.log"
        stderr_path = self.output_dir / "proxy.stderr.log"
        self.process = RequestProxyProcess(stdout_path, stderr_path)
        self.launch_config: RequestProxyLaunchConfig | None = None
        self.handle: RequestProxyHandle | None = None
        self._close_result: RequestProxyCloseResult | None = None
        self._close_lock = asyncio.Lock()

    async def start(self) -> RequestProxyHandle:
        port = self.config.port or _available_port(self.config.host)
        base_url = f"http://{_url_host(self.config.host)}:{port}"
        self.launch_config = RequestProxyLaunchConfig(
            host=self.config.host,
            port=port,
            upstream_url=self.upstream_url,
            run_id=self.run_id,
            trace_path=self.trace_path,
            request_timeout_seconds=self.config.request_timeout_seconds,
            shutdown_timeout_seconds=self.config.shutdown_timeout_seconds,
            shutdown_token=self.token,
            stdout_path=self.process.stdout_path,
            stderr_path=self.process.stderr_path,
            endpoint=self.endpoint,
        )
        await self.process.start(self.launch_config)
        self.handle = RequestProxyHandle(base_url, self.trace_path)
        return self.handle

    async def wait_ready(self, timeout_seconds: float) -> None:
        if self.handle is None:
            raise RuntimeError("proxy has not started")
        deadline = time.monotonic() + timeout_seconds
        async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline:
                return_code = await self.process.poll()
                if return_code is not None:
                    raise RuntimeError(f"request proxy exited before readiness with code {return_code}")
                try:
                    if (await client.get(f"{self.handle.base_url}/health")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
        raise RuntimeError(f"request proxy did not become ready within {timeout_seconds}s")

    async def close(self) -> RequestProxyCloseResult:
        async with self._close_lock:
            if self._close_result is not None:
                return self._close_result
            if self.handle is None:
                self._close_result = RequestProxyCloseResult(None, None, True, None)
                return self._close_result
            trace_health = None
            try:
                async with httpx.AsyncClient(timeout=self.config.shutdown_timeout_seconds, trust_env=False) as client:
                    response = await client.post(
                        f"{self.handle.base_url}/shutdown", headers={"authorization": f"Bearer {self.token}"}
                    )
                    response.raise_for_status()
                    trace_health = _TRACE_HEALTH_ADAPTER.validate_python(response.json())
                exit_result = await self.process.wait(self.config.shutdown_timeout_seconds)
                if exit_result.return_code != 0:
                    self._close_result = RequestProxyCloseResult(
                        trace_health,
                        exit_result,
                        False,
                        f"request proxy exited with code {exit_result.return_code}",
                    )
                else:
                    self._close_result = RequestProxyCloseResult(trace_health, exit_result, True, None)
            except asyncio.CancelledError:
                await asyncio.shield(self.process.terminate(self.config.shutdown_timeout_seconds))
                raise
            except Exception as exc:
                try:
                    exit_result = await self.process.terminate(self.config.shutdown_timeout_seconds)
                except Exception as terminate_exc:
                    self._close_result = RequestProxyCloseResult(
                        trace_health,
                        None,
                        False,
                        f"{exc}; process termination failed: {terminate_exc}",
                    )
                else:
                    self._close_result = RequestProxyCloseResult(trace_health, exit_result, False, str(exc))
            return self._close_result


def create_request_proxy_lifecycle(
    config: RequestProxyConfig,
    upstream_url: str,
    run_id: str,
    output_dir: Path,
    endpoint: str = "/v1/messages",
) -> RequestProxyLifecycle:
    return RequestProxyLifecycle(config, upstream_url, run_id, output_dir, endpoint)


def _available_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host
