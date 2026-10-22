# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Test-only fixtures for hermetic AgentBench contract runs."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest, AgentRunResult, TerminationReason
from agentinfer.agentbench.benchkit.common import utc_now
from agentinfer.agentbench.benchkit.dataset import Task
from agentinfer.agentbench.benchkit.workspace import export_patch
from agentinfer.agentbench.request_proxy.observers import _OpenAISSEUsageObserver
from agentinfer.agentbench.request_proxy.process import ProcessExit
from agentinfer.agentbench.request_proxy.server import RequestProxyLaunchConfig
from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity

from .mock_server import MockServerProcess


@dataclass(frozen=True)
class FakeProfile:
    name: str = "code.normal"

    @staticmethod
    def build_prompt(task: Task) -> str:
        return task.problem_statement


class ThreadBarrier:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.entered: set[str] = set()
        self._ready = asyncio.Event()

    def arrive(self, task_id: str) -> None:
        self.loop.call_soon_threadsafe(self._arrive, task_id)

    def _arrive(self, task_id: str) -> None:
        self.entered.add(task_id)
        if len(self.entered) == 2:
            self._ready.set()

    async def wait(self) -> None:
        await asyncio.wait_for(self._ready.wait(), 5)


class ThreadRequestProxyProcess:
    """Run the production proxy server in-process for fast cross-platform tests."""

    def __init__(self, stdout_path: Path, stderr_path: Path) -> None:
        from agentinfer.agentbench.request_proxy.process import ProcessExit

        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._exit_type = ProcessExit
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._exit = None

    async def start(self, config: RequestProxyLaunchConfig) -> None:
        from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter
        from agentinfer.agentbench.request_proxy.server import RequestProxyServer

        if self._thread is not None:
            raise RuntimeError("process has already started")
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_path.touch()
        self.stderr_path.touch()

        def serve() -> None:
            async def run() -> None:
                writer = RequestTraceWriter(config.trace_path)
                proxy = RequestProxyServer(
                    config.upstream_url,
                    config.run_id,
                    writer,
                    config.request_timeout_seconds,
                    config.shutdown_token,
                    config.endpoint,
                    config.shutdown_timeout_seconds,
                    lambda: setattr(self._server, "should_exit", True),
                )
                self._server = uvicorn.Server(
                    uvicorn.Config(proxy.app, host=config.host, port=config.port, log_level="error", access_log=False)
                )
                await writer.start()
                try:
                    await self._server.serve()
                finally:
                    await proxy.close()
                    await writer.close()

            try:
                asyncio.run(run())
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=serve, name="request-proxy", daemon=True)
        self._thread.start()

    async def poll(self) -> int | None:
        if self._thread is None:
            raise RuntimeError("process has not started")
        if self._thread.is_alive():
            return None
        return 1 if self._error else 0

    async def wait(self, timeout_seconds: float) -> ProcessExit:
        if self._exit is not None:
            return self._exit
        if self._thread is None:
            raise RuntimeError("process has not started")
        await asyncio.to_thread(self._thread.join, timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError(f"request proxy did not exit within {timeout_seconds}s")
        self._exit = self._exit_type(1 if self._error else 0, self.stdout_path, self.stderr_path)
        return self._exit

    async def terminate(self, timeout_seconds: float) -> ProcessExit:
        if self._server is not None:
            self._server.should_exit = True
        return await self.wait(timeout_seconds)


class FakeAgentRuntime:
    """Exercise runtime dispatch, proxy requests, workspace writes, and patches."""

    agent_type = "jiuwenswarm"
    required_endpoint = "/v1/chat/completions"
    usage_observer_class = _OpenAISSEUsageObserver

    def __init__(
        self,
        *,
        barrier: ThreadBarrier | None = None,
        delays: dict[str, float] | None = None,
        failed_tasks: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.barrier = barrier
        self.delays = delays or {}
        self.failed_tasks = frozenset(failed_tasks or ())
        self.requests: list[AgentRunRequest] = []
        self.completion_order: list[str] = []

    def get_profile(self, name: str) -> FakeProfile:
        if name != "code.normal":
            raise ValueError(f"Unsupported fake profile: {name}")
        return FakeProfile()

    async def preflight(self, _executable: Path) -> None:
        return None

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.requests.append(request)
        started_at = utc_now()
        if self.barrier is not None:
            self.barrier.arrive(request.task.instance_id)
            await self.barrier.wait()
        if delay := self.delays.get(request.task.instance_id, 0):
            await asyncio.sleep(delay)
        self.completion_order.append(request.task.instance_id)

        identity = AgentIdentity(
            program_id=f"{request.session_id}:lead",
            task_id=request.session_id,
            session_id=request.session_id,
            agent_id="lead",
            blocks_parent=False,
            expected_resume=True,
            agent_role="lead",
        )
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.task.problem_statement}],
            "vllm_xargs": {"agentic_context": encode_agent_identity(identity)},
        }
        async with httpx.AsyncClient(timeout=request.timeout_seconds, trust_env=False) as client:
            response = await client.post(f"{request.api_base_url}{self.required_endpoint}", json=payload)
            response.raise_for_status()

        changed = request.workspace / "tracked.txt"
        changed.write_text(f"changed by {request.task.instance_id}\n", encoding="utf-8")
        patch = await asyncio.to_thread(export_patch, request.workspace)
        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        (request.artifact_dir / "model.patch").write_text(patch, encoding="utf-8")
        finished_at = utc_now()
        failed = request.task.instance_id in self.failed_tasks
        return AgentRunResult(
            agent_type=self.agent_type,
            profile_name=request.profile_name,
            outcome=AgentRunOutcome.FAILED if failed else AgentRunOutcome.COMPLETED,
            termination_reason=TerminationReason.INTERRUPTED if failed else None,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            session_id=request.session_id,
            instance_id=request.task.instance_id,
            patch_bytes=len(patch.encode()),
            has_patch=bool(patch),
        )


class FakeModelServer(MockServerProcess):
    """Run the original fixed-usage fixture through the shared child service."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, {"profile": "run_legacy", "model": "fake-model"})
