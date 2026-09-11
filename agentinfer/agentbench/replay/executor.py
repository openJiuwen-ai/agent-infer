# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Execute structural Replay dependency graphs with node-level isolation.

This module owns release timing, dependency readiness, and execution outcomes;
Prompt construction and HTTP transport remain delegated to their owners.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass

from .config import ReplayBenchConfig
from .planner import ReplayPlan, ReplayPlanNode, ReplayTaskPlan
from .prompt import PromptBuilder, PromptExchange, SyntheticPrompt
from .transport import ReplayTransport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeExecution:
    source_key: str
    runtime_request_id: str
    node_type: str
    prompt_kind: str
    status: str
    planned_send_offset_seconds: float | None
    actual_send_offset_seconds: float | None
    scheduler_lag_seconds: float | None
    error: str | None
    context_mode: str = "none"
    prompt_calibration: dict[str, object] | None = None


@dataclass(frozen=True)
class ReplayTaskExecution:
    task_id: str
    runtime_session_id: str
    status: str
    duration_seconds: float
    error: str | None
    nodes: tuple[NodeExecution, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["nodes"] = [asdict(node) for node in self.nodes]
        return value


@dataclass(frozen=True)
class _NodeCompletion:
    """Resolve one node for timing and context consumers, including failures."""

    finished_clock: float
    exchange: PromptExchange | None
    status: str = "success"
    error: str | None = None


class _DependencyFailed(RuntimeError):
    """A planned node cannot run because one of its dependencies failed."""


class ReplayExecutor:
    """Execute one structural Replay plan with bounded Session concurrency."""

    def __init__(
        self,
        config: ReplayBenchConfig,
        plan: ReplayPlan,
        prompts: PromptBuilder,
        transport: ReplayTransport,
    ) -> None:
        self.config = config
        self.plan = plan
        self.prompts = prompts
        self.transport = transport

    async def execute(self) -> tuple[ReplayTaskExecution, ...]:
        """Run every planned Session and return its finalized execution record."""

        semaphore = asyncio.Semaphore(self.config.experiment.max_concurrency)

        async def bounded(task: ReplayTaskPlan) -> ReplayTaskExecution:
            async with semaphore:
                return await self._execute_task(task)

        return tuple(await asyncio.gather(*(bounded(task) for task in self.plan.tasks)))

    async def _execute_task(
        self,
        task: ReplayTaskPlan,
    ) -> ReplayTaskExecution:
        started = time.monotonic()
        error = None
        nodes: list[NodeExecution] = []
        try:
            await asyncio.wait_for(
                self._execute_graph(task, nodes),
                self.config.experiment.task_timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        failed_nodes = any(node.status != "success" for node in nodes)
        return ReplayTaskExecution(
            task.task_id,
            task.runtime_session_id,
            "failed" if error or failed_nodes else "completed",
            time.monotonic() - started,
            error,
            tuple(nodes),
        )

    async def _execute_graph(
        self,
        task: ReplayTaskPlan,
        results: list[NodeExecution],
    ) -> None:
        roots = [node for node in task.requests if node.node_type == "request" and node.context_after is None]
        prepared = await asyncio.gather(
            *(self.prompts.build(task, node, None) for node in roots),
            return_exceptions=True,
        )
        root_prompts = dict(zip((node.source_key for node in roots), prepared, strict=True))
        started = time.monotonic()
        completions = {node.source_key: asyncio.get_running_loop().create_future() for node in task.requests}
        running = [
            asyncio.create_task(self._execute_node(task, node, started, root_prompts, completions, results))
            for node in task.requests
        ]
        try:
            await asyncio.gather(*running)
        finally:
            for execution in running:
                if not execution.done():
                    execution.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def _execute_node(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        task_started: float,
        root_prompts: dict[str, SyntheticPrompt | BaseException],
        completions: dict[str, asyncio.Future[_NodeCompletion]],
        results: list[NodeExecution],
    ) -> None:
        """Execute one node and convert ordinary failures into terminal outcomes.

        Every handled failure resolves the node completion future so dependency
        consumers cannot deadlock. Context-dependent descendants may skip, while
        timing-only consumers and unrelated nodes remain runnable.
        """

        try:
            await self._execute_node_inner(task, node, task_started, root_prompts, completions, results)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "skipped_dependency_failed" if isinstance(exc, _DependencyFailed) else "failed"
            if isinstance(exc, _DependencyFailed):
                logger.warning(
                    "Replay node skipped because a context dependency failed: session_id=%s source_key=%s error=%s",
                    task.runtime_session_id,
                    node.source_key,
                    error,
                )
            else:
                logger.exception(
                    "Replay node execution failed: session_id=%s source_key=%s node_type=%s",
                    task.runtime_session_id,
                    node.source_key,
                    node.node_type,
                )
            completion = _NodeCompletion(time.monotonic(), None, status, error)
            if not completions[node.source_key].done():
                completions[node.source_key].set_result(completion)
            results.append(
                NodeExecution(
                    node.source_key,
                    node.runtime_request_id,
                    node.node_type,
                    node.prompt_kind,
                    status,
                    None,
                    None,
                    None,
                    error,
                    getattr(node, "context_mode", "none"),
                    None,
                )
            )

    async def _execute_node_inner(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        task_started: float,
        root_prompts: dict[str, SyntheticPrompt | BaseException],
        completions: dict[str, asyncio.Future[_NodeCompletion]],
        results: list[NodeExecution],
    ) -> None:
        """Prepare, release, and send a node after its required context is ready.

        ``send_after`` supplies an ordering clock even when that predecessor
        failed. Only ``context_after`` requires a successful Prompt exchange.
        The method records successful transport outcomes and lets the wrapper
        convert construction or dependency errors into node results.
        """

        predecessor_future = completions[node.send_after] if node.send_after else None
        prompt = None
        if node.node_type == "request":

            async def prepare_prompt() -> SyntheticPrompt:
                if node.context_after:
                    context_completion = await completions[node.context_after]
                    if context_completion.status != "success" or context_completion.exchange is None:
                        raise _DependencyFailed(f"context request {node.context_after} failed")
                    return await self.prompts.build(task, node, context_completion.exchange)
                root_prompt = root_prompts[node.source_key]
                if isinstance(root_prompt, BaseException):
                    raise RuntimeError(f"root prompt construction failed: {root_prompt}") from root_prompt
                return root_prompt

            if predecessor_future:
                predecessor, prompt = await asyncio.gather(predecessor_future, prepare_prompt())
            else:
                predecessor = None
                prompt = await prepare_prompt()
        else:
            predecessor = await predecessor_future if predecessor_future else None
        release_clock = (predecessor.finished_clock if predecessor else task_started) + node.effective_interval_seconds
        planned_offset = release_clock - task_started

        if node.node_type == "timing_dependency":
            await asyncio.sleep(max(0, release_clock - time.monotonic()))
            await asyncio.sleep(node.effective_duration_seconds or 0)
            completion = _NodeCompletion(time.monotonic(), None)
            completions[node.source_key].set_result(completion)
            results.append(
                NodeExecution(
                    node.source_key,
                    node.runtime_request_id,
                    node.node_type,
                    node.prompt_kind,
                    "success",
                    planned_offset,
                    None,
                    None,
                    None,
                    getattr(node, "context_mode", "none"),
                    None,
                )
            )
            return

        assert prompt is not None
        await asyncio.sleep(max(0, release_clock - time.monotonic()))
        sent_clock = time.monotonic()
        response = await self.transport.send(task, node, prompt)
        exchange = PromptExchange(prompt, response.assistant_content) if response.success else None
        completion_status = "success" if response.success else "failed"
        completions[node.source_key].set_result(
            _NodeCompletion(response.finished_clock, exchange, completion_status, response.error)
        )
        results.append(
            NodeExecution(
                node.source_key,
                node.runtime_request_id,
                node.node_type,
                node.prompt_kind,
                "success" if response.success else "failed",
                planned_offset,
                sent_clock - task_started,
                max(0, sent_clock - release_clock),
                response.error,
                getattr(node, "context_mode", "none"),
                asdict(prompt.calibration) if prompt.calibration is not None else None,
            )
        )
