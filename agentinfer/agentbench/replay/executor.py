# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Execute structural Replay dependency graphs with node-level isolation.

This module owns release timing, dependency readiness, and execution outcomes;
Prompt construction and HTTP transport remain delegated to their owners.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass

from .config import ReplayBenchConfig
from .observation import RealClock, ReplayClock, ReplayObserver, emit
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
    prompt_prepare_start_offset_seconds: float | None = None
    prompt_prepare_end_offset_seconds: float | None = None
    prompt_prepare_seconds: float | None = None
    prompt_prepare_mode: str | None = None


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
        *,
        clock: ReplayClock | None = None,
        observer: ReplayObserver | None = None,
    ) -> None:
        self.clock = clock or RealClock()
        self.observer = observer
        self.config = config
        self.plan = plan
        self.prompts = prompts
        self.transport = transport
        self._preparation: dict[str, tuple[float, float]] = {}

    async def _prepare(
        self, task: ReplayTaskPlan, node: ReplayPlanNode, context: PromptExchange | None
    ) -> SyntheticPrompt:
        """Build one prompt and record preparation time even if construction fails."""
        started = self.clock.monotonic()
        emit(self.observer, self.clock, "prompt_prepare_started", task, node, at=started)
        try:
            return await self.prompts.build(task, node, context)
        finally:
            finished = self.clock.monotonic()
            self._preparation[node.runtime_request_id] = (started, finished)
            emit(self.observer, self.clock, "prompt_prepare_finished", task, node, at=finished)

    def _preparation_metadata(self, node, task_started: float) -> dict[str, object]:
        times = self._preparation.get(node.runtime_request_id)
        if times is None:
            return {}
        tokenizer = getattr(self.prompts, "tokenizer", None)
        mode = (
            "completion_token_ids"
            if self.config.replay.uses_token_ids
            else ("chat_text_local_full" if getattr(tokenizer, "local_tokenizer", None) else "chat_text_backend_full")
        )
        return {
            "prompt_prepare_start_offset_seconds": times[0] - task_started,
            "prompt_prepare_end_offset_seconds": times[1] - task_started,
            "prompt_prepare_seconds": times[1] - times[0],
            "prompt_prepare_mode": mode,
        }

    @property
    def _uses_agentx_snapshots(self) -> bool:
        return self.config.replay.prompt_shape == "agentX_snapshot"

    async def execute(self) -> tuple[ReplayTaskExecution, ...]:
        """Run every planned Session and return its finalized execution record."""

        semaphore = asyncio.Semaphore(self.config.experiment.max_concurrency)

        async def bounded(task: ReplayTaskPlan) -> ReplayTaskExecution:
            emit(self.observer, self.clock, "task_submitted", task)
            async with semaphore:
                return await self._execute_task(task)

        return tuple(await asyncio.gather(*(bounded(task) for task in self.plan.tasks)))

    async def _execute_task(
        self,
        task: ReplayTaskPlan,
    ) -> ReplayTaskExecution:
        started = self.clock.monotonic()
        emit(self.observer, self.clock, "task_admitted", task, at=started)
        error = None
        nodes: list[NodeExecution] = []
        try:
            if self.config.experiment.task_timeout_seconds is None:
                await self._execute_graph(task, nodes)
            else:
                await asyncio.wait_for(
                    self._execute_graph(task, nodes),
                    self.config.experiment.task_timeout_seconds,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            release_task = getattr(self.prompts, "release_task", None)
            if release_task is not None:
                await release_task(task)
            for node in task.requests:
                self._preparation.pop(node.runtime_request_id, None)
        failed_nodes = any(node.status != "success" for node in nodes)
        finished = self.clock.monotonic()
        emit(self.observer, self.clock, "task_finished", task, at=finished)
        return ReplayTaskExecution(
            task.task_id,
            task.runtime_session_id,
            "failed" if error or failed_nodes else "completed",
            finished - started,
            error,
            tuple(nodes),
        )

    async def _execute_graph(
        self,
        task: ReplayTaskPlan,
        results: list[NodeExecution],
    ) -> None:
        roots = (
            []
            if self._uses_agentx_snapshots
            else [node for node in task.requests if node.node_type == "request" and node.context_after is None]
        )
        prepared = await asyncio.gather(
            *(self._prepare(task, node, None) for node in roots),
            return_exceptions=True,
        )
        root_prompts = dict(zip((node.source_key for node in roots), prepared, strict=True))
        started = self.clock.monotonic()
        emit(self.observer, self.clock, "graph_started", task, at=started)
        completions = {node.source_key: asyncio.get_running_loop().create_future() for node in task.requests}
        inflight = asyncio.Semaphore(self.config.replay.max_inflight_requests)
        running = [
            asyncio.create_task(self._execute_node(task, node, started, root_prompts, completions, results, inflight))
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
        inflight: asyncio.Semaphore,
    ) -> None:
        """Execute one node and convert ordinary failures into terminal outcomes.

        Every handled failure resolves the node completion future so dependency
        consumers cannot deadlock. Context-dependent descendants may skip, while
        timing-only consumers and unrelated nodes remain runnable.
        """

        try:
            await self._execute_node_inner(task, node, task_started, root_prompts, completions, results, inflight)
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
            completion = _NodeCompletion(self.clock.monotonic(), None, status, error)
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
                    **self._preparation_metadata(node, task_started),
                )
            )

        finally:
            future = completions[node.source_key]
            if future.done() and not future.cancelled():
                completion = future.result()
                emit(
                    self.observer,
                    self.clock,
                    "node_finished",
                    task,
                    node,
                    at=completion.finished_clock,
                    status=completion.status,
                )

    async def _execute_node_inner(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        task_started: float,
        root_prompts: dict[str, SyntheticPrompt | BaseException],
        completions: dict[str, asyncio.Future[_NodeCompletion]],
        results: list[NodeExecution],
        inflight: asyncio.Semaphore,
    ) -> None:
        """Prepare, release, and send a node after its required context is ready.

        ``send_after`` supplies an ordering clock even when that predecessor
        failed. Only ``context_after`` requires a successful Prompt exchange.
        The method records successful transport outcomes and lets the wrapper
        convert construction or dependency errors into node results.
        """

        predecessor_future = completions[node.send_after] if node.send_after else None
        if predecessor_future is not None:
            emit(self.observer, self.clock, "dependency_wait_started", task, node, kind="send_after")
        prompt = None
        agentx = self._uses_agentx_snapshots
        if node.node_type == "request" and not agentx:

            async def prepare_prompt() -> SyntheticPrompt:
                if node.context_after:
                    emit(self.observer, self.clock, "dependency_wait_started", task, node, kind="context_after")
                    context_completion = await completions[node.context_after]
                    if context_completion.status != "success" or context_completion.exchange is None:
                        raise _DependencyFailed(f"context request {node.context_after} failed")
                    return await self._prepare(task, node, context_completion.exchange)
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
        emit(self.observer, self.clock, "release_scheduled", task, node, release_clock=release_clock)
        planned_offset = release_clock - task_started

        if node.node_type == "timing_dependency":
            await self.clock.sleep(max(0, release_clock - self.clock.monotonic()))
            await self.clock.sleep(node.effective_duration_seconds or 0)
            completion = _NodeCompletion(self.clock.monotonic(), None)
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

        await self.clock.sleep(max(0, release_clock - self.clock.monotonic()))
        if agentx:
            async with inflight:
                # Bound prepared snapshots; semaphore wait and render time count as scheduler lag.
                prompt = await self._prepare(task, node, None)
                sent_clock = self.clock.monotonic()
                emit(self.observer, self.clock, "transport_entered", task, node, at=sent_clock)
                response = await self.transport.send(task, node, prompt)
        else:
            assert prompt is not None
            sent_clock = self.clock.monotonic()
            emit(self.observer, self.clock, "transport_entered", task, node, at=sent_clock)
            response = await self.transport.send(task, node, prompt)
        exchange = (
            PromptExchange(
                prompt,
                response.assistant_content,
                response.assistant_reasoning_content,
                response.assistant_token_ids,
            )
            if response.success and not agentx
            else None
        )
        completion_status = "success" if response.success else "failed"
        completions[node.source_key].set_result(
            _NodeCompletion(response.finished_clock, exchange, completion_status, response.error)
        )
        calibration = asdict(prompt.calibration) if prompt.calibration is not None else None
        if calibration is not None:
            actual = response.input_tokens
            calibration["backend_input_tokens"] = actual
            calibration["backend_input_residual_tokens"] = (
                actual - node.planned_input_tokens
                if actual is not None and node.planned_input_tokens is not None
                else None
            )
            calibration["source_output_tokens"] = node.source_output_tokens
            calibration["effective_output_tokens"] = node.planned_output_tokens
            calibration["backend_output_tokens"] = response.output_tokens
            calibration["backend_target_met"] = (
                actual == node.planned_input_tokens and response.output_tokens == node.planned_output_tokens
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
                calibration,
                **self._preparation_metadata(node, task_started),
            )
        )
