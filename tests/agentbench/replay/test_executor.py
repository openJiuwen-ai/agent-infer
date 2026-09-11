# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import time
from types import SimpleNamespace

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentbench.replay.prompt import SyntheticPrompt
from agentinfer.agentbench.replay.transport import TransportResult


def _node(
    key: str,
    prompt_kind: str,
    *,
    send_after: str | None = None,
    context_after: str | None = None,
    effective_interval: float = 0,
) -> object:
    return SimpleNamespace(
        source_key=key,
        runtime_request_id=f"runtime-{key}",
        node_type="request",
        prompt_kind=prompt_kind,
        send_after=send_after,
        context_after=context_after,
        effective_interval_seconds=effective_interval,
    )


def test_parallel_lead_roots_start_together_and_only_main_supplies_context() -> None:
    async def execute() -> tuple[object, list[tuple[str, float]], list[tuple[str, str | None]]]:
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_timeout_seconds": 5}, "replay": {"trace_path": "source.jsonl"}}
        )
        title = _node("title", "lead_title")
        main = _node("main", "lead_main")
        continuation = _node("next", "continuation", send_after="title", context_after="main")
        task = SimpleNamespace(
            task_id="task",
            runtime_session_id="session",
            requests=(title, main, continuation),
        )
        plan = SimpleNamespace(tasks=(task,))
        starts: list[tuple[str, float]] = []
        contexts: list[tuple[str, str | None]] = []
        roots_started = asyncio.Event()

        class Prompts:
            async def build(self, task: object, node: object, context: object) -> SyntheticPrompt:
                content = context.assistant_content if context is not None else None
                contexts.append((node.prompt_kind, content))
                return SyntheticPrompt("", (), ({"role": "user", "content": node.prompt_kind},))

        class Transport:
            async def send(self, task: object, node: object, prompt: object) -> TransportResult:
                starts.append((node.prompt_kind, time.monotonic()))
                if len(starts) == 2:
                    roots_started.set()
                if node.prompt_kind != "continuation":
                    await roots_started.wait()
                return TransportResult(True, node.prompt_kind, time.monotonic(), 200, None)

        result = await ReplayExecutor(config, plan, Prompts(), Transport()).execute()  # type: ignore[arg-type]
        return result[0], starts, contexts

    result, starts, contexts = asyncio.run(execute())

    root_starts = [started for kind, started in starts if kind != "continuation"]
    assert max(root_starts) - min(root_starts) < 0.05
    assert ("continuation", "lead_main") in contexts
    assert result.status == "completed"


def test_interval_release_waits_for_predecessor_completion() -> None:
    async def execute() -> dict[str, float]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_timeout_seconds": 5},
                "replay": {"trace_path": "source.jsonl"},
            }
        )
        task = SimpleNamespace(
            task_id="task",
            runtime_session_id="session",
            requests=(
                _node("first", "lead_main"),
                _node("second", "lead_title", send_after="first", effective_interval=0.03),
            ),
        )
        starts: dict[str, float] = {}

        class Prompts:
            async def build(self, task: object, node: object, context: object) -> SyntheticPrompt:
                return SyntheticPrompt("", (), ({"role": "user", "content": node.prompt_kind},))

        class Transport:
            async def send(self, task: object, node: object, prompt: object) -> TransportResult:
                starts[node.source_key] = time.monotonic()
                if node.source_key == "first":
                    await asyncio.sleep(0.04)
                return TransportResult(True, node.prompt_kind, time.monotonic(), 200, None)

        results = await ReplayExecutor(  # type: ignore[arg-type]
            config,
            SimpleNamespace(tasks=(task,)),
            Prompts(),
            Transport(),
        ).execute()
        assert all(result.status == "completed" for result in results)
        return starts

    starts = asyncio.run(execute())

    assert starts["second"] - starts["first"] >= 0.06


def test_node_failure_skips_dependents_without_cancelling_independent_siblings() -> None:
    async def execute() -> tuple[object, list[str]]:
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_timeout_seconds": 5}, "replay": {"trace_path": "source.jsonl"}}
        )
        failing = _node("failing", "lead_main")
        independent = _node("independent", "lead_title", send_after="failing")
        dependent = _node(
            "dependent",
            "continuation",
            send_after="failing",
            context_after="failing",
        )
        task = SimpleNamespace(
            task_id="task",
            runtime_session_id="session",
            requests=(failing, independent, dependent),
        )
        sent: list[str] = []

        class Prompts:
            async def build(self, task: object, node: object, context: object) -> SyntheticPrompt:
                if node.source_key == "failing":
                    raise ValueError("calibration failed")
                return SyntheticPrompt("", (), ({"role": "user", "content": node.prompt_kind},))

        class Transport:
            async def send(self, task: object, node: object, prompt: object) -> TransportResult:
                sent.append(node.source_key)
                return TransportResult(True, node.prompt_kind, time.monotonic(), 200, None)

        results = await ReplayExecutor(  # type: ignore[arg-type]
            config,
            SimpleNamespace(tasks=(task,)),
            Prompts(),
            Transport(),
        ).execute()
        return results[0], sent

    result, sent = asyncio.run(execute())
    statuses = {node.source_key: node.status for node in result.nodes}

    assert result.status == "failed"
    assert result.error is None
    assert statuses == {
        "failing": "failed",
        "independent": "success",
        "dependent": "skipped_dependency_failed",
    }
    assert sent == ["independent"]
    assert len(result.nodes) == 3
    executed = {node.source_key: node for node in result.nodes}
    assert executed["failing"].planned_send_offset_seconds is None
    assert executed["dependent"].planned_send_offset_seconds is None
    assert executed["independent"].planned_send_offset_seconds is not None
