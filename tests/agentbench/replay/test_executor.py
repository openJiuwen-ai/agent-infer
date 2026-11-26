# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import gc
import json
import time
import weakref
from types import SimpleNamespace

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentbench.replay.prompt import PromptBuilder, SyntheticPrompt, TokenizerClient
from agentinfer.agentbench.replay.transport import ReplayTransport, TransportResult
from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


@pytest.mark.parametrize("concurrency", [1, 3])
@pytest.mark.parametrize("fail", [False, True])
def test_root_prompts_and_private_caches_are_limited_to_active_tasks(concurrency, fail) -> None:
    async def exercise():
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 12, "max_concurrency": concurrency},
                "replay": {
                    "prompt_transport": "completion_token_ids",
                    "lead_1st_sys_shared_prefix": 3,
                    "lead_1st_sys_session_prefix": 2,
                },
            }
        )
        tasks = []
        for index in range(12):
            node = _node(f"root-{index}", "lead_main")
            node.__dict__.update(
                prompt_recipe_key="root",
                actor_role="lead",
                context_mode="independent",
                planned_input_tokens=32768,
                planned_output_tokens=1,
                source_output_tokens=1,
            )
            tasks.append(SimpleNamespace(task_id=f"t{index}", runtime_session_id=f"s{index}", requests=(node,)))
        plan = SimpleNamespace(tasks=tuple(tasks))
        refs = []
        peak = 0

        class Prompts(PromptBuilder):
            async def build(self, *args):
                nonlocal peak
                await asyncio.sleep(0.005)
                prompt = await super().build(*args)
                gc.collect()
                refs.append(weakref.ref(prompt))
                peak = max(peak, sum(ref() is not None for ref in refs))
                return prompt

        class Transport:
            async def send(self, task, node, prompt):
                await asyncio.sleep(0.005)
                return TransportResult(
                    not fail,
                    "answer",
                    time.monotonic(),
                    500 if fail else 200,
                    "backend error" if fail else None,
                    input_tokens=len(prompt.token_ids),
                    assistant_token_ids=(42,),
                    output_tokens=1,
                )

        async def respond(request):
            await asyncio.sleep(0)
            return httpx.Response(200, json={"tokens": [10, 20, 30]})

        tokenizer = TokenizerClient(config)
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        prompts = Prompts(config, tokenizer)
        try:
            await prompts.prepare_tasks(plan)
            assert refs == []
            assert all(key.startswith("shared:") for key in tokenizer._token_ids)
            results = await ReplayExecutor(config, plan, prompts, Transport()).execute()
            gc.collect()
            assert len(refs) == 12
            assert 0 < peak <= concurrency
            assert all(ref() is None for ref in refs)
            assert all(key.startswith("shared:") for key in tokenizer._token_ids)
            assert tokenizer._inflight == {}
            assert all(result.status == ("failed" if fail else "completed") for result in results)
            for result in results:
                root = result.nodes[0]
                assert root.prompt_prepare_seconds >= 0.005
                assert result.duration_seconds >= root.prompt_prepare_seconds
                assert root.prompt_prepare_end_offset_seconds <= 0
        finally:
            await tokenizer.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_interrupted_task_releases_private_preparation_before_releasing_its_slot(cancel) -> None:
    async def exercise():
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_num": 1, "task_timeout_seconds": 1}, "replay": {}}
        )
        task = SimpleNamespace(task_id="t", runtime_session_id="s", requests=(_node("root", "lead_main"),))
        entered, stopped = asyncio.Event(), asyncio.Event()
        released = []

        class Prompts:
            async def build(self, *args):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            async def release_task(self, finished):
                assert stopped.is_set()
                released.append(finished.runtime_session_id)

        executor = ReplayExecutor(config, SimpleNamespace(tasks=(task,)), Prompts(), None)
        running = asyncio.create_task(executor.execute())
        await entered.wait()
        if cancel:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            results = await running
            assert results[0].status == "failed"
            assert "TimeoutError" in results[0].error
        assert released == ["s"]

    asyncio.run(asyncio.wait_for(exercise(), 5))


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("release_path", [None, "/programs/release"])
def test_failed_context_then_interruption_leaves_no_dependency_tasks(cancel, release_path) -> None:
    """A failed context must not leave a waiter on a different, interrupted predecessor."""

    async def exercise():
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1, "task_timeout_seconds": 1},
                "replay": {"program_release_path": release_path},
            }
        )
        task = SimpleNamespace(
            task_id="task",
            runtime_session_id="session",
            requests=(
                _node("slow", "lead_main"),
                _node("failed", "lead_title"),
                _node("dependent", "continuation", send_after="slow", context_after="failed"),
            ),
        )
        dependency_finished = asyncio.Event()
        sent, released = [], []

        class Prompts:
            async def build(self, *args):
                return SyntheticPrompt("", (), ())

            async def release_task(self, task):
                released.append(task.runtime_session_id)

        class Transport:
            async def release_program(self, task, actor_id):
                pytest.fail("an interrupted Program must not be released")

            async def send(self, task, node, prompt):
                sent.append(node.source_key)
                if node.source_key == "slow":
                    await asyncio.Event().wait()
                return TransportResult(False, "", time.monotonic(), 500, "injected failure")

        class Executor(ReplayExecutor):
            async def _execute_node(self, task, node, *args):
                await super()._execute_node(task, node, *args)
                if node.source_key == "dependent":
                    dependency_finished.set()

        before = asyncio.all_tasks()
        running = asyncio.create_task(
            Executor(config, SimpleNamespace(tasks=(task,)), Prompts(), Transport()).execute()
        )
        try:
            await dependency_finished.wait()
            if cancel:
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            else:
                result = (await running)[0]
                assert result.status == "failed" and "TimeoutError" in result.error
                statuses = {node.source_key: node.status for node in result.nodes}
                assert statuses["dependent"] == "skipped_dependency_failed"
            assert set(sent) == {"slow", "failed"}
            assert released == ["session"]
            assert not asyncio.all_tasks() - before
        finally:
            remaining = asyncio.all_tasks() - before
            for pending in remaining:
                pending.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)

    asyncio.run(asyncio.wait_for(exercise(), 5))


@pytest.mark.parametrize("prepare_seconds", [0.005, 0.06])
def test_preparation_excludes_dependency_wait_and_keeps_original_release(prepare_seconds):
    async def execute():
        config = ReplayBenchConfig.model_validate({"experiment": {"task_num": 1}, "replay": {}})
        root = _node("root", "lead_main")
        next_node = _node("next", "continuation", send_after="root", context_after="root", effective_interval=0.03)
        task = SimpleNamespace(task_id="task", runtime_session_id="s", requests=(root, next_node))
        root_finished = 0

        class Prompts:
            async def build(self, task, node, context):
                if context is not None:
                    assert root_finished > 0
                    await asyncio.sleep(prepare_seconds)
                return SyntheticPrompt("", (), ())

        class Transport:
            async def send(self, task, node, prompt):
                nonlocal root_finished
                if node.source_key == "root":
                    await asyncio.sleep(0.1)
                    root_finished = time.monotonic()
                return TransportResult(True, "answer", time.monotonic(), 200, None)

        result = (await ReplayExecutor(config, SimpleNamespace(tasks=(task,)), Prompts(), Transport()).execute())[0]
        nodes = {node.source_key: node for node in result.nodes}
        continuation = nodes["next"]
        assert continuation.prompt_prepare_seconds >= prepare_seconds
        assert continuation.prompt_prepare_start_offset_seconds > 0.09
        assert continuation.planned_send_offset_seconds >= 0.13
        assert continuation.actual_send_offset_seconds >= continuation.planned_send_offset_seconds
        assert continuation.prompt_prepare_end_offset_seconds <= continuation.actual_send_offset_seconds
        if prepare_seconds > 0.03:
            assert continuation.scheduler_lag_seconds >= prepare_seconds - 0.03
        else:
            assert continuation.prompt_prepare_end_offset_seconds < continuation.planned_send_offset_seconds

    asyncio.run(execute())


def _node(
    key: str,
    prompt_kind: str,
    *,
    send_after: str | None = None,
    context_after: str | None = None,
    effective_interval: float = 0,
    actor_id: str = "lead",
) -> object:
    return SimpleNamespace(
        source_key=key,
        runtime_request_id=f"runtime-{key}",
        node_type="request",
        prompt_kind=prompt_kind,
        actor_id=actor_id,
        send_after=send_after,
        context_after=context_after,
        effective_interval_seconds=effective_interval,
    )


@pytest.mark.parametrize("release_status", [None, 204, 404, 200])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
def test_program_release_follows_actor_completion(tmp_path, monkeypatch, release_status, endpoint) -> None:
    async def exercise():
        monkeypatch.setenv("REPLAY_TEST_KEY", "test-token")
        token_ids = endpoint == "/v1/completions"
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend", "endpoint": endpoint, "api_key_env": "REPLAY_TEST_KEY"},
                "replay": {
                    "trace_type": "agentX" if token_ids else "agentinfer",
                    "program_release_path": "/control/release" if release_status else None,
                },
            }
        )
        child_first = _node("child-first", "subagent_first", actor_id="child")
        child_last = _node("child-last", "continuation", actor_id="child")
        lead = _node("lead-tail", "continuation", send_after="child-last")
        task = SimpleNamespace(task_id="task", runtime_session_id="session", requests=(child_first, child_last, lead))
        for node in task.requests:
            child = node.actor_id == "child"
            node.__dict__.update(
                actor_role="subagent" if child else "lead",
                parent_actor_id="lead" if child else None,
                planned_input_tokens=1,
                planned_output_tokens=1,
                backend_sampling_seed=7,
            )

        class Prompts:
            async def build(self, task, node, context):
                if token_ids:
                    return SyntheticPrompt("", (), (), token_ids=(task.requests.index(node),))
                return SyntheticPrompt("", (), ({"role": "user", "content": node.source_key},))

        allow_child_first_to_finish = asyncio.Event()
        child_first_finished = asyncio.Event()
        child_released = asyncio.Event()
        releases = []

        async def handle(request):
            body = json.loads(request.content)
            assert request.headers["authorization"] == "Bearer test-token"
            if request.url.path == "/control/release":
                assert request.method == "POST"
                assert set(body) == {"program_id"}
                releases.append(body["program_id"])
                if body["program_id"] == "session:child":
                    assert child_first_finished.is_set()
                    child_released.set()
                    if release_status == 200:
                        return httpx.Response(200, json={"released": False})
                    return httpx.Response(release_status)
                return httpx.Response(204)

            assert request.url.path == endpoint
            key = task.requests[body["prompt"][0]].source_key if token_ids else body["messages"][0]["content"]
            actor_id = "child" if key.startswith("child-") else "lead"
            identity = json.loads(body["vllm_xargs"]["agentic_context"])
            assert identity["program_id"] == f"session:{actor_id}"
            if release_status:
                assert body["program_id"] == identity["program_id"]
            else:
                assert "program_id" not in body
            if key == "child-first":
                await allow_child_first_to_finish.wait()
                child_first_finished.set()
            if key == "lead-tail":
                assert not child_released.is_set()
                allow_child_first_to_finish.set()
                if release_status:
                    await child_released.wait()
            return httpx.Response(
                200,
                content='data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n',
            )

        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.close()
        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            (result,) = await ReplayExecutor(config, SimpleNamespace(tasks=(task,)), Prompts(), transport).execute()
        finally:
            await transport.close()
            await writer.close()

        assert releases == (["session:child", "session:lead"] if release_status else [])
        assert len(result.nodes) == 3
        assert all(node.status == "success" for node in result.nodes)
        if release_status in {404, 200}:
            assert result.status == "failed"
            assert "Program release failed for session:child" in result.error
        else:
            assert result.status == "completed"
            assert result.error is None

    asyncio.run(asyncio.wait_for(exercise(), 5))


def test_failed_terminal_request_does_not_release_program() -> None:
    async def exercise():
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_num": 1}, "replay": {"program_release_path": "/programs/release"}}
        )
        first = _node("first", "lead_main")
        last = _node("last", "continuation", send_after="first")
        task = SimpleNamespace(task_id="task", runtime_session_id="session", requests=(first, last))

        class Prompts:
            async def build(self, task, node, context):
                return SyntheticPrompt("", (), ())

        class Transport:
            async def send(self, task, node, prompt):
                success = node is first
                return TransportResult(success, "", time.monotonic(), 200 if success else 500, None)

            async def release_program(self, task, actor_id):
                pytest.fail("a failed terminal request must not trigger Program release")

        (result,) = await ReplayExecutor(config, SimpleNamespace(tasks=(task,)), Prompts(), Transport()).execute()
        assert result.status == "failed"
        assert [node.status for node in result.nodes] == ["success", "failed"]

    asyncio.run(exercise())


def test_parallel_lead_roots_start_together_and_only_main_supplies_context() -> None:
    async def execute() -> tuple[object, list[tuple[str, float]], list[tuple[str, str | None]]]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1, "task_timeout_seconds": 5},
                "replay": {"trace_path": "source.jsonl"},
            }
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
                "experiment": {"task_num": 1, "task_timeout_seconds": 5},
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
            {
                "experiment": {"task_num": 1, "task_timeout_seconds": 5},
                "replay": {"trace_path": "source.jsonl"},
            }
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
