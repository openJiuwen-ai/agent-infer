# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.prompt import PromptBuilder, PromptExchange, SyntheticPrompt, TokenizerClient


class _Tokenizer:
    async def token_text(self, namespace: str, count: int) -> str:
        return f"[{namespace}:{count}]" if count else ""

    async def count(self, prompt: object) -> int:
        return 100


def _config() -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "backend": {"endpoint": "/v1/messages"},
            "replay": {
                "trace_path": "source.jsonl",
                "lead_title_sys_shared_prefix": 11,
                "lead_name_sys_shared_prefix": 7,
                "lead_1st_tool_shared_prefix": 22,
                "lead_1st_msg_shared_prefix": 33,
            },
        }
    )


def test_tokenizer_client_preserves_default_keepalive() -> None:
    with patch("agentinfer.agentbench.replay.prompt.httpx.AsyncClient", wraps=httpx.AsyncClient) as client_class:
        tokenizer = TokenizerClient(_config())

    try:
        limits = client_class.call_args.kwargs["limits"]
        assert limits.max_connections is None
        defaults = httpx.Limits()
        assert limits.max_keepalive_connections == defaults.max_keepalive_connections
        assert limits.keepalive_expiry == defaults.keepalive_expiry
    finally:
        asyncio.run(tokenizer.close())


def test_tokenizer_count_retries_transient_transport_error() -> None:
    attempts = 0
    sleep = AsyncMock()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url.path == "/v1/messages/count_tokens"
        if attempts == 1:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(200, json={"input_tokens": 123})

    async def count() -> int:
        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with patch("agentinfer.agentbench.replay.prompt.asyncio.sleep", sleep):
                return await tokenizer.count(SyntheticPrompt("", (), ()))
        finally:
            await tokenizer.close()

    assert asyncio.run(count()) == 123
    assert attempts == 2
    sleep.assert_awaited_once_with(0.1)


def test_tokenizer_count_exhausts_transport_retries() -> None:
    attempts = 0
    sleep = AsyncMock()

    async def fail(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("connection reset", request=request)

    async def count() -> None:
        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        try:
            with (
                patch("agentinfer.agentbench.replay.prompt.asyncio.sleep", sleep),
                pytest.raises(httpx.ReadError, match="connection reset"),
            ):
                await tokenizer.count(SyntheticPrompt("", (), ()))
        finally:
            await tokenizer.close()

    asyncio.run(count())
    assert attempts == 3
    assert [call.args for call in sleep.await_args_list] == [(0.1,), (0.2,)]


def test_tokenizer_count_does_not_retry_http_status_error() -> None:
    attempts = 0
    sleep = AsyncMock()

    async def reject(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request, json={"error": "invalid request"})

    async def count() -> None:
        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        try:
            with (
                patch("agentinfer.agentbench.replay.prompt.asyncio.sleep", sleep),
                pytest.raises(httpx.HTTPStatusError),
            ):
                await tokenizer.count(SyntheticPrompt("", (), ()))
        finally:
            await tokenizer.close()

    asyncio.run(count())
    assert attempts == 1
    sleep.assert_not_awaited()


def test_tokenizer_count_forwards_chat_template_kwargs() -> None:
    captured: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"count": 12})

    async def count() -> int:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {
                    "endpoint": "/v1/chat/completions",
                    "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True},
                },
                "replay": {"trace_path": "source.jsonl"},
            }
        )
        tokenizer = TokenizerClient(config)
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            return await tokenizer.count(SyntheticPrompt("", (), ({"role": "user", "content": "hello"},)))
        finally:
            await tokenizer.close()

    assert asyncio.run(count()) == 12
    assert captured["chat_template_kwargs"] == {"enable_thinking": False, "clear_thinking": True}


def test_tokenizer_client_reuses_validated_local_counter() -> None:
    calls: list[list[dict[str, object]]] = []

    def chat_token_ids(messages: list[dict[str, object]], *, tools=None) -> tuple[int, ...]:
        calls.append(messages)
        return tuple(range(42))

    local = SimpleNamespace(
        chat_token_ids=chat_token_ids,
        text_token_ids=lambda text, *, cache=True: (1, 2),
        detokenize_tokens=lambda tokens: "decoded",
    )
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "backend": {"endpoint": "/v1/chat/completions", "model": "local-model"},
            "replay": {"trace_path": "source.jsonl"},
        }
    )

    async def exercise() -> tuple[int, tuple[int, ...], str]:
        tokenizer = TokenizerClient(config, local)  # type: ignore[arg-type]
        try:
            count = await tokenizer.count(SyntheticPrompt("", (), ({"role": "user", "content": "hello"},)))
            token_ids = await tokenizer.text_token_ids("text")
            decoded = await tokenizer.detokenize_tokens([1, 2])
            return count, token_ids, decoded
        finally:
            await tokenizer.close()

    assert asyncio.run(exercise()) == (42, (1, 2), "decoded")
    assert calls == [[{"role": "user", "content": "hello"}]]


@pytest.mark.parametrize("cancel", [False, True])
def test_local_tokenizer_probe_waits_for_slow_backend_and_can_cancel(monkeypatch, cancel) -> None:
    def discover(config, *, get_json, backend_tokens):
        assert get_json("http://backend/tokenizer_info") is None
        return None

    monkeypatch.setattr("agentinfer.agentbench.replay.prompt.discover_local_tokenizer", discover)

    async def exercise():
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def respond(request):
            entered.set()
            try:
                await release.wait()
                return httpx.Response(404)
            finally:
                stopped.set()

        config = ReplayBenchConfig.model_validate({"experiment": {"task_num": 1}, "replay": {}})
        tokenizer = TokenizerClient(config)
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        probe = asyncio.create_task(tokenizer.prepare_local_tokenizer())
        try:
            await entered.wait()
            await asyncio.sleep(0.2)  # Exceed the thread bridge's 100 ms polling interval.
            assert not probe.done()
            if cancel:
                probe.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await probe
                await stopped.wait()
            else:
                release.set()
                await probe
                assert tokenizer.local_tokenizer is None
        finally:
            release.set()
            probe.cancel()
            await asyncio.gather(probe, return_exceptions=True)
            await tokenizer.close()

    asyncio.run(asyncio.wait_for(exercise(), 5))


def test_tokenizer_concurrent_text_misses_share_requests_and_survive_waiter_cancellation() -> None:
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = Counter()

        async def respond(request):
            calls[request.url.path] += 1
            if request.url.path == "/tokenize":
                entered.set()
                await release.wait()
                return httpx.Response(200, json={"tokens": [10, 20]})
            return httpx.Response(200, json={"prompt": "shared text"})

        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        waiters = [asyncio.create_task(tokenizer.token_text("shared:prefix", 8)) for _ in range(8)]
        try:
            await entered.wait()
            waiters[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiters[0]
            release.set()
            assert await asyncio.gather(*waiters[1:]) == ["shared text"] * 7
            assert await tokenizer.token_text("shared:prefix", 8) == "shared text"
            assert calls == {"/tokenize": 1, "/detokenize": 1}
        finally:
            release.set()
            await tokenizer.close()
            await asyncio.gather(*waiters, return_exceptions=True)

    asyncio.run(asyncio.wait_for(exercise(), 5))


def test_tokenizer_failed_shared_request_can_be_retried() -> None:
    async def exercise():
        calls = 0

        async def respond(request):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            if calls == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"tokens": [11, 22]})

        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            results = await asyncio.gather(
                *(tokenizer.token_sequence("shared:prefix", 4) for _ in range(4)), return_exceptions=True
            )
            assert all(isinstance(result, httpx.HTTPStatusError) for result in results)
            assert calls == 1
            assert await tokenizer.token_sequence("shared:prefix", 4) == (11, 22, 11, 22)
            assert calls == 2
        finally:
            await tokenizer.close()

    asyncio.run(exercise())


def test_tokenizer_limit_covers_tokenize_detokenize_and_count() -> None:
    async def exercise():
        active = peak = 0
        paths = set()

        async def respond(request):
            nonlocal active, peak
            paths.add(request.url.path)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json={"tokens": [1, 2], "prompt": "xx", "input_tokens": 2})

        config = _config()
        config.replay.tokenizer_max_concurrency = 2
        tokenizer = TokenizerClient(config)
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            await asyncio.gather(
                *(tokenizer.token_text(f"private:{index}", 2) for index in range(8)),
                *(tokenizer.count(SyntheticPrompt("", (), ())) for _ in range(8)),
            )
            assert peak == 2
            assert paths == {"/tokenize", "/detokenize", "/v1/messages/count_tokens"}
        finally:
            await tokenizer.close()

    asyncio.run(exercise())


def test_task_cleanup_cancels_private_work_and_keeps_other_sessions_and_shared_cache() -> None:
    async def exercise():
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = Counter()

        async def respond(request):
            calls[request.url.path] += 1
            body = json.loads(request.content)
            if request.url.path == "/detokenize" and len(body["tokens"]) == 99:
                entered.set()
                try:
                    await release.wait()
                finally:
                    stopped.set()
            return httpx.Response(200, json={"tokens": [1, 2], "prompt": "text"})

        tokenizer = TokenizerClient(_config())
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        task = SimpleNamespace(runtime_session_id="s1", requests=(SimpleNamespace(runtime_request_id="r1"),))
        private_names = ("s1:root", "session:s1:subagents:system-private", "trace-record:r1:repair:0")
        retained_names = ("s10:root", "shared:prefix")
        try:
            for namespace in (*private_names, *retained_names):
                await tokenizer.token_text(namespace, 2)
            private = asyncio.create_task(tokenizer.token_text("s1:root", 99))
            await entered.wait()
            await tokenizer.release_task(task)
            with pytest.raises(asyncio.CancelledError):
                await private
            assert stopped.is_set()
            before = calls.copy()
            for namespace in retained_names:
                assert await tokenizer.token_text(namespace, 2) == "text"
            assert calls == before
            for namespace in private_names:
                assert await tokenizer.token_text(namespace, 2) == "text"
            assert calls["/tokenize"] == before["/tokenize"] + 3
            assert calls["/detokenize"] == before["/detokenize"] + 3
        finally:
            release.set()
            await tokenizer.close()

    asyncio.run(asyncio.wait_for(exercise(), 5))


def test_lead_title_name_and_main_build_distinct_prompt_shapes() -> None:
    async def build() -> tuple[object, object, object]:
        builder = PromptBuilder(_config(), _Tokenizer())  # type: ignore[arg-type]
        task = SimpleNamespace(runtime_session_id="session")
        title_node = SimpleNamespace(
            prompt_kind="lead_title",
            prompt_recipe_key="title",
            planned_input_tokens=100,
        )
        name_node = SimpleNamespace(
            prompt_kind="lead_name",
            prompt_recipe_key="name",
            planned_input_tokens=100,
        )
        main_node = SimpleNamespace(
            prompt_kind="lead_main",
            prompt_recipe_key="main",
            planned_input_tokens=100,
        )
        return (
            await builder.build(task, title_node, None),  # type: ignore[arg-type]
            await builder.build(task, name_node, None),  # type: ignore[arg-type]
            await builder.build(task, main_node, None),  # type: ignore[arg-type]
        )

    title, name, main = asyncio.run(build())

    assert title.system == ({"type": "text", "text": "[shared:lead-title-system:11]"},)
    assert title.tools == ()
    assert "session:title" in str(title.messages[0]["content"])
    assert name.system == ({"type": "text", "text": "[shared:lead-name-system:7]"},)
    assert name.tools == ()
    assert "session:name" in str(name.messages[0]["content"])
    assert "shared:lead-title-system" not in main.system
    assert "shared:lead-name-system" not in main.system
    assert "shared:lead:tool:22" in str(main.tools[0]["function"]["description"])
    assert "shared:lead:message:33" in str(main.messages[0]["content"])


def test_minimal_claude_shape_preserves_context_without_cache_control() -> None:
    async def build() -> tuple[SyntheticPrompt, SyntheticPrompt]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {
                    "trace_path": "source.jsonl",
                    "lead_1st_sys_shared_prefix": 2,
                    "lead_1st_sys_session_prefix": 3,
                    "lead_1st_tool_shared_prefix": 4,
                    "lead_1st_msg_shared_prefix": 5,
                    "lead_1st_trailing_system_prefix": 6,
                    "lead_continuation_extra_system_ratio": 1,
                    "lead_continuation_system_tokens": 7,
                },
            }
        )
        builder = PromptBuilder(config, _Tokenizer())  # type: ignore[arg-type]
        task = SimpleNamespace(runtime_session_id="session")
        root_node = SimpleNamespace(
            actor_id="lead",
            actor_role="lead",
            source_key="root",
            prompt_kind="lead_main",
            prompt_recipe_key="root",
            planned_input_tokens=100,
            context_mode="independent",
        )
        root = await builder.build(task, root_node, None)  # type: ignore[arg-type]
        next_node = SimpleNamespace(
            actor_id="lead",
            actor_role="lead",
            source_key="next",
            prompt_kind="continuation",
            prompt_recipe_key="next",
            planned_input_tokens=100,
            context_mode="append",
        )
        continuation = await builder.build(  # type: ignore[arg-type]
            task,
            next_node,
            PromptExchange(root, "assistant"),
        )
        return root, continuation

    root, continuation = asyncio.run(build())

    assert isinstance(root.system, tuple) and len(root.system) == 3
    for prompt in (root, continuation):
        assert '"cache_control"' not in json.dumps(prompt.anthropic_payload("model"))
        assert '"cache_control"' not in json.dumps(prompt.tokenizer_payload("model"))
    assert root.tools[0]["function"]["name"] == "replay_lead_tool"
    assert isinstance(root.messages[0]["content"], list)
    assert root.messages[-1]["role"] == "system"
    assert root.extra_body is not None and root.extra_body["thinking"] == {"type": "adaptive"}
    assert continuation.messages[: len(root.messages)] == root.messages
    latest_user = continuation.messages[-2]["content"]
    assert isinstance(latest_user, list) and latest_user[0]["type"] == "text"
    assert continuation.messages[-1]["role"] == "system"


class _CharacterTokenizer:
    async def token_text(self, namespace: str, count: int) -> str:
        return "x" * count

    async def count(self, prompt: SyntheticPrompt) -> int:
        total = 0
        for message in prompt.messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(str(block.get("text", ""))) for block in content if isinstance(block, dict))
        return total

    async def text_token_ids(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _BlockTextTokenizer:
    async def token_text(self, namespace: str, count: int) -> str:
        return "x" * count

    async def count(self, prompt: SyntheticPrompt) -> int:
        total = 0
        for message in prompt.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(
                    len(str(block.get("text", "")))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        return total


class _BoundaryMergeTokenizer:
    """Model a suffix whose repeated filler always leaves one missing token."""

    def __init__(self, target: int, *, repairable: bool, residual: int = 1) -> None:
        self.target = target
        self.repairable = repairable
        self.residual = residual

    async def token_text(self, namespace: str, count: int) -> str:
        if self.repairable and namespace.endswith(":repair:1"):
            return "y" * count
        return "x" * count

    async def count(self, prompt: SyntheticPrompt) -> int:
        content = str(prompt.messages[0]["content"])
        if len(content) == 1:
            return 1
        if set(content[1:]) <= {"x"}:
            return min(len(content) - 1, self.target - self.residual)
        return self.target


class _OvershootTokenizer:
    """Model filler that overshoots before deterministic suffix repair."""

    def __init__(self, target: int, *, repairable: bool) -> None:
        self.target = target
        self.repairable = repairable

    async def token_text(self, namespace: str, count: int) -> str:
        if self.repairable and namespace.endswith(":repair:1"):
            return "y" * count
        return "x" * count

    async def count(self, prompt: SyntheticPrompt) -> int:
        content = str(prompt.messages[0]["content"])
        if content == "a":
            return self.target - 2
        if "y" in content[1:]:
            return self.target
        return self.target + 1


def _calibration_config(*, tolerance: int = 1) -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "replay": {
                "trace_path": "source.jsonl",
                "prompt_calibration_tolerance_tokens": tolerance,
            },
        }
    )


@pytest.mark.parametrize("target", [66_556, 59_580, 12_817])
def test_prompt_calibration_repairs_boundary_merge_fixed_point(target: int) -> None:
    async def calibrate() -> SyntheticPrompt:
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(),
            _BoundaryMergeTokenizer(target, repairable=True),
        )
        return await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    prompt = asyncio.run(calibrate())

    assert prompt.calibration is not None
    assert prompt.calibration.final_tokens == target
    assert prompt.calibration.target_met is True
    assert prompt.calibration.accepted_with_tolerance is False
    assert prompt.calibration.repair_attempts == 2
    assert prompt.calibration.count_history[-3:] == (target - 1, target - 1, target)
    assert prompt.calibration.requested_filler_tokens == target + 1
    assert prompt.calibration.actual_prompt_token_gain == target - 1


def test_prompt_calibration_accepts_audited_one_token_residual() -> None:
    async def calibrate() -> SyntheticPrompt:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(),
            _BoundaryMergeTokenizer(target, repairable=False),
        )
        return await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    prompt = asyncio.run(calibrate())

    assert prompt.calibration is not None
    assert prompt.calibration.final_tokens == 99
    assert prompt.calibration.residual_tokens == -1
    assert prompt.calibration.target_met is False
    assert prompt.calibration.accepted_with_tolerance is True
    assert prompt.calibration.repair_attempts == 32


def test_prompt_calibration_strict_tolerance_rejects_fixed_residual() -> None:
    async def calibrate() -> None:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(tolerance=0),
            _BoundaryMergeTokenizer(target, repairable=False),
        )
        await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    with pytest.raises(ValueError, match="Prompt calibration produced 99 tokens for target 100"):
        asyncio.run(calibrate())


def test_prompt_calibration_rejects_residual_above_nonzero_tolerance() -> None:
    async def calibrate() -> None:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(tolerance=1),
            _BoundaryMergeTokenizer(target, repairable=False, residual=2),
        )
        await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    with pytest.raises(ValueError, match="Prompt calibration produced 98 tokens for target 100"):
        asyncio.run(calibrate())


def test_prompt_calibration_repairs_overshoot_before_tolerance() -> None:
    async def calibrate() -> SyntheticPrompt:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(),
            _OvershootTokenizer(target, repairable=True),
        )
        return await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    prompt = asyncio.run(calibrate())

    assert prompt.calibration is not None
    assert prompt.calibration.final_tokens == 100
    assert prompt.calibration.residual_tokens == 0
    assert prompt.calibration.target_met is True
    assert prompt.calibration.accepted_with_tolerance is False
    assert prompt.calibration.repair_attempts == 2
    assert prompt.calibration.requested_filler_tokens == 2


def test_prompt_calibration_accepts_audited_positive_residual_after_overshoot_repair() -> None:
    async def calibrate() -> SyntheticPrompt:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(),
            _OvershootTokenizer(target, repairable=False),
        )
        return await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    prompt = asyncio.run(calibrate())

    assert prompt.calibration is not None
    assert prompt.calibration.final_tokens == 101
    assert prompt.calibration.residual_tokens == 1
    assert prompt.calibration.target_met is False
    assert prompt.calibration.accepted_with_tolerance is True
    assert prompt.calibration.repair_attempts == 32
    assert prompt.calibration.requested_filler_tokens == 2


def test_prompt_calibration_strict_tolerance_rejects_positive_residual_after_repair() -> None:
    async def calibrate() -> None:
        target = 100
        builder = PromptBuilder(  # type: ignore[arg-type]
            _calibration_config(tolerance=0),
            _OvershootTokenizer(target, repairable=False),
        )
        await builder._calibrate(  # noqa: SLF001
            SyntheticPrompt("", (), ({"role": "user", "content": "a"},)),
            "session:request",
            target,
            context_mode="append",
        )

    with pytest.raises(ValueError, match="Prompt calibration produced 101 tokens for target 100"):
        asyncio.run(calibrate())


def test_claude_shape_calibration_changes_only_private_remainder() -> None:
    async def build() -> SyntheticPrompt:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {
                    "trace_path": "source.jsonl",
                    "lead_1st_msg_shared_prefix": 5,
                    "lead_1st_trailing_system_prefix": 6,
                },
            }
        )
        node = SimpleNamespace(
            actor_id="lead",
            actor_role="lead",
            source_key="root",
            prompt_kind="lead_main",
            prompt_recipe_key="root",
            planned_input_tokens=20,
            context_mode="independent",
        )
        return await PromptBuilder(config, _BlockTextTokenizer()).build(  # type: ignore[arg-type]
            SimpleNamespace(runtime_session_id="session"),
            node,
            None,
        )

    prompt = asyncio.run(build())

    content = prompt.messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "x" * 5}
    assert content[1] == {"type": "text", "text": "x" * 9}
    assert prompt.calibration is not None
    assert prompt.calibration.added_filler_tokens == 8


def _continuation_node(*, target: int, context_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        actor_role="lead",
        prompt_kind="continuation",
        prompt_recipe_key="next",
        planned_input_tokens=target,
        context_mode=context_mode,
    )


def test_adaptive_prompt_calibration_trims_historical_synthetic_user_filler() -> None:
    async def build() -> SyntheticPrompt:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {
                    "trace_path": "source.jsonl",
                    "context_adjustment_mode": "adaptive",
                },
            }
        )
        previous = SyntheticPrompt("", (), ({"role": "user", "content": "a" * 20},))
        exchange = PromptExchange(previous, "b" * 10)
        return await PromptBuilder(config, _CharacterTokenizer()).build(  # type: ignore[arg-type]
            SimpleNamespace(runtime_session_id="session"),
            _continuation_node(target=25, context_mode="trim"),
            exchange,
        )

    prompt = asyncio.run(build())

    assert asyncio.run(_CharacterTokenizer().count(prompt)) == 25
    assert prompt.calibration is not None
    assert prompt.calibration.trimmed_filler_tokens == 6
    assert prompt.calibration.adjustment == "trim"


def test_strict_prompt_calibration_rejects_non_append_only_target() -> None:
    async def build() -> None:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {"trace_path": "source.jsonl", "context_adjustment_mode": "strict"},
            }
        )
        previous = SyntheticPrompt("", (), ({"role": "user", "content": "a" * 20},))
        exchange = PromptExchange(previous, "b" * 10)
        await PromptBuilder(config, _CharacterTokenizer()).build(  # type: ignore[arg-type]
            SimpleNamespace(runtime_session_id="session"),
            _continuation_node(target=25, context_mode="append"),
            exchange,
        )

    with pytest.raises(ValueError, match="Prompt minimum 31 tokens exceeds target 25"):
        asyncio.run(build())
