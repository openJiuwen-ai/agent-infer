# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
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
                "replay": {
                    "trace_path": "source.jsonl",
                    "prompt_shape": "claude_code_minimal_v1",
                    "lead_1st_sys_shared_prefix": 2,
                    "lead_1st_sys_session_prefix": 3,
                    "lead_1st_tool_shared_prefix": 4,
                    "lead_1st_msg_shared_prefix": 5,
                    "lead_1st_trailing_system_prefix": 6,
                    "lead_continuation_extra_system_ratio": 1,
                    "lead_continuation_system_tokens": 7,
                }
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
            "replay": {
                "trace_path": "source.jsonl",
                "prompt_calibration_tolerance_tokens": tolerance,
            }
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
                "replay": {
                    "trace_path": "source.jsonl",
                    "prompt_shape": "claude_code_minimal_v1",
                    "lead_1st_msg_shared_prefix": 5,
                    "lead_1st_trailing_system_prefix": 6,
                }
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
                "replay": {
                    "trace_path": "source.jsonl",
                    "context_adjustment_mode": "adaptive",
                }
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
            {"replay": {"trace_path": "source.jsonl", "context_adjustment_mode": "strict"}}
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
