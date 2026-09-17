# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.converters import (
    ConverterSummary,
    DeterministicBlockRenderer,
    ReplayDatasetConverter,
)
from agentinfer.agentbench.replay.converters.codex_swebenchpro import (
    CodexSwebenchProConverter,
    _iter_json_array,
)
from agentinfer.agentbench.replay.planner import build_replay_plan
from agentinfer.agentbench.replay.prompt import PromptBuilder, PromptExchange, SyntheticPrompt
from agentinfer.agentbench.replay.runner import _prepare_replay_source, run_replay
from agentinfer.agentbench.replay.unified_trace_ir import canonical_sha256, validate_trace_ir


class _CharacterTokenizer:
    def close(self) -> None:
        """This in-memory tokenizer has no resources to release."""

    def content_tokens(self, text: str) -> int:
        return len(text)

    def input_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


class _RuntimeCharacterTokenizer:
    async def prompt_token_ids(self, prompt: SyntheticPrompt) -> tuple[int, ...]:
        return tuple(ord(c) for message in prompt.messages for c in str(message["content"]))

    async def text_token_ids(self, text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    async def detokenize_tokens(self, tokens) -> str:
        return "".join(map(chr, tokens))

    async def token_text(self, namespace: str, count: int) -> str:
        return "p" * count

    async def count(self, prompt: SyntheticPrompt) -> int:
        total = 0
        for message in prompt.messages:
            content = message["content"]
            if isinstance(content, str):
                total += len(content)
            else:
                total += sum(len(str(block.get("text", ""))) for block in content)
        return total


class _BlockCharacterTokenizer:
    def text_token_ids(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def detokenize_tokens(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _BoundarySensitiveTokenizer(_BlockCharacterTokenizer):
    def text_token_ids(self, text: str) -> list[int]:
        tokens = super().text_token_ids(text)
        if len(text) == 18:
            tokens[7] += 1000
        return tokens


def test_converter_public_contract_imports() -> None:
    assert ConverterSummary.__module__.endswith("converters.base")
    assert ReplayDatasetConverter.__module__.endswith("converters.base")
    assert DeterministicBlockRenderer.__module__.endswith("converters.base")


def _trace_ir(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "xy"},
                        {"from": "human", "value": "tool\nWall time: 0.2 seconds"},
                        {"from": "gpt", "value": "z"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "trace-ir"
    converter = CodexSwebenchProConverter(_CharacterTokenizer())
    summary = converter.convert(source, output)
    trace_ir = validate_trace_ir(output / "requests.jsonl", output / "texts")
    return output, summary, trace_ir


@pytest.mark.parametrize("tampered", ["requests", "text"])
def test_converter_writes_closed_trace_ir_contract_and_detects_tampering(
    tmp_path: Path,
    tampered: str,
) -> None:
    output, summary, trace_ir = _trace_ir(tmp_path)

    assert summary.sessions == 1
    assert summary.requests == 2
    assert {path.name for path in output.iterdir()} == {"requests.jsonl", "texts", "manifest.json"}
    assert trace_ir.root == output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2"
    assert manifest["converter"] == {"name": "codex_swebenchpro"}
    assert "capabilities" not in manifest
    assert manifest["summary"]["source_records_consumed"] == 1
    assert manifest["summary"]["tokenizer_counting_mode"] == "custom"
    assert manifest["summary"]["tokenizer_operations"] is None
    assert len(manifest["texts"]) == 2
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["actor_id"] == row["actor_role"] == "lead" for row in rows)
    assert all(row["started_at"] == row["finished_at"] for row in rows)
    assert [row["request_purpose"] for row in rows] == ["lead_main", "continuation"]

    targets = {
        "requests": output / "requests.jsonl",
        "text": output / "texts" / "codex-session-0000" / "turn_1.txt",
    }
    targets[tampered].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_trace_ir(output / "requests.jsonl", output / "texts")


def test_trace_ir_accepts_legacy_converter_metadata_and_still_checks_its_digest(tmp_path: Path) -> None:
    output, _, _ = _trace_ir(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["converter"]["version"] = "legacy-converter"
    del manifest["bundle_sha256"]
    manifest["bundle_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    trace_ir = validate_trace_ir(output / "requests.jsonl", output / "texts")
    assert trace_ir.root == output

    del manifest["converter"]["version"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle_sha256"):
        validate_trace_ir(output / "requests.jsonl", output / "texts")


def test_trace_ir_rejects_tampered_manifest_digest(tmp_path: Path) -> None:
    output, _, _ = _trace_ir(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["requests"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle_sha256"):
        validate_trace_ir(output / "requests.jsonl", output / "texts")


def test_trace_ir_rejects_old_manifest_schema(tmp_path: Path) -> None:
    """Require reconversion instead of silently accepting a version-1 contract."""

    output, _, _ = _trace_ir(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported unified Trace IR schema_version"):
        validate_trace_ir(output / "requests.jsonl", output / "texts")


def test_trace_record_uses_human_sidecar_and_live_length_aligned_assistant(tmp_path: Path) -> None:
    output, _, trace_ir = _trace_ir(tmp_path)
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": tmp_path / "source.json",
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {
                    "p50_seconds": 2,
                    "p95_seconds": 30,
                    "p99_seconds": 90,
                },
            },
        }
    )
    plan = build_replay_plan(config, analyze_replay_trace(output / "requests.jsonl"), trace_ir)
    assert plan.source == str(config.replay.trace_path)
    assert "dataset_capabilities" not in plan.to_dict()
    task = plan.tasks[0]
    serialized = task.to_dict()["requests"]
    assert serialized[0]["prompt_ref"] == {"session_id": "codex-session-0000", "turn_index": 0}
    assert all("dependency_kind" not in node for node in serialized)
    assert all("same_agent_gap_seconds" not in node for node in serialized)
    builder = PromptBuilder(config, _RuntimeCharacterTokenizer(), trace_ir)  # type: ignore[arg-type]

    async def build_prompts() -> tuple[SyntheticPrompt, SyntheticPrompt]:
        first = await builder.build(task, task.requests[0], None)
        second = await builder.build(task, task.requests[1], PromptExchange(first, "AB"))
        return first, second

    first, second = asyncio.run(build_prompts())
    assert first.messages == ({"role": "user", "content": "hello"},)
    assert second.messages[1] == {"role": "assistant", "content": "AB"}
    assert second.messages[2]["content"] == "tool\nWall time: 0.2 seconds"
    assert second.calibration is not None
    assert second.calibration.adjustment == "none"
    assert second.calibration.target_met is True
    assert all(node.context_mode in {"independent", "append"} for node in task.requests)


def test_trace_record_repairs_live_assistant_drift_by_default(tmp_path: Path) -> None:
    output, _, trace_ir = _trace_ir(tmp_path)
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": tmp_path / "source.json",
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
            },
        }
    )
    plan = build_replay_plan(config, analyze_replay_trace(output / "requests.jsonl"), trace_ir)
    task = plan.tasks[0]
    first_node, second_node = task.requests

    async def build_prompt() -> SyntheticPrompt:
        builder = PromptBuilder(config, _RuntimeCharacterTokenizer(), trace_ir)  # type: ignore[arg-type]
        first = await builder.build(task, first_node, None)
        return await builder.build(task, second_node, PromptExchange(first, "A" * 15))

    prompt = asyncio.run(build_prompt())

    assert prompt.messages[0] == {"role": "user", "content": "hello"}
    assert prompt.messages[1] == {"role": "assistant", "content": "A" * 15}
    assert prompt.calibration is not None
    assert prompt.calibration.residual_tokens == 0
    assert prompt.calibration.trimmed_current_user_tokens == 13
    assert prompt.calibration.accepted_with_tolerance is False


def test_block_renderer_preserves_scope_prefix_and_tail_boundaries() -> None:
    tokenizer = _BlockCharacterTokenizer()
    session_scoped = DeterministicBlockRenderer(tokenizer, block_size=8, block_id_scope="session")
    first = session_scoped.render_request("s1", ["a", "b", "tail"], 19)
    same_prefix = session_scoped.render_request("s1", ["a", "b", "other"], 18)
    other_session = session_scoped.render_request("s2", ["a", "b", "tail"], 19)
    dataset_scoped = DeterministicBlockRenderer(tokenizer, block_size=8, block_id_scope="dataset")

    assert len(first) == 19
    assert first[:16] == same_prefix[:16]
    assert first[:8] != other_session[:8]
    assert dataset_scoped.render_block("s1", "a", 8) == dataset_scoped.render_block("s2", "a", 8)
    with pytest.raises(ValueError, match="inconsistent"):
        session_scoped.render_request("s1", ["a", "tail"], 17)


def test_block_renderer_rejects_common_prefix_token_drift() -> None:
    renderer = DeterministicBlockRenderer(
        _BoundarySensitiveTokenizer(),
        block_size=8,
        block_id_scope="session",
    )
    renderer.render_request("s1", ["a", "b", "tail"], 19)

    with pytest.raises(ValueError, match="shared hash prefix"):
        renderer.render_request("s1", ["a", "b", "other"], 18)


@pytest.mark.parametrize("tampered", [None, "requests", "text", "manifest"])
def test_inferact_trace_is_validated_before_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered: str | None
) -> None:
    """Accept complete conversion output and reject damage at the Runner boundary."""
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "answer"},
                        {"from": "human", "value": "next"},
                        {"from": "gpt", "value": "done"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    converter = CodexSwebenchProConverter(_CharacterTokenizer())
    convert = converter.convert

    def convert_with_damage(source: Path, output_dir: Path) -> ConverterSummary:
        """Simulate a converter returning an artifact with inconsistent contents."""

        summary = convert(source, output_dir)
        if tampered is not None:
            targets = {
                "requests": output_dir / "requests.jsonl",
                "text": output_dir / "texts" / "codex-session-0000" / "turn_1.txt",
                "manifest": output_dir / "manifest.json",
            }
            targets[tampered].write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(converter, "convert", convert_with_damage)
    monkeypatch.setattr(
        CodexSwebenchProConverter,
        "from_backend",
        classmethod(lambda cls, config: converter),
    )
    config = ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": source,
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
            }
        }
    )

    if tampered is not None:
        with pytest.raises(ValueError, match="SHA256 mismatch|schema_version"):
            _prepare_replay_source(config, tmp_path / "result")
        return

    analysis_source, trace_ir, captures, local_tokenizer = _prepare_replay_source(config, tmp_path / "result")

    assert analysis_source == tmp_path / "result" / "convert_result" / "requests.jsonl"
    assert trace_ir is not None and trace_ir.root == tmp_path / "result" / "convert_result"
    assert local_tokenizer is None
    assert {capture.source for capture in captures} == {"replay_conversion_manifest"}


def test_trace_record_runner_rejects_backend_residuals_and_reports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject even a one-token backend mismatch and retain the evidence in summaries."""

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([{"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "xy"}]}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        CodexSwebenchProConverter,
        "from_backend",
        classmethod(lambda cls, config: cls(_CharacterTokenizer())),
    )
    residuals = iter((0, 1, -1, 2, -2))

    def respond(request: httpx.Request) -> httpx.Response:
        """Calibrate exactly, then return drifting backend usage."""

        if request.url.path == "/metrics":
            return httpx.Response(503)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 5})
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["min_tokens"] == body["max_tokens"] == 2
        chunk = {
            "choices": [{"delta": {"content": "AB"}}],
            "usage": {"prompt_tokens": 5 + next(residuals), "completion_tokens": 2},
        }
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 5, "max_concurrency": 1, "result_dir": tmp_path / "result"},
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": source,
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
                "prompt_calibration_tolerance_tokens": 0,
            },
        }
    )

    result = run_replay(config)
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    execution = json.loads((result / "replay-execution.json").read_text(encoding="utf-8"))
    validation = json.loads((result / "trace-record-validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))

    assert not (result / "convert_result" / "capabilities.json").exists()
    assert "replay_conversion_capabilities" not in {capture["source"] for capture in manifest["evidence"]}
    assert "replay_conversion_capabilities" not in summary["source_health"]["sources"]
    assert not (result / "replay-normalization.json").exists()
    assert "replay_normalization" not in {capture["source"] for capture in manifest["evidence"]}
    assert "replay_normalization" not in summary["source_health"]["sources"]
    assert summary["requests"]["successful_requests"] == 1
    assert summary["requests"]["failed_requests"] == 4
    assert summary["execution"]["metadata"] == execution["summary"]
    assert (
        validation["requests_over_tolerance"] == execution["summary"]["prompt_calibration_requests_over_tolerance"] == 0
    )
    assert execution["summary"]["prompt_calibration_exact_requests"] == 5
    assert execution["summary"]["prompt_calibration_tolerated_requests"] == 0
    assert validation["tolerance_tokens"] == 0
    assert validation["input_length_comparable"] is False
    assert validation["backend_input_requests_over_tolerance"] == 4
    assert execution["summary"]["backend_input_max_absolute_residual_tokens"] == 2


@pytest.mark.parametrize("bad_usage", [False, True])
def test_current_turn_replay_repeats_targets_with_different_live_answers(tmp_path, monkeypatch, bad_usage):
    """Run 8 tasks at concurrency 4 twice, preserving history with pad/trim in both orders."""
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "xy"},
                        {"from": "human", "value": "abcdefghij"},
                        {"from": "gpt", "value": "xy"},
                        {"from": "human", "value": "klmnopqrst"},
                        {"from": "gpt", "value": "xy"},
                    ]
                }
            ]
            * 8
        )
    )
    monkeypatch.setattr(
        CodexSwebenchProConverter, "from_backend", classmethod(lambda cls, config: cls(_CharacterTokenizer()))
    )
    client = httpx.AsyncClient
    totals = []
    adjustments = []
    for run_index, replies in enumerate([["A", "BBBB", "ZZ"], ["CCCC", "D", "ZZ"]]):
        captured = {}

        def respond(request, replies=replies, captured=captured):
            if request.url.path == "/metrics":
                return httpx.Response(503)
            body = json.loads(request.content)
            if request.url.path == "/detokenize":
                return httpx.Response(200, json={"prompt": "".join(map(chr, body["tokens"]))})
            text = body.get("prompt")
            if text is None:
                text = "".join(message["content"] for message in body["messages"])
            count = len(text)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": count, "tokens": list(map(ord, text))})
            assert request.url.path == "/v1/chat/completions"
            session = captured.setdefault(request.headers["x-claude-code-session-id"], [])
            index = len(session)
            assert count == [5, 17, 29][index]
            if session:
                assert body["messages"][:-2] == session[-1]["messages"]
                assert body["messages"][-2] == {
                    "role": "assistant",
                    "content": replies[index - 1],
                    "reasoning_content": "thought",
                }
            session.append(body)
            chunk = {
                "choices": [{"delta": {"content": replies[index], "reasoning_content": "thought"}}],
                "usage": {"prompt_tokens": count + int(bad_usage), "completion_tokens": 2},
            }
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda handler=respond, **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
        )
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 8, "max_concurrency": 4, "result_dir": tmp_path / f"run-{run_index}"},
                "replay": {
                    "trace_type": "inferact_codex_swebenchpro",
                    "trace_path": source,
                    "prompt_shape": "trace_record",
                    "interval_mode": "lognormal",
                    "interval_lognormal": {"p50_seconds": 0.002, "p95_seconds": 0.03, "p99_seconds": 0.09},
                },
            }
        )
        result = run_replay(config)
        summary = json.loads((result / "summary.json").read_text())
        validation = json.loads((result / "trace-record-validation.json").read_text())
        if bad_usage:
            assert summary["tasks"]["failed"] == 8
            assert summary["requests"]["successful_requests"] == 0
            assert summary["requests"]["failed_requests"] == 8
            assert summary["execution"]["metadata"]["dependency_skipped_requests"] == 16
            assert validation["input_length_comparable"] is False
            assert validation["backend_input_requests_over_tolerance"] == 8
            continue
        assert summary["requests"]["successful_requests"] == 24
        assert summary["tasks"]["completed"] == 8
        assert summary["tasks"]["failed"] == 0
        assert validation["input_length_comparable"] is True
        assert validation["requests_over_tolerance"] == 0
        assert validation["backend_input_checked_requests"] == 24
        assert validation["trimmed_current_user_characters"] == 16
        assert summary["execution"]["metadata"]["trimmed_filler_tokens"] == 0
        adjustments.append(summary["execution"]["metadata"]["prompt_calibration_adjustments"])
        totals.append(summary["requests"]["input_tokens"])
    if not bad_usage:
        assert totals == [408, 408]
        assert all(item["pad"] == item["trim"] == 8 for item in adjustments)


@pytest.mark.parametrize("trace_type", ["agentX", "tracelab"])
def test_reserved_trace_types_fail_explicitly(tmp_path: Path, trace_type: str) -> None:
    config = ReplayBenchConfig.model_validate({"replay": {"trace_type": trace_type, "trace_path": tmp_path / "trace"}})

    with pytest.raises(NotImplementedError, match="reserved for future integration"):
        _prepare_replay_source(config, tmp_path / "result")


def test_runtime_converter_uses_configured_backend_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Response:
        def __init__(self, count: int) -> None:
            self.count = count

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, int]:
            return {"count": self.count}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        def post(self, url: str, json: dict[str, object]) -> Response:
            calls.append((url, json))
            return Response(37)

        def close(self) -> None:
            pass

    monkeypatch.setattr("agentinfer.agentbench.replay.converters.codex_swebenchpro.httpx.Client", Client)
    config = ReplayBenchConfig.model_validate(
        {
            "backend": {
                "base_url": "http://backend",
                "tokenizer_base_url": "http://tokenizer",
                "model": "example/custom-chat-model",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": "source.json",
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
            },
        }
    )

    converter = CodexSwebenchProConverter.from_backend(config)
    messages = [
        {"role": "user", "content": "human"},
        {"role": "assistant", "content": "assistant"},
        {"role": "user", "content": "next"},
    ]
    input_tokens = converter.tokenizer.input_tokens(messages)
    converter.close()

    assert input_tokens == 37
    assert calls == [
        (
            "http://tokenizer/tokenize",
            {
                "model": "example/custom-chat-model",
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    ]


def test_runtime_converter_counts_each_turn_with_full_backend_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    message_calls: list[list[dict[str, str]]] = []

    class Response:
        def __init__(self, count: int) -> None:
            self.count = count

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, int]:
            return {"count": self.count}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        def post(self, url: str, json: dict[str, object]) -> Response:
            if "messages" in json:
                messages = json["messages"]
                assert isinstance(messages, list)
                message_calls.append([dict(message) for message in messages])
                return Response(100 + len(messages))
            return Response(len(str(json["prompt"])))

        def close(self) -> None:
            pass

    monkeypatch.setattr("agentinfer.agentbench.replay.converters.codex_swebenchpro.httpx.Client", Client)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "first"},
                        {"from": "gpt", "value": "answer"},
                        {"from": "human", "value": "second"},
                        {"from": "gpt", "value": "done"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    config = ReplayBenchConfig.model_validate({"replay": {"trace_path": source}})
    converter = CodexSwebenchProConverter.from_backend(config)
    converter.convert(source, tmp_path / "output")
    converter.close()

    assert message_calls == [
        [{"role": "user", "content": "first"}],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
    ]
    rows = [
        json.loads(line) for line in (tmp_path / "output" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["input_tokens"] for row in rows] == [101, 103]


def test_converter_rejects_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "empty.json"
    source.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="no replayable conversation turns"):
        CodexSwebenchProConverter(_CharacterTokenizer()).convert(source, tmp_path / "output")


def test_streaming_source_rejects_trailing_data_in_later_chunk(tmp_path: Path) -> None:
    source = tmp_path / "trailing.json"
    source.write_text('[{"conversations": []}] trailing', encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected data after"):
        list(_iter_json_array(source, chunk_size=4))


@pytest.mark.parametrize(
    "content",
    [
        '[{"conversations": []},]',
        '[{"conversations": []} {"conversations": []}]',
    ],
)
def test_streaming_source_rejects_invalid_array_delimiters(tmp_path: Path, content: str) -> None:
    source = tmp_path / "invalid-delimiter.json"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="comma"):
        list(_iter_json_array(source, chunk_size=4))
