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
from agentinfer.agentbench.replay.unified_trace_ir import validate_trace_ir


class _CharacterTokenizer:
    def close(self) -> None:
        """This in-memory tokenizer has no resources to release."""

    def content_tokens(self, text: str) -> int:
        return len(text)

    def trace_turn_tokens(self, completed_tokens: int, human: str, assistant: str) -> tuple[int, int]:
        input_tokens = completed_tokens + len(human)
        return input_tokens, completed_tokens + len(human) + len(assistant)


class _RuntimeCharacterTokenizer:
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
    assert "capabilities" not in manifest
    assert manifest["summary"]["source_records_consumed"] == 1
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


def test_trace_record_audits_large_live_assistant_residual_without_rewriting_text(tmp_path: Path) -> None:
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

    class DriftTokenizer(_RuntimeCharacterTokenizer):
        async def count(self, prompt: SyntheticPrompt) -> int:
            count = await super().count(prompt)
            return count - 13 if len(prompt.messages) > 1 else count

    async def build_prompt() -> SyntheticPrompt:
        builder = PromptBuilder(config, DriftTokenizer(), trace_ir)  # type: ignore[arg-type]
        first = await builder.build(task, first_node, None)
        return await builder.build(task, second_node, PromptExchange(first, "AB"))

    prompt = asyncio.run(build_prompt())

    assert prompt.messages[1] == {"role": "assistant", "content": "AB"}
    assert prompt.calibration is not None
    assert prompt.calibration.residual_tokens == -13
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

    analysis_source, trace_ir, captures = _prepare_replay_source(config, tmp_path / "result")

    assert analysis_source == tmp_path / "result" / "convert_result" / "requests.jsonl"
    assert trace_ir is not None and trace_ir.root == tmp_path / "result" / "convert_result"
    assert {capture.source for capture in captures} == {"replay_conversion_manifest"}


def test_trace_record_runner_reports_consistent_residual_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the real Runner and HTTP adapters to audit signed residuals across artifacts."""

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
        """Return token counts at and beyond tolerance, plus length-constrained SSE."""

        if request.url.path == "/metrics":
            return httpx.Response(503)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 5 + next(residuals)})
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["min_tokens"] == body["max_tokens"] == 2
        chunk = {"choices": [{"delta": {"content": "AB"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
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
                "prompt_calibration_tolerance_tokens": 1,
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
    assert summary["requests"]["successful_requests"] == 5
    assert summary["execution"]["metadata"] == execution["summary"]
    assert (
        validation["requests_over_tolerance"] == execution["summary"]["prompt_calibration_requests_over_tolerance"] == 2
    )
    assert execution["summary"]["prompt_calibration_exact_requests"] == 1
    assert execution["summary"]["prompt_calibration_tolerated_requests"] == 2
    assert validation["max_absolute_residual_tokens"] == 2
    assert validation["sum_absolute_residual_tokens"] == 6


@pytest.mark.parametrize("trace_type", ["agentX", "tracelab"])
def test_reserved_trace_types_fail_explicitly(tmp_path: Path, trace_type: str) -> None:
    config = ReplayBenchConfig.model_validate({"replay": {"trace_type": trace_type, "trace_path": tmp_path / "trace"}})

    with pytest.raises(NotImplementedError, match="reserved for future integration"):
        _prepare_replay_source(config, tmp_path / "result")


def test_runtime_converter_uses_configured_backend_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

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
            if "prompt" in json:
                prompt = str(json["prompt"])
            else:
                prompt = (
                    "".join(
                        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                        for message in json["messages"]
                    )
                    + "<|im_start|>assistant\n"
                )
            calls.append((url, prompt))
            return Response(len(prompt))

        def close(self) -> None:
            pass

    monkeypatch.setattr("agentinfer.agentbench.replay.converters.codex_swebenchpro.httpx.Client", Client)
    config = ReplayBenchConfig.model_validate(
        {
            "backend": {
                "base_url": "http://backend",
                "tokenizer_base_url": "http://tokenizer",
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
    input_tokens, completed_tokens = converter.tokenizer.trace_turn_tokens(0, "human", "assistant")
    converter.close()

    assert input_tokens > 0 and completed_tokens > input_tokens
    assert calls and all(url == "http://tokenizer/tokenize" for url, _ in calls)


def test_runtime_converter_rejects_incompatible_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False

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
            return Response(1 if "messages" in json else len(str(json["prompt"])))

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr("agentinfer.agentbench.replay.converters.codex_swebenchpro.httpx.Client", Client)
    config = ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": "source.json",
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
            }
        }
    )

    with pytest.raises(ValueError, match="incompatible"):
        CodexSwebenchProConverter.from_backend(config)
    assert closed is True


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
