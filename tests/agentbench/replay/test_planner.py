# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.benchkit.compare import load_summary
from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture
from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.executor import NodeExecution, ReplayTaskExecution
from agentinfer.agentbench.replay.planner import ReplayPlan, ReplayPlanNode, ReplayTaskPlan, build_replay_plan
from agentinfer.agentbench.replay.runner import _execution_metadata, run_replay
from agentinfer.agentbench.request_proxy.request_trace import RequestFact


def _row(
    request_id: str,
    session_id: str,
    actor_id: str,
    started: int,
    finished: int,
    input_tokens: int | None = 10,
    output_tokens: int | None = 2,
    status: str = "success",
    cached_tokens: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "request_id": request_id,
        "session_id": session_id,
        "actor_id": actor_id,
        "started_at": f"2026-01-01T00:00:{started:02d}+00:00",
        "finished_at": f"2026-01-01T00:00:{finished:02d}+00:00",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status,
    }
    if cached_tokens is not None:
        row["cached_tokens"] = cached_tokens
    return row


def test_execution_metadata_partitions_sent_and_pre_send_failures() -> None:
    nodes = (
        NodeExecution("sent-ok", "runtime-ok", "request", "lead_main", "success", 0, 0, 0, None),
        NodeExecution("sent-failed", "runtime-failed", "request", "lead_main", "failed", 0, 0, 0, "read"),
        NodeExecution("pre-send", "runtime-pre", "request", "lead_main", "failed", None, None, None, "prompt"),
        NodeExecution(
            "skipped",
            "runtime-skipped",
            "request",
            "continuation",
            "skipped_dependency_failed",
            None,
            None,
            None,
            "dependency",
        ),
    )
    planned = tuple(SimpleNamespace(node_type="request", context_mode="append") for _ in nodes)
    plan = SimpleNamespace(
        workload_fingerprint="fingerprint",
        tasks=(SimpleNamespace(requests=planned),),
    )
    tasks = (ReplayTaskExecution("task", "session", "failed", 1, None, nodes),)

    metadata = _execution_metadata(plan, tasks)  # type: ignore[arg-type]

    assert metadata["materialized_request_nodes"] == 4
    assert metadata["attempted_requests"] == 2
    assert metadata["successful_requests"] == 1
    assert metadata["failed_requests"] == 1
    assert metadata["pre_send_failed_requests"] == 1
    assert metadata["dependency_skipped_requests"] == 1


def _source(path: Path) -> None:
    rows = [
        _row("s1-lead", "s1", "lead", 0, 1, cached_tokens=6),
        _row(
            "s1-failed",
            "s1",
            "child",
            2,
            3,
            None,
            None,
            "failed",
        ),
        _row("s1-retry", "s1", "child", 4, 5, 20, 4),
        _row("s2-lead", "s2", "lead", 0, 1, 12, 3),
        _row(
            "s2-timing",
            "s2",
            "orphan",
            2,
            3,
            None,
            None,
            "failed",
        ),
        _row("s2-next", "s2", "lead", 4, 5, 16, 3),
        _row(
            "s2-terminal",
            "s2",
            "tail",
            6,
            7,
            None,
            None,
            "failed",
        ),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(
    source: Path,
    result_dir: Path,
    *,
    endpoint: str = "/v1/chat/completions",
) -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "experiment": {
                "task_num": 5,
                "result_dir": result_dir,
                "max_concurrency": 2,
            },
            "backend": {"endpoint": endpoint},
            "replay": {
                "trace_path": source,
                "sample_seed": 9,
            },
        }
    )


def test_plan_is_stable_isolated_and_endpoint_independent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    analysis = analyze_replay_trace(source)

    chat = build_replay_plan(
        _config(source, tmp_path / "chat"),
        analysis,
    )
    messages = build_replay_plan(
        _config(
            source,
            tmp_path / "messages",
            endpoint="/v1/messages",
        ),
        analysis,
    )
    repeated = build_replay_plan(
        _config(source, tmp_path / "other"),
        analysis,
    )
    different_concurrency_config = _config(
        source,
        tmp_path / "concurrency",
    )
    different_concurrency_config.experiment.max_concurrency = 3
    different_concurrency = build_replay_plan(
        different_concurrency_config,
        analysis,
    )
    different_same_agent_gap_config = _config(
        source,
        tmp_path / "same-agent-gap",
    )
    different_same_agent_gap_config.replay.trace_same_agent_gap_scale = 2
    different_same_agent_gap = build_replay_plan(
        different_same_agent_gap_config,
        analysis,
    )

    assert isinstance(chat, ReplayPlan)
    assert all(isinstance(task, ReplayTaskPlan) for task in chat.tasks)
    assert all(isinstance(request, ReplayPlanNode) for task in chat.tasks for request in task.requests)
    assert chat == repeated
    assert chat.workload_fingerprint == messages.workload_fingerprint
    assert chat.workload_fingerprint != different_concurrency.workload_fingerprint
    assert chat.workload_fingerprint != different_same_agent_gap.workload_fingerprint
    assert chat.interval_model.fit_version == "agentinfer-replay-trace/v2"
    assert chat.planner_version == "agentinfer-replay-structural/v9"
    assert chat.execution_ready is False
    assert len(chat.tasks) == 5
    assert len({task.runtime_session_id for task in chat.tasks}) == 5
    assert {task.source_session_id for task in chat.tasks[:2]} == {"s1", "s2"}
    assert {task.source_session_id for task in chat.tasks[2:4]} == {"s1", "s2"}


def test_plan_materializes_retry_targets_and_prunes_terminal_timing_nodes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    config = _config(source, tmp_path / "result")
    config.replay.trace_same_agent_gap_scale = 10
    config.replay.trace_same_agent_gap_offset_seconds = 7
    analysis = analyze_replay_trace(source)
    source_keys = {
        request.source_request_id: request.key for session in analysis.sessions for request in session.requests
    }
    plan = build_replay_plan(config, analysis)
    task_by_source = {task.source_session_id: task for task in plan.tasks[:2]}
    s1_requests = task_by_source["s1"].requests
    retry_failure = next(request for request in s1_requests if request.source_key == source_keys["s1-failed"])
    assert retry_failure.planned_input_tokens == 20
    assert retry_failure.planned_output_tokens == 4
    assert retry_failure.prompt_recipe_key != retry_failure.source_key

    s2_requests = task_by_source["s2"].requests
    assert {request.source_key for request in s2_requests} == {
        source_keys[name] for name in ("s2-lead", "s2-timing", "s2-next")
    }
    timing_dependency = next(request for request in s2_requests if request.source_key == source_keys["s2-timing"])
    assert timing_dependency.node_type == "timing_dependency"
    assert timing_dependency.effective_duration_seconds == 1.0


def test_plan_artifact_contains_only_materialized_execution_fields(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    plan = build_replay_plan(
        _config(source, tmp_path / "result"),
        analyze_replay_trace(source),
    ).to_dict()
    nodes = [node for task in plan["tasks"] for node in task["requests"]]
    request = next(node for node in nodes if node["node_type"] == "request")
    timing_dependency = next(node for node in nodes if node["node_type"] == "timing_dependency")

    assert set(request) == {
        "runtime_request_id",
        "source_key",
        "prompt_recipe_key",
        "node_type",
        "prompt_kind",
        "actor_id",
        "actor_role",
        "parent_actor_id",
        "send_after",
        "context_after",
        "context_mode",
        "effective_interval_seconds",
        "planned_input_tokens",
        "planned_output_tokens",
        "backend_sampling_seed",
    }
    assert set(timing_dependency) == {
        "runtime_request_id",
        "source_key",
        "node_type",
        "send_after",
        "effective_interval_seconds",
        "effective_duration_seconds",
    }
    assert "dependency_kind" not in request
    assert "same_agent_gap_seconds" not in request


def test_adaptive_context_planning_preserves_trim_reset_and_send_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "context.jsonl"
    rows = [
        _row("first", "context", "lead", 0, 1, 100, 10),
        _row("micro", "context", "lead", 2, 3, 105, 10),
        _row("reset", "context", "lead", 4, 5, 20, 2),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1, "result_dir": tmp_path / "result"},
            "replay": {
                "trace_path": source,
                "context_adjustment_mode": "adaptive",
            },
        }
    )

    first, micro, reset = build_replay_plan(config, analyze_replay_trace(source)).tasks[0].requests

    assert first.context_mode == "independent"
    assert micro.context_mode == "trim"
    assert micro.context_after == first.source_key
    assert micro.send_after == first.source_key
    assert reset.context_mode == "reset"
    assert reset.context_after is None
    assert reset.send_after == micro.source_key


def test_strict_context_planning_preserves_append_only_failure_surface(tmp_path: Path) -> None:
    source = tmp_path / "strict.jsonl"
    rows = [
        _row("first", "strict", "lead", 0, 1, 100, 10),
        _row("next", "strict", "lead", 2, 3, 20, 2),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = ReplayBenchConfig.model_validate({"experiment": {"task_num": 1}, "replay": {"trace_path": source}})

    requests = build_replay_plan(config, analyze_replay_trace(source)).tasks[0].requests

    assert requests[1].context_mode == "append"
    assert requests[1].context_after == requests[0].source_key


def test_runner_writes_completed_execution_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    result_dir = tmp_path / "result"

    class FakeTokenizer:
        def __init__(self, config: object) -> None:
            pass

        async def close(self) -> None:
            pass

    class FakeTransport:
        def __init__(self, config: object, run_id: str, writer: object) -> None:
            self.run_id = run_id
            self.writer = writer

        async def close(self) -> None:
            pass

    class FakeExecutor:
        def __init__(self, config: object, plan: ReplayPlan, prompts: object, transport: FakeTransport) -> None:
            self.plan = plan
            self.transport = transport

        async def execute(self) -> tuple[ReplayTaskExecution, ...]:
            results = []
            now = datetime.now(timezone.utc).isoformat()
            for task in self.plan.tasks:
                nodes = []
                for node in task.requests:
                    nodes.append(
                        NodeExecution(
                            node.source_key,
                            node.runtime_request_id,
                            node.node_type,
                            node.prompt_kind,
                            "success",
                            0,
                            0 if node.node_type == "request" else None,
                            0 if node.node_type == "request" else None,
                            None,
                        )
                    )
                    if node.node_type == "request":
                        self.transport.writer.submit(
                            RequestFact(
                                "1",
                                self.transport.run_id,
                                node.runtime_request_id,
                                task.runtime_session_id,
                                node.actor_id,
                                node.actor_role,
                                now,
                                now,
                                "success",
                                200,
                                0.01,
                                0.001,
                                node.planned_input_tokens,
                                node.planned_output_tokens,
                                0,
                                0,
                                "http://backend",
                                None,
                            )
                        )
                results.append(
                    ReplayTaskExecution(
                        task.task_id,
                        task.runtime_session_id,
                        "completed",
                        0.01,
                        None,
                        tuple(nodes),
                    )
                )
            return tuple(results)

    async def no_metrics(url: str) -> EvidenceCapture:
        return EvidenceCapture("vllm", None, False, "test", {})

    monkeypatch.setattr("agentinfer.agentbench.replay.runner.TokenizerClient", FakeTokenizer)  # type: ignore[attr-defined]
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.ReplayTransport", FakeTransport)  # type: ignore[attr-defined]
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.ReplayExecutor", FakeExecutor)  # type: ignore[attr-defined]
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.capture_vllm_metrics", no_metrics)  # type: ignore[attr-defined]

    actual = run_replay(
        _config(source, result_dir),
        cli_metadata={"entrypoint": "test"},
    )

    manifest = json.loads((actual / "manifest.json").read_text(encoding="utf-8"))
    source_analysis = json.loads((actual / "replay-source-analysis.json").read_text(encoding="utf-8"))
    plan = json.loads((actual / "replay-plan.json").read_text(encoding="utf-8"))
    planned_requests = sum(request["node_type"] == "request" for task in plan["tasks"] for request in task["requests"])
    planned_timing_dependency_nodes = sum(
        request["node_type"] == "timing_dependency" for task in plan["tasks"] for request in task["requests"]
    )
    summary = load_summary(actual)
    source_requests = [request for session in source_analysis["sessions"] for request in session["requests"]]
    planned_nodes = [request for task in plan["tasks"] for request in task["requests"]]
    assert manifest["status"] == "completed"
    assert (
        next(request for request in source_requests if request["source_request_id"] == "s1-lead")[
            "source_cached_tokens"
        ]
        == 6
    )
    assert all("source_cached_tokens" not in request for request in planned_nodes)
    assert manifest["config"]["implementation_stage"] == "executable"
    assert plan["execution_ready"] is True
    assert summary["execution"]["available"] is True
    assert summary["execution"]["metadata"]["planned_tasks"] == 5
    assert summary["execution"]["metadata"]["planned_requests"] == planned_requests
    assert summary["execution"]["metadata"]["planned_timing_dependency_nodes"] == planned_timing_dependency_nodes
    assert summary["tasks"]["completed"] == 5
    assert summary["requests"]["requests"] == planned_requests
    assert summary["vllm"]["available"] is False
    assert summary["correctness"]["available"] is False
    assert summary["lifecycle"]["status"] == "completed"
    assert summary["cli"] == {"entrypoint": "test"}
    assert (actual / "summary.json").is_file()
    assert (actual / "requests.jsonl").read_text(encoding="utf-8")
    assert (actual / "evidence").is_dir()
    assert (actual / "replay-source-analysis.json").is_file()
    assert (actual / "replay-plan.json").is_file()
    assert not (actual / "replay-normalization.json").exists()
    assert "replay_normalization" not in {capture["source"] for capture in manifest["evidence"]}
    assert "replay_normalization" not in summary["source_health"]["sources"]
    assert (actual / "replay-execution.json").is_file()


def test_runner_failure_preserves_evidence_and_closes_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    result_dir = tmp_path / "failed-result"
    state = {"tokenizer_closed": False, "transport_closed": False}

    class FakeTokenizer:
        def __init__(self, config: object) -> None:
            pass

        async def close(self) -> None:
            state["tokenizer_closed"] = True

    class FakeTransport:
        def __init__(self, config: object, run_id: str, writer: object) -> None:
            pass

        async def close(self) -> None:
            state["transport_closed"] = True
            raise RuntimeError("transport close failed")

    class FailingExecutor:
        def __init__(self, config: object, plan: object, prompts: object, transport: object) -> None:
            pass

        async def execute(self) -> tuple[ReplayTaskExecution, ...]:
            raise RuntimeError("executor failed")

    async def no_metrics(url: str) -> EvidenceCapture:
        return EvidenceCapture("vllm", None, False, "test", {})

    monkeypatch.setattr("agentinfer.agentbench.replay.runner.TokenizerClient", FakeTokenizer)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.ReplayTransport", FakeTransport)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.ReplayExecutor", FailingExecutor)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.capture_vllm_metrics", no_metrics)

    with pytest.raises(RuntimeError, match="executor failed"):
        run_replay(_config(source, result_dir))

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    error = json.loads((result_dir / "replay-error.json").read_text(encoding="utf-8"))
    sources = {capture["source"] for capture in manifest["evidence"]}
    assert manifest["status"] == "failed"
    assert error["error_type"] == "RuntimeError"
    assert error["error"] == "executor failed"
    assert error["cleanup_errors"] == ["transport close failed: RuntimeError: transport close failed"]
    assert {
        "request_trace",
        "replay_source_analysis",
        "replay_plan",
        "vllm_start",
        "replay_error",
    } <= sources
    assert "replay_normalization" not in sources
    assert not (result_dir / "replay-normalization.json").exists()
    assert state == {"tokenizer_closed": True, "transport_closed": True}


def test_writer_start_failure_finalizes_without_creating_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "requests.jsonl"
    _source(source)
    result_dir = tmp_path / "writer-failed-result"

    class FailingWriter:
        def __init__(self, path: Path) -> None:
            pass

        async def start(self) -> None:
            raise RuntimeError("writer start failed")

    class UnexpectedClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("clients must not be created before the writer starts")

    async def no_metrics(url: str) -> EvidenceCapture:
        return EvidenceCapture("vllm", None, False, "test", {})

    monkeypatch.setattr("agentinfer.agentbench.replay.runner.RequestTraceWriter", FailingWriter)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.TokenizerClient", UnexpectedClient)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.ReplayTransport", UnexpectedClient)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.capture_vllm_metrics", no_metrics)

    with pytest.raises(RuntimeError, match="writer start failed"):
        run_replay(_config(source, result_dir))

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    error = json.loads((result_dir / "replay-error.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert error["error"] == "writer start failed"
