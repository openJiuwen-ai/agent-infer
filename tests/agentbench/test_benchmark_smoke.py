# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Hermetic end-to-end guards for the AgentBench benchmark contract."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.registry import RUNTIMES
from agentinfer.agentbench.benchkit.cli import main
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from agentinfer.agentbench.benchkit.runner import run_benchmark

from .helpers.benchmark_smoke import FakeAgentRuntime, FakeModelServer, ThreadBarrier, ThreadRequestProxyProcess

pytestmark = [pytest.mark.cpu_test, pytest.mark.benchmark_smoke]


@pytest.fixture(autouse=True)
def _threaded_proxy_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.request_proxy.lifecycle.RequestProxyProcess",
        ThreadRequestProxyProcess,
    )


def _seed_benchmark(tmp_path: Path, task_ids: tuple[str, ...]) -> tuple[Path, Path, Path]:
    """Create a benchmark fixture repo, task index, and task selection.

    Side effects: builds a one-commit git repo under ``tmp_path/source``,
    copies it into the workspace cache, and writes instances.jsonl +
    tasks.txt. Returns ``(index_path, selection_path, cache_dir)``.
    """
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "base"],
        cwd=source,
        check=True,
    )
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()

    cache = tmp_path / "cache" / "owner__repo"
    cache.parent.mkdir()
    shutil.copytree(source, cache)
    index_path = tmp_path / "instances.jsonl"
    rows = [
        {
            "instance_id": task_id,
            "repo": "owner/repo",
            "base_commit": base_commit,
            "problem_statement": f"fix {task_id}",
        }
        for task_id in task_ids
    ]
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selection_path = tmp_path / "tasks.txt"
    selection_path.write_text("".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8")
    return index_path, selection_path, cache.parent


def _config(
    tmp_path: Path,
    backend: FakeModelServer,
    task_ids: tuple[str, ...],
    *,
    run_name: str,
) -> AgentBenchConfig:
    """Build an AgentBenchConfig wired to the fake backend and seeded tasks."""
    index_path, selection_path, cache_dir = _seed_benchmark(tmp_path, task_ids)
    return AgentBenchConfig.model_validate(
        {
            "experiment": {
                "result_dir": str(tmp_path / run_name),
                "max_concurrency": len(task_ids),
                "task_timeout_seconds": 10,
            },
            "dataset": {
                "index_path": str(index_path),
                "selection_path": str(selection_path),
                "cache_dir": str(cache_dir),
            },
            "agent": {
                "type": "jiuwenswarm",
                "profile": "code.normal",
                "executable": "unused-fake-agent",
            },
            "backend": {
                "base_url": backend.base_url,
                "metrics_url": f"{backend.base_url}/metrics",
                "model": "fake-model",
                "endpoint": "/v1/chat/completions",
            },
            "request_proxy": {
                "startup_timeout_seconds": 10,
                "shutdown_timeout_seconds": 10,
            },
        }
    )


def _write_config(config: AgentBenchConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_cli_runs_full_contract_with_fake_agent_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeAgentRuntime()
    monkeypatch.setitem(RUNTIMES, "jiuwenswarm", runtime)
    with FakeModelServer() as backend:
        config = _config(tmp_path, backend, ("task-a",), run_name="run")
        config_path = tmp_path / "benchmark.yaml"
        _write_config(config, config_path)

        assert main(["run", "--config", str(config_path)]) == 0

    run_dir = tmp_path / "run"
    result = _json(run_dir / "tasks" / "task-a" / "result.json")
    task_index = _json(run_dir / "task_index.json")
    summary = _json(run_dir / "summary.json")
    manifest = _json(run_dir / "manifest.json")
    traces = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(runtime.requests) == 1
    session_id = result["session_id"]
    expected_identity = {
        "program_id": f"{session_id}:lead",
        "task_id": session_id,
        "session_id": session_id,
        "agent_id": "lead",
        "blocks_parent": False,
        "expected_resume": True,
        "agent_role": "lead",
    }
    assert backend.requests[0]["model"] == "fake-model"
    assert backend.requests[0]["vllm_xargs"] == {
        "agentic_context": json.dumps(expected_identity, separators=(",", ":"))
    }
    assert len(traces) == 1
    assert (traces[0]["session_id"], traces[0]["actor_id"], traces[0]["input_tokens"]) == (
        session_id,
        "lead",
        7,
    )
    assert (run_dir / "workspaces" / "task-a" / "tracked.txt").read_text() == "changed by task-a\n"
    assert "+changed by task-a" in (run_dir / "tasks" / "task-a" / "model.patch").read_text(encoding="utf-8")
    assert result["outcome"] == "completed"
    assert result["has_patch"] is True
    assert result["topology"] == {"session_id": session_id, "agent_ids": ["lead"], "agent_roles": {"lead": "lead"}}
    assert task_index["tasks"][0]["instance_id"] == "task-a"
    assert task_index["tasks"][0]["session_id"] == session_id
    assert summary["tasks"]["completed"] == 1
    assert summary["requests"]["requests"] == 1
    assert summary["vllm"]["counters"]["vllm:prompt_tokens_total"] == 7.0
    assert summary["lifecycle"]["proxy_close"]["graceful"] is True
    assert summary["lifecycle"]["proxy_close"]["trace_health"]["pending"] == 0
    assert manifest["status"] == "completed"
    trace_evidence = next(row for row in manifest["evidence"] if row["source"] == "request_trace")
    assert trace_evidence["available"] is True


def test_two_tasks_overlap_keep_identity_and_finalize_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise(backend: FakeModelServer) -> FakeAgentRuntime:
        runtime = FakeAgentRuntime(
            barrier=ThreadBarrier(asyncio.get_running_loop()),
            delays={"task-a": 0.5},
            failed_tasks={"task-b"},
        )
        monkeypatch.setitem(RUNTIMES, "jiuwenswarm", runtime)
        config = _config(tmp_path, backend, ("task-a", "task-b"), run_name="run-concurrent")
        await run_benchmark(config)
        return runtime

    with FakeModelServer() as backend:
        runtime = asyncio.run(exercise(backend))
    run_dir = tmp_path / "run-concurrent"
    first = _json(run_dir / "tasks" / "task-a" / "result.json")
    second = _json(run_dir / "tasks" / "task-b" / "result.json")
    task_index = _json(run_dir / "task_index.json")
    summary = _json(run_dir / "summary.json")
    manifest = _json(run_dir / "manifest.json")
    traces = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]

    assert runtime.completion_order == ["task-b", "task-a"]
    assert {request.task.instance_id for request in runtime.requests} == {"task-a", "task-b"}
    assert first["session_id"] != second["session_id"]
    assert first["outcome"] == "completed"
    assert second["outcome"] == "failed"
    assert second["termination_reason"] == "interrupted"
    assert second["error"] is None
    assert [row["instance_id"] for row in task_index["tasks"]] == ["task-a", "task-b"]
    assert [row["task_position"] for row in task_index["tasks"]] == [0, 1]
    assert summary["tasks"]["completed"] == 1
    assert summary["tasks"]["failed"] == 1
    assert summary["tasks"]["with_patch"] == 2
    assert summary["requests"]["requests"] == 2
    assert {row["session_id"] for row in traces} == {first["session_id"], second["session_id"]}
    assert {body["messages"][0]["content"] for body in backend.requests} == {"fix task-a", "fix task-b"}
    assert (run_dir / "workspaces" / "task-a" / "tracked.txt").read_text() == "changed by task-a\n"
    assert (run_dir / "workspaces" / "task-b" / "tracked.txt").read_text() == "changed by task-b\n"
    assert manifest["status"] == "completed"
