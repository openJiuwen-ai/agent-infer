# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise the Terminal-Bench boundary with a fake Harbor executable."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentinfer.agentbench.benchkit import terminalbench
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from agentinfer.agentbench.benchkit.dataset import Task, load_tasks
from agentinfer.agentbench.benchkit.runner import RunContext
from agentinfer.agentbench.benchkit.terminalbench import (
    _forward_model_request,
    _run_one_trial,
    prepare_terminalbench,
    run_terminalbench_tasks,
)

from .helpers.benchmark_smoke import FakeModelServer, ThreadRequestProxyProcess


def _fake_harbor(path: Path, *, reward: bool = True, timeout: bool = False) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
if args == ['--version']:
    print('claude 1.2.3')
elif args[:2] == ['datasets', 'download']:
    root = pathlib.Path(args[args.index('--output-dir') + 1]) / 'terminal-bench' / 'example'
    (root / 'environment').mkdir(parents=True)
    (root / 'tests').mkdir()
    (root / 'task.toml').write_text('[task]\\nname = "terminal-bench/example"\\n[agent]\\ntimeout_sec = 28800\\n[environment]\\ngpus = 0\\n')
    (root / 'instruction.md').write_text('Create a file in /app.\\n')
    (root / 'environment' / 'Dockerfile').write_text('FROM python:3.12-slim\\n')
    (root / 'tests' / 'test.sh').write_text('exit 0\\n')
elif args[0] == 'run':
    out = pathlib.Path(args[args.index('--jobs-dir') + 1]) / args[args.index('--job-name') + 1] / 'trial'
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps({'task_name': 'terminal-bench/example', 'verifier_result': {'rewards': {'reward': 1}}, 'exception_info': None}))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    if not reward:
        path.write_text(path.read_text().replace("{'rewards': {'reward': 1}}", "{'rewards': None}"))
    if timeout:
        path.write_text(
            path.read_text().replace(
                "'exception_info': None", "'exception_info': {'exception_type': 'AgentTimeoutError'}"
            )
        )
    path.chmod(0o755)


def test_prepare_and_import_official_reward(tmp_path: Path, monkeypatch) -> None:
    from agentinfer.agentbench.benchkit.cli import main

    harbor = tmp_path / "harbor"
    _fake_harbor(harbor)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    assert main(["prepare", "terminalbench", "--output-dir", str(tmp_path / "data-cli")]) == 0
    assert (tmp_path / "data-cli" / "instances.jsonl").is_file()
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    assert prepared.rows == 1
    assert json.loads(prepared.manifest_path.read_text())["source"] == "terminal-bench/terminal-bench@4.0.0"
    assert prepared.selection_path.read_text().strip() == "terminal-bench__example"
    tasks = load_tasks(prepared.index_path, prepared.selection_path)
    assert len(tasks) == 1
    assert tasks[0].task_path is not None

    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
    config.agent.executable = harbor
    config.experiment.task_timeout_seconds = 28800
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = RunContext(config, run_dir, run_dir / "requests.jsonl", tuple(tasks))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.terminalbench._docker_gateway", lambda: "127.0.0.1")
    result = asyncio.run(_run_one_trial(context, tasks[0], "http://127.0.0.1:9", 0))
    assert result.outcome.value == "completed"
    assert json.loads((run_dir / "tasks" / tasks[0].instance_id / "verifier.json").read_text()) == {
        "available": True,
        "rewards": {"reward": 1},
        "exception": None,
    }
    command = json.loads((run_dir / "tasks" / tasks[0].instance_id / "harbor-command.json").read_text())
    assert "--base-url" in command and "--path" in command and "--skip-model-parsing-and-provider-mapping" in command
    assert "version=1.2.3" in command
    assert "--agent-timeout-multiplier" not in command


def test_container_gateway_preserves_request_and_sets_task_identity() -> None:
    seen = []

    async def exercise() -> None:
        async def upstream(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            body = await reader.readexactly(5)
            seen.append((headers, body))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
            await writer.drain()
            writer.close()

        proxy = await asyncio.start_server(upstream, "127.0.0.1", 0)
        proxy_port = proxy.sockets[0].getsockname()[1]
        gateway = await asyncio.start_server(
            lambda reader, writer: _forward_model_request(reader, writer, "127.0.0.1", proxy_port, "task-session"),
            "127.0.0.1",
            0,
        )
        try:
            port = gateway.sockets[0].getsockname()[1]
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    content=b"hello",
                    headers={"X-Claude-Code-Session-Id": "native-session"},
                )
            assert response.text == "OK"
        finally:
            gateway.close()
            proxy.close()
            await gateway.wait_closed()
            await proxy.wait_closed()

    asyncio.run(exercise())
    assert len(seen) == 1
    assert seen[0][1] == b"hello"
    assert b"X-Claude-Code-Session-Id: task-session\r\n" in seen[0][0]
    assert b"native-session" not in seen[0][0]


def test_run_imports_reward_into_finalized_summary(tmp_path: Path, monkeypatch) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    harbor = tmp_path / "harbor"
    _fake_harbor(harbor)
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.terminalbench._docker_gateway", lambda: "127.0.0.1")
    monkeypatch.setattr("agentinfer.agentbench.request_proxy.lifecycle.RequestProxyProcess", ThreadRequestProxyProcess)

    async def preflight(_config):
        return None

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.check_preflight", preflight)
    with FakeModelServer(tmp_path / "backend") as backend:
        config = AgentBenchConfig()
        config.dataset.name = "terminalbench"
        config.dataset.index_path = prepared.index_path
        config.dataset.selection_path = prepared.selection_path
        monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
        config.agent.executable = harbor
        config.experiment.result_dir = tmp_path / "run"
        config.backend.base_url = backend.base_url
        config.backend.metrics_url = f"{backend.base_url}/metrics"
        asyncio.run(run_benchmark(config))

    correctness = json.loads((tmp_path / "run" / "correctness.json").read_text())
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert correctness["trials"][0]["rewards"] == {"reward": 1}
    assert correctness["evaluation_coverage"] == 1.0
    assert correctness["custom_timeout"] is True
    assert correctness["agent_timeout_seconds"] == 3600
    command = json.loads((tmp_path / "run" / "tasks" / "terminal-bench__example" / "harbor-command.json").read_text())
    assert "--agent-timeout-multiplier" in command
    assert summary["correctness"]["available"] is True
    assert summary["correctness"]["metadata"]["evaluation_coverage"] == 1.0
    assert summary["tasks"]["completed"] == 1


def test_missing_reward_is_not_reported_as_correctness(tmp_path: Path, monkeypatch) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    harbor = tmp_path / "harbor"
    _fake_harbor(harbor, reward=False)
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.terminalbench._docker_gateway", lambda: "127.0.0.1")
    monkeypatch.setattr("agentinfer.agentbench.request_proxy.lifecycle.RequestProxyProcess", ThreadRequestProxyProcess)

    async def preflight(_config):
        return None

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.check_preflight", preflight)
    with FakeModelServer(tmp_path / "backend") as backend:
        config = AgentBenchConfig()
        config.dataset.name = "terminalbench"
        config.dataset.index_path = prepared.index_path
        config.dataset.selection_path = prepared.selection_path
        config.agent.executable = harbor
        config.experiment.result_dir = tmp_path / "run"
        config.backend.base_url = backend.base_url
        config.backend.metrics_url = f"{backend.base_url}/metrics"
        asyncio.run(run_benchmark(config))

    correctness = json.loads((tmp_path / "run" / "correctness.json").read_text())
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert correctness["evaluation_coverage"] == 0.0
    assert summary["correctness"]["available"] is False
    assert summary["correctness"]["reason"] == "0/1 trials have valid rewards"


def test_missing_docker_compose_reports_clear_preflight_error(monkeypatch) -> None:
    monkeypatch.setattr(terminalbench.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="Docker Compose CLI plugin"):
        terminalbench._docker_gateway()


def test_unsupported_task_is_rejected_before_harbor_starts(tmp_path: Path) -> None:
    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    task = Task("outside", "", "", "", task_path=tmp_path, cpu_candidate=False)
    context = RunContext(config, tmp_path, tmp_path / "requests.jsonl", (task,))
    with pytest.raises(ValueError, match="outside the supported CPU candidate subset"):
        asyncio.run(run_terminalbench_tasks(context, "http://127.0.0.1:1", []))


def test_timeout_can_have_a_valid_official_reward(tmp_path: Path, monkeypatch) -> None:
    harbor = tmp_path / "harbor"
    _fake_harbor(harbor, timeout=True)
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    task = load_tasks(prepared.index_path, prepared.selection_path)[0]
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.terminalbench._docker_gateway", lambda: "127.0.0.1")
    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    config.agent.executable = harbor
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = RunContext(config, run_dir, run_dir / "requests.jsonl", (task,))
    result = asyncio.run(_run_one_trial(context, task, "http://127.0.0.1:9", 0))
    verifier = json.loads((run_dir / "tasks" / task.instance_id / "verifier.json").read_text())
    assert result.outcome.value == "failed"
    assert result.termination_reason.value == "timeout"
    assert verifier["available"] is True
    assert verifier["rewards"] == {"reward": 1}
