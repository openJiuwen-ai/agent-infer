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
    assert args[2] == 'terminal-bench/terminal-bench@4.1.0'
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


def test_prepare_accepts_official_tasks_with_default_zero_gpus(tmp_path: Path) -> None:
    harbor = tmp_path / "harbor"
    _fake_harbor(harbor)
    harbor.write_text(harbor.read_text().replace(r"\ngpus = 0\n", r"\n"))
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    assert prepared.selection_path.read_text().strip() == "terminal-bench__example"
    assert load_tasks(prepared.index_path, prepared.selection_path)[0].cpu_candidate


def test_prepare_and_import_official_reward(tmp_path: Path, monkeypatch) -> None:
    from agentinfer.agentbench.benchkit.cli import main

    harbor = tmp_path / "harbor"
    _fake_harbor(harbor)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    assert main(["prepare", "terminalbench", "--output-dir", str(tmp_path / "data-cli")]) == 0
    assert (tmp_path / "data-cli" / "instances.jsonl").is_file()
    assert (tmp_path / "data-cli" / "task-lists" / "selected.txt").is_file()
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    assert prepared.rows == 1
    assert prepared.selection_path == tmp_path / "data" / "task-lists" / "selected.txt"
    assert json.loads(prepared.manifest_path.read_text())["source"] == "terminal-bench/terminal-bench@4.1.0"
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


@pytest.mark.parametrize(
    ("agent_type", "header_name", "path"),
    [
        ("claude", "X-Claude-Code-Session-Id", "/v1/messages"),
        ("dsh", "X-DeepSeek-Harness-Session-Id", "/v1/chat/completions"),
    ],
)
def test_container_gateway_preserves_request_and_sets_task_identity(
    agent_type: str, header_name: str, path: str
) -> None:
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
            lambda reader, writer: _forward_model_request(
                reader, writer, "127.0.0.1", proxy_port, "task-session", agent_type
            ),
            "127.0.0.1",
            0,
        )
        try:
            port = gateway.sockets[0].getsockname()[1]
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}{path}",
                    content=b"hello",
                    headers={header_name: "native-session"},
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
    assert f"{header_name}: task-session\r\n".encode() in seen[0][0]
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


@pytest.fixture
def terminalbench_host(tmp_path: Path, monkeypatch):
    """Provide container prerequisites without host tmux or sandbox utilities."""

    def executable(path: Path, script: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n" + script + "\n")
        path.chmod(0o755)
        return path

    executable(tmp_path / "harbor", "echo 'harbor 0.25.0'")
    executable(
        tmp_path / "docker",
        'if [ "$1" = compose ]; then echo "Docker Compose version v2.40.0"; else echo 172.17.0.1; fi',
    )
    executable(tmp_path / "claude", "echo '2.1.0 (Claude Code)'")
    package = tmp_path / "dsh"
    executable(package / "lib" / "bin.js", "exit 99")
    (package / "package.json").write_text('{"version": "0.2.1-rc.2"}')
    executable(
        tmp_path / "node22",
        'if [ "$1" = --version ]; then echo v22.23.5; else echo 0.2.1-rc.2; fi',
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("DSH_NODE_EXECUTABLE", str(tmp_path / "node22"))
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(tmp_path / "harbor"))
    return tmp_path


@pytest.mark.parametrize("agent_type,profile", [("claude", "single"), ("dsh", "single"), ("dsh", "plan-subagent")])
def test_terminalbench_preflight_uses_container_requirements(terminalbench_host, monkeypatch, agent_type, profile):
    from agentinfer.agentbench.benchkit import runner

    root = terminalbench_host
    config = AgentBenchConfig.model_validate(
        {
            "dataset": {"name": "terminalbench"},
            "agent": {
                "type": agent_type,
                "profile": profile,
                "executable": root / ("claude" if agent_type == "claude" else "dsh/lib/bin.js"),
            },
            "backend": {"endpoint": "/v1/messages" if agent_type == "claude" else "/v1/chat/completions"},
        }
    )
    monkeypatch.setattr(runner, "check_agent_preflight", lambda *_args: pytest.fail("host sandbox probe was called"))
    urls = []

    def models(request):
        urls.append(str(request.url))
        return httpx.Response(200)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        runner.httpx, "AsyncClient", lambda **kwargs: client_class(transport=httpx.MockTransport(models), **kwargs)
    )
    report = asyncio.run(runner.check_preflight(config))
    assert urls == ["http://127.0.0.1:8000/v1/models"]
    assert report.runtime == agent_type
    assert report.versions[agent_type] == ("2.1.0" if agent_type == "claude" else "0.2.1-rc.2")
    assert report.capabilities["execution_environment"] == "docker"
    assert report.capabilities["container_setup_verified"] is False
    if agent_type == "dsh":
        assert report.installation["node_executable"] == str(root / "node22")


@pytest.mark.parametrize("agent_type", ["zcode", "jiuwenswarm"])
def test_terminalbench_rejects_missing_agent_adapters_before_host_probes(monkeypatch, agent_type):
    from agentinfer.agentbench.benchkit.runner import check_preflight

    config = AgentBenchConfig.model_validate(
        {
            "dataset": {"name": "terminalbench"},
            "agent": {"type": agent_type, "profile": "single" if agent_type == "zcode" else "code.normal"},
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )
    monkeypatch.setattr(terminalbench.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("host probe was called"))
    message = "zcode: a Harbor agent adapter is required" if agent_type == "zcode" else "Terminal-Bench supports"
    with pytest.raises(ValueError, match=message):
        asyncio.run(check_preflight(config))


@pytest.mark.parametrize("missing", ["harbor", "docker", "claude"])
def test_terminalbench_preflight_reports_missing_host_dependency(terminalbench_host, missing):
    root = terminalbench_host
    (root / missing).unlink()
    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    config.agent.executable = root / "claude"
    with pytest.raises(
        (FileNotFoundError, RuntimeError), match={"harbor": "Harbor", "docker": "Docker", "claude": "Claude"}[missing]
    ):
        terminalbench.check_terminalbench_preflight(config)


def test_terminalbench_preflight_reports_docker_daemon_failure(terminalbench_host):
    root = terminalbench_host
    (root / "docker").write_text(
        '#!/bin/sh\nif [ "$1" = compose ]; then echo v2; else echo "daemon permission denied" >&2; exit 1; fi\n'
    )
    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    config.agent.executable = root / "claude"
    with pytest.raises(RuntimeError, match="Docker bridge.*daemon permission denied"):
        terminalbench.check_terminalbench_preflight(config)


@pytest.mark.parametrize("failure", ["package", "node", "dsh"])
def test_terminalbench_dsh_preflight_rejects_unusable_container_runtime(terminalbench_host, failure):
    root = terminalbench_host
    if failure == "package":
        (root / "dsh" / "package.json").write_text('{"version": "0.1.0"}')
    elif failure == "node":
        (root / "node22").write_text("#!/bin/sh\necho v20.19.0\n")
    else:
        (root / "node22").write_text(
            '#!/bin/sh\nif [ "$1" = --version ]; then echo v22.23.5; else echo "DSH startup failed" >&2; exit 1; fi\n'
        )
    config = AgentBenchConfig.model_validate(
        {
            "dataset": {"name": "terminalbench"},
            "agent": {"type": "dsh", "executable": root / "dsh/lib/bin.js"},
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )
    message = {"package": "DSH executable", "node": "Node.js >=22.19", "dsh": "DSH startup failed"}[failure]
    with pytest.raises((ValueError, RuntimeError), match=message):
        terminalbench.check_terminalbench_preflight(config)


def test_terminalbench_preflight_bounds_probe_timeouts(monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 10
        raise terminalbench.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(terminalbench.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Docker Compose CLI plugin.*TimeoutExpired"):
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


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
def test_dsh_harbor_command_uses_container_agent_and_chat_endpoint(tmp_path: Path, monkeypatch, profile: str) -> None:
    harbor = tmp_path / "harbor"
    _fake_harbor(harbor)
    prepared = prepare_terminalbench(tmp_path / "data", harbor_executable=str(harbor))
    task = load_tasks(prepared.index_path, prepared.selection_path)[0]
    package = tmp_path / "node_modules" / "@deepseek-ai" / "dsh"
    (package / "lib").mkdir(parents=True)
    (package / "package.json").write_text('{"version":"0.2.1-rc.2"}')
    executable = package / "lib" / "bin.js"
    executable.write_text("#!/usr/bin/env node\n")
    executable.chmod(0o755)
    node = tmp_path / "node22"
    node.write_text("#!/bin/sh\necho v22.23.5\n")
    node.chmod(0o755)
    monkeypatch.setenv("DSH_NODE_EXECUTABLE", str(node))
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.terminalbench._docker_gateway", lambda: "127.0.0.1")
    config = AgentBenchConfig.model_validate(
        {
            "dataset": {"name": "terminalbench"},
            "agent": {"type": "dsh", "profile": profile, "executable": executable, "context_window": 65536},
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = RunContext(config, run_dir, run_dir / "requests.jsonl", (task,))
    result = asyncio.run(_run_one_trial(context, task, "http://127.0.0.1:9", 0))
    command = json.loads((run_dir / "tasks" / task.instance_id / "harbor-command.json").read_text())
    assert result.outcome.value == "completed"
    assert "agentinfer.agentbench.benchkit.harbor_dsh:DshHarborAgent" in command
    assert f"profile={profile}" in command
    assert "version=0.2.1-rc.2" in command
    assert any(value.startswith("task_session_id=") for value in command)
    assert f"host_dsh_package={package}" in command
    assert f"host_node_executable={node}" in command
    assert "context_window=65536" in command
    assert "max_tokens=8192" in command
    assert "--base-url" not in command
    assert any(value.startswith("proxy_url=http://127.0.0.1:") for value in command)
