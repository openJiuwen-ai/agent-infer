# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from urllib.parse import urlsplit

import httpx
import pytest

from agentinfer.agentbench.benchkit import runner, terminalbench, terminalbench_images
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from agentinfer.agentbench.request_proxy import lifecycle
from agentinfer.scheduling.headers import CLAUDE_SESSION_HEADER, DSH_SESSION_HEADER
from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity
from tests.helpers.marks import test_marks

from .helpers.mock_benchmark import FakeModelServer, ThreadRequestProxyProcess

pytestmark = test_marks(hardware="cpu", ci="ready")


@pytest.fixture
def proxy_process(monkeypatch):
    monkeypatch.setattr(lifecycle, "RequestProxyProcess", ThreadRequestProxyProcess)


@pytest.mark.parametrize("agent_type", ["claude", "dsh", "jiuwenswarm"])
def test_gateway_attributes_multiple_requests(tmp_path, proxy_process, agent_type):
    with FakeModelServer(tmp_path / "backend") as backend:

        async def exercise():
            config = AgentBenchConfig()
            endpoint = "/v1/chat/completions"
            proxy = lifecycle.create_request_proxy_lifecycle(
                config.request_proxy, backend.base_url, "gateway-test", tmp_path, endpoint
            )
            handle = await proxy.start()
            connections = 0
            try:
                await proxy.wait_ready(10)
                upstream = urlsplit(handle.base_url)

                async def forward(reader, writer):
                    nonlocal connections
                    connections += 1
                    await terminalbench._forward_model_request(
                        reader, writer, upstream.hostname, upstream.port, "trial", agent_type
                    )

                gateway = await asyncio.start_server(forward, "127.0.0.1", 0)
                async with gateway, httpx.AsyncClient(trust_env=False) as client:
                    url = f"http://127.0.0.1:{gateway.sockets[0].getsockname()[1]}{endpoint}"
                    for i in range(2):
                        header = DSH_SESSION_HEADER if agent_type == "dsh" else CLAUDE_SESSION_HEADER
                        body = {"model": "fake-model", "messages": [{"role": "user", "content": str(i)}]}
                        if agent_type == "jiuwenswarm":
                            body["vllm_xargs"] = {
                                "agentic_context": encode_agent_identity(
                                    AgentIdentity(program_id=f"actor-{i}", task_id="trial", agent_id=f"actor-{i}")
                                )
                            }
                        response = await client.post(url, headers={header: "native"}, json=body)
                        assert response.status_code == 200
            finally:
                await proxy.close()
            return connections

        assert asyncio.run(exercise()) == (1 if agent_type == "jiuwenswarm" else 2)
    facts = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [fact["session_id"] for fact in facts] == ["trial", "trial"]
    if agent_type == "jiuwenswarm":
        assert [fact["actor_id"] for fact in facts] == ["actor-0", "actor-1"]


@pytest.fixture
def prepared_export(tmp_path, monkeypatch):
    harbor = tmp_path / "harbor"
    harbor.write_text("""#!/usr/bin/env python3
import json
import pathlib
import sys
import time

args = sys.argv[1:]
if args == ['--version']:
    print('version 0.25.0')
elif args[:2] == ['datasets', 'download']:
    assert args[2] == 'terminal-bench/terminal-bench@4.1.0'
    root = pathlib.Path(args[args.index('--output-dir') + 1])
    for name in ('a', 'b', 'c'):
        task = root / name
        task.mkdir(parents=True)
        (task / 'task.toml').write_text("[metadata]\\nname = 'meta'\\n[task]\\nname = 'terminal-bench/" + name + "'\\n")
        (task / 'instruction.md').write_text('Write a file.')
else:
    target = pathlib.Path(args[args.index('--path') + 1])
    marker = target / 'fault'
    mode = marker.read_text() if marker.exists() else ''
    if mode == 'cancel':
        (target / 'started').touch()
        time.sleep(60)
    if mode == 'fail':
        raise SystemExit(17)
    job = args[args.index('--job-name') + 1]
    out = pathlib.Path(args[args.index('--jobs-dir') + 1]) / job / 'trial'
    out.mkdir(parents=True)
    rewards = None if '--disable-verification' in args else {'rewards': {'reward': 0}}
    (out / 'result.json').write_text(json.dumps({
        'task_name': job, 'verifier_result': rewards,
        'exception_info': {'exception_type': 'CancelledError'} if mode == 'cancel-scored' else None,
    }))
    if mode == 'cancel-scored':
        (target / 'started').touch()
        time.sleep(60)
""")
    harbor.chmod(0o755)
    monkeypatch.setenv("HARBOR_EXECUTABLE", str(harbor))
    return harbor, terminalbench.prepare_terminalbench(tmp_path / "dataset", harbor_executable=str(harbor))


@pytest.mark.parametrize("mode", ["unscored", "success", "fail", "cancel", "cancel-scored", "write-fails"])
def test_run_preserves_correctness_and_closes_task_gateway(
    tmp_path, monkeypatch, capsys, proxy_process, prepared_export, mode
):
    harbor, prepared = prepared_export
    assert prepared.rows == 3
    config = AgentBenchConfig()
    config.dataset.name = "terminalbench"
    config.dataset.index_path = prepared.index_path
    config.dataset.selection_path = prepared.selection_path
    config.agent.executable = harbor
    config.experiment.result_dir = tmp_path / "run"
    paths = tmp_path / "dataset/tasks"
    if mode in {"fail", "cancel", "cancel-scored"}:
        (paths / "b/fault").write_text(mode)
    monkeypatch.setattr(terminalbench, "_docker_gateway", lambda: "127.0.0.1")
    monkeypatch.setattr(terminalbench, "validate_claude_executable", lambda *_: None)
    monkeypatch.setattr(terminalbench_images, "docker_platform", lambda: "linux/amd64")
    monkeypatch.setattr(terminalbench_images, "prepare_task_images", lambda *_, **_kwargs: None)
    listeners = []
    start_server = asyncio.start_server

    async def record_listener(*args, **kwargs):
        listener = await start_server(*args, **kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(asyncio, "start_server", record_listener)
    if mode == "write-fails":
        write = terminalbench.atomic_write_json

        def fail_command(path, value):
            if path.parent.name == "terminal-bench__b" and path.name == "harbor-command.json":
                raise OSError("cannot record command")
            write(path, value)

        monkeypatch.setattr(terminalbench, "atomic_write_json", fail_command)

    async def exercise():
        options = {} if mode == "unscored" else {"with_correctness": True}
        batch = asyncio.create_task(runner.run_benchmark(config, **options))
        if mode in {"cancel", "cancel-scored"}:

            async def wait_for_harbor():
                while not (paths / "b/started").exists():
                    if batch.done():
                        await batch
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_for_harbor(), 10)
            batch.cancel()
        await batch

    with FakeModelServer(tmp_path / "backend") as backend:
        config.backend.base_url = backend.base_url
        expected_error = {
            "cancel": asyncio.CancelledError,
            "cancel-scored": asyncio.CancelledError,
            "fail": RuntimeError,
            "write-fails": OSError,
        }.get(mode)
        if expected_error:
            with pytest.raises(expected_error):
                asyncio.run(exercise())
        else:
            asyncio.run(exercise())

    assert listeners and not any(listener.is_serving() for listener in listeners)
    output = config.experiment.result_dir
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["with_correctness"] == (mode != "unscored")
    directory = output / "tasks/terminal-bench__a"
    command = json.loads((directory / "harbor-command.json").read_text())
    assert ("--disable-verification" in command) == (mode == "unscored")
    if mode == "unscored":
        assert summary["tasks"]["completed"] == 3
        assert summary["correctness"]["available"] is False
        assert summary["correctness"]["reason"] == "skipped"
        assert not (output / "correctness.json").exists()
        assert not (directory / "verifier.json").exists()
        return
    result = json.loads((output / "correctness.json").read_text())
    assert result["planned_trials"] == 3
    assert result["scored_trials"] == {"cancel": 1, "success": 3}.get(mode, 2)
    assert result["trials"][0]["rewards"] == {"reward": 0}
    assert summary["correctness"]["available"] == (mode == "success")
    assert summary["correctness"]["metadata"]["scored_trials"] == result["scored_trials"]
    assert "running=a" in capsys.readouterr().err
    if expected_error:
        index = json.loads((output / "task_index.json").read_text())
        assert index["tasks"][1]["outcome"] == "failed"
        assert result["trials"][1]["available"] == (mode == "cancel-scored")
    if mode in {"cancel", "cancel-scored"}:
        assert result["trials"][2]["exception"] == {"exception_type": "NotRun"}
    if mode == "cancel-scored":
        assert result["trials"][1]["rewards"] == {"reward": 0}
        assert result["trials"][1]["exception"]["exception_type"] == "CancelledError"


def test_claude_harbor_host_environment_only_contains_local_model_credentials(monkeypatch):
    monkeypatch.setenv("DOCKER_CONTEXT", "benchmark")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-host-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-host-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-oauth-token")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-cloud-key")
    env = terminalbench._harbor_env("claude")
    assert env["DOCKER_CONTEXT"] == "benchmark"
    assert env["ANTHROPIC_API_KEY"] == "agentbench"
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert (
        not {"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "AWS_ACCESS_KEY_ID"}
        & env.keys()
    )
