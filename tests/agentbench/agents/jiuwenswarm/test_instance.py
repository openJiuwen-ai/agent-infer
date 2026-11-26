# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import os
import socket
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agentinfer.agentbench.agents.jiuwenswarm.instance import JiuwenInstance
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


def _instance(tmp_path: Path) -> JiuwenInstance:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return JiuwenInstance(
        executable=Path("jiuwenswarm"),
        artifact_dir=tmp_path / "artifacts",
        workspace=workspace,
        session_id="root-session",
        api_base_url="http://127.0.0.1:18180",
        model="model",
        completion_timeout_seconds=14400,
    )


def _write_minimal_config(instance: JiuwenInstance) -> None:
    instance.config_path.parent.mkdir(parents=True)
    instance.config_path.write_text(
        yaml.safe_dump(
            {
                "models": {"defaults": [{"model_client_config": {}}]},
                "react": {
                    "model_name": "",
                    "subagents": {"browser_agent": {"enabled": True}},
                },
                "setup_guide": {"enabled": True},
                "auto_recap": {"enabled": True},
                "updater": {"enabled": True},
                "permissions": {},
                "modes": {
                    "code": {
                        "memory": {"enabled": True},
                        "rails": ["SkillUseRail", "WorktreeRail"],
                        "tools": ["web_free_search"],
                    }
                },
                "hooks": {"disable_all_hooks": False},
            }
        ),
        encoding="utf-8",
    )


def test_task_config_uses_identity_provider_and_disables_permissions(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    _write_minimal_config(instance)

    instance._configure()

    config = yaml.safe_load(instance.config_path.read_text(encoding="utf-8"))
    model = config["models"]["defaults"][0]["model_client_config"]
    assert model == {
        "api_base": "http://127.0.0.1:18180/v1",
        "api_key": "agentbench",
        "model_name": "model",
        "client_provider": "AgentBenchAffinity",
        "max_retries": 1,
        "timeout": 1800,
        "stream_first_chunk_timeout": 1500,
        "stream_idle_timeout": 1200,
    }
    extension_search_root = Path(config["extensions"]["extension_dirs"])
    assert extension_search_root.name == "agents"
    assert (extension_search_root / "jiuwenswarm" / "extension.py").is_file()
    assert config["permissions"] == {"enabled": False}
    assert config["react"]["completion_timeout"] == 14400
    assert config["kv_cache_affinity_config"] == {
        "enable_kv_cache_affinity": False,
    }
    assert config["react"]["subagent_runtime"] == {"enabled": True}
    assert config["modes"]["code"] == {
        "memory": {"enabled": False},
        "rails": [],
        "tools": [],
    }
    assert config["react"]["subagents"]["browser_agent"]["enabled"] is False
    assert config["hooks"]["disable_all_hooks"] is True


def test_task_config_writes_mandatory_sandbox_section(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    instance.jiuwenbox_url = "http://127.0.0.1:21004"
    _write_minimal_config(instance)

    instance._configure()

    config = yaml.safe_load(instance.config_path.read_text(encoding="utf-8"))
    assert config["sandbox"] == {
        "enabled": True,
        "url": instance.jiuwenbox_url,
        "type": "jiuwenbox",
        "startup_mode": "external",
    }
    assert config["sandbox"]["url"] == "http://127.0.0.1:21004"


def test_team_uses_same_model_workspace_policy_and_inprocess_runtime(tmp_path):
    instance = _instance(tmp_path)
    _write_minimal_config(instance)
    data = yaml.safe_load(instance.config_path.read_text())
    data["modes"]["team"] = {"jiuwen_team": {"agents": {"leader": "$agent_leader"}}}
    instance.config_path.write_text(yaml.safe_dump(data))
    instance._configure()
    config = yaml.safe_load(instance.config_path.read_text())
    team = config["modes"]["team"]["jiuwen_team"]
    assert team["spawn_mode"] == "inprocess"
    assert team["dispatch_mode"] == "autonomous"
    assert team["worktree"] == {"enabled": False}
    assert team["enable_swarmflow"] is False
    for role in ("leader", "teammate"):
        member = team["agents"][role]
        assert member["model"]["model_client_config"] == config["models"]["defaults"][0]["model_client_config"]
        assert member["enable_subagent_runtime"] is True
        assert member["sys_operation"]["mode"] == "sandbox"
        assert member["sys_operation"]["gateway_config"]["launcher_config"]["base_url"] == instance.jiuwenbox_url


def test_task_config_rejects_unexpected_mapping_shape(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    instance.config_path.parent.mkdir(parents=True)
    instance.config_path.write_text("models: []\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Jiuwen config models must be an object"):
        instance._configure()


def test_runtime_published_fallback_ports_update_gateway_url(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    env_path = instance.data_dir / "config" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "WEB_PORT=21001\nAGENT_SERVER_PORT=21002\nGATEWAY_PORT=21003\n",
        encoding="utf-8",
    )

    instance._refresh_runtime_ports()

    assert instance.gateway_url == "ws://127.0.0.1:21003/tui"
    assert instance.environment["AGENT_SERVER_PORT"] == "21002"


def test_init_is_explicitly_non_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)
    instance.artifact_dir.mkdir()
    captured = {}

    class Process:
        returncode = 0

        async def communicate(self):
            return b"initialized", b""

    async def create_process(*command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(instance_module.asyncio, "create_subprocess_exec", create_process)

    asyncio.run(instance._initialize())

    assert Path(captured["command"][-1]).name == "jiuwenswarm-init"
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.DEVNULL


@pytest.mark.skipif(sys.platform != "linux", reason="the WorkSwarm benchmark requires Linux abstract sockets")
def test_wait_ready_rejects_another_tasks_listener_and_follows_fallback(tmp_path, monkeypatch):
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)
    # Use actual listening children, with the same module names as WorkSwarm 0.2.8.
    package = tmp_path / "jiuwenswarm"
    package.mkdir()
    (package / "__init__.py").touch()
    with ExitStack() as stack:
        ports = []
        for role in ("gateway.app_gateway", "server.app_agentserver"):
            module = package / (role.replace(".", "/") + ".py")
            module.parent.mkdir()
            module.write_text(
                "import socket, sys\n"
                "with socket.create_server(('127.0.0.1', 0)) as sock:\n"
                "    print(sock.getsockname()[1], flush=True)\n"
                "    sys.stdin.read()\n"
            )
            child = stack.enter_context(
                subprocess.Popen(
                    [sys.executable, "-m", f"jiuwenswarm.{role}"],
                    cwd=tmp_path,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            )
            assert child.stdout is not None
            ports.append(int(child.stdout.readline()))

        gateway, agent_server = ports
        launcher = SimpleNamespace(pid=os.getpid(), returncode=None)
        foreign = stack.enter_context(socket.create_server(("127.0.0.1", 0)))
        instance.gateway_port = foreign.getsockname()[1]
        instance.agent_server_port = agent_server
        monkeypatch.setattr(instance, "_process", launcher)
        assert not instance_module._owns_runtime_ports(launcher.pid, instance.gateway_port, agent_server)
        assert not instance_module._owns_runtime_ports(launcher.pid, agent_server, gateway)

        async def check():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(instance._wait_ready(), 0.3)
            assert not await instance.interrupt()
            published = instance.data_dir / "config" / ".env"
            published.parent.mkdir(parents=True)
            published.write_text(f"GATEWAY_PORT={gateway}\n")
            previous_claim = instance_module._claim_gateway(gateway)
            assert previous_claim is not None
            with previous_claim, pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(instance._wait_ready(), 0.3)
            await asyncio.wait_for(instance._wait_ready(), 5)
            assert instance.gateway_url == f"ws://127.0.0.1:{gateway}/tui"
            assert instance_module._claim_gateway(gateway) is None

        try:
            asyncio.run(check())
        finally:
            monkeypatch.setattr(instance, "_process", None)
            asyncio.run(instance.stop())

        claim = instance_module._claim_gateway(gateway)
        assert claim is not None
        claim.close()


def test_extension_readiness_requires_the_owned_worker_marker(tmp_path, monkeypatch):
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)
    instance._runtime_dir = SimpleNamespace(name=str(tmp_path))
    instance._process = SimpleNamespace(pid=123, returncode=None)
    instance.environment["JIUWENSWARM_RUNTIME_SOCKET"] = str(tmp_path / "worker.sock")
    worker = SimpleNamespace(pid=456)
    monkeypatch.setattr(instance_module, "_service_children", lambda _: [])
    monkeypatch.setattr(instance_module, "_find_runtime_worker", lambda *_: (worker, sys.executable))
    monkeypatch.setattr(instance_module, "_STARTUP_TIMEOUT_SECONDS", 0.01)
    (tmp_path / "extension-123.ready").touch()

    with pytest.raises(RuntimeError, match="extension did not initialize"):
        asyncio.run(instance._wait_worker_extension())

    (tmp_path / "extension-456.ready").touch()
    asyncio.run(instance._wait_worker_extension())
    assert instance._runtime_worker == (worker, sys.executable)


def test_worker_stop_uses_worker_interpreter_and_falls_back(tmp_path, monkeypatch):
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)
    instance.artifact_dir.mkdir()
    instance.environment["JIUWENSWARM_RUNTIME_SOCKET"] = str(tmp_path / "worker.sock")
    worker = SimpleNamespace(pid=456)
    instance._runtime_worker = (worker, "/agent-env/bin/python3")
    monkeypatch.setattr(instance_module, "_worker_alive", lambda _: True)
    stopped_workers = []
    monkeypatch.setattr(instance_module, "_stop_owned_worker", stopped_workers.append)

    async def missing_interpreter(executable, *args, **kwargs):
        assert executable == "/agent-env/bin/python3"
        assert args[:2] == ("-I", "-c")
        assert kwargs["cwd"] == instance.artifact_dir
        raise FileNotFoundError("interpreter unavailable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_interpreter)
    asyncio.run(instance._stop_runtime_worker())

    assert stopped_workers == [worker]
    assert "interpreter unavailable" in (instance.artifact_dir / "jiuwenswarm-worker-stop.log").read_text()


@pytest.mark.timeout(20)
def test_stop_reaps_owned_services_and_keeps_logs(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    instance.artifact_dir.mkdir()
    log_path = instance.artifact_dir / "jiuwenswarm-services.stdout.log"

    async def exercise() -> None:
        processes: list[asyncio.subprocess.Process] = []

        async def start_process() -> asyncio.subprocess.Process:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import time; time.sleep(60)", start_new_session=os.name != "nt"
            )
            processes.append(process)
            return process

        with log_path.open("wb") as log:
            log.write(b"agent server started\n")
            instance._stdout = log
            try:
                server = await start_process()
                jiuwenbox = await start_process()
                sibling = await start_process()
                instance._process = server
                instance._jiuwenbox_process = jiuwenbox

                await instance.stop()

                assert server.returncode is not None
                assert jiuwenbox.returncode is not None
                assert sibling.returncode is None
                assert log.closed
            finally:
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()

    asyncio.run(exercise())
    assert log_path.read_bytes() == b"agent server started\n"


def test_jiuwenbox_server_uses_supported_listen_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)
    instance.artifact_dir.mkdir()
    captured = {}

    class Process:
        returncode = None

    async def create_process(*command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(instance_module, "_companion_executable", lambda *_args: "/venv/bin/jiuwenbox-server")
    monkeypatch.setattr(instance_module.shutil, "which", lambda name: name)
    monkeypatch.setattr(instance_module.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(instance_module, "_jiuwenbox_healthy", lambda _url: True)

    asyncio.run(instance._start_jiuwenbox_server())

    assert captured["command"] == (
        "/venv/bin/jiuwenbox-server",
        "--listen",
        instance.jiuwenbox_url,
        "--save-logs",
        str(instance.artifact_dir / "jiuwenbox-audit"),
    )
    assert captured["kwargs"]["cwd"] == instance.artifact_dir
    assert captured["kwargs"]["env"]["JIUWENBOX_POLICY_PATH"] == str(instance.jiuwenbox_policy_path)


@pytest.mark.parametrize("profile", ["agent.code.normal", "team.code.normal"])
def test_interrupt_sends_actual_profile(tmp_path, monkeypatch, profile):
    import json

    import websockets

    instance = _instance(tmp_path)
    instance.profile_name = profile
    sent = []

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            if not sent:
                return json.dumps({"type": "event", "event": "connection.ack"})
            return json.dumps({"type": "res", "id": sent[0]["id"], "ok": True, "payload": {"accepted": True}})

        async def send(self, message):
            sent.append(json.loads(message))

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: Connection())
    assert asyncio.run(instance._interrupt_once(5))
    assert sent[0]["params"]["mode"] == profile
