# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.jiuwenswarm.instance import JiuwenInstance


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
        "api_base": "http://127.0.0.1:18180",
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
    assert config["react"]["kv_cache_affinity_config"] == {
        "enable_kv_cache_affinity": False,
        "enable_kv_cache_release": False,
    }
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


def test_wait_ready_checks_both_runtime_ports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import instance as instance_module

    instance = _instance(tmp_path)

    class Process:
        returncode = None

    instance._process = Process()
    checked = []

    def port_open(port: int) -> bool:
        checked.append(port)
        return True

    monkeypatch.setattr(instance_module, "_port_open", port_open)

    asyncio.run(instance._wait_ready())

    assert checked == [instance.gateway_port, instance.agent_server_port]


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
