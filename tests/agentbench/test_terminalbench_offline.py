# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise prepared-image startup against the installed Harbor interfaces."""

import asyncio
import importlib
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.benchkit.terminalbench_images import local_build_image, sidecar_specs
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")
pytest.importorskip("harbor")

DockerEnvironment = importlib.import_module("harbor.environments.docker.docker").DockerEnvironment
EnvironmentConfig = importlib.import_module("harbor.models.task.config").EnvironmentConfig
TrialPaths = importlib.import_module("harbor.models.trial.paths").TrialPaths
harbor_claude = importlib.import_module("agentinfer.agentbench.benchkit.harbor_claude")
harbor_dsh = importlib.import_module("agentinfer.agentbench.benchkit.harbor_dsh")
harbor_environment = importlib.import_module("agentinfer.agentbench.benchkit.harbor_environment")
harbor_jiuwenswarm = importlib.import_module("agentinfer.agentbench.benchkit.harbor_jiuwenswarm")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    directory = tmp_path / "environment"
    directory.mkdir()
    (directory / "Dockerfile").write_text("FROM example:base\n")
    monkeypatch.setattr(harbor_environment, "docker_platform", lambda: "linux/amd64")
    monkeypatch.setattr(
        harbor_environment.PreparedDockerEnvironment, "_egress_control_kernel_support", staticmethod(lambda: False)
    )
    return harbor_environment.PreparedDockerEnvironment(
        environment_dir=directory,
        environment_name="task",
        session_id="offline-trial",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="example:prepared"),
    )


def test_dockerfile_run_uses_exact_prepared_image_without_build(environment, monkeypatch):
    environment.task_env_config.docker_image = None
    expected = local_build_image(environment.environment_dir, "linux/amd64")
    called = []

    async def start(self, force_build):
        called.append((self.task_env_config.docker_image, self._env_vars.prebuilt_image_name, force_build))

    monkeypatch.setattr(DockerEnvironment, "start", start)
    asyncio.run(environment.start(force_build=False))
    assert called == [(expected, expected, False)]


def test_compose_start_never_pulls_or_builds(environment, monkeypatch):
    commands = []

    async def run(self, command):
        commands.append(command)

    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", run)
    asyncio.run(environment._run_docker_compose_command(["up", "--detach", "--wait"]))
    asyncio.run(environment._run_docker_compose_command(["down", "--rmi", "all"]))
    assert commands == [["up", "--pull", "never", "--no-build", "--detach", "--wait"], ["down"]]


def test_harbor_egress_sidecar_uses_only_the_prepared_image(environment, monkeypatch):
    async def platform():
        return "linux/amd64"

    monkeypatch.setattr(harbor_environment, "default_docker_platform", platform)
    spec = harbor_environment.sidecar_image_spec("linux/amd64")
    monkeypatch.setattr(harbor_environment, "inspect_local_image", lambda image: None)
    with pytest.raises(RuntimeError, match="Prepared Harbor egress image missing"):
        asyncio.run(environment._ensure_egress_control_sidecar_image_built())
    monkeypatch.setattr(harbor_environment, "inspect_local_image", lambda image: {} if image == spec.image else None)
    asyncio.run(environment._ensure_egress_control_sidecar_image_built())
    assert environment._env_vars.egress_control_sidecar_image_name == spec.image


def test_sidecar_specs_query_the_harbor_installation():
    harbor = shutil.which(os.environ.get("HARBOR_EXECUTABLE", "harbor"))
    if harbor is None:
        pytest.skip("The Harbor CLI is required for the installed sidecar probe")
    sidecar, probe = sidecar_specs("linux/amd64", harbor)
    assert sidecar == harbor_environment.sidecar_image_spec("linux/amd64")
    assert probe.image == DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE


def test_claude_deploys_local_binary_and_disables_updates(tmp_path):
    native = tmp_path / "claude"
    native.write_bytes(b"prepared native cli")
    commands = []
    uploads = {}
    agent = harbor_claude.LocalClaudeCode(
        logs_dir=tmp_path,
        model_name="model",
        skip_model_parsing=True,
        base_url="http://proxy",
        version="2.1.0",
        host_executable=str(native),
    )
    assert agent._resolve_auth_env() == {"ANTHROPIC_BASE_URL": "http://proxy", "ANTHROPIC_API_KEY": "agentbench"}

    class Environment:
        default_user = None

        async def exec(self, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="2.1.0 (Claude Code)", stderr="")

        async def upload_file(self, source, destination):
            uploads[destination] = Path(source).read_bytes()

    asyncio.run(agent.install(Environment()))
    assert uploads[harbor_claude.REMOTE_BINARY] == native.read_bytes()
    assert b"DISABLE_AUTOUPDATER=1" in uploads["/usr/local/bin/claude"]
    assert all(re.search(r"\b(curl|npm|pip|apt|wget)\b", command) is None for command in commands)
    with pytest.raises(RuntimeError, match="ACP runtime downloads"):
        asyncio.run(agent.acp_install(Environment()))


@pytest.mark.parametrize(("version", "supported"), [("v23.0.0", False), ("v24.0.0", True)])
def test_dsh_container_checks_node_before_running_dsh(tmp_path, version, supported):
    agent = harbor_dsh.DshHarborAgent(
        logs_dir=tmp_path,
        model_name="model",
        skip_model_parsing=True,
        proxy_url="http://proxy",
        task_session_id="trial",
        host_dsh_package=str(tmp_path / "dsh"),
        host_node_executable=str(tmp_path / "node"),
    )
    commands = []

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append(command)
            output = version if command.endswith("node --version") else ""
            return SimpleNamespace(return_code=0, stdout=output, stderr="")

        async def upload_dir(self, source, destination):
            pass

        async def upload_file(self, source, destination):
            pass

    if supported:
        asyncio.run(agent.setup(Environment()))
        assert commands[-2].endswith("node --version")
        assert commands[-1] == "dsh --version"
    else:
        with pytest.raises(RuntimeError, match="DSH requires Node.js"):
            asyncio.run(agent.setup(Environment()))
        assert all("dsh --version" not in command for command in commands)


def test_jiuwenswarm_deploys_current_entry_and_forwards_prompt_timeout(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"prepared runtime")
    agent = harbor_jiuwenswarm.JiuwenHarborAgent(
        logs_dir=tmp_path,
        model_name="model",
        skip_model_parsing=True,
        proxy_url="http://proxy",
        task_session_id="trial",
        host_runtime=str(tmp_path),
        prompt_delivery_timeout_seconds=7,
    )
    events = []
    uploads = {}

    class Environment:
        async def exec(self, command, **kwargs):
            events.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, destination):
            events.append(destination)
            uploads[destination] = Path(source).read_bytes()

    async def exercise():
        environment = Environment()
        await agent.setup(environment)
        await agent.run("Run the fixture.", environment, SimpleNamespace(metadata={}))

    asyncio.run(exercise())
    entry = next(name for name in uploads if name.endswith("/terminalbench_entry.py"))
    assert events.index(entry) > next(i for i, command in enumerate(events) if command.startswith("tar -xzf"))
    assert b"build_chat_command" in uploads[entry]
    config = json.loads(uploads[f"{harbor_jiuwenswarm.REMOTE_LOGS}/config.json"])
    assert config["profile"] == "agent.code.normal"
    assert config["prompt_delivery_timeout_seconds"] == 7
