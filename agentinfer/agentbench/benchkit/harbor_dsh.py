# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Harbor agent adapter that runs DSH inside the Terminal-Bench task container."""

import shlex

import yaml
from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from ..agents.dsh.bridge import AGENTBENCH_BRIDGE
from ..agents.dsh.instance import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, build_dsh_settings
from ..agents.dsh.profiles import get_profile
from .terminalbench import DSH_VERSION

_REMOTE_HOME = "/tmp/agentbench-dsh-home"
_REMOTE_LOGS = "/logs/agent/agentbench-dsh"


class DshHarborOptions(AgentOptions):
    profile: str = "single"
    proxy_url: str
    task_session_id: str
    host_dsh_package: str
    host_node_executable: str
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, ge=1)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    version: str = Field(default=DSH_VERSION)


class DshHarborAgent(BaseAgent):
    """Use Harbor's task container for both DSH tools and the official verifier."""

    DEFAULT_SETUP_TIMEOUT_SEC = 1200.0
    options_model = DshHarborOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = get_profile(self.options.profile)
        self.proxy_url = self.options.proxy_url.rstrip("/")
        self._version = self.options.version
        if self._version != DSH_VERSION:
            raise ValueError(f"Unsupported DSH version: {self._version}")

    @staticmethod
    def name() -> str:
        return "agentbench-dsh"

    def version(self) -> str:
        return self._version

    async def setup(self, environment: BaseEnvironment) -> None:
        """Copy the pinned host DSH and Node runtime into the task image."""

        prepared = await environment.exec(
            "mkdir -p /opt/agentbench-dsh /opt/agentbench-node/bin /usr/local/bin",
            timeout_sec=30,
            user="root",
        )
        if prepared.return_code:
            raise RuntimeError(f"DSH setup failed: {prepared.stderr or prepared.stdout}")
        await environment.upload_dir(self.options.host_dsh_package, "/opt/agentbench-dsh")
        await environment.upload_file(self.options.host_node_executable, "/opt/agentbench-node/bin/node")
        command = (
            "set -eu; "
            "chmod 755 /opt/agentbench-node/bin/node; "
            "ln -sfn /opt/agentbench-node/bin/node /usr/local/bin/node; "
            "ln -sfn /opt/agentbench-dsh/lib/bin.js /usr/local/bin/dsh; "
            'node -e \'const [major, minor] = process.versions.node.split(".").map(Number); '
            "if (major < 22 || (major === 22 && minor < 19)) process.exit(1)'; "
            "dsh --version"
        )
        result = await environment.exec(command, timeout_sec=60, user="root")
        if result.return_code:
            raise RuntimeError(f"DSH setup failed: {result.stderr or result.stdout}")

    async def _upload_text(self, environment: BaseEnvironment, name: str, content: str) -> None:
        source = self.logs_dir / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
        await environment.upload_file(source, f"{_REMOTE_HOME}/{name}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Configure the isolated DSH profile and execute the supplied task."""

        if not self.model_name:
            raise ValueError("DSH requires a model name")
        remote_url = self.proxy_url
        env = {
            "DSH_HOME": _REMOTE_HOME,
            "DSH_PERMISSION_MODE": self.profile.permission_mode,
            "AGENTBENCH_DSH_API_KEY": "agentbench",
            "AGENTBENCH_ROOT_SESSION_ID": self.options.task_session_id,
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
        }
        init = await environment.exec(
            f"mkdir -p {_REMOTE_HOME} {_REMOTE_LOGS} && dsh --profile headless --dump-config"
            f" > {_REMOTE_LOGS}/init-config.yaml 2> {_REMOTE_LOGS}/init.stderr.log",
            env=env,
            timeout_sec=120,
        )
        if init.return_code:
            raise RuntimeError(f"DSH profile initialization failed: {init.stderr or init.stdout}")
        settings = build_dsh_settings(
            remote_url,
            self.model_name,
            context_window=self.options.context_window,
            max_tokens=self.options.max_tokens,
        )
        await self._upload_text(
            environment,
            "cordis.patch.yml",
            yaml.safe_dump([{"id": name, "config": value} for name, value in settings.items()], sort_keys=False),
        )
        await self._upload_text(environment, "agentbench-bridge.mjs", AGENTBENCH_BRIDGE)
        bridge_patch = (
            "\n- insert:\n"
            "    - id: agentbench-bridge\n"
            f"      name: {_REMOTE_HOME}/agentbench-bridge.mjs\n"
            "      config:\n"
            f"        enforcePlanMode: {str(self.profile.enforce_plan_mode).lower()}\n"
            f"        apiBaseURL: {remote_url!r}\n"
            f"        lineagePath: {_REMOTE_LOGS}/dsh-lineage.json\n"
        )
        await self._upload_text(
            environment,
            "agentbench.patch.yml",
            "- id: session-title-llm\n  disabled: true\n" + self.profile.policy_patch + bridge_patch,
        )
        if self.profile.enforce_plan_mode:
            prompt = (
                "You are the lead agent solving this Terminal-Bench task. "
                "Your first tool call must be subagent; delegate focused exploration before inspecting files yourself. "
                "Each subagent is read-only: ask it to inspect the task environment and return concise findings. "
                "Wait for its report, then call exit_plan_mode with your implementation plan. "
                "Implement the task yourself after planning. Do not ask the user questions.\n\n"
            )
        else:
            prompt = (
                "You are an agent solving this Terminal-Bench task. Work directly in the task environment. "
                "Do not create subagents or ask the user questions.\n\n"
            )
        await self._upload_text(environment, "prompt.txt", prompt + instruction + "\n")
        patch = shlex.quote(f"{_REMOTE_HOME}/agentbench.patch.yml")
        checked = await environment.exec(
            f"dsh --profile headless --patch {patch} --dump-config"
            f" > {_REMOTE_LOGS}/effective-config.yaml 2> {_REMOTE_LOGS}/config.stderr.log",
            env=env,
            timeout_sec=120,
        )
        if checked.return_code:
            raise RuntimeError(f"DSH profile configuration failed: {checked.stderr or checked.stdout}")
        completed = await environment.exec(
            f"dsh --profile headless --patch {patch} --json < {_REMOTE_HOME}/prompt.txt"
            f" > {_REMOTE_LOGS}/stdout.jsonl 2> {_REMOTE_LOGS}/stderr.log",
            env=env,
        )
        context.metadata = {"profile": self.profile.name, "dsh_version": self._version}
        if completed.return_code:
            raise RuntimeError(
                f"DSH exited with status {completed.return_code}: {completed.stderr or completed.stdout}"
            )
