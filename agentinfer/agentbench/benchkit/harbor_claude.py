# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Reuse Harbor's Claude protocol with offline, local binary deployment."""

from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode, ClaudeCodeOptions
from harbor.environments.base import BaseEnvironment

REMOTE_BINARY = "/opt/agentbench-claude/claude"
WRAPPER = (
    "#!/bin/sh\n"
    "export DISABLE_AUTOUPDATER=1\n"
    "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1\n"
    f'exec {REMOTE_BINARY} "$@"\n'
)


class LocalClaudeOptions(ClaudeCodeOptions):
    host_executable: str


class LocalClaudeCode(ClaudeCode):
    options_model = LocalClaudeOptions

    def _resolve_auth_env(self) -> dict[str, str]:
        """Use only the task gateway address and its local placeholder credential."""

        gateway = self.model_connection.execution_base_url
        if not gateway:
            raise ValueError("Terminal-Bench Claude requires a task model gateway")
        return {"ANTHROPIC_API_KEY": "agentbench", "ANTHROPIC_BASE_URL": gateway}

    async def install(self, environment: BaseEnvironment) -> None:
        executable = Path(self.options.host_executable).resolve()
        await self.exec_as_root(environment, "mkdir -p /opt/agentbench-claude /usr/local/bin")
        await environment.upload_file(executable, REMOTE_BINARY)
        wrapper = self.logs_dir / "claude-offline-wrapper.sh"
        wrapper.write_text(WRAPPER, encoding="utf-8")
        await environment.upload_file(wrapper, "/usr/local/bin/claude")
        await self.exec_as_root(environment, f"chmod 755 {REMOTE_BINARY} /usr/local/bin/claude")
        await self.exec_as_agent(
            environment, 'mkdir -p "$HOME/.local/bin" && ln -sfn /usr/local/bin/claude "$HOME/.local/bin/claude"'
        )
        if not await self._installed_claude_satisfies_version(environment):
            raise RuntimeError(
                "Local Claude binary cannot run at the requested version in this task image; no download fallback"
            )

    async def acp_install(self, environment: BaseEnvironment) -> None:
        raise RuntimeError("Terminal-Bench offline Claude adapter does not support ACP runtime downloads")
