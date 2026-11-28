# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run JiuwenSwarm in the original Harbor task container."""

import json
import shlex

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from .jiuwenswarm_runtime import REMOTE_RUNTIME, RUNTIME_IMPORT_CHECK, WORKSWARM_VERSION

REMOTE_LOGS = "/logs/agent/agentbench-jiuwenswarm"


class JiuwenHarborOptions(AgentOptions):
    profile: str = "agent.code.normal"
    proxy_url: str
    task_session_id: str
    host_runtime: str
    timeout_seconds: int = Field(default=28800, ge=1)
    max_subagents: int = Field(default=1, ge=1)
    max_concurrent_subagents: int = Field(default=1, ge=1)
    version: str = WORKSWARM_VERSION


class JiuwenHarborAgent(BaseAgent):
    DEFAULT_SETUP_TIMEOUT_SEC = 1200.0
    options_model = JiuwenHarborOptions

    @staticmethod
    def name() -> str:
        return "agentbench-jiuwenswarm"

    def version(self) -> str:
        return WORKSWARM_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        from pathlib import Path

        prepared = await environment.exec(f"mkdir -p {REMOTE_RUNTIME} {REMOTE_LOGS}", user="root", timeout_sec=30)
        if prepared.return_code:
            raise RuntimeError(f"JiuwenSwarm runtime directory setup failed: {prepared.stderr}")
        await environment.upload_file(
            Path(self.options.host_runtime) / "runtime.tar.gz", f"{REMOTE_RUNTIME}/runtime.tar.gz"
        )
        checked = await environment.exec(
            f"tar -xzf {REMOTE_RUNTIME}/runtime.tar.gz -C {REMOTE_RUNTIME} && "
            f"{REMOTE_RUNTIME}/python/bin/python3 -c "
            + shlex.quote(
                "from importlib.metadata import version; "
                "assert version('workswarm') == '0.2.8'; assert version('openjiuwen') == '0.1.20'; "
                + RUNTIME_IMPORT_CHECK
            ),
            user="root",
            timeout_sec=300,
        )
        if checked.return_code:
            raise RuntimeError(f"JiuwenSwarm container runtime is incompatible: {checked.stderr or checked.stdout}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("JiuwenSwarm requires a model name")
        config = {
            "executable": f"{REMOTE_RUNTIME}/python/bin/jiuwenswarm",
            "logs": REMOTE_LOGS,
            "prompt": f"{REMOTE_LOGS}/prompt.txt",
            "model": self.model_name,
            "proxy_url": self.options.proxy_url,
            "session_id": self.options.task_session_id,
            "timeout_seconds": self.options.timeout_seconds,
            "max_subagents": self.options.max_subagents,
            "max_concurrent_subagents": self.options.max_concurrent_subagents,
        }
        prompt = (
            "Solve this Terminal-Bench task in the provided container environment. "
            "Do not ask the user questions or request interactive approvals. "
            "Complete the task and report your changes and checks.\n\n" + instruction + "\n"
        )
        for name, content in (("prompt.txt", prompt), ("config.json", json.dumps(config))):
            local = self.logs_dir / name
            local.write_text(content, encoding="utf-8")
            await environment.upload_file(local, f"{REMOTE_LOGS}/{name}")
        completed = await environment.exec(
            f"{REMOTE_RUNTIME}/python/bin/python3 -m "
            f"agentinfer.agentbench.agents.jiuwenswarm.terminalbench_entry {REMOTE_LOGS}/config.json"
            f" > {REMOTE_LOGS}/entry.stdout.log 2> {REMOTE_LOGS}/entry.stderr.log",
            env={
                "AGENTBENCH_HARBOR_CONTAINER": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        context.metadata = {
            "profile": self.options.profile,
            "workswarm": WORKSWARM_VERSION,
            "isolation": "harbor-docker",
        }
        if completed.return_code:
            raise RuntimeError(f"JiuwenSwarm failed (exit {completed.return_code}); see {REMOTE_LOGS}/entry.stderr.log")
