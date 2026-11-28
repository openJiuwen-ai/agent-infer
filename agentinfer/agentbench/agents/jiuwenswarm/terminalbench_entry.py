# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Container-only JiuwenSwarm entry point; Harbor owns isolation and scoring."""

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

import yaml

from .instance import JiuwenInstance
from .runner import _communicate_with_timeout, _kill_cli_process_group, _write_outputs, build_chat_command


class HarborJiuwenInstance(JiuwenInstance):
    """Use LOCAL tools inside Harbor's container, with no host fallback."""

    def __init__(self, **kwargs):
        if not Path("/.dockerenv").exists() or os.environ.get("AGENTBENCH_HARBOR_CONTAINER") != "1":
            raise RuntimeError("Harbor Jiuwen instance may only run inside a Docker task container")
        super().__init__(**kwargs)
        # Empty proxy variables are rejected by the SDK's HTTP transport too.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            self.environment.pop(name, None)

    def _write_jiuwenbox_policy(self) -> None:
        pass  # The task's Docker container is the explicit isolation boundary.

    async def _start_jiuwenbox_server(self) -> None:
        pass

    def _configure(self) -> None:
        super()._configure()
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        data["sandbox"] = {"enabled": False, "type": "", "url": ""}
        self.config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


async def run(config: dict) -> None:
    logs = Path(config["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    instance = HarborJiuwenInstance(
        executable=Path(config["executable"]),
        artifact_dir=logs,
        workspace=Path.cwd(),
        session_id=config["session_id"],
        api_base_url=config["proxy_url"],
        model=config["model"],
        completion_timeout_seconds=config["timeout_seconds"],
        profile_name=config["profile"],
        max_subagents=config["max_subagents"],
        max_concurrent_subagents=config["max_concurrent_subagents"],
    )
    deadline = time.monotonic() + config["timeout_seconds"]
    process = None
    try:
        await asyncio.wait_for(instance.start(), max(0, deadline - time.monotonic()))
        command = build_chat_command(
            instance.executable,
            instance.profile_name,
            instance.session_id,
            instance.workspace,
            instance.gateway_url,
            config["timeout_seconds"],
        )
        (logs / "jiuwenswarm-launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
        process = await asyncio.create_subprocess_exec(
            *command,
            env=instance.environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        prompt = Path(config["prompt"]).read_bytes()
        stdout, stderr, timed_out, reason = await _communicate_with_timeout(
            process, prompt, deadline, config["prompt_delivery_timeout_seconds"]
        )
        await _write_outputs(logs, stdout, stderr)
        if timed_out:
            await instance.interrupt()
            raise TimeoutError("JiuwenSwarm task timed out")
        if reason is not None:
            raise RuntimeError(f"JiuwenSwarm terminated without completion: {reason.value}")
    finally:
        if process is not None and process.returncode is None:
            await _kill_cli_process_group(process)
        try:
            await asyncio.wait_for(instance.stop(), 30)
        finally:
            (logs / "container-runtime.json").write_text(
                json.dumps({"isolation": "harbor-docker", "profile": instance.profile_name}) + "\n"
            )


async def main(config: dict) -> None:
    task = asyncio.create_task(run(config))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    await task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    asyncio.run(main(json.loads(parser.parse_args().config.read_text(encoding="utf-8"))))
