# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify DSH's native Linux workspace sandbox through a real headless process."""

import asyncio
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest
from agentinfer.agentbench.agents.dsh.runner import run_dsh
from agentinfer.agentbench.benchkit.dataset import Task

pytestmark = pytest.mark.skipif(os.name != "posix", reason="DSH native Linux sandbox is unavailable")


class _CanaryServer(ThreadingHTTPServer):
    commands: list[str]
    requests: list[dict]


class _CanaryHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        tool_results = [message for message in body.get("messages", []) if message.get("role") == "tool"]
        index = len(tool_results)
        if index < len(self.server.commands):  # type: ignore[attr-defined]
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": self.server.commands[index],  # type: ignore[attr-defined]
                                    "description": f"Run sandbox canary {index}",
                                }
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            delta = {"role": "assistant", "content": "Sandbox canaries complete."}
            finish_reason = "stop"
        chunks = [
            {
                "id": "canary",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "canary-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "canary",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "canary-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        encoded = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.mark.cpu_test
def test_dsh_native_workspace_sandbox_blocks_escape_canaries(tmp_path: Path) -> None:
    executable = shutil.which("dsh")
    if executable is None:
        pytest.skip("dsh is not installed")
    config = subprocess.run(
        [executable, "--profile", "headless", "--dump-default-config"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout
    required = ("dsh-sandbox-local", "dsh-bash-sandbox", "dsh-fs-sandbox")
    assert all(plugin in config for plugin in required), "installed DSH lacks the native sandbox composition"

    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    external.mkdir()
    targets = {name: external / f"{name}.txt" for name in ("relative", "absolute", "symlink")}
    for target in targets.values():
        target.write_text("original\n", encoding="utf-8")
    link = workspace / "escape-link.txt"
    link.symlink_to(targets["symlink"])
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "canary"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.email", "canary@example.invalid"], cwd=workspace, check=True)
    (workspace / "README").write_text("canary\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)

    server = _CanaryServer(("127.0.0.1", 0), _CanaryHandler)
    server.requests = []
    server.commands = [
        "printf workspace-ok > workspace-ok.txt",
        "printf relative-escape > ../external/relative.txt",
        f"printf absolute-escape > {targets['absolute']}",
        f"printf symlink-escape > {link}",
    ]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = AgentRunRequest(
        agent_type="dsh",
        task=Task("sandbox-canary", "local/canary", "HEAD", "Run the requested sandbox canaries."),
        profile_name="single",
        executable=Path(executable),
        model="canary-model",
        api_base_url=f"http://127.0.0.1:{server.server_port}",
        workspace=workspace,
        artifact_dir=artifacts,
        session_id="sandbox-canary-root",
        timeout_seconds=120,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=2,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )
    try:
        result = asyncio.run(run_dsh(request))
    finally:
        server.shutdown()
        thread.join()

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert result.termination_reason is None
    assert (workspace / "workspace-ok.txt").read_text(encoding="utf-8") == "workspace-ok"
    assert all(target.read_text(encoding="utf-8") == "original\n" for target in targets.values())
    assert link.is_symlink()
    assert len(server.requests) == 5
    assert (artifacts / "dsh-launch-command.json").exists()
    assert (artifacts / "transcript.jsonl").exists()
    assert list((artifacts / "dsh-sessions").glob("session-*.jsonl"))
