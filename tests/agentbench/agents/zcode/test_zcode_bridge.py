# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise the adapter's wire identity and pre-execution hook policy."""

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.agents.contracts import AgentRunRequest
from agentinfer.agentbench.agents.zcode.bridge import ZCodeBridge
from agentinfer.agentbench.agents.zcode.runner import run_zcode
from agentinfer.agentbench.agents.zcode.runtime import ZCodeRuntime
from agentinfer.agentbench.benchkit.dataset import Task
from tests.agentbench.agents.zcode.test_zcode_integration import ZCodeMockServer
from tests.agentbench.helpers.dsh_mock_server import seed_tasks


@pytest.fixture
def request_data(tmp_path):
    (tmp_path / ".git/info").mkdir(parents=True)
    return AgentRunRequest(
        agent_type="zcode",
        task=Task("task", "owner/repo", "HEAD", "fix"),
        profile_name="plan-subagent",
        executable=Path("zcode"),
        model="dsh-test",
        api_base_url="http://unused",
        workspace=tmp_path,
        artifact_dir=tmp_path,
        session_id="bench-task",
        timeout_seconds=20,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=5,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )


def hook(kind, name=None, **values):
    return {
        "hookEventName": kind,
        "sessionId": "sess_root",
        "toolName": name,
        "toolCallId": "call-1",
        "toolInput": {},
        **values,
    }


def test_plan_requires_completed_exploration_and_approval(request_data):
    bridge = ZCodeBridge(request_data)
    bridge.observe_hook(hook("SessionStart"))
    assert bridge.observe_hook(hook("PreToolUse", "Write"))["decision"] == "block"
    assert (
        bridge.observe_hook(hook("PreToolUse", "Agent", toolInput={"subagent_type": "agentbench-explore-bench-task"}))[
            "hookSpecificOutput"
        ]["permissionDecision"]
        == "allow"
    )
    plan = {"plan": "Fix the bug"}
    assert bridge.observe_hook(hook("PreToolUse", "ExitPlanMode", toolInput=plan))["decision"] == "block"
    bridge.observe_hook(hook("PostToolUse", "Agent", toolResponse={"status": "completed"}))
    assert bridge.observe_hook(hook("PreToolUse", "Write"))["decision"] == "block"
    decision = bridge.observe_hook(hook("PreToolUse", "ExitPlanMode", toolInput=plan))
    assert decision["hookSpecificOutput"]["permissionDecision"] == "allow"
    bridge.observe_hook(hook("PostToolUse", "ExitPlanMode", toolResponse={"approved": True}))
    assert bridge.observe_hook(hook("PreToolUse", "Write"))["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert (
        bridge.observe_hook(hook("PreToolUse", "Agent", toolInput={"subagent_type": "agentbench-explore-bench-task"}))[
            "decision"
        ]
        == "block"
    )


def test_child_and_background_tool_calls_are_rejected(request_data):
    bridge = ZCodeBridge(request_data)
    bridge.observe_hook(hook("SessionStart"))
    tools = [
        {"type": "function", "function": {"name": name}}
        for name in ("Read", "Glob", "Grep", "RespondToCoordinator", "Write", "Agent")
    ]
    advertised = bridge.model_tools(tools, child=True)
    assert {tool["function"]["name"] for tool in advertised} == {"Read", "Glob", "Grep"}
    for tool in advertised:
        assert bridge.denial(True, tool["function"]["name"], {}) is None
    for name in ("Write", "Bash", "Agent", "ExitPlanMode", "RespondToCoordinator"):
        response = bridge.observe_hook(hook("PreToolUse", name, sessionId="sess_subagent_agent_child"))
        assert response["decision"] == "block"
    assert (
        bridge.observe_hook(
            hook(
                "PreToolUse",
                "Agent",
                toolInput={"subagent_type": "agentbench-explore-bench-task", "run_in_background": True},
            )
        )["decision"]
        == "block"
    )


def test_hook_subprocess_and_model_stream_preserve_request(request_data):
    with ZCodeMockServer("single") as server:
        request = replace(request_data, profile_name="single", api_base_url=server.base_url)
        bridge = ZCodeBridge(request)

        async def scenario():
            async with bridge.serve() as base_url:
                import agentinfer.agentbench.agents.zcode.hook as hook_module

                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    hook_module.__file__,
                    f"{base_url}/hook",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate(json.dumps(hook("SessionStart")).encode())
                assert process.returncode == 0 and json.loads(stdout) == {}
                payload = {
                    "model": "dsh-test",
                    "messages": [{"role": "user", "content": "fix"}],
                    "tools": [],
                    "stream": True,
                    "vllm_xargs": {"existing": "kept"},
                }
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.post(
                        f"{base_url}/v1/chat/completions",
                        json=payload,
                        headers={"x-session-id": "root", "x-zcode-session-type": "main"},
                    )
                assert response.status_code == 200
                assert "data: [DONE]" in response.text and '"prompt_tokens": 100' in response.text
                body = server.requests[0]["body"]
                assert body["messages"] == payload["messages"] and body["tools"] == []
                assert body["vllm_xargs"]["existing"] == "kept"
                assert json.loads(body["vllm_xargs"]["agentic_context"])["task_id"] == "bench-task"

        asyncio.run(scenario())


def test_native_context_error_is_preserved(request_data, tmp_path):
    seed_tasks(tmp_path / "fixture", 1)
    error = {
        "code": "MODEL_CONTEXT_EXCEEDED",
        "type": "model_context_exceeded",
        "message": "Autocompact stopped",
        "attribution": {"reason": "compact_rapid_refill_breaker"},
    }
    executable = tmp_path / "failed-zcode"
    event = {"type": "turn.failed", "sessionId": "sess_root", "payload": {"error": error}}
    executable.write_text(f"#!{sys.executable}\nimport sys\nprint({json.dumps(event)!r})\nsys.exit(1)\n")
    executable.chmod(0o755)
    result = asyncio.run(
        run_zcode(
            replace(
                request_data, executable=executable, workspace=tmp_path / "fixture/cache/agentbench-fixture__tiny-calc"
            )
        )
    )
    assert result.termination_reason.value == "model_context_exceeded"
    status = json.loads((tmp_path / "zcode-run-status.json").read_text())
    assert status["native_error"] == error
    assert status["exit_code"] == 1


@pytest.mark.parametrize("version", ["0.17.1", "0.17.10", "10.17.1"])
def test_preflight_requires_exact_version(tmp_path, version):
    executable = tmp_path / "zcode"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"outputs = {{'--help': '--prompt --cwd', '--version': {version!r}, 'doctor': '{{}}'}}\n"
        "print(outputs[sys.argv[1]])\n"
    )
    executable.chmod(0o755)
    if version == "0.17.1":
        report = asyncio.run(ZCodeRuntime().preflight(executable))
        assert report.versions["zcode"] == version
    else:
        with pytest.raises(RuntimeError, match="Expected ZCode CLI"):
            asyncio.run(ZCodeRuntime().preflight(executable))


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_exit_without_completion_is_failed(request_data, tmp_path, profile, exit_code):
    seed_tasks(tmp_path / "fixture", 1)
    executable = tmp_path / "incomplete-zcode"
    executable.write_text(f"#!{sys.executable}\nimport sys\nsys.exit({exit_code})\n")
    executable.chmod(0o755)
    result = asyncio.run(
        run_zcode(
            replace(
                request_data,
                executable=executable,
                profile_name=profile,
                workspace=tmp_path / "fixture/cache/agentbench-fixture__tiny-calc",
            )
        )
    )
    assert result.outcome.value == "failed"
    expected = "plan_not_approved" if profile == "plan-subagent" and exit_code == 0 else "harness_error"
    assert result.termination_reason.value == expected
    status = json.loads((tmp_path / "zcode-run-status.json").read_text())
    assert status["exit_code"] == exit_code
    assert status["termination_reason"] == expected
