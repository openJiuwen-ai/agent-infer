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
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


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


def test_sse_heartbeat_prevents_body_timeout(request_data, monkeypatch):
    import time

    from agentinfer.agentbench.agents.zcode import stream
    from tests.agentbench.helpers.dsh_mock_server import _Handler

    monkeypatch.setattr(stream, "SSE_HEARTBEAT_SECONDS", 0.03)
    wire = b'data: {"choices":[],"usage":{"prompt_tokens":100}}\n\ndata: [DONE]\n\n'

    class SlowBody(_Handler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                # A partial frame must not be forwarded before inserting a heartbeat.
                self.wfile.write(wire[:12])
                self.wfile.flush()
                time.sleep(0.6)
                self.wfile.write(wire[12:])
            except BrokenPipeError:
                pass  # Expected from the direct client's negative control.

    with ZCodeMockServer("single") as server:
        server.RequestHandlerClass = SlowBody
        bridge = ZCodeBridge(replace(request_data, profile_name="single", api_base_url=server.base_url))
        bridge.register("sess_root")

        async def scenario():
            async with httpx.AsyncClient(timeout=httpx.Timeout(5, read=0.2), trust_env=False) as client:
                with pytest.raises(httpx.ReadTimeout):
                    await client.post(f"{server.base_url}/v1/chat/completions", json={})
                async with bridge.serve() as base_url:
                    response = await client.post(
                        f"{base_url}/v1/chat/completions",
                        json={},
                        headers={"x-session-id": "root", "x-zcode-session-type": "main"},
                    )
                assert response.status_code == 200
                assert b": agentbench keep-alive\n\n" in response.content
                assert response.content.replace(b": agentbench keep-alive\n\n", b"") == wire
                from agentinfer.agentbench.request_proxy.observers import _OpenAISSEUsageObserver

                observer = _OpenAISSEUsageObserver()
                observer.observe(response.content)
                assert observer.usage["input_tokens"] == 100
                assert observer.completed

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
        f"#!{sys.executable}\nimport sys, os\nfrom pathlib import Path\n"
        f"outputs = {{'--help': '--prompt --cwd', '--version': {version!r}, 'doctor': '{{}}'}}\n"
        "if '--prompt' in sys.argv:\n"
        "    path = Path(os.environ['AGENTBENCH_ZCODE_PROMPT_FILE'])\n"
        "    Path(os.environ['AGENTBENCH_ZCODE_PROMPT_RECEIPT']).write_text(str(path.stat().st_size))\n"
        "    print(outputs['--help'])\n"
        "    sys.exit(0)\n"
        "print(outputs[sys.argv[1]])\n"
    )
    executable.chmod(0o755)
    if version == "0.17.1":
        report = asyncio.run(ZCodeRuntime().preflight(executable))
        assert report.versions["zcode"] == version
    else:
        with pytest.raises(RuntimeError, match="Expected ZCode CLI"):
            asyncio.run(ZCodeRuntime().preflight(executable))


def test_preflight_rejects_launcher_without_prompt_loading(tmp_path):
    executable = tmp_path / "zcode"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nprint('0.17.1' if '--version' in sys.argv else '--prompt --cwd')\n"
    )
    executable.chmod(0o755)
    with pytest.raises(RuntimeError, match="must preserve NODE_OPTIONS"):
        asyncio.run(ZCodeRuntime().preflight(executable))


def test_preflight_isolates_all_cli_probes(tmp_path, monkeypatch):
    original_home = tmp_path / "caller-home"
    original_home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(original_home))
    monkeypatch.setenv("ZCODE_STORAGE_DIR", str(original_home / "state"))
    monkeypatch.setenv("ZCODE_PERSONAL_PROVIDER_CONFIG_FILE", str(original_home / "provider.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-value")
    monkeypatch.setenv("NODE_OPTIONS", "caller-preload-must-not-be-inherited")
    log = tmp_path / "probes.jsonl"
    executable = tmp_path / "bin/zcode"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\nimport sys, os, json\nfrom pathlib import Path\n"
        "paths = {k: os.environ[k] for k in ['HOME', 'USERPROFILE', 'TMPDIR', 'ZCODE_DATA_BASE_DIR', "
        "'ZCODE_STORAGE_DIR', 'ZCODE_LOG_DIR', 'ZCODE_SESSION_DB_PATH']}\n"
        "row = {'args': sys.argv[1:], 'cwd': os.getcwd(), 'paths': paths, "
        "'personal': 'ZCODE_PERSONAL_PROVIDER_CONFIG_FILE' in os.environ, "
        "'credential': 'ANTHROPIC_API_KEY' in os.environ, 'node_options': os.environ.get('NODE_OPTIONS')}\n"
        f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(row)+'\\n')\n"
        "(Path(os.environ['HOME'])/'probe-marker').write_text('private')\n"
        "if '--prompt' in sys.argv:\n"
        "    Path(os.environ['AGENTBENCH_ZCODE_PROMPT_RECEIPT']).write_text('loaded')\n"
        "print('0.17.1' if '--version' in sys.argv else '{}' if 'doctor' in sys.argv else '--prompt --cwd')\n"
    )
    executable.chmod(0o755)
    asyncio.run(ZCodeRuntime().preflight(Path("bin/zcode")))
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["args"][0] for row in rows} == {"--help", "--version", "--prompt", "doctor"}
    for row in rows:
        assert Path(row["cwd"]) != tmp_path
        assert all(Path(value).resolve().is_relative_to(Path(row["cwd"]).resolve()) for value in row["paths"].values())
        assert not row["personal"] and not row["credential"]
        if row["args"][0] != "--prompt":
            assert row["node_options"] is None
        else:
            assert "prompt.cjs" in row["node_options"]
    assert not (original_home / "probe-marker").exists()
    assert not Path(rows[0]["paths"]["HOME"]).exists()


def test_subagent_total_and_concurrent_budgets(request_data):
    bridge = ZCodeBridge(replace(request_data, max_subagents=4, max_concurrent_subagents=2))
    args = {"subagent_type": "agentbench-explore-bench-task"}

    def launch(key):
        return bridge.observe_hook(hook("PreToolUse", "Agent", toolCallId=key, toolInput=args))

    def finish(key, status="completed"):
        bridge.observe_hook(hook("PostToolUse", "Agent", toolCallId=key, toolResponse={"status": status}))

    for key in ("a", "b"):
        assert "decision" not in launch(key)
    launch("a")  # Repeated delivery of the same pending call must not consume another launch.
    assert len(bridge.delegations) == 2
    assert launch("c")["decision"] == "block"
    finish("a")
    assert bridge.denial(False, "ExitPlanMode", {"plan": "fix"})
    assert "decision" not in launch("c")
    finish("b")
    finish("c")
    assert "decision" not in launch("d")
    finish("d")
    assert launch("e")["decision"] == "block"
    assert bridge.denial(False, "ExitPlanMode", {"plan": "fix"}) is None
    assert bridge.reports == 4
    lineage = json.loads((request_data.artifact_dir / "zcode-lineage.json").read_text())
    assert lineage["started_subagents"] == 4
    assert lineage["peak_concurrent_subagents"] == 2
    assert not lineage["pending"]


@pytest.mark.parametrize("failure_event", ["PostToolUse", "PostToolUseFailure"])
def test_failed_child_consumes_budget_and_prevents_approval(request_data, failure_event):
    bridge = ZCodeBridge(replace(request_data, max_subagents=4, max_concurrent_subagents=2))
    args = {"subagent_type": "agentbench-explore-bench-task"}
    for key in ("a", "b"):
        bridge.observe_hook(hook("PreToolUse", "Agent", toolCallId=key, toolInput=args))
    bridge.observe_hook(hook("PostToolUse", "Agent", toolCallId="a", toolResponse={"status": "completed"}))
    bridge.observe_hook(hook(failure_event, "Agent", toolCallId="b", toolResponse={"status": "failed"}))
    assert bridge.failure.value == "subagent_failed"
    assert len(bridge.delegations) == 2 and not bridge.pending
    assert bridge.denial(False, "Agent", args)
    assert bridge.denial(False, "ExitPlanMode", {"plan": "fix"})
    bridge.observe_hook(hook("Stop", stopHookActive=True))
    assert bridge.failure.value == "subagent_failed"


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
