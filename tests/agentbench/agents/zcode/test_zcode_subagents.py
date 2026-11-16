# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify parallel, bounded exploration through the real ZCode CLI."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.contracts import AgentRunRequest
from agentinfer.agentbench.agents.zcode.runner import run_zcode
from agentinfer.agentbench.benchkit.dataset import Task
from tests.agentbench.agents.zcode.test_zcode_integration import ZCodeMockServer
from tests.agentbench.helpers.dsh_mock_server import _Handler, seed_tasks, write_chat_completion_stream

pytestmark = pytest.mark.timeout(180)


@pytest.fixture
def zcode_executable():
    value = os.environ.get("ZCODE_TEST_EXECUTABLE")
    if not value:
        pytest.skip("Set ZCODE_TEST_EXECUTABLE to the pinned ZCode launcher")
    return str(Path(value).resolve())


class MultiHandler(_Handler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        try:
            calls, text = self.server.respond(body)
        except Exception as exc:
            self.server.errors.append(repr(exc))
            self.send_error(500, "Mock response failed")
            return
        delta = (
            {"content": text}
            if not calls
            else {
                "tool_calls": [
                    {
                        "index": i,
                        "id": f"call-{len(self.server.requests)}-{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for i, (name, args) in enumerate(calls)
                ]
            }
        )
        write_chat_completion_stream(
            self,
            [
                {"id": "multi", "object": "chat.completion.chunk", "model": "dsh-test", "created": 0, **chunk}
                for chunk in [
                    {"choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
                ]
            ],
        )

    def log_message(self, *_args):
        pass


class MultiExplorer(ZCodeMockServer):
    def __init__(self, *, hold_children=False, finish_early=False):
        super().__init__("plan-subagent")
        self.RequestHandlerClass = MultiHandler
        self.children_barrier = threading.Barrier(2)
        self.hold_children = hold_children
        self.finish_early = finish_early
        self.errors = []

    def respond(self, body):
        identity = json.loads(body["vllm_xargs"]["agentic_context"])
        key = identity["program_id"]
        stage = self.states.get(key, 0)
        self.states[key] = stage + 1
        self.requests.append({"identity": identity, "body": body})
        if identity["agent_role"] == "subagent":
            if stage == 0:
                self.children_barrier.wait(15)  # Sequential execution cannot pass this barrier.
                self.blocked.set()
                if self.hold_children:
                    self.release.wait(30)
                return [("Read", {"file_path": "calc.py"})], None
            return [], "The add function subtracts; replace a - b with a + b."
        delegate = (
            "Agent",
            {
                "subagent_type": f"agentbench-explore-{identity['task_id']}",
                "description": "Inspect addition",
                "prompt": "Read calc.py and report.",
                "run_in_background": False,
            },
        )
        plan = ("ExitPlanMode", {"plan": "Change subtraction to addition, then run the tests."})
        if stage == 0:
            assert body["tool_choice"] == "required"
            assert body["parallel_tool_calls"] is True
            assert {tool["function"]["name"] for tool in body["tools"]} <= {"Agent", "Task"}
            return [delegate, delegate, delegate], None
        if stage == 1:
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert "Agent" in names and "ExitPlanMode" in names
            assert body["tool_choice"] == "auto"
            return ([plan] if self.finish_early else [delegate, delegate]), None
        if self.finish_early and stage >= 2:
            stage += 2
        if stage == 2:
            assert {tool["function"]["name"] for tool in body["tools"]} == {"ExitPlanMode"}
            return [delegate], None  # A fifth launch must be blocked even if the model emits it.
        if stage == 3:
            return [plan], None
        if stage == 4:
            return [("Read", {"file_path": "calc.py"})], None
        if stage == 5:
            return [("Write", {"file_path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})], None
        if stage == 6:
            return [("Bash", {"command": "python3 -m unittest -v", "description": "Run tests"})], None
        return [], "Fixed and tested."


def make_request(tmp_path, executable, base_url, *, timeout=60):
    return AgentRunRequest(
        agent_type="zcode",
        task=Task("calc-0", "agentbench-fixture/tiny-calc", "HEAD", "Fix add"),
        profile_name="plan-subagent",
        executable=Path(executable),
        model="dsh-test",
        api_base_url=base_url,
        workspace=tmp_path / "cache/agentbench-fixture__tiny-calc",
        artifact_dir=tmp_path / "artifacts",
        session_id="multi-task",
        timeout_seconds=timeout,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=10,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
        max_subagents=4,
        max_concurrent_subagents=2,
    )


@pytest.mark.parametrize("finish_early", [False, True])
def test_parallel_batches_and_limits(tmp_path, zcode_executable, finish_early):
    dataset = seed_tasks(tmp_path, 1)
    with MultiExplorer(finish_early=finish_early) as server:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "dataset": dataset,
                    "agent": {"type": "zcode", "profile": "plan-subagent", "executable": zcode_executable},
                    "backend": {"base_url": server.base_url, "model": "dsh-test", "endpoint": "/v1/chat/completions"},
                }
            )
        )
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from agentinfer.agentcache.entrypoints.cli.main import main; raise SystemExit(main(sys.argv[1:]))",
                "bench",
                "serve",
                "--agentinfer",
                "run",
                "--config",
                str(config_path),
                "--max-subagents",
                "4",
                "--max-concurrent-subagents",
                "2",
                "--timeout",
                "60",
                "--result-dir",
                str(tmp_path / "run"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        (tmp_path / "cli.log").write_text(run.stdout + run.stderr)
        (tmp_path / "wire-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in server.requests))
    assert run.returncode == 0, run.stdout + run.stderr
    artifact_dir = tmp_path / "run/tasks/calc-0"
    result = json.loads((artifact_dir / "result.json").read_text())
    assert result["outcome"] == "completed", result
    assert result["has_patch"]
    assert not server.errors, server.errors
    lineage = json.loads((artifact_dir / "zcode-lineage.json").read_text())
    assert lineage["started_subagents"] == lineage["exploration_reports"] == (2 if finish_early else 4)
    assert lineage["peak_concurrent_subagents"] == 2
    assert not lineage["pending"]
    children = [row for row in lineage["sessions"] if row["agent_role"] == "subagent"]
    assert len(children) == (2 if finish_early else 4)
    assert all(row["parent_program_id"] == f"{lineage['task_id']}:lead" for row in children)
    events = list(map(json.loads, (artifact_dir / "zcode-hooks.jsonl").read_text().splitlines()))
    blocked = [row for row in events if row["response"].get("decision") == "block"]
    if not finish_early:
        assert any("total subagent budget" in row["response"]["reason"] for row in blocked)
    assert any("slots are occupied" in row["response"]["reason"] for row in blocked)
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v"],
        cwd=tmp_path / "run/workspaces/calc-0",
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_parallel_children_timeout_and_cancel(tmp_path, zcode_executable, cancel):
    seed_tasks(tmp_path, 1)
    with MultiExplorer(hold_children=True) as server:
        request = make_request(tmp_path, zcode_executable, server.base_url, timeout=12)

        async def scenario():
            task = asyncio.create_task(run_zcode(request))
            assert await asyncio.to_thread(server.blocked.wait, 10)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.termination_reason.value == "timeout"

        try:
            asyncio.run(scenario())
        finally:
            server.release.set()
    status = json.loads((request.artifact_dir / "zcode-run-status.json").read_text())
    assert status["termination_reason"] == ("cancelled" if cancel else "timeout")
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)
