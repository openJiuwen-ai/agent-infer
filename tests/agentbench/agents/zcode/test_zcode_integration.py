# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Real ZCode tools through AgentBench, using deterministic model responses."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.contracts import AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.zcode.runner import run_zcode
from agentinfer.agentbench.benchkit.dataset import Task
from tests.agentbench.helpers.dsh_mock_server import DshMockServer, seed_tasks

pytestmark = pytest.mark.timeout(180)


class ZCodeMockServer(DshMockServer):
    def respond(self, body):
        identity = json.loads(body["vllm_xargs"]["agentic_context"])
        key = identity["program_id"]
        stage = self.states.get(key, 0)
        self.states[key] = stage + 1
        self.requests.append({"identity": identity, "body": body})
        if identity["agent_role"] == "subagent":
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert names <= {"Read", "Glob", "Grep"}, names
            if stage == 0:
                return "Write", {"file_path": "forbidden-child-write", "content": "must not execute"}
            if stage == 1:
                return "Read", {"file_path": "calc.py"}
            return None, "The add function subtracts. Change a - b to a + b."
        if stage == 0:
            self.barrier.wait(timeout=30)
        if self.profile == "plan-subagent":
            names = {tool["function"]["name"] for tool in body["tools"]}
            if stage <= 1:
                assert names <= {"Agent", "Task"}
                assert body["tool_choice"]["function"]["name"] in names
                for tool in body["tools"]:
                    assert tool["function"]["parameters"]["properties"]["subagent_type"]["enum"] == [
                        f"agentbench-explore-{identity['task_id']}"
                    ]
            elif stage == 2:
                assert names == {"ExitPlanMode"}
                assert body["tool_choice"]["function"]["name"] == "ExitPlanMode"
            else:
                assert not names & {"Agent", "Task", "ExitPlanMode"}
            if stage == 0:
                return "Bash", {
                    "command": "touch forbidden-plan-write",
                    "description": "Attempt write before delegation",
                }
            if stage == 1:
                return "Agent", {
                    "subagent_type": f"agentbench-explore-{identity['task_id']}",
                    "description": "Inspect add",
                    "prompt": "Read calc.py and report the bug.",
                    "run_in_background": False,
                }
            if stage == 2:
                return "ExitPlanMode", {"plan": "Change add to return a + b, then run the existing unit tests."}
            stage -= 3
        if stage == 0:
            return "Read", {"file_path": "calc.py"}
        if stage == 1:
            return "Write", {"file_path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}
        if self.hold_after_patch:
            self.blocked.set()
            self.release.wait(30)
        if stage == 2:
            return "Bash", {"command": "python3 -m unittest -v", "description": "Run existing tests"}
        return None, "Fixed add and ran the tests."


@pytest.fixture
def zcode_executable():
    value = os.environ.get("ZCODE_TEST_EXECUTABLE")
    if not value:
        pytest.skip("Set ZCODE_TEST_EXECUTABLE to the pinned ZCode launcher")
    return str(Path(value).resolve())


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
def test_two_concurrent_tasks_through_benchmark(tmp_path, zcode_executable, profile):
    dataset = seed_tasks(tmp_path, 2)
    with ZCodeMockServer(profile, concurrent_tasks=2) as server:
        config = {
            "experiment": {"task_num": 2, "max_concurrency": 2, "task_timeout_seconds": 90, "patch_flush_seconds": 0},
            "dataset": dataset,
            "agent": {"type": "zcode", "profile": profile, "executable": zcode_executable, "context_window": 32768},
            "backend": {"base_url": server.base_url, "model": "dsh-test", "endpoint": "/v1/chat/completions"},
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        output = tmp_path / "run"
        command = [
            sys.executable,
            "-c",
            "import sys; from agentinfer.agentcache.entrypoints.cli.main import main; raise SystemExit(main(sys.argv[1:]))",
            "bench",
            "serve",
            "--agentinfer",
            "run",
            "--config",
            str(config_path),
            "--result-dir",
            str(output),
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=150)
        (tmp_path / "cli.log").write_text(run.stdout + run.stderr)
        (tmp_path / "wire-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in server.requests))
    assert run.returncode == 0, run.stdout + run.stderr
    facts = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(facts) == len(server.requests)
    assert all(row["body"]["max_tokens"] == 8192 for row in server.requests)
    assert all(row["status"] == "success" for row in facts)
    assert len({row["identity"]["task_id"] for row in server.requests}) == 2
    for directory in (output / "tasks").iterdir():
        result = json.loads((directory / "result.json").read_text())
        assert result["outcome"] == "completed", (result, (directory / "zcode.stderr.log").read_text())
        lineage = json.loads((directory / "zcode-lineage.json").read_text())
        assert result["has_patch"]
        assert lineage["auto_plan_approvals"] == int(profile == "plan-subagent")
        if profile == "plan-subagent":
            children = [row for row in lineage["sessions"] if row["agent_role"] == "subagent"]
            assert children and all(row["parent_program_id"] == f"{lineage['task_id']}:lead" for row in children)
        workspace = output / "workspaces" / directory.name
        assert not (workspace / "forbidden-plan-write").exists()
        assert not (workspace / "forbidden-child-write").exists()
        assert ".zcode" not in (directory / "model.patch").read_text()
        subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=workspace, check=True, capture_output=True)
        subprocess.run(
            ["git", "apply", "--check", "--reverse", str(directory / "model.patch")], cwd=workspace, check=True
        )


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_preserve_patch(tmp_path, zcode_executable, cancel):
    seed_tasks(tmp_path, 1)
    with ZCodeMockServer("single") as server:
        server.hold_after_patch = True
        request = AgentRunRequest(
            agent_type="zcode",
            task=Task("calc-0", "agentbench-fixture/tiny-calc", "HEAD", "Fix add"),
            profile_name="single",
            executable=Path(zcode_executable),
            model="dsh-test",
            api_base_url=server.base_url,
            workspace=tmp_path / "cache/agentbench-fixture__tiny-calc",
            artifact_dir=tmp_path / "artifacts",
            session_id="timeout-task",
            timeout_seconds=20,
            patch_flush_seconds=0,
            prompt_delivery_timeout_seconds=10,
            tmux_startup_seconds=0,
            terminal_capture_interval_seconds=1,
        )

        async def scenario():
            task = asyncio.create_task(run_zcode(request))
            assert await asyncio.to_thread(server.blocked.wait, 15), "ZCode did not write the patch"
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.termination_reason is TerminationReason.TIMEOUT

        try:
            asyncio.run(scenario())
        finally:
            server.release.set()
    assert (request.artifact_dir / "model.patch").exists()
    status = json.loads((request.artifact_dir / "zcode-run-status.json").read_text())
    assert status["termination_reason"] == ("cancelled" if cancel else "timeout")
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)


def test_explore_returns_report_after_tool_budget(tmp_path, zcode_executable):
    class RepeatingExplorer(ZCodeMockServer):
        def respond(self, body):
            identity = json.loads(body["vllm_xargs"]["agentic_context"])
            if identity["agent_role"] != "subagent":
                return super().respond(body)
            self.requests.append({"identity": identity, "body": body})
            if body.get("tool_choice") == "none":
                assert "tools" not in body
                return None, "The function subtracts instead of adding. Replace a - b with a + b."
            return "Read", {"file_path": "calc.py"}

    seed_tasks(tmp_path, 1)
    with RepeatingExplorer("plan-subagent") as server:
        request = AgentRunRequest(
            agent_type="zcode",
            task=Task("calc-0", "agentbench-fixture/tiny-calc", "HEAD", "Fix add"),
            profile_name="plan-subagent",
            executable=Path(zcode_executable),
            model="dsh-test",
            api_base_url=server.base_url,
            workspace=tmp_path / "cache/agentbench-fixture__tiny-calc",
            artifact_dir=tmp_path / "artifacts",
            session_id="explore-budget",
            timeout_seconds=60,
            patch_flush_seconds=0,
            prompt_delivery_timeout_seconds=10,
            tmux_startup_seconds=0,
            terminal_capture_interval_seconds=1,
        )
        result = asyncio.run(run_zcode(request))
    assert result.outcome.value == "completed", (request.artifact_dir / "zcode.stderr.log").read_text()
    assert result.has_patch
    children = [r for r in server.requests if r["identity"]["agent_role"] == "subagent"]
    assert len(children) == 9
    assert children[-1]["body"]["tool_choice"] == "none"
