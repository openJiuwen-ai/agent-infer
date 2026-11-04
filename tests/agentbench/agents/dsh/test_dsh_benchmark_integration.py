# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Real pinned DSH through the production CLI, process proxy and workspaces."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.dsh.runner import run_dsh
from agentinfer.agentbench.agents.dsh.transcript import DshTranscriptNormalizer, load_session_log
from agentinfer.agentbench.benchkit.dataset import Task
from tests.agentbench.helpers.dsh_mock_server import DshMockServer, seed_tasks

pytestmark = [pytest.mark.dsh_runtime, pytest.mark.timeout(180)]


@pytest.fixture
def dsh_executable():
    path = os.environ.get("DSH_TEST_EXECUTABLE") or shutil.which("dsh")
    if path is None:
        pytest.skip("Pinned DSH is not installed")
    return str(Path(path).resolve())


@pytest.fixture
def real_request(tmp_path, dsh_executable):
    seed_tasks(tmp_path, 1)
    return AgentRunRequest(
        agent_type="dsh",
        task=Task("fixture", "agentbench-fixture/tiny-calc", "HEAD", "fix add"),
        profile_name="single",
        executable=Path(dsh_executable),
        model="dsh-test",
        api_base_url="http://proxy",
        workspace=tmp_path / "cache" / "agentbench-fixture__tiny-calc",
        artifact_dir=tmp_path / "artifacts",
        session_id="fixture-task",
        timeout_seconds=60,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=1,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
def test_real_dsh_benchmark_two_concurrent_tasks(tmp_path, dsh_executable, profile):
    dataset = seed_tasks(tmp_path, 2)
    with DshMockServer(profile, concurrent_tasks=2) as server:
        config = {
            "experiment": {
                "task_num": 2,
                "max_concurrency": 2,
                "task_timeout_seconds": 90,
                "run_timeout_seconds": 120,
                "patch_flush_seconds": 0,
            },
            "dataset": dataset,
            "agent": {"type": "dsh", "profile": profile, "executable": dsh_executable, "context_window": 32768},
            "backend": {"base_url": server.base_url, "model": "dsh-test", "endpoint": "/v1/chat/completions"},
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        run_dir = tmp_path / "run"
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
            str(run_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=150)
        (tmp_path / "cli.log").write_text(completed.stdout + completed.stderr)
        (tmp_path / "wire-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in server.requests))
        assert completed.returncode == 0, completed.stderr
    facts = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    assert len(facts) == len(server.requests)
    assert all(row["status"] == "success" for row in facts)
    for task_dir in sorted((run_dir / "tasks").iterdir()):
        result = json.loads((task_dir / "result.json").read_text())
        assert result["outcome"] == "completed", (result, (task_dir / "dsh-run-status.json").read_text())
        assert result["has_patch"]
        for log in (task_dir / "dsh-sessions").glob("session-*.jsonl"):
            events = load_session_log(log)
            assert events[0]["version"] == 4
            normalized = DshTranscriptNormalizer().normalize(events)
            assert {"assistant", "tool_call", "tool_result"} <= {event.type for event in normalized}
        assert not (run_dir / "workspaces" / task_dir.name / "forbidden-child-write").exists()
        assert not (run_dir / "workspaces" / task_dir.name / "forbidden-plan-write").exists()
        lineage = json.loads((task_dir / "dsh-lineage.json").read_text())
        rows = lineage["sessions"]
        roots = [row for row in rows if row["agent_role"] == "lead"]
        assert len(roots) == 1
        if profile == "plan-subagent":
            children = [row for row in rows if row["agent_role"] == "subagent"]
            assert children and all(row["parent_program_id"] == roots[0]["program_id"] for row in children)
            assert lineage["auto_plan_approvals"] == 1
        else:
            assert len(rows) == 1
        observed = [row["identity"] for row in server.requests if row["identity"]["task_id"] == lineage["task_id"]]
        assert {row["program_id"] for row in observed} == {row["program_id"] for row in rows}
        workspace = run_dir / "workspaces" / task_dir.name
        subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=workspace, check=True, capture_output=True)
        subprocess.run(
            ["git", "apply", "--check", "--reverse", str(task_dir / "model.patch")], cwd=workspace, check=True
        )
    assert (run_dir / "evidence" / "dsh-runtime.json").is_file()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("cancel", [False, True])
def test_real_dsh_preserves_patch_after_timeout_or_cancel(real_request, cancel):
    with DshMockServer("single") as server:
        server.hold_after_patch = True
        request = replace(real_request, api_base_url=server.base_url, timeout_seconds=8)

        async def scenario():
            task = asyncio.create_task(run_dsh(request))
            ready = await asyncio.to_thread(server.blocked.wait, 6)
            assert ready, "DSH did not execute the patch before the controlled wait"
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.termination_reason is TerminationReason.TIMEOUT
                assert result.has_patch

        try:
            asyncio.run(scenario())
        finally:
            server.release.set()
    artifact = request.artifact_dir
    assert (artifact / "model.patch").is_file()
    assert (artifact / "dsh-sessions" / "raw" / "index.json").is_file()
    status = json.loads((artifact / "dsh-run-status.json").read_text())
    assert status["termination_reason"] == ("cancelled" if cancel else "timeout")
    for line in (artifact / "dsh-processes.jsonl").read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(json.loads(line)["pid"], 0)


def test_real_dsh_long_prompt_uses_stdin(real_request):
    with DshMockServer("single") as server:
        request = replace(
            real_request,
            api_base_url=server.base_url,
            task=replace(real_request.task, problem_statement="中文\n" * 50000),
        )
        result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.COMPLETED
    prompt = (request.artifact_dir / "prompt.txt").read_text()
    assert len(prompt.encode()) > 131072
    messages = server.requests[0]["body"]["messages"]
    assert any(message.get("content") == prompt for message in messages)
    command = (request.artifact_dir / "dsh-launch-command.json").read_text()
    assert len(command) < 2048 and "中文" not in command


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("repeat", TerminationReason.SUBAGENT_READ_LOOP),
        ("plan-recovery", None),
        ("plan-refusal", TerminationReason.PLAN_NOT_APPROVED),
        ("lead-loop", TerminationReason.VALIDATION_ERROR_LOOP),
        ("lead-invalid-args", TerminationReason.VALIDATION_ERROR_LOOP),
    ],
)
def test_real_dsh_enforces_bounded_exploration_and_plan_recovery(real_request, scenario, expected):
    class ScriptedServer(DshMockServer):
        skipped_plan = False
        lead_calls = 0

        def respond(self, body):
            identity = json.loads(body["vllm_xargs"]["agentic_context"])
            tool, value = super().respond(body)
            if (
                scenario in {"lead-loop", "lead-invalid-args"}
                and identity["agent_role"] == "lead"
                and tool is None
                and self.states.get((identity["task_id"], identity["program_id"]), 0) == 3
            ):
                self.lead_calls += 1
                if self.lead_calls % 2:
                    return ("exit_plan_mode" if scenario == "lead-loop" else "read"), {}
                return "write", {"file_path": "completion.txt", "content": "Finished."}
            if scenario == "repeat" and identity["agent_role"] == "subagent":
                return "read", {"file_path": "calc.py"}
            if (
                scenario in {"plan-recovery", "plan-refusal"}
                and tool == "exit_plan_mode"
                and (scenario == "plan-refusal" or not self.skipped_plan)
            ):
                self.skipped_plan = True
                self.states[(identity["task_id"], identity["program_id"])] = 1
                return None, "The exploration is complete; here is my final analysis."
            return tool, value

    with ScriptedServer("plan-subagent") as server:
        request = replace(real_request, api_base_url=server.base_url, profile_name="plan-subagent")
        result = asyncio.run(run_dsh(request))
    assert result.termination_reason is expected, (request.artifact_dir / "dsh-run-status.json").read_text()
    assert (result.outcome is AgentRunOutcome.COMPLETED) == (expected is None)
    lineage = json.loads((request.artifact_dir / "dsh-lineage.json").read_text())
    if expected is None:
        assert result.has_patch
        assert lineage["auto_plan_approvals"] == 1
        assert lineage["plan_recoveries"] == 1
        subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=request.workspace, check=True, capture_output=True)
    else:
        assert lineage["failure"]["reason"] == expected
        assert lineage["auto_plan_approvals"] == (1 if scenario in {"lead-loop", "lead-invalid-args"} else 0)
    if scenario in {"lead-loop", "lead-invalid-args"}:
        assert result.has_patch
        assert server.lead_calls <= 8
        subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=request.workspace, check=True, capture_output=True)
    if scenario == "repeat":
        assert sum(row["identity"]["agent_role"] == "subagent" for row in server.requests) <= 5
    assert (request.artifact_dir / "transcript.jsonl").is_file()
