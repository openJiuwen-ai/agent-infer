# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise runner deadlines, stdin, evidence and cleanup with real processes."""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.dsh.runner import _classify, run_dsh
from agentinfer.agentbench.benchkit.dataset import Task

pytestmark = pytest.mark.timeout(20)


def _request(tmp_path: Path) -> AgentRunRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunRequest(
        agent_type="dsh",
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="single",
        executable=Path("dsh"),
        model="model",
        api_base_url="http://proxy",
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="root-session",
        timeout_seconds=10,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=2,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )


def _raw_event(event_type: str, data: dict) -> dict:
    return {"seq": 0, "type": event_type, "data": data}


def _completed_turn_events() -> list[dict]:
    return [_raw_event("turn/end", {"reason": {"kind": "completed"}})]


@pytest.fixture
def request_factory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calc.py").write_text("old\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"],
        cwd=workspace,
        check=True,
    )
    script = tmp_path / "fake-dsh"
    script.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, sys, time, subprocess
from pathlib import Path
home = Path(os.environ['DSH_HOME'])
home.mkdir(parents=True, exist_ok=True)
artifact = home.parent
mode = os.environ.get('TEST_AGENT_SCENARIO', 'complete')
if '--dump-config' in sys.argv:
    if mode == 'startup-timeout': time.sleep(60)
    if mode == 'config-timeout' and (home / 'cordis.patch.yml').exists(): time.sleep(60)
    if mode == 'startup-error':
        sys.stderr.write('broken profile')
        sys.exit(2)
    config = home / 'cordis.patch.yml'
    print(config.read_text() if config.exists() else '[]')
    sys.exit(0)
if mode == 'input-block': time.sleep(60)
prompt = sys.stdin.buffer.read()
(artifact / 'stdin.bin').write_bytes(prompt)
(artifact / 'argv.json').write_text(json.dumps(sys.argv))
(artifact / 'pid').write_text(str(os.getpid()))
print(json.dumps({'type': 'session', 'sessionId': 'root'}), flush=True)
if mode == 'no-log': sys.exit(0)
folder = home / 'sessions' / 'project' / 'root'
folder.mkdir(parents=True)
events = [{'type': 'session', 'version': 4, 'id': 'root', 'delegationDepth': 0},
          {'type': 'turn/end', 'data': {'reason': {'kind': 'completed'}}}]
(folder / 'session.v4.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
(artifact / 'dsh-lineage.json').write_text(json.dumps({'task_id': os.environ['AGENTBENCH_ROOT_SESSION_ID'],
    'sessions': [{'session_id': 'root', 'program_id': 'root', 'agent_role': 'lead'}], 'auto_plan_approvals': 0}))
Path('calc.py').write_text('fixed 中文\n')
if mode == 'large-output':
    for _ in range(32):
        print(json.dumps({'type': 'text', 'text': 'x' * 65536}), flush=True)
        sys.stderr.write('y' * 65536)
        sys.stderr.flush()
if mode == 'child-pipe':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (artifact / 'child-pid').write_text(str(child.pid))
(artifact / 'ready').touch()
if mode == 'wait': time.sleep(60)
if mode == 'nonzero': sys.exit(1)
"""
    )
    script.chmod(0o755)

    def build(mode="complete", **changes):
        monkeypatch.setenv("TEST_AGENT_SCENARIO", mode)
        return replace(_request(tmp_path), executable=script, **{"timeout_seconds": 5, **changes})

    return build


def test_runner_completes_and_exports_patch(request_factory):
    request = request_factory()
    result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.COMPLETED
    assert result.has_patch
    assert result.patch_bytes == len((request.artifact_dir / "model.patch").read_bytes())
    assert (request.artifact_dir / "dsh-sessions" / "raw" / "index.json").is_file()
    assert (request.artifact_dir / "transcript.jsonl").is_file()
    assert result.session_id == "root-session"
    subprocess.run(
        ["git", "apply", "--check", "--reverse", str(request.artifact_dir / "model.patch")],
        cwd=request.workspace,
        check=True,
    )


@pytest.mark.parametrize("budget,expected", [(0.01, TerminationReason.CLEANUP_TIMEOUT), (2.0, None)])
def test_slow_collection_uses_configured_cleanup_budget(request_factory, monkeypatch, budget, expected):
    from agentinfer.agentbench.agents.dsh import runner

    collect = runner._copy_transcript

    def slow_collection(*args, **kwargs):
        time.sleep(0.04)
        return collect(*args, **kwargs)

    monkeypatch.setattr(runner, "_copy_transcript", slow_collection)
    request = request_factory(cleanup_timeout_seconds=budget)
    result = asyncio.run(run_dsh(request))
    status = json.loads((request.artifact_dir / "dsh-run-status.json").read_text())
    assert result.termination_reason is expected
    assert status["cleanup_budget_seconds"] == budget
    assert (result.outcome is AgentRunOutcome.COMPLETED) == (expected is None)
    if expected is None:
        assert result.has_patch
    else:
        assert "TimeoutError" in status["collection"]


def test_collection_timeout_preserves_execution_timeout(request_factory, monkeypatch):
    def timed_out(*args, **kwargs):
        raise TimeoutError("session collection expired")

    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runner._copy_transcript", timed_out)
    request = request_factory("wait", timeout_seconds=2.0)
    result = asyncio.run(run_dsh(request))
    assert result.termination_reason is TerminationReason.TIMEOUT
    assert result.has_patch
    status = json.loads((request.artifact_dir / "dsh-run-status.json").read_text())
    assert "session collection expired" in status["collection"]


@pytest.mark.parametrize(
    "mode,expected", [("complete", TerminationReason.CLEANUP_TIMEOUT), ("wait", TerminationReason.TIMEOUT)]
)
def test_patch_timeout_preserves_existing_failure(request_factory, monkeypatch, mode, expected):
    def timed_out(*args, **kwargs):
        raise TimeoutError("patch export expired")

    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runner.export_patch", timed_out)
    request = request_factory(mode, timeout_seconds=2.0)
    result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is expected
    status = json.loads((request.artifact_dir / "dsh-run-status.json").read_text())
    assert "patch export expired" in status["patch"]


def test_stdin_preserves_long_utf8_prompt_and_drains_both_outputs(request_factory):
    task = Task("instance", "owner/repo", "abc", '中文 "quoted"\n' * 200000 + "\n")
    request = request_factory("large-output", task=task)
    result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.COMPLETED
    assert (request.artifact_dir / "stdin.bin").read_bytes() == (request.artifact_dir / "prompt.txt").read_bytes()
    command = json.loads((request.artifact_dir / "dsh-launch-command.json").read_text())
    assert len(json.dumps(command)) < 1024 and "--json" in command
    assert (request.artifact_dir / "dsh.stdout.log").stat().st_size > 2_000_000
    assert (request.artifact_dir / "dsh.stderr.log").stat().st_size > 2_000_000


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("startup-timeout", TerminationReason.AGENT_STARTUP_TIMEOUT),
        ("startup-error", TerminationReason.AGENT_STARTUP_FAILED),
        ("config-timeout", TerminationReason.TIMEOUT),
        ("input-block", TerminationReason.TIMEOUT),
        ("wait", TerminationReason.TIMEOUT),
        ("child-pipe", TerminationReason.TIMEOUT),
        ("no-log", TerminationReason.HARNESS_ERROR),
        ("nonzero", TerminationReason.INTERRUPTED),
    ],
)
def test_failures_do_not_become_completed(request_factory, mode, expected):
    request = request_factory(mode, timeout_seconds=2.0)
    started = time.monotonic()
    result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is expected
    assert time.monotonic() - started < 5
    if mode in {"wait", "child-pipe"}:
        assert result.has_patch
    assert (request.artifact_dir / "dsh-run-status.json").is_file()


def test_cancellation_collects_patch_before_reraising(request_factory):
    request = request_factory("wait")

    async def scenario():
        task = asyncio.create_task(run_dsh(request))
        while not (request.artifact_dir / "ready").exists():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    status = json.loads((request.artifact_dir / "dsh-run-status.json").read_text())
    assert status["termination_reason"] == "cancelled"
    assert status["patch"] == "exported"
    assert (request.artifact_dir / "transcript.jsonl").is_file()
    pid = int((request.artifact_dir / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_transcript_selects_current_generation_and_keeps_raw_history(tmp_path):
    from agentinfer.agentbench.agents.dsh.runner import _copy_transcript
    from agentinfer.agentbench.agents.dsh.transcript import load_session_log

    root_id = "root"
    events = [
        {"type": "session", "version": 4, "id": root_id, "delegationDepth": 0},
        *_completed_turn_events(),
    ]
    home = tmp_path / "home"
    folder = home / "sessions" / "project" / root_id
    folder.mkdir(parents=True)
    (folder / "session.v4.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (folder / "session.jsonl").write_text(json.dumps({"type": "session", "version": 0, "id": root_id}) + "\n")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "dsh-lineage.json").write_text(
        json.dumps(
            {"task_id": "task", "sessions": [{"session_id": root_id, "program_id": root_id, "agent_role": "lead"}]}
        )
    )
    trace = artifact / "requests.jsonl"
    trace.write_bytes(b"online identity facts must not be rewritten\n")
    path, collected, lineage = _copy_transcript(
        home, artifact, root_id=root_id, task_id="task", deadline=time.monotonic() + 5
    )
    assert load_session_log(path) == collected == events
    assert lineage["task_id"] == "task"
    index = json.loads((artifact / "dsh-sessions" / "raw" / "index.json").read_text())
    assert {Path(name).name for name in index[0]["generations"]} == {"session.jsonl", "session.v4.jsonl"}
    assert trace.read_bytes() == b"online identity facts must not be rewritten\n"


def test_classify_ask_user_loop_is_confirmation_hang() -> None:
    filler = [
        _raw_event("assistant/message", {"message": {"content": [{"type": "text", "text": "step"}]}}) for _ in range(60)
    ]
    asks = [_raw_event("tool/call", {"name": "ask_user_question", "arguments": '{"questions":[]}'}) for _ in range(3)]
    events = filler + asks + [_raw_event("turn/end", {"reason": {"kind": "error"}})]

    assert _classify(1, b"", events) is TerminationReason.CONFIRMATION_HANG


def test_classify_zero_exit_and_completed_turn_complete() -> None:
    assert _classify(0, b"", []) is TerminationReason.HARNESS_ERROR
    assert _classify(1, b"", _completed_turn_events()) is TerminationReason.INTERRUPTED
    assert _classify(0, b"", _completed_turn_events()) is None
    assert _classify(0, b"", [{"type": "final", "text": "done"}]) is TerminationReason.INTERRUPTED


@pytest.mark.parametrize(
    "events",
    [
        [_raw_event("approval/asked", {"action": "bash"})],
        [_raw_event("turn/end", {"reason": {"kind": "blocked"}})],
    ],
)
def test_classify_confirmation_hang(events: list[dict]) -> None:
    assert _classify(1, b"", events) is TerminationReason.CONFIRMATION_HANG


@pytest.mark.parametrize("return_code", [0, 1])
def test_token_limit_has_an_explicit_termination_reason(return_code):
    events = [_raw_event("turn/end", {"reason": {"kind": "max-tokens"}})]
    assert _classify(return_code, b"", events) is TerminationReason.MAX_TOKENS


def test_unapproved_plan_cannot_become_completed(request_factory):
    request = request_factory(profile_name="plan-subagent")
    result = asyncio.run(run_dsh(request))
    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.PLAN_NOT_APPROVED
    assert result.has_patch
