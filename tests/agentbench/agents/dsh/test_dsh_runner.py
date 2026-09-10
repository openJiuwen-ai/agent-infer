# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the DSH headless runner lifecycle and termination classification."""

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.dsh.runner import _classify, _communicate_with_deadline
from agentinfer.agentbench.benchkit.dataset import Task


def _request(tmp_path: Path) -> AgentRunRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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


class _FakeInstance:
    def __init__(self, **kwargs):
        self.artifact_dir = kwargs["artifact_dir"]
        self.home_dir = self.artifact_dir / "dsh-home"
        self.policy_patch_path = self.home_dir / "agentbench.patch.yml"

    def bootstrap(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)

    def environment(self) -> dict[str, str]:
        return {"DSH_HOME": str(self.home_dir), "DSH_PERMISSION_MODE": "workspace-write"}


class _EmptyStream:
    async def read(self, _limit: int = -1) -> bytes:
        return b""


class _FakeProcess:
    returncode: int | None = 0
    pid = 99999
    stderr = _EmptyStream()

    async def wait(self) -> int:
        return self.returncode or 0


class _HangingProcess(_FakeProcess):
    returncode = None

    async def wait(self) -> int:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _completed_run_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    timed_out: bool = False,
) -> tuple[AgentRunRequest, dict]:
    from agentinfer.agentbench.agents.dsh import runner

    state: dict = {}

    async def create_process(*command, **kwargs):
        state["command"] = command
        state["kwargs"] = kwargs
        return _FakeProcess()

    async def communicate(process, deadline):
        return b"answer", b"", timed_out

    async def copy_transcript(*_args):
        return None

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_deadline", communicate)
    monkeypatch.setattr(runner, "_copy_transcript", copy_transcript)
    monkeypatch.setattr(runner, "load_session_log", lambda _path: _completed_turn_events())
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "diff --git a/a b/a\n")

    request = _request(tmp_path)
    result = runner.asyncio.run(runner.run_dsh(request))
    return request, state, result


def test_runner_initializes_profile_before_writing_task_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    events: list[str] = []

    class Instance(_FakeInstance):
        def bootstrap(self) -> None:
            events.append("bootstrap")
            super().bootstrap()

    async def create_process(*command, **kwargs):
        if "--dump-config" in command:
            events.append("initialize")
        return _FakeProcess()

    async def communicate(process, deadline):
        return b"answer", b"", False

    async def copy_transcript(*_args):
        return None

    monkeypatch.setattr(runner, "DshInstance", Instance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_deadline", communicate)
    monkeypatch.setattr(runner, "_copy_transcript", copy_transcript)
    monkeypatch.setattr(runner, "load_session_log", lambda _path: _completed_turn_events())
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert events == ["initialize", "bootstrap"]


def test_runner_completes_and_exports_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, state, result = _completed_run_test(tmp_path, monkeypatch)

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert result.termination_reason is None
    assert result.has_patch is True
    assert state["command"][1:5] == (
        "--profile",
        "headless",
        "--patch",
        str(request.artifact_dir / "dsh-home" / "agentbench.patch.yml"),
    )
    assert state["kwargs"]["cwd"] == request.workspace
    assert state["kwargs"]["env"]["DSH_PERMISSION_MODE"] == "workspace-write"
    assert state["kwargs"]["stdin"] is asyncio.subprocess.DEVNULL
    assert (tmp_path / "artifacts" / "model.patch").exists()
    assert (tmp_path / "artifacts" / "dsh.stdout.log").read_bytes() == b"answer"


def test_runner_profile_initialization_respects_task_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    calls: list[tuple[str, ...]] = []

    async def create_process(*command, **_kwargs):
        calls.append(command)
        return _HangingProcess()

    async def kill(process):
        process.returncode = 137

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_kill_process_group", kill)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    request = replace(_request(tmp_path), timeout_seconds=0.01)
    result = runner.asyncio.run(runner.run_dsh(request))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED
    assert calls == [("dsh", "--profile", "headless", "--dump-config")]


def test_runner_profile_initialization_nonzero_exit_is_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    class Process(_FakeProcess):
        returncode = 2

    async def create_process(*command, **_kwargs):
        assert "--dump-config" in command
        return Process()

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED
    error_log = tmp_path / "artifacts" / "dsh-runner-error.log"
    assert error_log.read_text(encoding="utf-8").startswith("_ProfileInitializationError:")
    assert "DSH profile initialization failed" in error_log.read_text(encoding="utf-8")


def test_runner_deadline_wins_over_completed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, _state, result = _completed_run_test(tmp_path, monkeypatch, timed_out=True)

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.TIMEOUT
    # DSH's SIGTERM graceful dispose can exit 0, so deadline state decides.
    assert (tmp_path / "artifacts" / "model.patch").exists()


def test_runner_rejects_prompt_exceeding_argv_limit_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    spawned: list[tuple[str, ...]] = []

    async def create_process(*command, **_kwargs):
        spawned.append(command)
        return _FakeProcess()

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_copy_transcript", lambda *_args: None)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")
    if hasattr(runner.os, "sysconf"):
        monkeypatch.setattr(runner.os, "sysconf", lambda _name: 1024)
    else:  # Windows: stub a sysconf for the guard under test.
        monkeypatch.setattr(runner.os, "sysconf", lambda _name: 1024, raising=False)

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED
    # The main launch never happens; the oversized prompt is rejected pre-spawn.
    assert all("--dump-config" in command for command in spawned)
    error_log = tmp_path / "artifacts" / "dsh-runner-error.log"
    assert error_log.exists()
    assert "argv limit" in error_log.read_text(encoding="utf-8")


def test_runner_reports_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    class Process(_FakeProcess):
        returncode = 1

    async def create_process(*command, **_kwargs):
        if "--dump-config" in command:
            return _FakeProcess()
        return Process()

    async def communicate(process, deadline):
        return b"", b"dsh: MISSING_CREDENTIAL: no API key", False

    async def copy_transcript(*_args):
        return None

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_deadline", communicate)
    monkeypatch.setattr(runner, "_copy_transcript", copy_transcript)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED
    # The stable transcript artifact exists even on early failure paths.
    assert result.transcript == str(tmp_path / "artifacts" / "transcript.jsonl")
    assert (tmp_path / "artifacts" / "transcript.jsonl").exists()


def test_runner_reports_interrupted_without_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    class Process(_FakeProcess):
        returncode = 1

    async def create_process(*command, **_kwargs):
        if "--dump-config" in command:
            return _FakeProcess()
        return Process()

    async def communicate(process, deadline):
        return b"", b"", False

    async def copy_transcript(*_args):
        return None

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_deadline", communicate)
    monkeypatch.setattr(runner, "_copy_transcript", copy_transcript)
    monkeypatch.setattr(runner, "load_session_log", lambda _path: [_raw_event("tool/call", {"name": "bash"})])
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.INTERRUPTED


def test_runner_cancellation_kills_process_and_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    calls: list[str] = []

    class Process(_FakeProcess):
        returncode = None

    async def create_process(*command, **_kwargs):
        if "--dump-config" in command:
            return _FakeProcess()
        return Process()

    async def communicate(process, deadline):
        raise asyncio.CancelledError

    async def kill(process):
        calls.append("kill")
        process.returncode = 1

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_deadline", communicate)
    monkeypatch.setattr(runner, "_kill_process_group", kill)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    request = _request(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        runner.asyncio.run(runner.run_dsh(request))

    assert calls == ["kill"]


def test_runner_bootstrap_failure_is_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    class FailingInstance:
        def __init__(self, **kwargs):
            self.artifact_dir = kwargs["artifact_dir"]
            self.home_dir = self.artifact_dir / "dsh-home"

        def environment(self) -> dict[str, str]:
            return {"DSH_HOME": str(self.home_dir)}

        def bootstrap(self):
            raise OSError("disk full")

    async def initialize(*_args) -> None:
        return None

    monkeypatch.setattr(runner, "DshInstance", FailingInstance)
    monkeypatch.setattr(runner, "_initialize_profile", initialize)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.HARNESS_ERROR
    error_log = tmp_path / "artifacts" / "dsh-runner-error.log"
    assert error_log.exists()
    assert "OSError" in error_log.read_text(encoding="utf-8")


def test_copy_transcript_prefers_root_over_larger_subagent_log(tmp_path: Path) -> None:
    import json as json_module

    from agentinfer.agentbench.agents.dsh import runner

    home = tmp_path / "home"
    root_dir = home / "sessions" / "proj" / "session-root"
    sub_dir = home / "sessions" / "proj" / "session-sub"
    root_dir.mkdir(parents=True)
    sub_dir.mkdir(parents=True)
    sub_events = [
        {"type": "session", "version": 0, "id": "session-sub", "parentSession": "session-root", "delegationDepth": 1},
        *[
            _raw_event("assistant/message", {"message": {"content": [{"type": "text", "text": f"sub {i}"}]}})
            for i in range(10)
        ],
    ]
    root_events = [
        {"type": "session", "version": 0, "id": "session-root", "delegationDepth": 0, "cwd": "/w"},
        _raw_event("turn/end", {"reason": {"kind": "completed"}}),
    ]
    (sub_dir / "session.jsonl").write_text(
        "\n".join(json_module.dumps(event) for event in sub_events) + "\n",
        encoding="utf-8",
    )
    (root_dir / "session.jsonl").write_text(
        "\n".join(json_module.dumps(event) for event in root_events) + "\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    transcript = runner.asyncio.run(runner._copy_transcript(home, artifacts))

    assert transcript is not None
    events = runner.load_session_log(transcript)
    assert events[0]["id"] == "session-root"
    preserved = list((artifacts / "dsh-sessions").glob("session-*.jsonl"))
    assert len(preserved) == 2


def test_runner_spawn_failure_is_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    async def create_process(*_command, **_kwargs):
        raise FileNotFoundError("no dsh")

    monkeypatch.setattr(runner, "DshInstance", _FakeInstance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_dsh(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED


def test_classify_ask_user_loop_is_confirmation_hang() -> None:
    filler = [
        _raw_event("assistant/message", {"message": {"content": [{"type": "text", "text": "step"}]}}) for _ in range(60)
    ]
    asks = [_raw_event("tool/call", {"name": "ask_user_question", "arguments": '{"questions":[]}'}) for _ in range(3)]
    events = filler + asks + [_raw_event("turn/end", {"reason": {"kind": "error"}})]

    assert _classify(1, b"", events) is TerminationReason.CONFIRMATION_HANG


def test_classify_zero_exit_and_completed_turn_complete() -> None:
    assert _classify(0, b"", []) is None
    assert _classify(1, b"", _completed_turn_events()) is None


@pytest.mark.parametrize(
    "events",
    [
        [_raw_event("approval/asked", {"action": "bash"})],
        [_raw_event("turn/end", {"reason": {"kind": "blocked"}})],
    ],
)
def test_classify_confirmation_hang(events: list[dict]) -> None:
    assert _classify(1, b"", events) is TerminationReason.CONFIRMATION_HANG


@pytest.mark.skipif(
    os.name == "nt", reason="os.killpg is POSIX-only; on Windows the runner takes the process.kill() path"
)
def test_communicate_kills_process_group_on_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.agents.dsh import runner

    class Process:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = None
            self.pid = 99999
            self.killed = False

        async def wait(self) -> int:
            while not self.killed:
                await asyncio.sleep(0.01)
            self.returncode = 137
            return 137

    holder: dict[str, Process] = {}

    def killpg(pid, sig):
        assert sig == __import__("signal").SIGKILL
        holder["process"].killed = True
        holder["process"].stdout.feed_eof()
        holder["process"].stderr.feed_eof()

    monkeypatch.setattr(runner.os, "killpg", killpg)

    async def scenario() -> tuple[bytes, bytes, bool]:
        # StreamReaders capture their loop at construction, so build the fake
        # process inside the running loop.
        fake = Process()
        holder["process"] = fake
        return await _communicate_with_deadline(fake, time.monotonic() + 0.05)

    stdout, stderr, timed_out = runner.asyncio.run(scenario())

    assert timed_out is True
    assert holder["process"].killed is True
    assert stdout == b""
    assert stderr == b""
