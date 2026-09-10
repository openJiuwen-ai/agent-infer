# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
import time
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.jiuwenswarm.runner import _JsonlStatusTracker
from agentinfer.agentbench.benchkit.dataset import Task


def _event(event: str, payload: dict) -> bytes:
    return (json.dumps({"type": "event", "event": event, "payload": payload}) + "\n").encode()


def _classify_jsonl(stdout: bytes, return_code: int) -> TerminationReason | None:
    tracker = _JsonlStatusTracker()
    tracker.feed(stdout)
    tracker.finish()
    return tracker.classify(return_code)


def test_terminal_jsonl_with_zero_exit_completes() -> None:
    stdout = _event("chat.delta", {"content": "x"}) + _event(
        "chat.final",
        {"event_type": "chat.final", "content": "done"},
    )

    assert _classify_jsonl(stdout, 0) is None
    # Terminal JSONL status takes precedence over non-zero exit code.
    # If CLI sent chat.final, the task completed even if the process was killed afterward.
    assert _classify_jsonl(stdout, 1) is None


def test_control_chat_final_is_not_terminal() -> None:
    stdout = _event("chat.final", {"event_type": "chat.llm_usage"})

    assert _classify_jsonl(stdout, 0) is TerminationReason.INTERRUPTED


def test_interactive_and_malformed_jsonl_fail_closed() -> None:
    assert (
        _classify_jsonl(_event("chat.ask_user_question", {"questions": []}), 0) is TerminationReason.CONFIRMATION_HANG
    )
    assert _classify_jsonl(b"not-json\n", 0) is TerminationReason.HARNESS_ERROR


def _request(tmp_path: Path) -> AgentRunRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentRunRequest(
        agent_type="jiuwenswarm",
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="code.normal",
        executable=Path("jiuwenswarm"),
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


def test_runner_passes_prompt_on_stdin_and_exports_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    state = {}

    class Instance:
        gateway_url = "ws://127.0.0.1:21003/tui"
        environment = {"ENV": "value"}

        def __init__(self, **kwargs):
            state["instance_kwargs"] = kwargs
            self.workspace = kwargs["workspace"]

        async def start(self):
            state["started"] = True
            (self.workspace / ".gitignore").write_bytes(b"# JiuwenSwarm runtime file operation logs\n.agent_history/\n")
            (self.workspace / ".agent_history").mkdir()
            (self.workspace / "prompt_attachment").mkdir()
            (self.workspace / "prompt_attachment" / "README.md").write_text("runtime", encoding="utf-8")

        async def stop(self):
            state["stopped"] = True

        async def interrupt(self):
            raise AssertionError("normal completion must not interrupt")

    class Process:
        returncode = 0

    async def create_process(*command, **kwargs):
        state["command"] = command
        state["process_kwargs"] = kwargs
        return Process()

    async def communicate(_process, prompt: bytes, _deadline: float, _prompt_timeout: float):
        state["prompt"] = prompt
        return (
            _event("chat.final", {"event_type": "chat.final", "content": "done"}),
            b"diagnostic",
            False,
            None,
        )

    monkeypatch.setattr(runner, "JiuwenInstance", Instance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_timeout", communicate)

    def export_clean_patch(workspace: Path) -> str:
        # Runtime artifacts are no longer cleaned by the runner.
        # Workspace isolation is enforced at the benchmark architecture level.
        assert (workspace / ".gitignore").exists()
        assert (workspace / ".agent_history").exists()
        assert (workspace / "prompt_attachment").exists()
        return "diff --git a/a b/a\n"

    monkeypatch.setattr(runner, "export_patch", export_clean_patch)

    result = runner.asyncio.run(runner.run_jiuwenswarm(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert result.termination_reason is None
    assert result.tmux_session is None
    assert result.has_patch is True
    assert state["started"] is state["stopped"] is True
    assert state["instance_kwargs"]["completion_timeout_seconds"] == 10
    assert state["prompt"].decode().startswith("You are an AI coding agent.")
    assert "--trusted-dir" not in state["command"]
    assert state["command"][-2:] == ("--timeout", "10")
    assert (tmp_path / "artifacts" / "model.patch").exists()


def test_runner_timeout_interrupts_before_instance_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    calls = []

    class Instance:
        gateway_url = "ws://127.0.0.1:21003/tui"
        environment = {}

        def __init__(self, **_kwargs):
            pass

        async def start(self):
            calls.append("start")

        async def interrupt(self):
            calls.append("interrupt")
            return True

        async def stop(self):
            calls.append("stop")

    class Process:
        returncode = None
        pid = 123

    async def create_process(*_args, **_kwargs):
        return Process()

    async def communicate(_process, _prompt: bytes, _deadline: float, _prompt_timeout: float):
        calls.append("kill")
        return b"", b"", True, None

    monkeypatch.setattr(runner, "JiuwenInstance", Instance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_timeout", communicate)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")

    result = runner.asyncio.run(runner.run_jiuwenswarm(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.TIMEOUT
    assert calls == ["start", "kill", "interrupt", "stop"]


def test_runner_cancellation_preserves_runtime_workspace_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    calls = []

    class Instance:
        gateway_url = "ws://127.0.0.1:21003/tui"
        environment = {}

        def __init__(self, **kwargs):
            self.workspace = kwargs["workspace"]

        async def start(self):
            calls.append("start")
            (self.workspace / ".gitignore").write_bytes(b"# JiuwenSwarm runtime file operation logs\n.agent_history/\n")
            (self.workspace / ".agent_history").mkdir()
            (self.workspace / "prompt_attachment").mkdir()

        async def interrupt(self):
            calls.append("interrupt")
            return True

        async def stop(self):
            calls.append("stop")

    class Process:
        returncode = 0

    async def create_process(*_args, **_kwargs):
        return Process()

    async def communicate(_process, _prompt: bytes, _deadline: float, _prompt_timeout: float):
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "JiuwenInstance", Instance)
    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "_communicate_with_timeout", communicate)
    monkeypatch.setattr(
        runner,
        "export_patch",
        lambda _workspace: pytest.fail("cancelled tasks must not export a patch"),
    )

    request = _request(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        runner.asyncio.run(runner.run_jiuwenswarm(request))

    assert calls == ["start", "interrupt", "stop"]
    # Runtime artifacts are preserved even on cancellation.
    assert (request.workspace / ".gitignore").exists()
    assert (request.workspace / ".agent_history").exists()
    assert (request.workspace / "prompt_attachment").exists()


@pytest.mark.parametrize(
    ("deadline_seconds", "prompt_delivery_timeout_seconds"),
    [(1, 0.01), (0.01, 1)],
)
def test_prompt_delivery_is_bounded_by_prompt_and_task_timeouts(
    deadline_seconds: float,
    prompt_delivery_timeout_seconds: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    class Stdin:
        closed = False

        def write(self, _prompt: bytes) -> None:
            pass

        async def drain(self) -> None:
            await asyncio.Event().wait()

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            pass

    class Process:
        def __init__(self) -> None:
            self.stdin = Stdin()
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = None
            self.pid = 123
            self.killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = 1
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            if self.returncode is None:
                await asyncio.Event().wait()
            return self.returncode

    async def run():
        process = Process()

        def killpg(_pid: int, _signal: int) -> None:
            process.kill()

        monkeypatch.setattr(runner.os, "killpg", killpg)
        result = await runner._communicate_with_timeout(
            process,
            b"prompt",
            deadline=time.monotonic() + deadline_seconds,
            prompt_delivery_timeout_seconds=prompt_delivery_timeout_seconds,
        )
        return process, result

    process, (stdout, stderr, timed_out, reason) = asyncio.run(run())

    assert process.stdin.closed is True
    assert process.killed is True
    assert stdout == stderr == b""
    assert timed_out is True
    assert reason is TerminationReason.INTERRUPTED


def test_kill_cli_process_group_is_idempotent_when_process_exits_before_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    class Process:
        returncode = None
        pid = 123

        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    async def run() -> None:
        process = Process()

        def killpg(_pid: int, _signal: int) -> None:
            process.returncode = 0
            raise ProcessLookupError

        monkeypatch.setattr(runner.os, "name", "posix")
        monkeypatch.setattr(runner.os, "killpg", killpg, raising=False)
        monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
        await runner._kill_cli_process_group(process)
        assert process.returncode == 0

    asyncio.run(run())


def test_communicate_timeout_cleanup_survives_process_exit_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentinfer.agentbench.agents.jiuwenswarm import runner

    class Stdin:
        closed = False

        def write(self, _prompt: bytes) -> None:
            pass

        async def drain(self) -> None:
            await asyncio.Event().wait()

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            pass

    async def run():
        class Process:
            def __init__(self) -> None:
                self.stdin = Stdin()
                self.stdout = asyncio.StreamReader()
                self.stderr = asyncio.StreamReader()
                self.returncode = None
                self.pid = 123

            def kill(self) -> None:
                raise ProcessLookupError

            async def wait(self) -> int:
                if self.returncode is None:
                    await asyncio.Event().wait()
                return self.returncode

        process = Process()

        def killpg(_pid: int, _signal: int) -> None:
            process.returncode = 0
            process.stdout.feed_eof()
            process.stderr.feed_eof()
            raise ProcessLookupError

        monkeypatch.setattr(runner.os, "name", "posix")
        monkeypatch.setattr(runner.os, "killpg", killpg, raising=False)
        monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
        return await runner._communicate_with_timeout(
            process,
            b"prompt",
            deadline=time.monotonic() + 0.01,
            prompt_delivery_timeout_seconds=1,
        )

    stdout, stderr, timed_out, reason = asyncio.run(run())

    assert timed_out is True
    assert reason is TerminationReason.INTERRUPTED
    assert stdout == stderr == b""


@pytest.mark.parametrize(
    ("trailing_output", "expected_reason"),
    [
        (_event("chat.final", {"event_type": "chat.final"}), None),
        (_event("chat.ask_user_question", {"questions": []}), TerminationReason.CONFIRMATION_HANG),
        (_event("chat.error", {"message": "failed"}), TerminationReason.INTERRUPTED),
        (b"not-json\n", TerminationReason.HARNESS_ERROR),
    ],
)
def test_jsonl_status_tracker_classifies_complete_stream(
    trailing_output: bytes,
    expected_reason: TerminationReason | None,
) -> None:
    """Verify that status tracker correctly classifies streams without truncation."""
    from agentinfer.agentbench.agents.jiuwenswarm.runner import _read_jsonl_and_track_status

    output = _event("chat.delta", {"content": "abcdefgh"}) + trailing_output

    async def run():
        stream = asyncio.StreamReader()
        stream.feed_data(output)
        stream.feed_eof()
        return await _read_jsonl_and_track_status(stream)

    captured, tracker = asyncio.run(run())

    assert captured == output
    assert tracker.classify(0) is expected_reason


def test_jsonl_status_tracker_handles_lines_split_across_chunks() -> None:
    output = _event("chat.final", {"event_type": "chat.final"})
    tracker = _JsonlStatusTracker()

    tracker.feed(output[:3])
    tracker.feed(output[3:-2])
    tracker.feed(output[-2:])
    tracker.finish()

    assert tracker.classify(0) is None
