"""Verify Claude runtime outcomes and owned-resource cleanup."""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents import AgentRunOutcome, AgentRunRequest, TerminationReason
from agentinfer.agentbench.agents.claude import runner
from agentinfer.agentbench.agents.claude.interaction import InteractionAction
from agentinfer.agentbench.benchkit.dataset import Task


def _request(
    tmp_path: Path,
    *,
    timeout_seconds: int = 2,
    patch_flush_seconds: int = 0,
) -> AgentRunRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentRunRequest(
        agent_type="claude",
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="single",
        executable=Path("claude"),
        model="model",
        api_base_url="http://proxy",
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="12345678-1234-1234-1234-123456789abc",
        timeout_seconds=timeout_seconds,
        patch_flush_seconds=patch_flush_seconds,
        prompt_delivery_timeout_seconds=1,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transcript: list[dict[str, object]],
    patch: str = "",
    action: InteractionAction | None = None,
) -> list[str]:
    cleanup: list[str] = []
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")
    monkeypatch.setattr(
        runner, "bootstrap_claude_state", lambda config_dir: (config_dir / ".claude.json").write_text("{}")
    )
    monkeypatch.setattr(runner, "start_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "remove_global_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "set_history_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "session_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "pane_has_exited", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(runner, "capture_pane", lambda *_args, **_kwargs: "❯ ")
    monkeypatch.setattr(runner, "load_and_paste_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "send_keys", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "kill_server", lambda target: cleanup.append(target.socket_name))
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: patch)
    monkeypatch.setattr(runner, "_find_transcript_path", lambda *_args: Path("transcript.jsonl"))
    monkeypatch.setattr(runner.TranscriptLoader, "load", lambda *_args: transcript)
    monkeypatch.setattr(runner, "_file_signature_async", lambda _path: asyncio.sleep(0, result=(1, 1)))
    monkeypatch.setattr(runner, "_sleep_with_deadline", lambda *_args: asyncio.sleep(0, result=True))
    if action is not None:

        def scripted_action(_controller: object, context: object) -> InteractionAction:
            if not context.state.startup_dismissed:
                context.state.startup_dismissed = True
                return InteractionAction(kind="none")
            return action

        monkeypatch.setattr(runner.InteractionController, "process", scripted_action)
    return cleanup


def test_completed_transcript_returns_normal_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup = _patch_runtime(
        monkeypatch,
        transcript=[{"type": "assistant", "message": {"stop_reason": "end_turn", "content": []}}],
    )

    result = asyncio.run(runner.run_claude(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert result.termination_reason is None
    assert len(cleanup) == 1


def test_idle_after_patch_is_failure_with_patch_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch = "diff --git a/file b/file\n"
    _patch_runtime(
        monkeypatch,
        transcript=[{"type": "assistant", "message": {"content": []}}],
        patch=patch,
        action=InteractionAction(kind="check_workspace"),
    )
    monkeypatch.setattr(runner, "workspace_has_changes", lambda _workspace: True)

    result = asyncio.run(runner.run_claude(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.IDLE_AFTER_PATCH
    assert result.has_patch is True
    assert result.patch_bytes == len(patch.encode())
    assert (result_path := tmp_path / "artifacts" / "model.patch").read_text(encoding="utf-8") == patch
    assert result_path.is_file()


def test_early_agent_exit_is_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")
    monkeypatch.setattr(
        runner, "bootstrap_claude_state", lambda config_dir: (config_dir / ".claude.json").write_text("{}")
    )
    monkeypatch.setattr(runner, "start_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "remove_global_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "set_history_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "session_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "pane_has_exited", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "capture_pane", lambda *_args, **_kwargs: "agent startup error")
    monkeypatch.setattr(runner, "kill_server", lambda *_args: None)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")
    monkeypatch.setattr(runner, "_sleep_with_deadline", lambda *_args: asyncio.sleep(0, result=True))

    result = asyncio.run(runner.run_claude(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.AGENT_STARTUP_FAILED
    assert (tmp_path / "artifacts" / "terminal-final.log").read_text(encoding="utf-8") == "agent startup error"


def test_cancellation_cleans_server_and_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup: list[str] = []
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")
    monkeypatch.setattr(
        runner, "bootstrap_claude_state", lambda config_dir: (config_dir / ".claude.json").write_text("{}")
    )
    monkeypatch.setattr(runner, "start_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "remove_global_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "set_history_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "kill_server", lambda target: cleanup.append(target.socket_name))

    async def cancel_during_startup(*_args: object) -> bool:
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "_sleep_with_deadline", cancel_during_startup)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_claude(_request(tmp_path)))

    assert len(cleanup) == 1


def test_cancellation_during_launch_waits_then_cleans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()
    cleanup: list[str] = []
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")
    monkeypatch.setattr(
        runner, "bootstrap_claude_state", lambda config_dir: (config_dir / ".claude.json").write_text("{}")
    )

    def blocked_start(*_args: object, **_kwargs: object) -> None:
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(runner, "start_session", blocked_start)
    monkeypatch.setattr(runner, "kill_server", lambda target: cleanup.append(target.socket_name))

    async def exercise() -> None:
        task = asyncio.create_task(runner.run_claude(_request(tmp_path)))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert len(cleanup) == 1


def test_final_evidence_stops_claude_before_patch_flush(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    request = _request(tmp_path, patch_flush_seconds=1)
    request.artifact_dir.mkdir()

    async def session_exists(*_args: object) -> bool:
        return True

    async def capture(*_args: object) -> str:
        return "terminal"

    async def pane_running(*_args: object) -> bool:
        return False

    async def send(_target: object, _deadline: float, *keys: str) -> None:
        events.append("send:" + ",".join(keys))

    async def interrupt_wait(*_args: object) -> bool:
        events.append("interrupt-wait")
        return True

    async def flush(_delay: float) -> None:
        events.append("flush")

    async def no_transcript(*_args: object) -> None:
        return None

    async def no_patch(*_args: object) -> str:
        return ""

    monkeypatch.setattr(runner, "_session_exists_async", session_exists)
    monkeypatch.setattr(runner, "_capture_pane_async", capture)
    monkeypatch.setattr(runner, "_pane_has_exited_async", pane_running)
    monkeypatch.setattr(runner, "_send_keys_async", send)
    monkeypatch.setattr(runner, "_sleep_with_deadline", interrupt_wait)
    monkeypatch.setattr(runner.asyncio, "sleep", flush)
    monkeypatch.setattr(runner, "_find_transcript_path_async", no_transcript)
    monkeypatch.setattr(runner, "_export_patch_async", no_patch)

    asyncio.run(
        runner._collect_final_evidence(
            request,
            runner.AgentRunResult(),
            runner.TmuxTarget("agentinfer-task"),
            runner.TranscriptLoader(),
            tmp_path,
            tmp_path / "transcript.jsonl",
        )
    )
    assert events == ["send:C-d", "interrupt-wait", "send:C-d", "flush"]


def test_transcript_is_reparsed_only_when_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0
    _patch_runtime(monkeypatch, transcript=[])
    signatures = iter([(1, 1), (1, 1), (2, 2)])

    async def signature(_path: Path) -> tuple[int, int]:
        return next(signatures)

    def load(*_args: object) -> list[dict[str, object]]:
        nonlocal loads
        loads += 1
        if loads == 2:
            return [{"type": "assistant", "message": {"stop_reason": "end_turn", "content": []}}]
        return []

    monkeypatch.setattr(runner, "_file_signature_async", signature)
    monkeypatch.setattr(runner.TranscriptLoader, "load", load)

    result = asyncio.run(runner.run_claude(_request(tmp_path)))

    assert result.outcome is AgentRunOutcome.COMPLETED
    assert loads == 3
