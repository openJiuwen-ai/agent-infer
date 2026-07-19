"""Verify the task-owned tmux process boundary."""

import subprocess
from pathlib import Path

import pytest

from agentcache.benchmarks.agents.claude.tmux import (
    TmuxCommandError,
    TmuxTarget,
    capture_pane,
    kill_server,
    remove_global_environment,
    start_session,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_start_session_passes_secret_only_through_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs["env"]))
        return _completed()

    monkeypatch.setattr("agentcache.benchmarks.agents.claude.tmux.subprocess.run", fake_run)
    target = TmuxTarget("agentinfer-task")
    secret = "unique-secret"

    start_session(
        target,
        command=["claude", "--model", "model"],
        cwd=tmp_path,
        launch_env={"ANTHROPIC_AUTH_TOKEN": secret},
    )

    args, env = calls[0]
    assert args[:3] == ["tmux", "-L", target.socket_name]
    assert secret not in " ".join(args)
    assert env["ANTHROPIC_AUTH_TOKEN"] == secret
    assert "remain-on-exit" in args
    assert "respawn-pane" in args


def test_control_command_removes_token_from_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_env: dict[str, str] = {}

    def fake_run(_args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_env.update(kwargs["env"])
        return _completed()

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")
    monkeypatch.setattr("agentcache.benchmarks.agents.claude.tmux.subprocess.run", fake_run)

    remove_global_environment(TmuxTarget("agentinfer-task"), "ANTHROPIC_AUTH_TOKEN")

    assert "ANTHROPIC_AUTH_TOKEN" not in seen_env


def test_capture_failure_raises_without_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentcache.benchmarks.agents.claude.tmux.subprocess.run",
        lambda *_args, **_kwargs: _completed(1, stderr="pane unavailable"),
    )

    with pytest.raises(TmuxCommandError, match="pane unavailable") as error:
        capture_pane(TmuxTarget("agentinfer-task"))

    assert "ANTHROPIC_AUTH_TOKEN" not in str(error.value)


def test_cleanup_attempts_dedicated_server_even_without_session(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed()

    monkeypatch.setattr("agentcache.benchmarks.agents.claude.tmux.subprocess.run", fake_run)

    kill_server(TmuxTarget("agentinfer-task"))

    assert calls == [["tmux", "-L", "agentinfer-task", "kill-server"]]


def test_cleanup_reports_failure_when_server_remains(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([_completed(1, stderr="kill failed"), _completed()])
    monkeypatch.setattr(
        "agentcache.benchmarks.agents.claude.tmux.subprocess.run",
        lambda *_args, **_kwargs: next(results),
    )

    with pytest.raises(TmuxCommandError, match="kill failed"):
        kill_server(TmuxTarget("agentinfer-task"))
