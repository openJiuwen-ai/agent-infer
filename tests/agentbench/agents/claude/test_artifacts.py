"""Verify Claude runtime artifacts do not persist authentication secrets."""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents import AgentRunRequest
from agentinfer.agentbench.agents.claude import runner
from agentinfer.agentbench.benchkit.dataset import Task


def test_runtime_artifacts_exclude_auth_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "unique-auth-secret"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = AgentRunRequest(
        agent_type="claude",
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="single",
        executable=Path("claude"),
        model="model",
        api_base_url="http://proxy",
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="12345678-1234-1234-1234-123456789abc",
        timeout_seconds=1,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=1,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)
    monkeypatch.setattr(
        runner, "bootstrap_claude_state", lambda config_dir: (config_dir / ".claude.json").write_text("{}")
    )
    monkeypatch.setattr(runner, "start_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "remove_global_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "set_history_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "session_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(runner, "kill_server", lambda *_args: None)
    monkeypatch.setattr(runner, "export_patch", lambda _workspace: "")
    monkeypatch.setattr(runner, "_sleep_with_deadline", lambda *_args: asyncio.sleep(0, result=True))

    asyncio.run(runner.run_claude(request))

    for artifact in request.artifact_dir.rglob("*"):
        if artifact.is_file():
            assert secret not in artifact.read_text(encoding="utf-8", errors="replace")
    assert "--debug-file" not in (request.artifact_dir / "claude-launch-command.txt").read_text(encoding="utf-8")
    assert not (request.artifact_dir / "claude-launch-env.json").exists()
    assert not (request.artifact_dir / "claude-debug.log").exists()
