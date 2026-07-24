"""Verify Claude state, settings, and launch environment."""

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.claude.settings import (
    bootstrap_claude_state,
    build_claude_env,
    build_claude_settings,
)


def test_bootstrap_writes_onboarding_and_trust_state(tmp_path: Path) -> None:
    bootstrap_claude_state(tmp_path)

    state = json.loads((tmp_path / ".claude.json").read_text(encoding="utf-8"))

    assert state["onboardingComplete"] is True
    assert state["hasCompletedOnboarding"] is True
    assert state["projectTrustAccepted"] is True


def test_settings_never_contain_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")

    settings = build_claude_settings("http://proxy")

    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://proxy"
    assert "unique-secret" not in json.dumps(settings)
    assert "ANTHROPIC_AUTH_TOKEN" not in settings["env"]


def test_launch_environment_contains_task_local_state_and_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unique-secret")

    env = build_claude_env(tmp_path, "http://proxy")

    assert env["ANTHROPIC_AUTH_TOKEN"] == "unique-secret"
    assert env["ANTHROPIC_BASE_URL"] == "http://proxy"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path)


def test_launch_environment_requires_auth_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_AUTH_TOKEN"):
        build_claude_env(tmp_path, "http://proxy")
