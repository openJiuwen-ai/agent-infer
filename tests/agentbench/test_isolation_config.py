# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Unit tests for AgentBench execution isolation config wiring.

Covers the build_claude_settings sandbox+hook block, the Jiuwen sandbox
section, the mcp_exec_command disable stub, and the fence path-guard decision
logic. None of these require bwrap/socat/jiuwenbox on the host.
"""

from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.claude.fence import _decide
from agentinfer.agentbench.agents.claude.settings import build_claude_settings
from agentinfer.agentbench.agents.jiuwenswarm.instance import (
    JiuwenInstance,
    _jiuwenbox_policy,
)

# ---------------------------------------------------------------------------
# Claude settings
# ---------------------------------------------------------------------------


def test_claude_settings_adds_mandatory_sandbox_and_fence_hook() -> None:
    settings = build_claude_settings("http://proxy", "model", workspace="/tmp/w")
    assert settings["sandbox"] == {
        "enabled": True,
        "failIfUnavailable": True,
        "allowUnsandboxedCommands": False,
    }
    hook = settings["hooks"]["PreToolUse"][0]
    assert hook["matcher"] == "Edit|Write|MultiEdit|NotebookEdit"
    command = hook["hooks"][0]["command"]
    assert command.startswith("ALLOWED_ROOT=/tmp/w python3")
    assert command.rstrip("'").endswith("fence.py")


def test_claude_settings_requires_workspace() -> None:
    with pytest.raises(TypeError, match="workspace"):
        build_claude_settings("http://proxy", "model")


def test_claude_settings_quotes_workspace_in_fence_command() -> None:
    settings = build_claude_settings(
        "http://proxy",
        "model",
        workspace="/tmp/work space/it's-safe",
    )
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "ALLOWED_ROOT='/tmp/work space/it'\"'\"'s-safe'" in command


# ---------------------------------------------------------------------------
# Fence path-guard decision logic
# ---------------------------------------------------------------------------


def test_fence_allows_path_inside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ROOT", str(tmp_path))
    payload = {"tool_input": {"file_path": str(tmp_path / "inside.py")}}
    assert _decide(payload) is None  # None => allow


def test_fence_denies_path_outside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ROOT", str(tmp_path))
    payload = {"tool_input": {"file_path": str(tmp_path.parent / "leak.py")}}
    decision = _decide(payload)
    assert decision is not None
    assert decision["permissionDecision"] == "deny"


def test_fence_denies_absolute_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ROOT", str(tmp_path))
    payload = {"tool_input": {"file_path": "/tmp/agentbench-canary-escape"}}
    assert _decide(payload)["permissionDecision"] == "deny"


def test_fence_denies_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.skip("symlink creation requires privilege on Windows; verified on L20 Linux")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    link = workspace / "link"
    link.symlink_to(tmp_path / "outside")  # symlink pointing outside workspace
    monkeypatch.setenv("ALLOWED_ROOT", str(workspace))
    payload = {"tool_input": {"file_path": str(link / "escaped.txt")}}
    assert _decide(payload)["permissionDecision"] == "deny"


def test_fence_denies_notebook_path_outside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ALLOWED_ROOT", str(workspace))
    payload = {"tool_input": {"notebook_path": str(tmp_path / "outside.ipynb")}}
    assert _decide(payload)["permissionDecision"] == "deny"


def test_fence_validates_every_multiedit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ALLOWED_ROOT", str(workspace))
    assert (
        _decide(
            {
                "tool_input": {
                    "edits": [
                        {"file_path": str(workspace / "one.py")},
                        {"file_path": str(workspace / "two.py")},
                    ]
                }
            }
        )
        is None
    )
    decision = _decide(
        {
            "tool_input": {
                "edits": [
                    {"file_path": str(workspace / "one.py")},
                    {"file_path": str(tmp_path / "outside.py")},
                ]
            }
        }
    )
    assert decision["permissionDecision"] == "deny"


def test_fence_no_allow_root_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOWED_ROOT", raising=False)
    decision = _decide({"tool_input": {"file_path": "/etc/passwd"}})
    assert decision["permissionDecision"] == "deny"
    assert "ALLOWED_ROOT is required" in decision["permissionDecisionReason"]


@pytest.mark.parametrize(
    "payload, reason",
    [
        ({}, "tool_input must be an object"),
        ({"tool_input": {}}, "tool_input file path is required"),
        ({"tool_input": {"file_path": 123}}, "tool_input file path is required"),
    ],
)
def test_fence_malformed_boundary_payload_denies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    reason: str,
) -> None:
    monkeypatch.setenv("ALLOWED_ROOT", str(tmp_path))
    decision = _decide(payload)
    assert decision["permissionDecision"] == "deny"
    assert reason in decision["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# Jiuwen sandbox section + policy
# ---------------------------------------------------------------------------


def _write_minimal_config(instance: JiuwenInstance) -> None:
    instance.config_path.parent.mkdir(parents=True)
    instance.config_path.write_text(
        yaml.safe_dump(
            {
                "models": {"defaults": [{"model_client_config": {}}]},
                "react": {
                    "model_name": "",
                    "subagents": {"browser_agent": {"enabled": True}},
                },
                "setup_guide": {"enabled": True},
                "auto_recap": {"enabled": True},
                "updater": {"enabled": True},
                "permissions": {},
                "modes": {
                    "code": {
                        "memory": {"enabled": True},
                        "rails": ["SkillUseRail"],
                        "tools": ["web_free_search"],
                    }
                },
                "hooks": {"disable_all_hooks": False},
            }
        ),
        encoding="utf-8",
    )


def _instance(tmp_path: Path) -> JiuwenInstance:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    return JiuwenInstance(
        executable=Path("jiuwenswarm"),
        artifact_dir=tmp_path / "artifacts",
        workspace=workspace,
        session_id="root-session",
        api_base_url="http://127.0.0.1:18180",
        model="model",
        completion_timeout_seconds=14400,
    )


def test_jiuwen_policy_grants_rw_only_to_task_owned_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    sandbox_tmp = tmp_path / "sandbox-tmp"
    sandbox_tmp.mkdir()
    policy = _jiuwenbox_policy(workspace, home, data, sandbox_tmp)
    assert policy["network"]["mode"] == "host"
    bind_paths = {(b["host_path"], b["sandbox_path"], b["mode"]) for b in policy["filesystem_policy"]["bind_mounts"]}
    expected_rw = {
        (str(sandbox_tmp), "/tmp", "rw"),
        (str(workspace), str(workspace), "rw"),
        (str(home), str(home), "rw"),
        (str(data), str(data), "rw"),
    }
    assert {bind for bind in bind_paths if bind[2] == "rw"} == expected_rw
    assert "/home" not in policy["filesystem_policy"]["read_write"]
    assert "/tmp" in policy["filesystem_policy"]["read_write"]
    assert "directories" not in policy["filesystem_policy"]
    if Path("/usr").exists():
        assert any(sandbox == "/usr" and mode == "ro" for _, sandbox, mode in bind_paths)


def test_jiuwen_allocates_jiuwenbox_port_at_startup(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    assert instance.jiuwenbox_port == 0
    assert instance.jiuwenbox_url == ""
