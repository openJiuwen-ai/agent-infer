# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise functional artifact checks without launching agent processes."""

import json
from pathlib import Path

import pytest

from tests.e2e.function.agentbench.artifact_validation import validate_agent_artifacts


@pytest.fixture(params=["claude", "jiuwenswarm", "dsh", "zcode"])
def run_artifacts(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, str, Path, Path]:
    return _make_run_artifacts(tmp_path, request.param)


def _make_run_artifacts(tmp_path: Path, agent: str) -> tuple[Path, str, Path, Path]:
    task = tmp_path / "tasks" / "task-1"
    task.mkdir(parents=True)
    (tmp_path / "workspaces" / "task-1").mkdir(parents=True)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "summary.json").write_text(json.dumps({"lifecycle": {"status": "completed"}}))
    (tmp_path / "requests.jsonl").write_text("{}\n")
    (tmp_path / "task_index.json").write_text(
        json.dumps({"tasks": [{"instance_id": "task-1", "outcome": "completed"}]})
    )
    transcript, state = {
        "claude": ("transcript.jsonl", "claude_state"),
        "jiuwenswarm": ("jiuwenswarm.jsonl", "jiuwenswarm-instance/home"),
        "dsh": ("transcript.jsonl", "dsh-home"),
        "zcode": ("transcript.jsonl", "zcode-home"),
    }[agent]
    (task / state).mkdir(parents=True)
    (task / transcript).write_text("{}\n")
    if agent == "zcode":
        artifacts = {
            "zcode-run-status.json": {
                "outcome": "completed",
                "exit_code": 0,
                "termination_reason": None,
                "native_error": None,
                "prompt_loaded": True,
                "execution_phase": "agent",
            },
            "zcode-lineage.json": {"sessions": [{"id": "lead"}], "failure": None, "pending": []},
            "zcode-config.json": {"schemaVersion": 1},
            "zcode-command.json": ["zcode", "--prompt", "prompt.txt"],
        }
        for name, content in artifacts.items():
            (task / name).write_text(json.dumps(content))
        (task / "zcode-hooks.jsonl").write_text("{}\n")
        (task / "zcode.stderr.log").write_text("")
    return tmp_path, agent, task / transcript, task / state


def test_accepts_runtime_artifacts(run_artifacts):
    root, agent, _, _ = run_artifacts
    validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize("missing", ["manifest", "transcript", "state", "workspace"])
def test_rejects_missing_evidence(run_artifacts, missing):
    root, agent, transcript, state = run_artifacts
    if missing == "manifest":
        (root / "manifest.json").unlink()
    elif missing == "transcript":
        transcript.write_text("")
    elif missing == "state":
        state.rmdir()
    else:
        (root / "workspaces" / "task-1").rmdir()
    with pytest.raises(RuntimeError):
        validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize("instance_id", ["../escape", "..", "C:escape", "task-1\\escape"])
def test_rejects_invalid_task_paths(run_artifacts, instance_id):
    root, agent, _, _ = run_artifacts
    (root / "task_index.json").write_text(json.dumps({"tasks": [{"instance_id": instance_id, "outcome": "completed"}]}))
    with pytest.raises(RuntimeError, match="instance_id"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


def test_rejects_task_count_mismatch(run_artifacts):
    root, agent, _, _ = run_artifacts
    with pytest.raises(RuntimeError, match="2 tasks"):
        validate_agent_artifacts(root, agent, expected_tasks=2)


@pytest.mark.parametrize("present", [False, True])
def test_zcode_reproduction_file_is_optional(tmp_path: Path, present):
    root, agent, _, _ = _make_run_artifacts(tmp_path, "zcode")
    if present:
        (root / "reproduce_issue.py").write_text("print('reproduction')\n")
    validate_agent_artifacts(root, agent, expected_tasks=1)


def test_rejects_other_extra_root_files(run_artifacts):
    root, agent, _, _ = run_artifacts
    (root / "unexpected.py").write_text("print('unexpected')\n")
    with pytest.raises(RuntimeError, match="unexpected entry"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


def test_rejects_workspace_escape(run_artifacts):
    root, agent, _, _ = run_artifacts
    (root / "leaked-workspace").mkdir()
    with pytest.raises(RuntimeError, match="unexpected entry"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


def test_accepts_dsh_session_log_fallback(run_artifacts):
    root, agent, transcript, _ = run_artifacts
    if agent != "dsh":
        return
    transcript.unlink()
    sessions = transcript.parent / "dsh-sessions"
    sessions.mkdir()
    (sessions / "session-1.jsonl").write_text("{}\n")
    validate_agent_artifacts(root, agent, expected_tasks=1)


def test_rejects_dsh_hmr_startup_error(tmp_path: Path):
    root, agent, transcript, _ = _make_run_artifacts(tmp_path, "dsh")
    (transcript.parent / "dsh.stderr.log").write_text("--expose-internals is required for HMR service")
    with pytest.raises(RuntimeError, match="HMR"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize("reason", ["agent_startup_failed", "agent_startup_timeout", "harness_error"])
def test_rejects_failure_summary_for_all_agents(run_artifacts, reason):
    root, agent, _, _ = run_artifacts
    path = root / "task_index.json"
    index = json.loads(path.read_text())
    index["failure_summary"] = {"by_termination_reason": {reason: 1}}
    path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match=reason):
        validate_agent_artifacts(root, agent, expected_tasks=1)


def test_rejects_isolated_state_outside_tasks_for_all_agents(run_artifacts):
    root, agent, _, _ = run_artifacts
    state_path = {
        "claude": "claude_state",
        "jiuwenswarm": "jiuwenswarm-instance/home",
        "dsh": "dsh-home",
        "zcode": "zcode-home",
    }[agent]
    (root / "evidence" / state_path).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="escapes"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize("field", ["termination_reason", "error"])
def test_rejects_failure_evidence_on_completed_tasks(run_artifacts, field):
    root, agent, _, _ = run_artifacts
    path = root / "task_index.json"
    index = json.loads(path.read_text())
    index["tasks"][0][field] = "harness_error" if field == "termination_reason" else {"message": "startup failed"}
    path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="failure evidence"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize(
    "name",
    [
        "zcode-run-status.json",
        "zcode-lineage.json",
        "zcode-config.json",
        "zcode-command.json",
        "zcode-hooks.jsonl",
        "zcode.stderr.log",
    ],
)
def test_rejects_missing_zcode_artifacts(tmp_path: Path, name):
    root, agent, transcript, _ = _make_run_artifacts(tmp_path, "zcode")
    (transcript.parent / name).unlink()
    with pytest.raises(RuntimeError, match="ZCode"):
        validate_agent_artifacts(root, agent, expected_tasks=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exit_code", 1),
        ("prompt_loaded", False),
        ("execution_phase", "startup"),
        ("native_error", {"code": "MODEL_CONTEXT_EXCEEDED"}),
    ],
)
def test_rejects_zcode_failed_status(tmp_path: Path, field, value):
    root, agent, transcript, _ = _make_run_artifacts(tmp_path, "zcode")
    path = transcript.parent / "zcode-run-status.json"
    status = json.loads(path.read_text())
    status[field] = value
    path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="run status"):
        validate_agent_artifacts(root, agent, expected_tasks=1)
