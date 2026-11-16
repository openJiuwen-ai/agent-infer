# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the common functional artifact contract without agent processes."""

from pathlib import Path

import pytest

from tests.e2e.function.agentbench.artifact_validation import validate_agent_artifacts

_DATA_FILES = (
    "agents.csv",
    "distribution_samples.csv",
    "manifest.json",
    "requests.jsonl",
    "sessions.csv",
    "summary.json",
    "task_index.json",
)
_LOGS = ("proxy.stdout.log", "proxy.stderr.log")
_DIRS = ("evidence", "tasks", "workspaces")


@pytest.fixture
def run_artifacts(tmp_path: Path) -> Path:
    for name in _DATA_FILES:
        (tmp_path / name).write_text("data\n")
    for name in _LOGS:
        (tmp_path / name).touch()
    for name in _DIRS:
        (tmp_path / name).mkdir()
    return tmp_path


def test_accepts_common_artifacts_with_empty_proxy_logs(run_artifacts):
    validate_agent_artifacts(run_artifacts)


@pytest.mark.parametrize("name", (*_DATA_FILES, *_LOGS, *_DIRS))
def test_rejects_missing_artifacts(run_artifacts, name):
    path = run_artifacts / name
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        validate_agent_artifacts(run_artifacts)


@pytest.mark.parametrize("name", _DATA_FILES)
def test_rejects_empty_data_files(run_artifacts, name):
    (run_artifacts / name).write_text("")
    with pytest.raises(RuntimeError, match="empty"):
        validate_agent_artifacts(run_artifacts)


def test_accepts_additional_files(run_artifacts):
    (run_artifacts / "reproduce_issue.py").write_text("print('reproduce')\n")
    validate_agent_artifacts(run_artifacts)


def test_rejects_additional_root_directories(run_artifacts):
    (run_artifacts / "extra-diagnostics").mkdir()
    with pytest.raises(RuntimeError, match="unexpected directory"):
        validate_agent_artifacts(run_artifacts)


@pytest.mark.parametrize("name", ["summary.json", "tasks"])
def test_rejects_wrong_artifact_type(run_artifacts, name):
    path = run_artifacts / name
    if path.is_dir():
        path.rmdir()
        path.touch()
    else:
        path.unlink()
        path.mkdir()
    with pytest.raises(RuntimeError, match="missing"):
        validate_agent_artifacts(run_artifacts)
