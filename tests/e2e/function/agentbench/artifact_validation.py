# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate the common artifacts produced by every BenchKit agent run."""

from pathlib import Path

_REQUIRED_FILES = (
    "agents.csv",
    "distribution_samples.csv",
    "manifest.json",
    "requests.jsonl",
    "sessions.csv",
    "summary.json",
    "task_index.json",
)
_PROXY_LOGS = ("proxy.stdout.log", "proxy.stderr.log")
_REQUIRED_DIRS = ("evidence", "tasks", "workspaces")


def validate_agent_artifacts(run_dir: Path) -> None:
    """Require common files and directories; proxy logs may be empty."""
    for name in (*_REQUIRED_FILES, *_PROXY_LOGS):
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing run artifact: {path}")
        if name in _REQUIRED_FILES and path.stat().st_size == 0:
            raise RuntimeError(f"empty run artifact: {path}")
    for name in _REQUIRED_DIRS:
        path = run_dir / name
        if not path.is_dir():
            raise RuntimeError(f"missing artifact directory: {path}")
    for path in run_dir.iterdir():
        if path.is_dir() and path.name not in _REQUIRED_DIRS:
            raise RuntimeError(f"unexpected directory at run root: {path}")
