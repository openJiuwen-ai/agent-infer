# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Post-run artifact validation for DSH SWE-bench functional E2E cases."""

from __future__ import annotations

import json
from pathlib import Path

_HMR_STARTUP_NEEDLE = "--expose-internals is required for HMR service"
_STARTUP_FAILURE = "agent_startup_failed"

_RUN_TOP_LEVEL = frozenset(
    {
        "manifest.json",
        "summary.json",
        "requests.jsonl",
        "task_index.json",
        "distribution_samples.csv",
        "agents.csv",
        "sessions.csv",
        # Request Proxy process logs written under the run directory.
        "proxy.stdout.log",
        "proxy.stderr.log",
    }
)
_RUN_TOP_DIRS = frozenset({"evidence", "tasks", "workspaces"})


def validate_dsh_swebench_run(run_dir: Path) -> None:
    """Assert a completed DSH SWE-bench run produced expected artifacts and no HMR startup failures."""

    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary.json under {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("lifecycle", {}).get("status") != "completed":
        raise RuntimeError(f"run lifecycle is not completed: {run_dir}")

    for name in ("manifest.json", "task_index.json", "requests.jsonl"):
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing {name} under {run_dir}")

    task_index = json.loads((run_dir / "task_index.json").read_text(encoding="utf-8"))
    failure_summary = task_index.get("failure_summary", {})
    by_reason = failure_summary.get("by_termination_reason", {})
    if isinstance(by_reason, dict) and by_reason.get(_STARTUP_FAILURE, 0) > 0:
        raise RuntimeError(
            f"run {run_dir} has {_STARTUP_FAILURE} tasks (regression for Cordis HMR boot); "
            f"failure_summary={failure_summary!r}"
        )

    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        raise RuntimeError(f"missing tasks/ directory under {run_dir}")

    task_rows = task_index.get("tasks", [])
    if not isinstance(task_rows, list) or not task_rows:
        raise RuntimeError(f"task_index.json has no tasks under {run_dir}")

    for row in task_rows:
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id")
        outcome = row.get("outcome")
        if not isinstance(instance_id, str) or not instance_id:
            continue
        task_dir = tasks_dir / instance_id
        if not task_dir.is_dir():
            raise RuntimeError(f"missing task artifacts directory {task_dir}")

        stderr_path = task_dir / "dsh.stderr.log"
        if stderr_path.is_file():
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
            if _HMR_STARTUP_NEEDLE in stderr_text:
                raise RuntimeError(f"HMR startup error in {stderr_path}")

        if outcome != "completed":
            reason = row.get("termination_reason")
            if reason == _STARTUP_FAILURE:
                raise RuntimeError(f"task {instance_id} failed with {_STARTUP_FAILURE}")
            continue

        _assert_dsh_trace_artifacts(task_dir, instance_id)
        _assert_dsh_home_confined(task_dir, instance_id)

    _assert_run_tree_confined(run_dir)
    _assert_workspaces_confined(run_dir)


def _assert_dsh_trace_artifacts(task_dir: Path, instance_id: str) -> None:
    transcript = task_dir / "transcript.jsonl"
    sessions_dir = task_dir / "dsh-sessions"
    session_logs = list(sessions_dir.glob("session-*.jsonl")) if sessions_dir.is_dir() else []
    if transcript.is_file() and transcript.stat().st_size > 0:
        return
    if session_logs and any(path.stat().st_size > 0 for path in session_logs):
        return
    raise RuntimeError(
        f"task {instance_id} completed but missing DSH trace "
        f"(expected non-empty transcript.jsonl or dsh-sessions/session-*.jsonl under {task_dir})"
    )


def _assert_dsh_home_confined(task_dir: Path, instance_id: str) -> None:
    home = task_dir / "dsh-home"
    if not home.is_dir():
        raise RuntimeError(f"task {instance_id} missing isolated dsh-home under {task_dir}")


def _assert_run_tree_confined(run_dir: Path) -> None:
    for home in run_dir.rglob("dsh-home"):
        try:
            home.relative_to(run_dir / "tasks")
        except ValueError as exc:
            raise RuntimeError(f"dsh-home found outside tasks/: {home}") from exc


def _assert_workspaces_confined(run_dir: Path) -> None:
    """Ensure agent workspaces stay under run_dir/workspaces/<instance_id>/ (no sibling leaks)."""

    workspaces_root = run_dir / "workspaces"
    if not workspaces_root.is_dir():
        return

    for entry in run_dir.iterdir():
        name = entry.name
        if entry.is_file() and name not in _RUN_TOP_LEVEL:
            raise RuntimeError(f"unexpected file at run root (possible workspace escape): {entry}")
        if entry.is_dir() and name not in _RUN_TOP_DIRS:
            raise RuntimeError(f"unexpected directory at run root (possible workspace escape): {entry}")

    for workspace in workspaces_root.iterdir():
        if not workspace.is_dir():
            raise RuntimeError(f"non-directory entry under workspaces/: {workspace}")
        try:
            workspace.resolve().relative_to(workspaces_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"workspace path escapes workspaces root: {workspace}") from exc
