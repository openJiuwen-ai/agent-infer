# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate shared run artifacts and runtime-specific task evidence."""

from __future__ import annotations

import json
from pathlib import Path

_TASK_ARTIFACTS = {
    "claude": ("transcript.jsonl", "claude_state"),
    "jiuwenswarm": ("jiuwenswarm.jsonl", "jiuwenswarm-instance/home"),
    "dsh": ("transcript.jsonl", "dsh-home"),
    "zcode": ("transcript.jsonl", "zcode-home"),
}
_ROOT_FILES = {
    "manifest.json",
    "summary.json",
    "requests.jsonl",
    "task_index.json",
    "distribution_samples.csv",
    "agents.csv",
    "sessions.csv",
    "proxy.stdout.log",
    "proxy.stderr.log",
}
_ROOT_DIRS = {"evidence", "tasks", "workspaces"}
_HMR_STARTUP_NEEDLE = "--expose-internals is required for HMR service"


def _confined(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes {root}: {path}") from exc


def validate_agent_artifacts(run_dir: Path, agent_type: str, *, expected_tasks: int) -> None:
    """Require complete task evidence and paths confined to their owning run."""
    transcript_name, state_name = _TASK_ARTIFACTS[agent_type]
    allowed_root_files = _ROOT_FILES | ({"reproduce_issue.py"} if agent_type == "zcode" else set())
    run_dir = run_dir.resolve()
    for name in ("manifest.json", "summary.json", "task_index.json", "requests.jsonl"):
        path = run_dir / name
        _confined(path, run_dir)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty run artifact: {path}")
    for entry in run_dir.iterdir():
        if (entry.is_file() and entry.name not in allowed_root_files) or (
            entry.is_dir() and entry.name not in _ROOT_DIRS
        ):
            raise RuntimeError(f"unexpected entry at run root: {entry}")
        _confined(entry, run_dir)
    tasks_root = run_dir / "tasks"
    workspaces_root = run_dir / "workspaces"
    for root in (tasks_root, workspaces_root):
        if not root.is_dir():
            raise RuntimeError(f"missing artifact directory: {root}")
        _confined(root, run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("lifecycle", {}).get("status") != "completed":
        raise RuntimeError(f"run lifecycle is not completed: {run_dir}")
    task_index = json.loads((run_dir / "task_index.json").read_text(encoding="utf-8"))
    failure_summary = task_index.get("failure_summary", {})
    by_reason = failure_summary.get("by_termination_reason", {})
    if failure_summary.get("total_failed", 0) > 0 or any(count > 0 for count in by_reason.values()):
        raise RuntimeError(f"{agent_type} task failures in {run_dir}: {failure_summary}")
    rows = task_index.get("tasks")
    if not isinstance(rows, list) or len(rows) != expected_tasks:
        raise RuntimeError(f"task index must contain {expected_tasks} tasks")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid task index row")
        instance_id = row.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in {".", ".."}
            or "/" in instance_id
            or "\\" in instance_id
            or ":" in instance_id
            or instance_id in seen
        ):
            raise RuntimeError(f"invalid or duplicate instance_id: {instance_id!r}")
        seen.add(instance_id)
        if row.get("outcome") != "completed":
            raise RuntimeError(f"task {instance_id} did not complete: {row.get('termination_reason')}")
        if row.get("termination_reason") is not None or row.get("error") is not None:
            raise RuntimeError(f"task {instance_id} has failure evidence despite completed outcome: {row}")
        task_dir = tasks_root / instance_id
        workspace = workspaces_root / instance_id
        for path, root in ((task_dir, tasks_root), (workspace, workspaces_root)):
            _confined(path, root)
            if not path.is_dir():
                raise RuntimeError(f"missing task directory: {path}")
        if agent_type == "dsh":
            _validate_dsh_startup(task_dir)
        elif agent_type == "zcode":
            _validate_zcode_artifacts(task_dir)
        state_dir = task_dir / state_name
        _confined(state_dir, task_dir)
        if not state_dir.is_dir():
            raise RuntimeError(f"missing isolated agent state: {state_dir}")
        transcripts = [task_dir / transcript_name]
        if agent_type == "dsh":
            transcripts.extend((task_dir / "dsh-sessions").glob("session-*.jsonl"))
        for path in transcripts:
            _confined(path, task_dir)
        if not any(path.is_file() and path.stat().st_size > 0 for path in transcripts):
            raise RuntimeError(f"missing non-empty {agent_type} transcript under {task_dir}")
    for workspace in workspaces_root.iterdir():
        _confined(workspace, workspaces_root)
        if not workspace.is_dir() or workspace.name not in seen:
            raise RuntimeError(f"unexpected workspace: {workspace}")
    for state_dir in run_dir.rglob(state_name):
        _confined(state_dir, tasks_root)


def _validate_dsh_startup(task_dir: Path) -> None:
    """Reject the DSH-specific Cordis HMR boot regression."""
    stderr_path = task_dir / "dsh.stderr.log"
    _confined(stderr_path, task_dir)
    if stderr_path.is_file():
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if _HMR_STARTUP_NEEDLE in stderr_text:
            raise RuntimeError(f"HMR startup error in {stderr_path}")


def _validate_zcode_artifacts(task_dir: Path) -> None:
    """Validate ZCode's own completion, prompt delivery, and adapter evidence."""
    documents = {}
    for name in ("zcode-run-status.json", "zcode-lineage.json", "zcode-config.json", "zcode-command.json"):
        path = task_dir / name
        _confined(path, task_dir)
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"missing non-empty ZCode artifact: {path}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    status = documents["zcode-run-status.json"]
    if (
        not isinstance(status, dict)
        or status.get("outcome") != "completed"
        or status.get("exit_code") != 0
        or status.get("termination_reason") is not None
        or status.get("native_error") is not None
        or status.get("prompt_loaded") is not True
        or status.get("execution_phase") != "agent"
    ):
        raise RuntimeError(f"ZCode run status does not confirm successful execution: {status}")
    lineage = documents["zcode-lineage.json"]
    if (
        not isinstance(lineage, dict)
        or not lineage.get("sessions")
        or lineage.get("failure") is not None
        or lineage.get("pending")
    ):
        raise RuntimeError(f"ZCode lineage contains missing sessions or unfinished work: {lineage}")
    if not isinstance(documents["zcode-config.json"], dict) or not documents["zcode-config.json"]:
        raise RuntimeError("invalid ZCode configuration artifact")
    command = documents["zcode-command.json"]
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise RuntimeError("invalid ZCode launch command artifact")
    for name in ("zcode-hooks.jsonl", "zcode.stderr.log"):
        path = task_dir / name
        _confined(path, task_dir)
        if not path.is_file() or (name == "zcode-hooks.jsonl" and not path.stat().st_size):
            raise RuntimeError(f"missing ZCode diagnostic artifact: {path}")
