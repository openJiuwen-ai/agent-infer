# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""PreToolUse fence hook for Claude Code under AgentBench isolation.

This script is wired into Claude's settings.json as a PreToolUse hook over the
Edit|Write|MultiEdit|NotebookEdit matchers. Claude invokes it before each matching tool call
passes its file path to the editor: paths that resolve outside the task
workspace are denied, so an agent running under ``bypassPermissions`` cannot
persist writes to the AgentBench host filesystem or pollute other tasks.

Why a package-shipped module, not a per-task generated script: a copy written
into ``artifact_dir`` would be self-modifiable (artifact_dir is reachable by
absolute path under ``bypassPermissions``), so an agent could edit the guard
out of its own way. This file lives under the installed package tree, which is
outside Claude's bubblewrap write-set and is itself denied by this hook, so it
is tamper-resistant.

Named ``fence.py`` (not ``extension.py``) because JiuwenSwarm's extension
loader auto-imports any ``agents/*/extension.py``; reserving that filename for
the Jiuwen integration is enforced by test_extension_py_reserved.
"""

import json
import os
import sys
from pathlib import Path


def _resolve(path_str: str) -> Path:
    """Resolve a path, following symlinks via realpath semantics."""

    if not path_str:
        return Path()
    expanded = Path(path_str).expanduser()
    # os.path.realpath follows all symlinks and collapses ".." components,
    # matching the PoC's symlink-escape coverage: a write via a symlink that
    # points outside the workspace is caught because the realpath lands
    # outside ALLOWED_ROOT.
    return Path(os.path.realpath(str(expanded)))


def _is_outside(child: Path, root: Path) -> bool:
    """Return True if ``child`` does not live at or under ``root``."""

    try:
        child.relative_to(root)
    except ValueError:
        return True
    return False


def _deny(reason: str) -> dict[str, str]:
    """Build one Claude PreToolUse deny decision."""

    return {
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def _tool_paths(tool_input: dict) -> list[str] | None:
    """Return every path carried by a supported Claude file-tool payload."""

    direct = tool_input.get("file_path") or tool_input.get("notebook_path") or tool_input.get("path")
    if direct is not None:
        return [direct] if isinstance(direct, str) and direct.strip() else None

    edits = tool_input.get("edits")
    if not isinstance(edits, list) or not edits:
        return None
    paths = []
    for edit in edits:
        if not isinstance(edit, dict):
            return None
        path = edit.get("file_path") or edit.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        paths.append(path)
    return paths


def _decide(payload: dict) -> dict[str, str] | None:
    """Return a deny decision unless every target resolves inside the workspace."""

    root = os.environ.get("ALLOWED_ROOT", "").strip()
    if not root:
        return _deny("ALLOWED_ROOT is required under AgentBench isolation")
    root_path = _resolve(root)
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return _deny("tool_input must be an object under AgentBench isolation")
    file_paths = _tool_paths(tool_input)
    if file_paths is None:
        return _deny("tool_input file path is required under AgentBench isolation")
    for file_path in file_paths:
        resolved = _resolve(file_path)
        if _is_outside(resolved, root_path):
            return _deny(f"path {resolved} is outside the isolated workspace {root_path}")
    return None


def main() -> None:
    """Read one PreToolUse payload and emit an allow or fail-closed deny decision.

    All stdin reading is confined here and only reached under ``__main__``, so
    importing this module (e.g. for testing ``_decide``) never blocks on stdin.
    """

    try:
        raw = sys.stdin.read()
    except Exception as exc:
        decision = _deny(f"failed to read PreToolUse payload: {type(exc).__name__}")
    else:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            decision = _deny("PreToolUse payload must be valid JSON")
        else:
            if not isinstance(payload, dict):
                decision = _deny("PreToolUse payload must be a JSON object")
            else:
                decision = _decide(payload)
    if decision is not None:
        print(json.dumps(decision))


if __name__ == "__main__":
    main()
