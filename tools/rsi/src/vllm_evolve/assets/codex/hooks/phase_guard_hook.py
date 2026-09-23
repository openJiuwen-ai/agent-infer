#!/usr/bin/env python3
"""Codex PreToolUse wrapper for the shared vllm-evolve phase guard."""
import json
import sys
from pathlib import Path

_WRITE_CAPABLE_TOOLS = {
    "Bash",
    "exec_command",
    "apply_patch",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
}


def _fail_closed(reason: str) -> int:
    sys.stderr.write(f"[vllm-evolve phase guard] blocked: guard unavailable: {reason}\n")
    return 2


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        return _fail_closed(f"invalid hook payload ({type(exc).__name__})")
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    # A repository-local install must work even when hooks.json was rendered by a system Python
    # rather than the project's virtualenv.  Prefer this checkout's source tree before falling back
    # to the interpreter's installed package.
    project_src = Path(__file__).resolve().parents[2] / "src"
    if project_src.is_dir():
        sys.path.insert(0, str(project_src))
    try:
        from vllm_evolve.install.hook_decision import decide
        from vllm_evolve.tools.phase_guard import read_phase

        decision = decide(tool_name, tool_input, read_phase())
    except Exception as exc:
        if tool_name in _WRITE_CAPABLE_TOOLS or not tool_name:
            return _fail_closed(f"{type(exc).__name__}: {exc}")
        return 0
    if not decision.allow:
        sys.stderr.write(f"[vllm-evolve phase guard] blocked: {decision.reason}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
