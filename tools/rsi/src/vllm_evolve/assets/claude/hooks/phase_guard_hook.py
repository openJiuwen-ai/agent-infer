#!/usr/bin/env python3
"""PreToolUse hook: enforce the vllm-evolve phase machine on Claude tool calls.

Installed by ``ve init`` into ``.claude/hooks/``. Reads Claude Code's tool-call
JSON on stdin; exits 2 (with a reason on stderr) to BLOCK the call, or 0 to
ALLOW it. The policy lives in ``vllm_evolve.install.hook_decision.decide`` so it
is unit-tested as a decision matrix.

Fails open: any internal error (e.g. vllm_evolve not importable outside a
project) exits 0 so the user's session is never hard-blocked by the hook.
"""
import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    try:
        from vllm_evolve.install.hook_decision import decide
        from vllm_evolve.tools.phase_guard import read_phase

        decision = decide(tool_name, tool_input, read_phase())
    except Exception:
        return 0
    if not decision.allow:
        sys.stderr.write(f"[vllm-evolve phase guard] blocked: {decision.reason}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
