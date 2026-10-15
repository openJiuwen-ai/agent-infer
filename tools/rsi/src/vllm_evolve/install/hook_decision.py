"""Pure decision logic for the PreToolUse phase-guard hook.

The hook script (``.claude/hooks/phase_guard_hook.py``) is a thin stdin→exit
wrapper; all the policy lives here so it is unit-testable as a decision matrix
without Claude Code in the loop. ``decide`` is a pure function of
``(tool_name, tool_input, phase)``.

Two things are enforced:

1. **Protected-path writes are always denied.** Ground-truth and provenance
   files (target seeds/skeletons, configs, schemas, the archive, the phase
   state, the installed harness assets) must never be edited by the agent.
2. **Editing a policy file is GENERATE-only**, and **`ar <verb>` is gated by
   phase** (mirrors ``phase_guard`` so the agent is blocked *before* the verb
   runs, with a clear reason).

Everything else is allowed — the hook must not get in the way of normal work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# The verb->phase table is the canonical one in phase_guard._VERB_PHASES (M4): the round-lifecycle
# verbs are declared there, so this hook no longer imports the CLI module for an import side effect.
from vllm_evolve.tools import phase_guard
from vllm_evolve.tools.phase_guard import Phase

_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
_PATCH_TOOLS = ("apply_patch",)

# Substrings (in a forward-slash-normalized path) that must never be edited.
PROTECTED_PATH_MARKERS: tuple[str, ...] = (
    "/seed.py",
    "/skeleton.py",
    "/config/",
    "/schemas/",
    "/archive_policies/",
    "/.ve/",
    "/.claude/",
    "/.codex/",
    "/.agents/",
    # auto-executed Python hooks: a sub-agent must not plant code here that could subvert the
    # deterministic bench/gate during the next test/import run.
    "conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
)


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str = ""


def _norm(path: str) -> str:
    return "/" + path.replace("\\", "/").lstrip("/")


def _is_run_provenance(norm: str) -> bool:
    """True for machine-written run provenance under ``runs/<target>/<run_id>/``:
    ``manifest.json`` / ``eval_result.json`` (the run's self-description + result) and
    anything under its ``bench/`` or ``gpu/`` subdirs. These are append-only machine
    artifacts whose sha the keep/verify-gain path verifies — an agent must not hand-edit
    them. Substring markers can't express this shape, and
    PurePosixPath.match can't recurse ``bench/**``, so match structurally on the
    forward-slash-normalized path (covers relative AND absolute/Windows inputs)."""
    parts = norm.lstrip("/").split("/")
    # locate the runs/<target>/<run_id>/ prefix anywhere in the path (absolute paths prepend
    # drive/dirs, e.g. /C:/.../runs/scheduling/<id>/manifest.json)
    if "runs" not in parts:
        return False
    i = parts.index("runs")
    rest = parts[i + 1:]                     # [<target>, <run_id>, <tail...>]
    if len(rest) < 3:
        return False
    tail = rest[2:]
    if len(tail) == 1 and tail[0] in ("manifest.json", "eval_result.json"):
        return True
    return tail[0] in ("bench", "gpu") and len(tail) >= 2


def _edit_path_decision(path: str, phase: Phase) -> Decision:
    norm = _norm(path)
    for marker in PROTECTED_PATH_MARKERS:
        if marker in norm:
            return Decision(
                False,
                f"writes to protected path are blocked ({marker.strip('/')}); "
                "ground-truth / provenance files must not be edited",
            )
    if _is_run_provenance(norm):
        return Decision(
            False,
            "writes to run provenance are blocked (runs/<target>/<run_id>/"
            "{manifest.json,eval_result.json,bench/**,gpu/**}); these are machine-written "
            "append-only artifacts whose sha keep/verify-gain verifies",
        )
    if "/targets/" in norm and norm.endswith(".py"):
        if phase != Phase.GENERATE:
            return Decision(
                False,
                f"editing a policy file is only allowed in GENERATE phase "
                f"(current: {phase.value}); run 've phase set GENERATE' first",
            )
        if not norm.endswith("/work.py"):
            return Decision(
                False,
                "in GENERATE, only the working policy file (targets/<target>/work.py) may be "
                "edited; other targets/*.py files are off-limits",
            )
    return Decision(True)


_PATCH_PATH_PREFIXES = (
    "*** Add File: ",
    "*** Update File: ",
    "*** Delete File: ",
    "*** Move to: ",
)


def _patch_paths(command: str) -> list[str]:
    """Extract every target path from Codex's apply_patch format."""
    paths: list[str] = []
    for line in command.splitlines():
        for prefix in _PATCH_PATH_PREFIXES:
            if line.startswith(prefix):
                path = line[len(prefix):].strip()
                if path:
                    paths.append(path)
                break
    return paths


def _verb_allowed(verb: str, phase: Phase) -> tuple[bool, str]:
    """Mirror phase_guard.check_or_fail's gating, but PURE (no transition side
    effect). Unknown/admin verbs are allowed."""
    required = phase_guard._VERB_PHASES.get(verb)
    if required is None:
        return True, ""
    if required == Phase.BENCHMARK:
        ok = phase == Phase.VERIFY_PASSED
        return ok, "" if ok else "bench requires VERIFY_PASSED (run 've verify' first)"
    if required == Phase.VERIFY:
        ok = phase in (Phase.GENERATE, Phase.VERIFY)
        return ok, "" if ok else "verify requires GENERATE or VERIFY phase"
    ok = phase == required
    return ok, "" if ok else f"'ve {verb}' requires phase {required.value} (current {phase.value})"


def _ve_verb(command: str) -> str | None:
    """Extract the verb from a Bash command invoking the ``ve`` CLI, else None.

    Recognizes the ``ve`` console script (renamed from ``ar``; ``ar`` kept too for
    back-compat during the transition) and the ``python -m vllm_evolve.cli.main`` module form."""
    toks = command.split()
    if not toks:
        return None
    head = toks[0].replace("\\", "/").rsplit("/", 1)[-1]
    rest: list[str]
    if head in ("ve", "ve.exe", "ar", "ar.exe"):
        rest = toks[1:]
    elif "vllm_evolve.cli.main" in command:
        idx = next((i for i, t in enumerate(toks) if "cli.main" in t), None)
        rest = toks[idx + 1:] if idx is not None else []
    else:
        return None
    for t in rest:
        if not t.startswith("-"):
            return t
    return None


_SHELL_WRITE_RE = re.compile(
    r"(?:^|\s)(?:rm|mv|cp|touch|truncate|tee|install)(?:\s|$)|"
    r"(?:^|\s)(?:sed|perl)\s+-[^\s]*i[^\s]*(?:\s|$)|"
    r"(?<![<])>{1,2}(?!>)"
)


def _bash_protected_write(command: str, phase: Phase) -> Decision | None:
    """Fail closed for common shell-write forms targeting protected/policy paths.

    The hook is not a shell parser, so it deliberately does not claim to prove arbitrary commands
    safe.  It catches the direct write forms produced by Codex (`>`, `tee`, `sed -i`, `rm`, `mv`,
    `cp`, `touch`, `truncate`, `install`) and delegates every mentioned protected path to the same
    path policy used by structured edit tools.
    """
    if not _SHELL_WRITE_RE.search(command):
        return None
    normalized_command = command.replace("\\", "/")
    mentions_protected = False
    for raw_token in re.split(r"[\s;|&<>]+", normalized_command):
        token = raw_token.strip("'\"")
        if not token:
            continue
        norm = _norm(token)
        protected_token = any(marker in norm for marker in PROTECTED_PATH_MARKERS)
        target_token = "/targets/" in norm
        if protected_token or target_token:
            mentions_protected = True
            decision = _edit_path_decision(token, phase)
            if not decision.allow:
                return decision
    if mentions_protected:
        return Decision(False, "shell write mentions a protected path but its target was ambiguous")
    return None


def decide(tool_name: str, tool_input: dict, phase: Phase) -> Decision:
    """Allow/deny a Claude tool call under the current phase. Pure."""
    if tool_name in _EDIT_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        return _edit_path_decision(str(path), phase)

    if tool_name in _PATCH_TOOLS:
        command = str(tool_input.get("command") or tool_input.get("patch") or "")
        paths = _patch_paths(command)
        if not paths:
            return Decision(
                False,
                "apply_patch input did not expose any file paths; refusing an unverifiable write",
            )
        for path in paths:
            decision = _edit_path_decision(path, phase)
            if not decision.allow:
                return decision
        return Decision(True)

    if tool_name in {"Bash", "exec_command"}:
        command = str(tool_input.get("command") or tool_input.get("cmd") or "")
        protected_write = _bash_protected_write(command, phase)
        if protected_write is not None:
            return protected_write
        verb = _ve_verb(command)
        if verb is not None:
            ok, reason = _verb_allowed(verb, phase)
            if not ok:
                return Decision(False, reason)
        return Decision(True)

    return Decision(True)
