"""Tool-level phase guard.

Persists the current phase under ``<repo>/.ve/state/phase.txt`` and rejects
any verb invocation whose required phase does not match. Implementation is
intentionally runtime-neutral: it does not depend on Claude Code hooks, so
the same guard is enforced for any agent that drives the harness through the
``ve`` CLI or imports the guard directly.

(The state dir was ``.ar/`` before the ``ar``→``ve`` command rename; a pre-rename
``.ar/`` tree is migrated to ``.ve/`` once, transparently, on first access.)

The lock primitive is a portable sentinel-file lock (``os.O_CREAT | O_EXCL``)
so it works on Windows and POSIX without ``fcntl`` / ``msvcrt`` divergence.
"""

from __future__ import annotations

import enum
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# --- State location ---------------------------------------------------------


def _repo_root() -> Path:
    """Resolve repo root from this module's location.

    ``tools/phase_guard.py`` -> ``src/vllm_evolve/tools/`` ->
    ``src/vllm_evolve/`` -> ``src/`` -> repo root (parents[3]).
    """
    return Path(__file__).resolve().parents[3]


# Runtime state lives under this repo-root dir (renamed .ar -> .ve with the command).
_STATE_ROOT = ".ve"
_LEGACY_STATE_ROOT = ".ar"


def _migrate_legacy_state(root: Path) -> None:
    """One-time, idempotent: move a pre-rename ``.ar/`` tree to ``.ve/`` so phase state
    (and design scratch) survives the ``ar``→``ve`` rename. Best-effort: if the move
    fails, a fresh ``.ve/`` is created by the caller instead."""
    legacy = root / _LEGACY_STATE_ROOT
    current = root / _STATE_ROOT
    if legacy.is_dir() and not current.exists():
        try:
            os.replace(legacy, current)
        except OSError:
            pass


def state_dir() -> Path:
    root = _repo_root()
    _migrate_legacy_state(root)
    return root / _STATE_ROOT / "state"


def state_file() -> Path:
    return state_dir() / "phase.txt"


def lock_file() -> Path:
    return state_dir() / "phase.lock"


# --- Phase enum + transition table -----------------------------------------


class Phase(str, enum.Enum):
    """Subset of the 7-phase contract wired in Round 0.

    Plan reference: ``.humanize/plans/vllm-evolve-rebuild.md`` section
    "Feasibility Hints / Conceptual Approach". ``VERIFY_PASSED`` is a state
    token recorded by ``ve verify`` so that ``ve bench`` (which is a stub
    this round) can refuse without it.
    """

    INIT = "INIT"
    READ_CONTEXT = "READ_CONTEXT"
    DESIGN = "DESIGN"
    GENERATE = "GENERATE"
    VERIFY = "VERIFY"
    VERIFY_PASSED = "VERIFY_PASSED"
    BENCHMARK = "BENCHMARK"
    KEEP_OR_DISCARD = "KEEP_OR_DISCARD"
    COMMIT_OR_ROLLBACK = "COMMIT_OR_ROLLBACK"
    FINISH = "FINISH"


_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.INIT: frozenset({Phase.READ_CONTEXT, Phase.FINISH}),
    Phase.READ_CONTEXT: frozenset({Phase.DESIGN, Phase.FINISH}),
    Phase.DESIGN: frozenset({Phase.GENERATE, Phase.READ_CONTEXT, Phase.FINISH}),
    Phase.GENERATE: frozenset({Phase.VERIFY, Phase.DESIGN, Phase.FINISH}),
    Phase.VERIFY: frozenset({Phase.VERIFY_PASSED, Phase.GENERATE, Phase.FINISH}),
    Phase.VERIFY_PASSED: frozenset({Phase.BENCHMARK, Phase.FINISH}),
    Phase.BENCHMARK: frozenset({Phase.KEEP_OR_DISCARD, Phase.FINISH}),
    Phase.KEEP_OR_DISCARD: frozenset({Phase.COMMIT_OR_ROLLBACK, Phase.FINISH}),
    Phase.COMMIT_OR_ROLLBACK: frozenset({Phase.READ_CONTEXT, Phase.FINISH}),
    Phase.FINISH: frozenset({Phase.INIT}),
}


# Verb -> required phase. ``None`` means administrative (no gate).
#
# ``legacy_tool`` is registered as admin in Round 1 so the existing
# ``vllm-evolve`` CLI workflows (which call into the pre-refactor tool
# modules under ``src/vllm_evolve/tools/*.py``) continue to work while the
# guard hook point is present. The hook point is what closes the AC-3
# direct-submodule-import bypass surfaced by Codex's Round-0 review --
# every public callable in those legacy modules now routes through
# ``check_or_fail('legacy_tool')`` even when imported via
# ``from vllm_evolve.tools.simulate import simulate``. Later rounds can
# tighten by re-registering ``legacy_tool`` with a non-admin phase once
# ``vllm-evolve`` is hard-cut in M5 (DEC-6).
# Canonical verb -> required-phase table (M4). This is the SINGLE source of truth: the phase
# machine owns it, so the hook / CLI know every gated verb's phase WITHOUT importing the CLI module
# for an import side effect (which was fragile — a missed import silently un-gated the round verbs).
# Round-lifecycle verbs are declared here (formerly registered from ar_cli at import time).
_VERB_PHASES: dict[str, Phase | None] = {
    "phase": None,
    "verify": Phase.VERIFY,
    "bench": Phase.BENCHMARK,
    "legacy_tool": None,
    # round-lifecycle verbs (moved here from ar_cli's import-time registration)
    "context": Phase.READ_CONTEXT,
    "design": Phase.DESIGN,
    # The high-level real search begins only after context/research/design are frozen. It
    # internally authors, verifies, and benchmarks proposals, but never adopts them.
    "real-evolve": Phase.DESIGN,
    "compare": Phase.KEEP_OR_DISCARD,
    "keep": Phase.COMMIT_OR_ROLLBACK,
    "discard": Phase.COMMIT_OR_ROLLBACK,
}


def register_verb(verb: str, required_phase: Phase | None) -> None:
    """Register a verb -> required-phase mapping at runtime.

    Future rounds extend this when adding new verbs (e.g. ``ve context``,
    ``ar sim``). Re-registration with a different phase is rejected to
    prevent silent regressions.
    """
    if verb in _VERB_PHASES and _VERB_PHASES[verb] != required_phase:
        raise ValueError(
            f"phase_guard: verb {verb!r} already registered with phase "
            f"{_VERB_PHASES[verb]}; refusing to overwrite with {required_phase}"
        )
    _VERB_PHASES[verb] = required_phase


# --- Error type returned to CLI / callers ----------------------------------


class PhaseError(RuntimeError):
    """Raised when a verb is invoked in a phase that does not allow it."""

    def __init__(
        self,
        phase_required: str,
        current_phase: str,
        hint: str,
    ):
        self.phase_required = phase_required
        self.current_phase = current_phase
        self.hint = hint
        super().__init__(hint)

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "outcome_class": "phase_violation",
            "phase_required": self.phase_required,
            "current_phase": self.current_phase,
            "hint": self.hint,
        }


# --- Portable advisory lock (sentinel file) --------------------------------


@dataclass
class FileLock:
    """Sentinel-file lock. Acquire by creating the file with O_EXCL; release
    by unlinking. Cleans up on context exit. Stale-lock detection is left to
    future rounds (Round 0 uses short timeouts so a leaked lock is loud).
    """

    path: Path
    timeout_s: float = 5.0
    poll_s: float = 0.005
    _fd: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self._fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.write(self._fd, f"{os.getpid()}\n".encode())
                return self
            except (FileExistsError, PermissionError):
                # FileExistsError on POSIX; PermissionError on Windows when
                # another holder is still inside the O_EXCL window.
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"phase_guard: lock acquisition timeout after "
                        f"{self.timeout_s}s: {self.path}"
                    )
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _acquire_lock(timeout_s: float = 5.0) -> FileLock:
    return FileLock(lock_file(), timeout_s=timeout_s)


# --- Atomic write ----------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    # ``Path.write_text(newline=...)`` was added after Python 3.9, while this
    # package explicitly supports Python >=3.9.  Use ``open`` so phase
    # transitions keep their stable LF encoding on every supported runtime.
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    # os.replace is atomic, but on Windows the destination can be transiently locked
    # (antivirus / indexer scanning the just-written temp file) -> PermissionError
    # [WinError 5]. Retry briefly; this is robustness only and does NOT change phase
    # semantics (the replaced content is identical on every attempt).
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.02)


# --- Public state ops ------------------------------------------------------


def read_phase() -> Phase:
    """Return current phase, defaulting to ``INIT`` when no state file exists."""
    path = state_file()
    if not path.exists():
        return Phase.INIT
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return Phase(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"phase_guard: malformed phase token {raw!r} in {path}"
        ) from exc


def transition(target: Phase) -> Phase:
    """Move from current phase to ``target``, enforcing the transition table."""
    with _acquire_lock():
        current = read_phase()
        allowed = _TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            allowed_str = ", ".join(sorted(p.value for p in allowed)) or "(none)"
            raise PhaseError(
                phase_required=allowed_str,
                current_phase=current.value,
                hint=(
                    f"Cannot transition {current.value} -> {target.value}. "
                    f"Allowed next phases: {{{allowed_str}}}"
                ),
            )
        _atomic_write(state_file(), target.value + "\n")
        return target


def reset_state() -> None:
    """Test/maintenance helper: drop the persisted phase. Not exposed via ar."""
    sf = state_file()
    if sf.exists():
        sf.unlink()
    lf = lock_file()
    if lf.exists():
        try:
            lf.unlink()
        except OSError:
            pass


def mark_verify_passed() -> Phase:
    """Convenience used by ``ve verify`` on success."""
    return transition(Phase.VERIFY_PASSED)


# --- Verb gate -------------------------------------------------------------


def check_or_fail(verb: str, *, override_phase: Phase | None = None) -> Phase:
    """Verify ``verb`` may run in the current phase.

    Raises :class:`PhaseError` on mismatch. The ``override_phase`` keyword is
    intended for tests so they need not touch the on-disk state file.
    """
    required = _VERB_PHASES.get(verb)
    current = override_phase if override_phase is not None else read_phase()
    if required is None:
        # Round-2 hardening per Codex review: ``legacy_tool`` is registered
        # as administrative so the surviving ``vllm-evolve`` CLI keeps
        # working through M5, but the guard MUST still be able to reject
        # when the phase machine is in pristine ``INIT`` state. Without
        # this, a malicious or accidental ``from vllm_evolve.tools.X import
        # Y`` from a freshly-cloned checkout would silently run work code
        # before the agent ever entered a round, defeating AC-3.
        if verb == "legacy_tool" and current == Phase.INIT:
            raise PhaseError(
                phase_required="any non-INIT phase",
                current_phase=current.value,
                hint=(
                    "legacy tool callable invoked from pristine INIT phase. "
                    "Start a round with 've phase set READ_CONTEXT' before "
                    "calling vllm-evolve commands or importing "
                    "vllm_evolve.tools.* work functions directly. M5 "
                    "(DEC-6 hard-cut) will remove the legacy surface entirely."
                ),
            )
        return current

    if required == Phase.BENCHMARK:
        if current != Phase.VERIFY_PASSED:
            raise PhaseError(
                phase_required=Phase.VERIFY_PASSED.value,
                current_phase=current.value,
                hint=(
                    "Run 've verify <policy> <target>' first; bench refuses "
                    "until VERIFY_PASSED is recorded."
                ),
            )
        return current

    if required == Phase.VERIFY:
        if current not in {Phase.GENERATE, Phase.VERIFY}:
            raise PhaseError(
                phase_required="GENERATE or VERIFY",
                current_phase=current.value,
                hint=(
                    "Verify can only follow GENERATE. Current phase: "
                    f"{current.value}. Use 've phase set GENERATE' first."
                ),
            )
        if current == Phase.GENERATE:
            transition(Phase.VERIFY)
        return Phase.VERIFY

    if current != required:
        raise PhaseError(
            phase_required=required.value,
            current_phase=current.value,
            hint=(
                f"Verb '{verb}' requires phase {required.value}; current "
                f"phase is {current.value}."
            ),
        )
    return current


def required(verb: str) -> Callable:
    """Decorator that gates a callable on ``verb``'s required phase.

    Even when callers ``from vllm_evolve.tools.<mod> import <fn>`` directly,
    invoking the decorated function triggers ``check_or_fail`` first.
    """

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            check_or_fail(verb)
            return fn(*args, **kwargs)

        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
        wrapper.__doc__ = fn.__doc__
        wrapper._phase_guarded_verb = verb  # sentinel inspected by tests
        return wrapper

    return decorator


# --- Introspection helpers (used by tests) ---------------------------------


def registered_verbs() -> dict[str, Phase | None]:
    """Return a copy of the verb -> phase registry for test inspection."""
    return dict(_VERB_PHASES)


def transitions() -> dict[Phase, frozenset[Phase]]:
    """Return a copy of the transition table for test inspection."""
    return dict(_TRANSITIONS)


__all__ = [
    "Phase",
    "PhaseError",
    "FileLock",
    "check_or_fail",
    "lock_file",
    "mark_verify_passed",
    "read_phase",
    "registered_verbs",
    "register_verb",
    "required",
    "reset_state",
    "state_dir",
    "state_file",
    "transition",
    "transitions",
]
