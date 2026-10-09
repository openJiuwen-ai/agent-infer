"""Tests for ``vllm_evolve.tools.phase_guard`` and its CLI integration.

Round 0 AC-3 success criteria:

* every ``ar`` verb calls ``check_or_fail`` at entry
* ``tools/__init__.py`` exports no work-doing callables
* concurrent writers cannot corrupt the phase state file
* a deliberate Python bypass still routes through the guard
* invalid phase transitions return structured ``{ok: false, ...}`` JSON
* ``ve bench`` refuses without ``VERIFY_PASSED`` recorded

State is isolated by monkey-patching ``phase_guard._repo_root`` so tests do
not pollute the real ``.ar/state/`` directory.
"""

from __future__ import annotations

import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.tools import phase_guard  # noqa: E402
from vllm_evolve.tools.phase_guard import (  # noqa: E402
    Phase,
    PhaseError,
)

SEED_POLICY = REPO_ROOT / "targets" / "scheduling" / "seed.py"


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_phase_state(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(phase_guard, "_repo_root", lambda: fake_root)
    phase_guard.reset_state()
    yield
    phase_guard.reset_state()


def test_legacy_ar_state_migrated_to_ve(tmp_path, monkeypatch):
    # P3b: the ar->ve rename moved runtime state .ar/ -> .ve/. A pre-rename .ar/ tree must
    # migrate once (state + design scratch preserved), idempotently, with no phase loss.
    root = tmp_path / "legacyrepo"
    (root / ".ar" / "state").mkdir(parents=True)
    (root / ".ar" / "state" / "phase.txt").write_text("DESIGN\n", encoding="utf-8")
    (root / ".ar" / "design.md").write_text("hyp", encoding="utf-8")
    monkeypatch.setattr(phase_guard, "_repo_root", lambda: root)

    sd = phase_guard.state_dir()                         # triggers one-time migration
    assert sd == root / ".ve" / "state"
    assert (root / ".ve" / "state" / "phase.txt").read_text(encoding="utf-8") == "DESIGN\n"
    assert (root / ".ve" / "design.md").read_text(encoding="utf-8") == "hyp"
    assert not (root / ".ar").exists()                   # moved, not copied
    assert phase_guard.state_dir() == root / ".ve" / "state"   # idempotent, no re-migrate/raise
    assert phase_guard.read_phase() == Phase.DESIGN            # migrated phase is read back


def _walk_to(target: Phase) -> None:
    """Transition the isolated state from INIT to ``target``."""
    chain: list[Phase] = []
    if target in {Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
                  Phase.VERIFY_PASSED}:
        chain.append(Phase.READ_CONTEXT)
    if target in {Phase.DESIGN, Phase.GENERATE, Phase.VERIFY, Phase.VERIFY_PASSED}:
        chain.append(Phase.DESIGN)
    if target in {Phase.GENERATE, Phase.VERIFY, Phase.VERIFY_PASSED}:
        chain.append(Phase.GENERATE)
    if target in {Phase.VERIFY, Phase.VERIFY_PASSED}:
        chain.append(Phase.VERIFY)
    if target == Phase.VERIFY_PASSED:
        chain.append(Phase.VERIFY_PASSED)
    for step in chain:
        phase_guard.transition(step)


# --- check_or_fail at every verb entry -------------------------------------


def test_each_ar_verb_calls_check_or_fail(monkeypatch, capsys):
    """Each Round-0 verb must hit ``check_or_fail`` before doing work."""

    called: list[str] = []
    original = phase_guard.check_or_fail

    def spy(verb: str, **kwargs):
        called.append(verb)
        return original(verb, **kwargs)

    monkeypatch.setattr(phase_guard, "check_or_fail", spy)
    monkeypatch.setattr(cli_main.phase_guard, "check_or_fail", spy)

    # ve phase status — administrative; always allowed.
    cli_main.main(["phase", "status"])
    capsys.readouterr()

    # ve verify needs GENERATE; seed policy is clean.
    _walk_to(Phase.GENERATE)
    cli_main.main(["verify", str(SEED_POLICY), "scheduling"])
    capsys.readouterr()

    # ve bench needs VERIFY_PASSED; the prior verify call already moved us there.
    cli_main.main(["bench", str(SEED_POLICY), "scheduling"])
    capsys.readouterr()

    assert called.count("phase") >= 1
    assert called.count("verify") >= 1
    assert called.count("bench") >= 1


# --- private tools/__init__.py + legacy submodule guard --------------------


def test_tools_init_does_not_reexport_work_callables():
    """tools/__init__.py must not re-export work-doing callables.

    This is the *package*-level check; the submodule-level check below
    is what actually closes the AC-3 bypass surface.
    """
    import vllm_evolve.tools as tools_pkg

    forbidden = {
        "simulate", "verify_code", "verify_config", "build_context",
        "format_flat", "get_store", "best", "history", "compare",
        "list_items", "stats", "apply_diff", "parse_diff",
        "apply_diff_to_program", "apply_full_code_to_program",
        "generate_candidates", "simulate_config", "export_vllm_yaml",
        "flatten_space", "sample_random", "sample_grid",
        "apply_constraints",
    }
    leaked = [name for name in forbidden if hasattr(tools_pkg, name)]
    assert not leaked, (
        f"tools/__init__.py leaks work-doing callables: {leaked}. Importing "
        "these directly would skip the phase guard."
    )


_LEGACY_SUBMODULE_BYPASS_CASES = [
    # ("vllm_evolve.tools.simulate", "simulate") removed (M3): module deleted.
    ("vllm_evolve.tools.verify_tool", "verify_code"),
    ("vllm_evolve.tools.verify_tool", "verify_config"),
    ("vllm_evolve.tools.store_tool", "get_store"),
    ("vllm_evolve.tools.store_tool", "best"),
    ("vllm_evolve.tools.context_tool", "build_context"),
    ("vllm_evolve.tools.configure_tool", "generate_candidates"),
]


@pytest.mark.parametrize("module_path,callable_name", _LEGACY_SUBMODULE_BYPASS_CASES)
def test_legacy_submodule_callable_rejected_from_init(
    module_path, callable_name
):
    """Round-2 follow-up to Codex's REQUIRED_CHANGE on AC-3: prove the
    guard hook point can actually reject, not merely fire as a no-op.

    The autouse fixture leaves the phase machine at INIT. From INIT,
    every legacy work callable -- regardless of which tool module it
    lives in -- must raise ``PhaseError`` rather than execute. Any non-
    INIT phase still allows the call so the surviving ``vllm-evolve``
    CLI workflow keeps working through M5 (DEC-6 hard-cut).
    """

    assert phase_guard.read_phase() == Phase.INIT, (
        "test pre-condition: autouse fixture must leave state at INIT"
    )
    module = __import__(module_path, fromlist=[callable_name])
    fn = getattr(module, callable_name)
    with pytest.raises(PhaseError) as exc:
        fn()  # any arguments -- guard fires first
    assert exc.value.current_phase == Phase.INIT.value
    assert "INIT" in exc.value.phase_required or "non-INIT" in exc.value.phase_required


@pytest.mark.parametrize("module_path,callable_name", _LEGACY_SUBMODULE_BYPASS_CASES)
def test_direct_submodule_import_routes_through_guard(
    module_path, callable_name, monkeypatch
):
    """Even when the caller does ``from vllm_evolve.tools.<module> import <fn>``,
    invoking ``<fn>(...)`` must call ``phase_guard.check_or_fail`` first.

    This closes the AC-3 bypass surface flagged by Codex's Round-0 review.
    The wrapper installed by ``tools/_legacy_guard._install_guard`` routes
    every call through the guard regardless of import path. ``legacy_tool``
    is registered as admin (no required phase) so this does not break the
    surviving ``vllm-evolve`` CLI workflows; what matters here is that the
    hook *fires*.
    """

    calls: list[str] = []
    original = phase_guard.check_or_fail

    def spy(verb: str, **kwargs):
        calls.append(verb)
        return original(verb, **kwargs)

    monkeypatch.setattr(phase_guard, "check_or_fail", spy)

    module = __import__(module_path, fromlist=[callable_name])
    fn = getattr(module, callable_name)

    assert getattr(fn, "_legacy_guard_wrapped", False), (
        f"{module_path}.{callable_name} is not guarded -- the legacy "
        "submodule import bypass remains open"
    )

    # Most legacy callables take arguments we cannot safely synthesise in
    # a unit test (e.g. simulate(policy_path, target, scenario, ...)). The
    # contract we are testing is that the guard fires *before* the
    # underlying function is reached, so a TypeError from the unguarded
    # body is acceptable proof the guard did its job. ``get_store`` takes
    # only a default path so it is safe to actually invoke. Either way, a
    # successful guard call ends up on ``calls``.
    try:
        if callable_name == "get_store":
            fn(":memory:")
        else:
            # Pass empty args; the guard runs first and the underlying
            # function then raises TypeError. Both outcomes are fine
            # because we only assert the guard fired.
            try:
                fn()
            except TypeError:
                pass
    except Exception:
        # The guard already fired; whatever the underlying body raised
        # is irrelevant to this test.
        pass

    assert "legacy_tool" in calls, (
        f"{module_path}.{callable_name} did not route through "
        "phase_guard.check_or_fail('legacy_tool')"
    )


def test_direct_import_of_ar_verb_still_routes_through_guard(capsys):
    """Even if a caller imports ``cmd_verify`` directly, the guard still fires."""

    import argparse

    from vllm_evolve.cli.main import cmd_verify

    # State is INIT; verify requires GENERATE/VERIFY.
    args = argparse.Namespace(policy=str(SEED_POLICY), target="scheduling")
    rc = cmd_verify(args)
    out = capsys.readouterr().out.strip()
    assert rc == 2
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["outcome_class"] == "phase_violation"
    assert payload["phase_required"] in {"GENERATE or VERIFY", "VERIFY"}


# --- atomic state writes under contention ----------------------------------


def test_atomic_phase_state_under_concurrent_writers():
    """Many writers racing must leave the state file parseable.

    The lock guarantees mutual exclusion of writes; ``_atomic_write`` uses
    ``tmp + os.replace`` so readers either see the pre-write content or the
    new content -- never a torn intermediate. The test exercises both
    properties at once.
    """

    phase_guard.transition(Phase.READ_CONTEXT)  # ensure file exists

    targets = list(Phase)
    rng = random.Random(0xC0FFEE)

    def writer(idx: int) -> str:
        chosen = rng.choice(targets)
        # Bypass transition() because we don't care about table validity here;
        # we are stressing the atomic-write + lock primitives.
        with phase_guard.FileLock(phase_guard.lock_file(), timeout_s=10.0):
            content = chosen.value + "\n"
            from vllm_evolve.tools.phase_guard import _atomic_write
            _atomic_write(phase_guard.state_file(), content)
        return chosen.value

    valid_values = {p.value for p in Phase}
    seen: list[str] = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(writer, i) for i in range(64)]
        for f in as_completed(futures):
            seen.append(f.result())

    # After all writes, the state file must contain a single valid phase.
    final = phase_guard.state_file().read_text(encoding="utf-8").strip()
    assert final in valid_values, f"torn or invalid final state: {final!r}"
    # Reader must succeed without raising.
    assert phase_guard.read_phase().value == final


def test_lock_times_out_if_held(monkeypatch):
    """If the lock is held, a competing acquire must time out cleanly."""

    holder_event = threading.Event()
    release_event = threading.Event()

    def holder():
        with phase_guard.FileLock(phase_guard.lock_file(), timeout_s=5.0):
            holder_event.set()
            release_event.wait(timeout=2.0)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert holder_event.wait(timeout=1.0)

    with pytest.raises(TimeoutError):
        with phase_guard.FileLock(phase_guard.lock_file(), timeout_s=0.1):
            pass  # pragma: no cover - should raise before entering body

    release_event.set()
    t.join(timeout=2.0)


# --- python-level bypass attempts -------------------------------------------


def test_python_bypass_attempt_fails(capsys):
    """``getattr``-style bypass cannot reach work without guard."""

    # Resolve cmd_verify dynamically to mimic a malicious agent doing
    # ``getattr(__import__("vllm_evolve.cli.main", ...), "cmd_verify")(args)``.
    module = __import__("vllm_evolve.cli.main", fromlist=["cmd_verify"])
    fn = getattr(module, "cmd_verify")
    import argparse
    args = argparse.Namespace(policy=str(SEED_POLICY), target="scheduling")
    rc = fn(args)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 2
    assert payload["outcome_class"] == "phase_violation"


# --- transition table integrity --------------------------------------------


def test_invalid_phase_transition_rejected(capsys):
    """``ve phase set GENERATE`` from INIT must be rejected with structured JSON."""

    rc = cli_main.main(["phase", "set", Phase.GENERATE.value])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 2
    assert payload["ok"] is False
    assert payload["outcome_class"] == "phase_violation"
    assert payload["current_phase"] == Phase.INIT.value
    # READ_CONTEXT is the only legal next phase from INIT (plus FINISH); the
    # message must surface that to the caller.
    assert "READ_CONTEXT" in payload["phase_required"]


def test_unknown_phase_token_rejected(capsys):
    rc = cli_main.main(["phase", "set", "NOT_A_PHASE"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 2
    assert payload["outcome_class"] == "unknown_phase"
    assert payload["given"] == "NOT_A_PHASE"


# --- bench refuses without VERIFY_PASSED -----------------------------------


def test_bench_phase_refuses_without_verify_passed(capsys):
    """``ve bench`` must reject any phase that is not VERIFY_PASSED."""

    rc = cli_main.main(["bench", str(SEED_POLICY), "scheduling"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 2
    assert payload["ok"] is False
    assert payload["outcome_class"] == "phase_violation"
    assert payload["phase_required"] == Phase.VERIFY_PASSED.value


def test_bench_phase_runs_after_verify_passes(capsys, monkeypatch):
    """Once verify succeeds, bench passes the guard and runs the real BenchConfig
    runner; with the box unreachable it reports an honest box_gated_blocked payload
    carrying the full provenance (never a fabricated pass)."""

    _walk_to(Phase.GENERATE)
    rc = cli_main.main(["verify", str(SEED_POLICY), "scheduling"])
    capsys.readouterr()
    assert rc == 0
    assert phase_guard.read_phase() == Phase.VERIFY_PASSED

    rc = cli_main.main(["bench", str(SEED_POLICY), "scheduling"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 1  # box-gated: the box is unreachable in this environment
    assert payload["ok"] is False
    assert payload["outcome_class"] == "box_gated_blocked"
    # the real runner built a BenchConfig + rendered the serve command (provenance)
    assert "provenance" in payload and payload["provenance"]["rendered_serve_args"]
    assert payload["provenance"]["runner_kind"] == "candidate"
