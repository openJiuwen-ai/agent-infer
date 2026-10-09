"""Decision matrix for the PreToolUse phase-guard hook (pure, no Claude Code)."""
from __future__ import annotations

import pytest

from vllm_evolve.install.hook_decision import decide
from vllm_evolve.tools.phase_guard import Phase

# ── protected-path writes are always denied ───────────────────

@pytest.mark.parametrize("path", [
    "targets/scheduling/seed.py",
    "targets/scheduling/skeleton.py",
    "config/bench/profiles/throughput.yaml",
    "schemas/eval_result.schema.json",
    "archive_policies/scheduling/abc/policy.py",
    ".ve/state.json",
    ".claude/hooks/phase_guard_hook.py",
])
@pytest.mark.parametrize("phase", [Phase.GENERATE, Phase.READ_CONTEXT, Phase.INIT])
def test_protected_paths_denied_in_any_phase(path, phase):
    d = decide("Edit", {"file_path": path}, phase)
    assert d.allow is False
    assert "protected" in d.reason


# ── run provenance (runs/<target>/<run_id>/...) is machine-written, denied ──

@pytest.mark.parametrize("path", [
    "runs/scheduling/20260610-120000-autopt-abc123/manifest.json",
    "runs/scheduling/20260610-120000-autopt-abc123/eval_result.json",
    "runs/scheduling/20260610-120000-autopt-abc123/bench/seed1.json",
    "runs/scheduling/20260610-120000-autopt-abc123/gpu/trace.ncu",
    # absolute / Windows path (hook _norm collapses backslashes) — R3-n1
    r"C:\Users\u\proj\runs\scheduling\r1\manifest.json",
    "/home/u/proj/runs/scheduling/r1/bench/x.json",
])
@pytest.mark.parametrize("phase", [Phase.GENERATE, Phase.BENCHMARK, Phase.INIT])
def test_run_provenance_denied_in_any_phase(path, phase):
    d = decide("Write", {"file_path": path}, phase)
    assert d.allow is False
    assert "provenance" in d.reason


@pytest.mark.parametrize("path", [
    "runs/scheduling/20260610-120000-autopt-abc123/design.md",   # human-editable working file
    "runs/scheduling/20260610-120000-autopt-abc123/notes.txt",
    "runs/README.md",                                            # not a run bundle file
    "src/runs_helper.py",                                        # 'runs' substring, not the dir
])
def test_run_working_files_allowed(path):
    assert decide("Write", {"file_path": path}, Phase.GENERATE).allow is True


# ── policy edits are GENERATE-only ────────────────────────────

def test_policy_edit_allowed_in_generate():
    d = decide("Write", {"file_path": "targets/scheduling/work.py"}, Phase.GENERATE)
    assert d.allow is True


def test_policy_edit_denied_outside_generate():
    d = decide("Write", {"file_path": "targets/scheduling/work.py"}, Phase.DESIGN)
    assert d.allow is False
    assert "GENERATE" in d.reason


def test_non_policy_edit_allowed():
    # editing a scratch file outside targets/ + protected paths is fine
    d = decide("Edit", {"file_path": "notes/scratch.md"}, Phase.READ_CONTEXT)
    assert d.allow is True


# ── ar verb gating via Bash ───────────────────────────────────

def test_bench_denied_without_verify_passed():
    d = decide("Bash", {"command": "ve bench targets/scheduling/work.py --profile throughput"},
               Phase.BENCHMARK)
    assert d.allow is False
    assert "VERIFY_PASSED" in d.reason


def test_bench_allowed_after_verify_passed():
    d = decide("Bash", {"command": "ve bench work.py"}, Phase.VERIFY_PASSED)
    assert d.allow is True


def test_context_verb_gated_by_phase():
    assert decide("Bash", {"command": "ve context scheduling"}, Phase.INIT).allow is False
    assert decide("Bash", {"command": "ve context scheduling"}, Phase.READ_CONTEXT).allow is True


def test_keep_verb_gated_by_phase():
    assert decide("Bash", {"command": "ve keep work.py"}, Phase.KEEP_OR_DISCARD).allow is False
    assert decide("Bash", {"command": "ve keep work.py"}, Phase.COMMIT_OR_ROLLBACK).allow is True


def test_python_module_invocation_is_gated_too():
    d = decide("Bash", {"command": "python -m vllm_evolve.cli.main bench work.py"}, Phase.DESIGN)
    assert d.allow is False


def test_ve_command_is_gated_like_ar():
    # after the ar->ve rename the hook gates `ve <verb>` (ar kept for back-compat)
    assert decide("Bash", {"command": "ve context scheduling"}, Phase.INIT).allow is False
    assert decide("Bash", {"command": "ve context scheduling"}, Phase.READ_CONTEXT).allow is True
    assert decide("Bash", {"command": "ve keep work.py"}, Phase.COMMIT_OR_ROLLBACK).allow is True


def test_admin_phase_verb_always_allowed():
    # `ve phase ...` is administrative -> never blocked
    assert decide("Bash", {"command": "ve phase status"}, Phase.INIT).allow is True


def test_experiment_admin_verb_is_phase_independent():
    # R6: `ve experiment` is an admin verb (register_verb("experiment", None)) -> allowed in EVERY
    # phase, WITHOUT changing the normal phase machine (a gated verb like bench stays gated).
    for ph in (Phase.INIT, Phase.READ_CONTEXT, Phase.GENERATE, Phase.BENCHMARK,
               Phase.COMMIT_OR_ROLLBACK):
        assert decide("Bash", {"command": "ve experiment run spec.json"}, ph).allow is True
        assert decide("Bash", {"command": "ve experiment list"}, ph).allow is True
    assert decide("Bash", {"command": "ve bench work.py"}, Phase.GENERATE).allow is False


def test_run_front_door_admin_verb_is_phase_independent():
    # VE_RUN: `ve run` is an admin verb (register_verb("run", None)) -> allowed in EVERY phase
    # without changing the phase machine (a gated verb like bench stays gated).
    cmd = 've run --model demo --hw none --metric "maximize throughput" --method evolve'
    for ph in (Phase.INIT, Phase.READ_CONTEXT, Phase.GENERATE, Phase.BENCHMARK,
               Phase.COMMIT_OR_ROLLBACK):
        assert decide("Bash", {"command": cmd}, ph).allow is True
    assert decide("Bash", {"command": "ve bench work.py"}, Phase.GENERATE).allow is False


def test_non_ar_bash_allowed():
    for cmd in ["git status", "ls -la", "python -m pytest tests/bench -q", "rg foo src"]:
        assert decide("Bash", {"command": cmd}, Phase.INIT).allow is True


# ── M4: verb→phase is canonical in phase_guard; hook doesn't import the CLI ──

def test_round_verb_phases_are_canonical_in_phase_guard():
    # the single source of truth (no import-time registration from the CLI module)
    from vllm_evolve.tools import phase_guard
    table = phase_guard._VERB_PHASES
    assert table["context"] == Phase.READ_CONTEXT
    assert table["design"] == Phase.DESIGN
    assert table["real-evolve"] == Phase.DESIGN
    assert table["compare"] == Phase.KEEP_OR_DISCARD
    assert table["keep"] == Phase.COMMIT_OR_ROLLBACK
    assert table["discard"] == Phase.COMMIT_OR_ROLLBACK


def test_hook_does_not_depend_on_cli_import_side_effect():
    # M4: hook_decision must NOT import the CLI module to learn verb phases
    import pathlib

    import vllm_evolve.install.hook_decision as hd
    src = pathlib.Path(hd.__file__).read_text(encoding="utf-8")
    assert "import vllm_evolve.cli.main" not in src
    # and gating still works for a round verb purely via the canonical table
    assert decide("Bash", {"command": "ve keep work.py"}, Phase.KEEP_OR_DISCARD).allow is False
    assert decide("Bash", {"command": "ve keep work.py"}, Phase.COMMIT_OR_ROLLBACK).allow is True


def test_real_evolve_is_design_phase_locked():
    command = "ve real-evolve --bench-config b.json --research-snapshot r.json"
    assert decide("Bash", {"command": command}, Phase.INIT).allow is False
    assert decide("Bash", {"command": command}, Phase.DESIGN).allow is True
