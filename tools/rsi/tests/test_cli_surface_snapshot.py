"""P0 safety-net: a characterization snapshot of the CLI surface.

This freezes the *current* top-level verb set and `--help` behavior so the
upcoming refactor can prove it does not silently change the command surface:

* **P3** renames the command (`ar` -> `ve`) and splits `ar_cli.py` into a `cli/`
  package. The verb set and help listing MUST be unchanged by that move — this
  test is the guard.
* **P4** intentionally ADDS verbs (`tune`, `port`, `run`). When that lands, the
  frozen set below is updated *consciously* in the same commit — an accidental
  drift fails here first.

No GPU; pure parser introspection. (Categories 2-4 of the P0 net — pure-handler
behavior, local_smoke plumbing, and @requires_gpu gating — are already covered
by test_ar_cli.py, test_e2e_local_smoke.py, and tests/bench/* respectively.)
"""
from __future__ import annotations

import pytest

from vllm_evolve.cli.main import build_parser
from vllm_evolve.cli.main import main as cli_main

# The frozen top-level verb surface as of the pre-refactor baseline.
# Sorted; update ONLY in the same commit that intentionally changes the surface.
EXPECTED_VERBS = frozenset({
    "autopt", "bench", "calibrate", "compare", "context", "design",
    "diagnose", "discard", "goal", "init", "inspect", "keep", "optimize",
    "phase", "profile", "targets", "verify", "verify-gain",
    # P4: new top-level mode entry points
    "tune", "port",
    # P5: run-area browse/gc
    "runs",
    # Research Harness (R6): open-world hypothesis research (admin verb)
    "experiment",
    # Codex + local real-Frontier generational search (sim-winner only)
    "frontier-evolve",
    # Strict real-vLLM-only generational search
    "real-evolve",
    # Internal Auto Research Expert compiler (admin utility, not a fourth mode)
    "research",
    # VE_RUN: the natural-language front door ({model,hw,metric,method} -> self-configured run)
    "run",
})


def _verb_choices() -> set[str]:
    parser = build_parser()
    return set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]


def test_cli_verb_surface_snapshot():
    assert _verb_choices() == set(EXPECTED_VERBS)


def test_help_exits_zero_and_lists_every_verb(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for verb in EXPECTED_VERBS:
        assert verb in out, f"help omits verb {verb!r}"
