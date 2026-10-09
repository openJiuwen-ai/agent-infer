"""Consistency: the PreToolUse hook gates `ar` verbs the same way phase_guard
does. Both read phase_guard._VERB_PHASES as the single source of truth; this
test catches drift if a verb is added to one side but not the other (M6)."""
from __future__ import annotations

# Importing ar_cli registers the round-lifecycle verbs into _VERB_PHASES.
import vllm_evolve.cli.main  # noqa: F401,E402
from vllm_evolve.install import hook_decision
from vllm_evolve.tools import phase_guard
from vllm_evolve.tools.phase_guard import Phase


def test_simulate_phase_is_gone():
    assert not hasattr(Phase, "SIMULATE")
    assert "SIMULATE" not in {p.value for p in Phase}


def test_verify_passed_goes_straight_to_benchmark():
    assert Phase.BENCHMARK in phase_guard._TRANSITIONS[Phase.VERIFY_PASSED]
    # no intermediate phase between VERIFY_PASSED and BENCHMARK
    assert all(p in (Phase.BENCHMARK, Phase.FINISH)
               for p in phase_guard._TRANSITIONS[Phase.VERIFY_PASSED])


def test_hook_gates_every_registered_round_verb():
    gated_verbs = [
        v for v, req in phase_guard._VERB_PHASES.items()
        if req is not None and v != "legacy_tool"
    ]
    # the M4 round verbs must all be present and gated
    for expected in (
        "verify", "bench", "context", "design", "real-evolve", "compare", "keep", "discard",
    ):
        assert expected in gated_verbs, f"{expected} not registered with a phase"

    for verb in gated_verbs:
        required = phase_guard._VERB_PHASES[verb]
        for phase in Phase:
            ok, _ = hook_decision._verb_allowed(verb, phase)
            if required == Phase.BENCHMARK:
                expected = phase == Phase.VERIFY_PASSED
            elif required == Phase.VERIFY:
                expected = phase in (Phase.GENERATE, Phase.VERIFY)
            else:
                expected = phase == required
            assert ok == expected, f"hook gating drift: {verb} @ {phase.value}"
