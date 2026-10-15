"""M4 / AC4: the ve-author -> run_autopt evolve_fn bridge. The keystone anti-fabrication proof: an
UNVERIFIED / marker-forging / local_smoke authored policy can NEVER adopt. The author returns source
TEXT only and has no channel to claim it ran — marker_verified comes ONLY from the deterministic
bench. Pure, off-GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.runtime import PLUGIN_INVOKED_MARKER  # noqa: E402
from vllm_evolve.core.schemas import Profile, Spec  # noqa: E402
from vllm_evolve.engine.evolve_target import author_evolve_fn  # noqa: E402

_GOOD = "def schedule_batch(running, state):\n    return None\n"
_SPEC = Spec(metric="output_throughput_tok_s", direction="max")
_BASE = {"n_requests": 50, "model": "facebook/opt-125m"}


def test_local_smoke_authored_policy_never_verifies():
    # default profile_fn is local_smoke (marker_verified always False) -> untrusted, value empty.
    cand = author_evolve_fn(lambda c, s: _GOOD)(_BASE, _SPEC)
    assert cand.marker_verified is False and cand.value == {}


def test_marker_forgery_rejected_before_any_bench():
    forging = ("def schedule_batch(running, state):\n"
               f"    print('{PLUGIN_INVOKED_MARKER}')\n    return None\n")
    benched = []

    def spy(config):  # must never be called for a forging source
        benched.append(config)
        raise AssertionError("bench must not run on a marker-forging source")

    cand = author_evolve_fn(lambda c, s: forging, profile_fn=spy)(_BASE, _SPEC)
    assert cand.marker_verified is False and cand.value == {}
    assert "forgery" in cand.note and benched == []


def test_marker_verified_comes_from_the_bench_not_the_author():
    # the author only returns a string; the bridge takes marker_verified FROM the deterministic
    # bench (profile_fn here stands in for it). A verified bench -> adoptable; unverified -> empty.
    import os
    ok = author_evolve_fn(lambda c, s: _GOOD,
                          profile_fn=lambda cfg: Profile(marker_verified=True, source="real_vllm"))
    cand_ok = ok(_BASE, _SPEC)
    assert cand_ok.marker_verified is True
    path = cand_ok.value["policy"]                          # value carries a PATH, not source text
    assert Path(path).read_text(encoding="utf-8") == _GOOD  # the bench loads this exact policy
    os.unlink(path)                                         # cleanup the kept policy file
    no = author_evolve_fn(lambda c, s: _GOOD,
                          profile_fn=lambda cfg: Profile(marker_verified=False, source="real_vllm"))
    cand_no = no(_BASE, _SPEC)
    assert cand_no.marker_verified is False and cand_no.value == {}


def test_empty_author_output_is_unverified():
    assert author_evolve_fn(lambda c, s: "")(_BASE, _SPEC).value == {}
    assert author_evolve_fn(lambda c, s: None)(_BASE, _SPEC).value == {}


def test_bridge_output_fails_the_orchestrator_code_candidate_gate():
    # run_autopt skips a code candidate unless (cand.value and cand.marker_verified). A local_smoke
    # bridge can never satisfy that -> no_verified_candidate -> never adopts.
    cand = author_evolve_fn(lambda c, s: _GOOD)(_BASE, _SPEC)
    assert not (cand.value and cand.marker_verified)
