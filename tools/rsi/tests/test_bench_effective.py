"""M-B2: effective gate (anti-fabrication) + vanilla provenance (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.runtime import (  # noqa: E402
    PLUGIN_DEFERRED_MARKER,
    PLUGIN_FALLBACK_MARKER,
    PLUGIN_FORCED_ADMIT_MARKER,
    PLUGIN_INACTIVE_MARKER,
    PLUGIN_INVOKED_MARKER,
    PLUGIN_POLICY_CALL_MARKER,
    PLUGIN_PREEMPTED_MARKER,
    PLUGIN_REORDERED_MARKER,
    assert_plugin_effective,
    contains_marker_forgery,
    plugin_provenance,
    vanilla_provenance,
)

INV, REO, FB = PLUGIN_INVOKED_MARKER, PLUGIN_REORDERED_MARKER, PLUGIN_FALLBACK_MARKER


def test_marker_forgery_detected_in_policy_source():
    forged = f'def schedule_batch(w, r, *a):\n    print("{REO}")\n    return None\n'
    assert contains_marker_forgery(forged) is True


def test_forgery_check_closes_bypasses():
    # ANY reference to the marker namespace / constants / nonce env is rejected (HIGH#3)
    assert contains_marker_forgery('x = "vllm-evolve: " + "waiting reordered"') is True
    assert contains_marker_forgery("from m import PLUGIN_REORDERED_MARKER as z; print(z)") is True
    assert contains_marker_forgery('import os; print(os.environ["VE_MARKER_NONCE"])') is True


def test_clean_policy_is_not_forgery():
    assert contains_marker_forgery("def schedule_batch(w, r):\n    return sorted(w)") is False


def test_nonce_blocks_static_marker_forgery():
    # a log with the BARE marker (no nonce) must not satisfy a nonce'd gate
    assert assert_plugin_effective(f"{INV}\n{REO}\n", nonce="abc123") is False
    # the real wrapper emits "<marker> <nonce>" -> satisfies the gate
    real = f"{INV} abc123\n{REO} abc123\n"
    assert assert_plugin_effective(real, nonce="abc123") is True
    assert assert_plugin_effective(f"{real}{FB} abc123\n", nonce="abc123") is False


def test_effective_requires_invoked_reordered_no_fallback():
    log = f"... {INV} ...\n... {REO} ...\n"
    assert assert_plugin_effective(log) is True
    assert plugin_provenance(log)["effective"] is True


def test_fallback_invalidates_even_if_reordered():
    # candidate reordered once then errored -> fell back -> NOT a valid candidate
    log = f"{INV}\n{REO}\n{FB}\n"
    assert assert_plugin_effective(log) is False
    p = plugin_provenance(log)
    assert p["effective"] is False and p["fallback"] is True and p["fallback_count"] == 1


def test_loaded_but_errored_immediately_is_not_effective():
    # the silent-fallback hole M-B2 closes: invoked, never reordered, fell back
    log = f"{INV}\n{FB}\n"
    assert assert_plugin_effective(log) is False
    assert plugin_provenance(log)["effective"] is False


def test_invoked_only_not_effective():
    # invoked but no reorder (e.g. priority heap) -> loaded != effective
    assert assert_plugin_effective(f"{INV}\n") is False


def test_declared_inactive_is_valid_execution_but_not_effective():
    log = f"{INV} abc\n{PLUGIN_INACTIVE_MARKER} abc\n"
    provenance = plugin_provenance(log, nonce="abc")
    assert provenance["execution_valid"] is True
    assert provenance["mechanism_applicable"] is False
    assert provenance["effective"] is False


def test_real_preemption_is_effective_without_fake_reorder():
    log = f"{INV} abc\n{PLUGIN_PREEMPTED_MARKER} abc\n"
    provenance = plugin_provenance(log, nonce="abc")
    assert provenance["preempted"] is True
    assert provenance["preemption_count"] == 1
    assert provenance["effective"] is True
    assert provenance["execution_valid"] is True


def test_defer_is_effective_and_exposes_action_counters():
    log = (
        f"{INV} abc\n"
        f"{PLUGIN_POLICY_CALL_MARKER} abc\n"
        f"{PLUGIN_POLICY_CALL_MARKER} abc\n"
        f"{PLUGIN_DEFERRED_MARKER} abc\n"
        f"{PLUGIN_DEFERRED_MARKER} abc\n"
        f"{PLUGIN_FORCED_ADMIT_MARKER} abc\n"
    )
    provenance = plugin_provenance(log, nonce="abc")
    assert provenance["effective"] is True
    assert provenance["policy_invocation_count"] == 2
    assert provenance["deferred_request_actions"] == 2
    assert provenance["starvation_forced_admissions"] == 1


def test_vanilla_provenance_trusted_when_no_plugin():
    log = "INFO vllm serve facebook/opt-125m ... started\nINFO Avg generation throughput..."
    p = vanilla_provenance(log, model="facebook/opt-125m", vllm_version="0.14.0")
    assert p["trusted_baseline"] is True and p["no_plugin_marker"] is True


def test_vanilla_provenance_rejects_if_plugin_present():
    # a "vanilla" run that somehow loaded our plugin is NOT a trusted baseline
    assert vanilla_provenance(f"... {INV} ...")["trusted_baseline"] is False
    assert vanilla_provenance("INFO Using custom scheduler class X")["trusted_baseline"] is False
