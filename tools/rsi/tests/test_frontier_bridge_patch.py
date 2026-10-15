"""R8/AC6 bridge-level regression guards (offline).

The real ve_policy bridge lives in `integrations/frontier/ve_policy.patch` (it is applied into a
Frontier checkout, so it cannot be imported here). These tests pin the ACTUAL patched bridge
behaviour by:

1. asserting the patch source dispatches every event through the single-source `DeferTracker`
   (the Round-5 bug was a `or not defer` short-circuit that skipped the tracker on empty defers,
   leaving stale streaks), exposes the tenant `session_id`, and records the `defers` marker;
2. exercising the policy-facing `ve_policy_api.ScheduleDecision(..., defer_ids=[...])` through the
   exact read+dispatch logic the bridge uses, proving the bridge reads those ids and that an
   empty-defer event resets streaks via the bridge path — not only the helper's.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTIER = REPO_ROOT / "integrations" / "frontier"
if str(FRONTIER) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(FRONTIER))

import ve_defer  # noqa: E402
from ve_policy_api import ScheduleDecision  # noqa: E402

PATCH = (FRONTIER / "ve_policy.patch").read_text(encoding="utf-8")


def _bridge_hunk_source(patch: str) -> str:
    """The '+'-prefixed body of the NEW bridge file hunk ONLY (`@@ -0,0 +1,N @@` ... next
    `diff --git`) — NOT every '+' line in the patch, which would interleave the other three files'
    added lines and corrupt the reconstructed module."""
    lines = patch.splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if re.match(r"^@@ -0,0 \+1,\d+ @@$", ln))
    end = next(i for i in range(hdr + 1, len(lines)) if lines[i].startswith("diff --git"))
    return "\n".join(ln[1:] for ln in lines[hdr + 1:end] if ln.startswith("+"))


_BRIDGE_SRC = _bridge_hunk_source(PATCH)


def test_patch_dispatch_always_delegates_to_the_tracker():
    # the buggy short-circuit must be gone: the only non-tracker fallback is `_ve_defer is None`.
    assert "or not defer" not in _BRIDGE_SRC
    assert "if self._ve_defer is None:" in _BRIDGE_SRC
    # and the tracker is actually called on the v2 path
    assert "self._ve_defer.reorder(" in _BRIDGE_SRC


def test_patch_exposes_session_and_records_defers():
    assert '"session_id"' in _BRIDGE_SRC and "req.session_id" in _BRIDGE_SRC
    assert '"has_prefix_hint"' in _BRIDGE_SRC and '"block_hash_ids"' in _BRIDGE_SRC
    assert '"defers": self._ve_defers' in _BRIDGE_SRC
    assert '"forced": self._ve_forced' in _BRIDGE_SRC
    assert '"generated_output_tokens"' in _BRIDGE_SRC
    assert '"waiting_age_s"' in _BRIDGE_SRC


def test_patch_executes_and_records_policy_preemption():
    assert 'getattr(decision, "preempt_ids"' in _BRIDGE_SRC
    assert "self._preempt_request(victim, preempted_requests)" in _BRIDGE_SRC
    assert '"preemptions": self._ve_preemptions' in _BRIDGE_SRC


def test_patch_body_is_valid_python():
    compile(_BRIDGE_SRC, "ve_policy_replica_scheduler.py", "exec")


# ── the bridge's read + dispatch, replicated faithfully from the patch source ──

def _bridge_dispatch(tracker, parent_ids, decision):
    """Exactly what `_get_sorted_waiting_queue` does after calling the policy: read prefill/defer
    off the decision via getattr, then dispatch through the tracker (or v1 fallback)."""
    chosen = [str(rid) for rid in getattr(decision, "prefill_batch", [])]
    defer = [str(rid) for rid in (getattr(decision, "defer_ids", ()) or ())]
    by_id = {rid: rid for rid in parent_ids}
    if tracker is None:
        chosen_reqs = [by_id.pop(rid) for rid in chosen if rid in by_id]
        return chosen_reqs + list(by_id.values())
    result = tracker.reorder(list(parent_ids), chosen, defer)
    return [by_id[rid] for rid in result.order if rid in by_id]


def test_bridge_reads_defer_ids_off_the_policy_api_decision():
    # a policy returning the sanctioned ScheduleDecision(..., defer_ids=[...]) is honored: the
    # deferred id is moved to the tail while remaining visible (no custom return object needed).
    d = ScheduleDecision(prefill_batch=["a"], decode_batch=[], defer_ids=["b"])
    order = _bridge_dispatch(ve_defer.DeferTracker(), ["a", "b"], d)
    assert order == ["a", "b"]                  # b soft-deferred -> visible at tail


def test_bridge_empty_defer_event_resets_streak():
    # drive the bridge dispatch (not just the helper): defer b 5x, one empty-defer event, then defer
    # b again -> force takes a fresh N, proving the empty-defer event reached the tracker.
    t = ve_defer.DeferTracker()
    parent = ["a", "b"]
    for _ in range(5):
        _bridge_dispatch(t, parent, ScheduleDecision(prefill_batch=["a"], defer_ids=["b"]))
    # empty-defer event: a policy that defers nothing this step
    _bridge_dispatch(t, parent, ScheduleDecision(prefill_batch=["a"]))
    forced_at = None
    for event in range(1, 12):
        order = _bridge_dispatch(t, parent, ScheduleDecision(prefill_batch=["a"], defer_ids=["b"]))
        if order and order[0] == "b":           # forced to the head
            forced_at = event
            break
    assert forced_at == t.force_after           # fresh 8, not 3


def test_default_decode_and_defer_fields_are_omittable():
    # backwards compatible: a policy can construct it like the skeleton's 2-field decision.
    d = ScheduleDecision(prefill_batch=["a"])
    assert d.decode_batch == [] and d.preempt_ids == [] and d.defer_ids == []
