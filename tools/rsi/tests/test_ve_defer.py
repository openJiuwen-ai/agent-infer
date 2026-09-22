"""R8/AC6: the ve_policy bridge-v2 two-channel admission reorder (anti-starvation decay).

These are the plan's two PROOF-OBLIGATION starvation tests (plan §5 test 6) plus composition /
precedence checks. They run FULLY OFFLINE against the pure `ve_defer` logic — no Frontier checkout
or venv — because `integrations/frontier/ve_defer.py` is the single source the bridge imports at
runtime. The real-Frontier wiring is exercised by the (Frontier-gated) e2e.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTIER = REPO_ROOT / "integrations" / "frontier"
if str(FRONTIER) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(FRONTIER))

from ve_defer import DEFAULT_FORCE_AFTER, DeferTracker  # noqa: E402


def test_default_force_after_is_eight():
    # Q3: N defaults to 8 scheduling events.
    assert DEFAULT_FORCE_AFTER == 8
    assert DeferTracker().force_after == 8


def test_forgotten_id_is_appended_and_never_omitted():
    # Channel 1: a policy that mentions only some ids; an id never mentioned and never deferred
    # must ALWAYS appear in the view (FCFS append behind chosen) — a forgetful policy cannot starve.
    t = DeferTracker()
    parent = ["a", "b", "c"]
    for _ in range(50):  # many events: "c" is forgotten every time
        r = t.reorder(parent, chosen_ids=["b", "a"], defer_ids=[])
        assert r.order == ["b", "a", "c"]      # chosen order first, forgotten FCFS behind
        assert r.deferred == [] and r.forced == []
        assert "c" in r.order                  # never omitted


def test_always_deferred_id_is_force_admitted_within_force_after():
    # Channel 2: defer the same id EVERY event. It stays visible at the tail for force_after-1
    # consecutive events, then is forced to the HEAD on the force_after-th.
    t = DeferTracker()  # force_after = 8
    parent = ["x", "y"]
    tail_streak = 0
    forced_events = []
    for event in range(1, 25):  # ~3 decay cycles
        r = t.reorder(parent, chosen_ids=["y"], defer_ids=["x"])
        if "x" in r.forced:
            forced_events.append(event)
            assert r.order[0] == "x"           # forced to the queue HEAD
            assert "x" not in r.deferred
            assert tail_streak == t.force_after - 1
            tail_streak = 0
        else:
            assert "x" in r.deferred            # soft-deferred this step
            assert r.order[-1] == "x"           # visible at the tail, never stranded
            tail_streak += 1
            assert tail_streak <= t.force_after - 1
    # forced exactly on events 8, 16, 24 -> deterministic, bounded
    assert forced_events == [8, 16, 24]


def test_deferred_streak_resets_when_id_is_later_chosen():
    # The streak is CONSECUTIVE: choosing (or forgetting) the id breaks it, so the decay clock
    # restarts — defer is a per-event decision, not a cumulative debt.
    t = DeferTracker()
    parent = ["x", "y"]
    for _ in range(5):
        t.reorder(parent, chosen_ids=["y"], defer_ids=["x"])   # build a streak of 5
    r = t.reorder(parent, chosen_ids=["x", "y"], defer_ids=[])  # now chosen -> streak breaks
    assert "x" in r.order and r.forced == []
    # re-deferring starts the count from 1 again: 7 more tail placements, force on the 8th.
    forced_at = None
    for event in range(1, 12):
        r = t.reorder(parent, chosen_ids=["y"], defer_ids=["x"])
        if "x" in r.forced:
            forced_at = event
            break
    assert forced_at == t.force_after


def test_empty_defer_event_resets_streak():
    # The bridge must call reorder() on EVERY event, including ones with NO defer_ids, so a streak
    # is cleared the moment an id is not deferred (the decay rule is CONSECUTIVE defers). This is
    # the exact Round-5 bridge bug: defer x for several events, then one no-defer event, then defer
    # again -> force must take a FRESH full N, not the stale remaining count.
    t = DeferTracker()  # force_after = 8
    parent = ["x", "y"]
    for _ in range(5):
        t.reorder(parent, chosen_ids=["y"], defer_ids=["x"])     # streak of 5
    r = t.reorder(parent, chosen_ids=["y"], defer_ids=[])        # NO defer event -> streak clears
    assert "x" in r.order and r.forced == [] and r.deferred == []
    forced_at = None
    for event in range(1, 12):
        r = t.reorder(parent, chosen_ids=["y"], defer_ids=["x"])
        if "x" in r.forced:
            forced_at = event
            break
    assert forced_at == t.force_after          # fresh 8 events, NOT 3 (8 - 5 stale)


def test_chosen_wins_over_a_contradictory_defer():
    # An id listed in BOTH prefill_batch and defer_ids is treated as CHOSEN (explicit pick wins):
    # it is always preferred, never accrues a defer count, and is never the one deferred/forced.
    # ("b" is the genuinely deferred id; its own decay is exercised by the dedicated test above.)
    t = DeferTracker()
    parent = ["a", "b"]
    for _ in range(20):
        r = t.reorder(parent, chosen_ids=["a"], defer_ids=["a", "b"])
        assert "a" in r.order                         # a is admitted every event
        assert "a" not in r.deferred and "a" not in r.forced


def test_composition_forced_then_chosen_then_forgotten_then_deferred():
    # A single event composes in the documented order:
    # forced -> chosen -> forgotten -> soft-deferred.
    t = DeferTracker(force_after=2)
    parent = ["f", "c1", "c2", "g"]
    # event 1: defer "f" (streak 1, at tail), choose c2 then c1, forget g
    r1 = t.reorder(parent, chosen_ids=["c2", "c1"], defer_ids=["f"])
    assert r1.order == ["c2", "c1", "g", "f"] and r1.deferred == ["f"]
    # event 2: defer "f" again -> streak 2 == force_after -> forced to head
    r2 = t.reorder(parent, chosen_ids=["c2", "c1"], defer_ids=["f"])
    assert r2.forced == ["f"]
    assert r2.order == ["f", "c2", "c1", "g"]   # forced, then chosen order, then forgotten


def test_view_never_invents_ids_and_drops_departed_counters():
    # The view only ever contains ids from parent; counters for ids that leave the queue are
    # forgotten (no unbounded growth, no stale force on a returning id).
    t = DeferTracker(force_after=3)
    t.reorder(["x", "y"], chosen_ids=["y"], defer_ids=["x"])  # x streak 1
    t.reorder(["x", "y"], chosen_ids=["y"], defer_ids=["x"])  # x streak 2
    # x leaves the queue entirely for an event -> its counter must be dropped
    r = t.reorder(["y"], chosen_ids=["y"], defer_ids=[])
    assert r.order == ["y"] and "x" not in r.order
    # x returns: the streak restarts from 1, NOT from the stale 2 (would-be force at 3).
    r = t.reorder(["x", "y"], chosen_ids=["y"], defer_ids=["x"])
    assert "x" in r.deferred and r.forced == [] and r.order == ["y", "x"]
