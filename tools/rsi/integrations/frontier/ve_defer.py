"""Pure two-channel admission reorder for the ve_policy Frontier bridge (P5 / bridge v2).

NO frontier and NO vllm_evolve imports — pure stdlib only. This module is the SINGLE SOURCE of
the anti-starvation logic: the Frontier bridge inserts ``<VE_POLICY_ROOT>/integrations/frontier``
on ``sys.path`` and imports it at runtime, while the vllm-evolve test suite imports it directly.
There is no copy embedded in ``ve_policy.patch`` — so there is nothing to drift out of sync, and
the two starvation proofs run fully offline (no Frontier checkout required).

Two channels, strictly separated (C3 — "forgotten" is NOT "deferred"):

* FORGOTTEN — a waiting id the policy never mentioned (neither chosen nor deferred). Kept in the
  parent's FCFS order BEHIND the policy's chosen ids. A forgetful policy can never starve a
  request — identical to bridge v1.
* DEFERRED — a waiting id the policy explicitly listed in ``decision.defer_ids``. Kept visible at
  the TAIL of the parent view, so capacity pressure delays it without relying on another simulator
  event to rediscover an omitted request. Anti-starvation decay: a per-id counter tracks CONSECUTIVE
  scheduling events an id has been deferred; once it reaches ``force_after`` (default 8) the id is
  FORCED to the HEAD of the view and its counter resets to 0.

Precedence: an id that is BOTH chosen and deferred is treated as CHOSEN (the explicit pick wins;
its defer counter clears). The view never invents ids — every returned id was in ``parent_ids``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_FORCE_AFTER = 8


@dataclass
class ReorderResult:
    """One scheduling event's outcome.

    ``order`` is the complete waiting-queue view handed to the parent scheduler (admission order);
    deferred ids that were not forced this step are placed at the tail. ``forced`` are deferred ids
    whose decay fired (placed at the head); ``deferred`` are ids soft-deferred this step.
    """

    order: list[str]
    forced: list[str]
    deferred: list[str]


@dataclass
class DeferTracker:
    """Stateful across scheduling events; owns the per-id consecutive-defer counters."""

    force_after: int = DEFAULT_FORCE_AFTER
    _counts: dict[str, int] = field(default_factory=dict)

    def reorder(self, parent_ids, chosen_ids, defer_ids) -> ReorderResult:
        """Compose the view for one scheduling event and advance the decay counters.

        ``parent_ids``  — the parent scheduler's FCFS order (the full waiting queue this step).
        ``chosen_ids``  — the policy's preferred prefill order (a subset, in policy order).
        ``defer_ids``   — the policy's explicit defer list (the throttle channel).
        """
        parent = [str(x) for x in parent_ids]
        present = set(parent)
        chosen_set = {str(x) for x in chosen_ids if str(x) in present}
        # chosen wins over a contradictory defer: only count a defer that is present AND not chosen.
        defer_set = {
            str(x) for x in defer_ids if str(x) in present and str(x) not in chosen_set
        }

        forced: list[str] = []
        deferred: list[str] = []
        for rid in parent:  # iterate in parent order -> deterministic forced/deferred ordering
            if rid in defer_set:
                self._counts[rid] = self._counts.get(rid, 0) + 1
                if self._counts[rid] >= self.force_after:
                    forced.append(rid)
                    self._counts[rid] = 0  # decay fired -> reset, bounded starvation
                else:
                    deferred.append(rid)
            else:
                # chosen or forgotten -> not deferred this step -> the streak breaks.
                self._counts.pop(rid, None)
        # forget counters for ids that left the queue entirely (admitted / completed).
        for rid in list(self._counts):
            if rid not in present:
                del self._counts[rid]

        chosen_order: list[str] = []
        seen: set[str] = set()
        for x in chosen_ids:  # preserve the POLICY's order, de-duplicated, present-only.
            rid = str(x)
            if rid in chosen_set and rid not in seen:
                chosen_order.append(rid)
                seen.add(rid)
        forgotten = [
            rid for rid in parent if rid not in chosen_set and rid not in defer_set
        ]
        # Frontier's event loop does not promise a future scheduling event for a request omitted
        # from the returned view.  Keep every parent id visible: forced defers go first, followed
        # by chosen and forgotten ids, while ordinary defers are delayed at the tail.
        order = forced + chosen_order + forgotten + deferred
        return ReorderResult(order=order, forced=forced, deferred=deferred)
