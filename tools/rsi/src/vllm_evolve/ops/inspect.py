"""ve inspect — read-only views over the store (top-N ranking) + the cited-fact contract.

Pure view assembly: every function takes a ``Store`` and only reads (SELECT) — no mutation, no
migration. ``inspect_top`` ranks via the ``policy_ranking`` view; ``verify_citations`` backs the
ve-research anti-fabrication contract (a cited policy_id must exist in the store).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_evolve.store.db import Store


def inspect_top(store: Store, target: str, n: int) -> dict:
    """Top-N policies for a target by average fitness (policy_ranking view)."""
    return {"view": "top", "target": target, "top": store.best_policies(target, n)}


def verify_citations(store: Store, policy_ids: list[str]) -> dict:
    """Cited-fact contract for ve-research.

    Every policy_id a research summary cites MUST exist in the store. Returns which cited ids are
    real vs fabricated (plus ``checked`` = how many were verified), so the orchestrator can REJECT a
    summary that invents a past result — the research agent is an untrusted producer and may not
    hallucinate a record. Read-only (SELECT via get_policy). An empty citation list is vacuously
    all-real; callers that require evidence should additionally check ``checked > 0``.
    """
    real: list[str] = []
    fabricated: list[str] = []
    for pid in policy_ids:
        if pid.startswith("lesson:"):
            # Evolution-v2 lesson citation: must point at an immutable evidence row.
            try:
                found = store.get_lesson(int(pid.split(":", 1)[1])) is not None
            except (ValueError, TypeError):
                found = False
            (real if found else fabricated).append(pid)
            continue
        if pid.startswith("hypothesis:"):
            # Research-harness hypothesis citation: must point at an immutable ledger event.
            try:
                found = store.get_hypothesis_event(int(pid.split(":", 1)[1])) is not None
            except (ValueError, TypeError):
                found = False
            (real if found else fabricated).append(pid)
            continue
        bare = pid.split(":", 1)[1] if pid.startswith("policy:") else pid
        (real if store.get_policy(bare) is not None else fabricated).append(pid)
    return {"view": "citations", "checked": len(policy_ids), "all_real": not fabricated,
            "real": real, "fabricated": fabricated}
