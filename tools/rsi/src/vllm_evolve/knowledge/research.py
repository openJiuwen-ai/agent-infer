"""Store-backed research synthesis: cited facts about prior policy attempts.

Read-only views over :class:`vllm_evolve.store.db.Store` for the ``ve-research``
step. No embeddings, no RAG, no new SQL schema — just the existing
``best_policies`` / ``history`` / ``lineage_chain`` queries, returned with explicit
``citations`` (policy ids + evaluation ids) so a downstream proposal is grounded in
real records.
"""
from __future__ import annotations

from vllm_evolve.store.db import Store


def gather(target: str, *, db: str | None = None, n: int = 5,
           regime: str | None = None, metric: str | None = None) -> dict:
    """Return cited prior-attempt facts for ``target`` (top-N policies + their
    evaluation histories + lineage) PLUS Evolution-v2 lessons (immutable evidence
    rows, deduped via the lessons_latest view), each tagged with its real id.

    Pure read; opens (and closes) the store. ``db`` overrides the default DB path."""
    store = Store(db) if db else Store()
    try:
        top = store.best_policies(target, n=n)
        citations: list[str] = []
        lineage: dict[str, list] = {}
        histories: dict[str, list] = {}
        for row in top:
            pid = row.get("policy_id")
            if not pid:
                continue
            citations.append(f"policy:{pid}")
            lineage[pid] = store.lineage_chain(pid)
            hist = store.history("policy", pid)
            histories[pid] = hist
            citations += [f"eval:{h['id']}" for h in hist if h.get("id") is not None]
        # Evolution-v2 lessons: each row is immutable evidence with a verifiable lesson_id;
        # `source` names the true origin (e.g. frontier_sim) — a lesson is never presented
        # as a real-vLLM result.
        lessons = store.get_lessons(regime=regime, metric=metric, limit=n)
        citations += [f"lesson:{r['lesson_id']}" for r in lessons]
        # Research-harness hypothesis ledger: prior hypotheses + their verdicts, ALL of them —
        # supported, falsified, AND inconclusive. A FALSIFIED hypothesis is first-class research
        # memory (it stops the loop re-proposing a refuted idea), not a discarded failure. Each is
        # cited by its immutable adjudication/prediction event id.
        hypotheses = store.list_hypotheses(limit=n)
        for h in hypotheses:
            ev = h.get("adjudication_event_id") or h.get("prediction_event_id")
            if ev is not None:
                citations.append(f"hypothesis:{ev}")
        return {
            "target": target,
            "top_policies": top,
            "lineage": lineage,
            "histories": histories,
            "lessons": lessons,
            "hypotheses": hypotheses,
            "citations": citations,
        }
    finally:
        store.close()
