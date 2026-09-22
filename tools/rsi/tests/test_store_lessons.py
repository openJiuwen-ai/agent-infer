"""D: immutable lesson_evidence + lessons_latest dedup view (Evolution v2). Offline, tmp DB."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.store.db import Store  # noqa: E402
from vllm_evolve.store.migrations import apply_migrations  # noqa: E402


def _lesson(store, sha="p1", score=10.0, conclusion="c", run="r1"):
    return store.put_lesson(run_id=run, source="frontier_sim", policy_sha=sha,
                            regime="throughput", metric="goodput_req_s",
                            score=score, conclusion=conclusion)


def test_cited_lesson_id_is_immutable(tmp_path):
    with Store(tmp_path / "s.db") as store:
        lid = _lesson(store, score=10.0, conclusion="first finding", run="r1")
        before = store.get_lesson(lid)
        # a LATER run on the same dedup key INSERTS new evidence — never rewrites the cited row
        _lesson(store, score=99.0, conclusion="revised claim", run="r2")
        after = store.get_lesson(lid)
        assert before == after and after["conclusion"] == "first finding"
        # but the dedup view serves the LATEST evidence for that key
        latest = store.get_lessons(regime="throughput", metric="goodput_req_s", limit=5)
        same_key = [r for r in latest if r["policy_sha"] == "p1"]
        assert len(same_key) == 1 and same_key[0]["conclusion"] == "revised claim"


def test_get_lessons_limit_is_hard(tmp_path):
    with Store(tmp_path / "s.db") as store:
        for i in range(8):
            _lesson(store, sha=f"p{i}", score=float(i))
        rows = store.get_lessons(limit=3)
        assert len(rows) == 3
        assert [r["score"] for r in rows] == [7.0, 6.0, 5.0]   # best-score-first


def test_cross_run_lessons_reach_gather_and_citations_verify(tmp_path):
    # E: run 1 persists lessons -> run 2's research gather() returns them with verifiable ids.
    from vllm_evolve.knowledge.research import gather
    from vllm_evolve.ops.inspect import verify_citations
    db = tmp_path / "s.db"
    with Store(db) as store:
        lid = _lesson(store, sha="w1", score=18.0,
                      conclusion="SJF beats LJF under TTFT SLO", run="run1")
    out = gather("scheduling", db=str(db), regime="throughput", metric="goodput_req_s")
    assert any(r["lesson_id"] == lid for r in out["lessons"])
    assert f"lesson:{lid}" in out["citations"]
    with Store(db) as store:
        v = verify_citations(store, [f"lesson:{lid}", "lesson:99999"])
        assert f"lesson:{lid}" in v["real"] and "lesson:99999" in v["fabricated"]


def test_migration_idempotent_on_existing_db(tmp_path):
    db = tmp_path / "s.db"
    with Store(db) as store:
        lid = _lesson(store)
    # re-opening (re-running apply_migrations) must not touch existing evidence
    with Store(db) as store:
        applied = apply_migrations(store._conn)
        assert "0005_lessons" not in [a for a in applied]   # already applied -> no-op
        assert store.get_lesson(lid) is not None
