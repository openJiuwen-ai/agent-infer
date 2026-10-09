"""R4/AC2: event-sourced hypothesis ledger (migration 0006). Append-only events chain
prediction_registered -> experiment_executed (prediction-first) -> adjudication_recorded (anchored
to the exact execution). Citations verifiable; research consumes falsified too. Pure/offline."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.knowledge.research import gather  # noqa: E402
from vllm_evolve.ops.inspect import verify_citations  # noqa: E402
from vllm_evolve.store.db import Store  # noqa: E402

_PRED = {"metric": {"column": "ttft", "agg": "p99"}, "comparator": "<", "arm_a": "B",
         "arm_b": "A", "margin": 10}


def _store(tmp_path) -> Store:
    return Store(tmp_path / "h.db")


def test_migration_0006_applies(tmp_path):
    s = _store(tmp_path)
    # the table exists after open (migration ran); inserting + reading round-trips
    eid = s.register_prediction(hypothesis_id="h1", run_id="r1", statement="H*",
                                prediction=_PRED, source="frontier_sim")
    assert isinstance(eid, int)
    assert s.get_hypothesis_event(eid)["event_type"] == "prediction_registered"
    s.close()


def test_prediction_first_is_enforced(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="prediction"):
        s.record_execution(hypothesis_id="h2", experiment_ref={"spec_sha": "abc"})  # no prediction
    s.register_prediction(hypothesis_id="h2", run_id="r", statement="H", prediction=_PRED)
    ex = s.record_execution(hypothesis_id="h2", experiment_ref={"spec_sha": "abc"})  # now allowed
    assert s.get_hypothesis_event(ex)["event_type"] == "experiment_executed"
    s.close()


def test_adjudication_anchors_to_exact_execution_event(tmp_path):
    s = _store(tmp_path)
    s.register_prediction(hypothesis_id="h3", run_id="r", statement="H", prediction=_PRED)
    ex = s.record_execution(hypothesis_id="h3", experiment_ref={"spec_sha": "s"})
    # cannot adjudicate against a non-execution event id (e.g. a prediction event) or another hyp
    with pytest.raises(ValueError, match="experiment_executed"):
        s.record_adjudication(hypothesis_id="h3", experiment_event_id=999999,
                              verdict="supported", measured={})
    adj = s.record_adjudication(hypothesis_id="h3", experiment_event_id=ex,
                                verdict="falsified", measured={"diff": 5})
    ev = s.get_hypothesis_event(adj)
    assert ev["event_type"] == "adjudication_recorded" and ev["ref_event_id"] == ex  # anchored
    s.close()


def test_get_hypothesis_aggregates_events(tmp_path):
    s = _store(tmp_path)
    s.register_prediction(hypothesis_id="h4", run_id="r", statement="prefix thrash",
                          prediction=_PRED)
    ex = s.record_execution(hypothesis_id="h4", experiment_ref={"spec_sha": "s"})
    s.record_adjudication(hypothesis_id="h4", experiment_event_id=ex, verdict="supported",
                          measured={"value_a": 200, "value_b": 260})
    view = s.get_hypothesis("h4")
    assert view["statement"] == "prefix thrash" and view["verdict"] == "supported"
    assert view["execution_event_ids"] == [ex] and view["measured"]["value_a"] == 200
    assert s.get_hypothesis("nope") is None
    s.close()


def test_falsified_is_first_class_in_list_and_research(tmp_path):
    s = _store(tmp_path)
    s.register_prediction(hypothesis_id="hf", run_id="r", statement="H", prediction=_PRED)
    ex = s.record_execution(hypothesis_id="hf", experiment_ref={"spec_sha": "s"})
    adj = s.record_adjudication(hypothesis_id="hf", experiment_event_id=ex, verdict="falsified",
                                measured={})
    assert [v["hypothesis_id"] for v in s.list_hypotheses(verdict="falsified")] == ["hf"]
    s.close()
    # research gather() surfaces the hypothesis + a verifiable citation to its event
    g = gather("scheduling", db=str(tmp_path / "h.db"))
    assert any(h["hypothesis_id"] == "hf" for h in g["hypotheses"])
    assert f"hypothesis:{adj}" in g["citations"]


def test_verify_citations_accepts_hypothesis_events(tmp_path):
    s = _store(tmp_path)
    eid = s.register_prediction(hypothesis_id="hc", run_id="r", statement="H", prediction=_PRED)
    res = verify_citations(s, [f"hypothesis:{eid}", "hypothesis:999999", "hypothesis:bad"])
    assert res["real"] == [f"hypothesis:{eid}"]
    assert set(res["fabricated"]) == {"hypothesis:999999", "hypothesis:bad"}
    s.close()


def test_events_are_immutable_cited_id_stays_stable(tmp_path):
    s = _store(tmp_path)
    eid = s.register_prediction(hypothesis_id="hi", run_id="r", statement="orig", prediction=_PRED)
    first = s.get_hypothesis_event(eid)
    # appending more events for the same hypothesis never alters the cited event
    s.record_execution(hypothesis_id="hi", experiment_ref={"spec_sha": "s"})
    assert s.get_hypothesis_event(eid) == first
    s.close()


def test_schema_triggers_block_raw_update_and_delete(tmp_path):
    # Codex R3: append-only must be enforced at the SCHEMA level, not just by helpers. Raw SQL
    # UPDATE/DELETE (via the same connection that exposes Store.query) must be rejected by triggers,
    # and the cited event row must remain byte-for-byte unchanged.
    s = _store(tmp_path)
    eid = s.register_prediction(hypothesis_id="ht", run_id="r", statement="orig", prediction=_PRED)
    before = s.get_hypothesis_event(eid)
    with pytest.raises(sqlite3.Error):
        s._conn.execute("UPDATE hypothesis_events SET payload='{}' WHERE event_id=?", (eid,))
    with pytest.raises(sqlite3.Error):
        s._conn.execute("DELETE FROM hypothesis_events WHERE event_id=?", (eid,))
    assert s.get_hypothesis_event(eid) == before                    # untouched
    s.close()


def test_one_prediction_per_hypothesis_and_verdict_is_validated(tmp_path):
    s = _store(tmp_path)
    s.register_prediction(hypothesis_id="hp", run_id="r", statement="H", prediction=_PRED)
    with pytest.raises(ValueError, match="already has a prediction"):
        s.register_prediction(hypothesis_id="hp", run_id="r", statement="H2", prediction=_PRED)
    ex = s.record_execution(hypothesis_id="hp", experiment_ref={"spec_sha": "s"})
    with pytest.raises(ValueError, match="verdict"):
        s.record_adjudication(hypothesis_id="hp", experiment_event_id=ex, verdict="winner",
                              measured={})
    # a valid verdict binds to its execution event (chain coherence) in the aggregate view
    s.record_adjudication(hypothesis_id="hp", experiment_event_id=ex, verdict="supported",
                          measured={})
    assert s.get_hypothesis("hp")["adjudicated_execution_event_id"] == ex
    s.close()
