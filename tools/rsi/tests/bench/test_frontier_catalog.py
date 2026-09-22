"""R2/AC1: the P2 measurement catalog exposes Frontier's full output, and the single MetricExpr
evaluator is honest — a column Frontier did not emit yields None + a missing list, never an invented
number. Pure/offline (fixture files on disk; no Frontier, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.frontier_catalog import (  # noqa: E402
    MetricExpr,
    evaluate,
    load_catalog,
    metrics_dir,
)

# real-shaped: latencies in MILLISECONDS; two tenants (session_id 1, 2), tenant 2 has the worse tail
_SYSTEM = """{
  "throughput_metrics": {"requests_per_second": 4.0, "tokens_per_second": 900.0,
                         "total_duration_seconds": 1.0},
  "request_e2e_time_statistics": {"p50": 120.0, "unit": "ms"}
}"""
_CSV = (
    "Request Id,ttft,request_e2e_time,request_session_id\n"
    "0,30.0,120.0,1\n"
    "1,40.0,130.0,1\n"
    "2,200.0,400.0,2\n"
    "3,260.0,500.0,2\n"
)
_LEDGER = (
    '{"batch_id": 0, "stage_start_ts": 0.0, "stage_end_ts": 1.0, "request_ids": [0, 1]}\n'
)


def _fixture(tmp_path) -> Path:
    d = tmp_path / "meta" / "online_serving" / "run0"      # exercise the nested-dir locator
    d.mkdir(parents=True)
    (d / "system_metrics.json").write_text(_SYSTEM, encoding="utf-8")
    (d / "request_metrics.csv").write_text(_CSV, encoding="utf-8")
    (d / "frontier_stage_batch_ledger.jsonl").write_text(_LEDGER, encoding="utf-8")
    (tmp_path / "ve_policy_marker.json").write_text(
        json.dumps({"scheduler": "ve_policy", "invocations": 9, "fallbacks": 0}), encoding="utf-8")
    return tmp_path


def test_load_catalog_reads_full_output(tmp_path):
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    assert cat.dir == metrics_dir(str(tmp_path), "run0")          # nested locator agrees
    assert cat.columns == ["Request Id", "ttft", "request_e2e_time", "request_session_id"]
    assert len(cat.rows) == 4 and len(cat.ledger) == 1
    assert cat.marker["invocations"] == 9                          # marker found in out_dir root
    assert cat.system["throughput_metrics"]["requests_per_second"] == 4.0


def test_evaluate_basic_aggregates(tmp_path):
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    assert evaluate({"column": "ttft", "agg": "max"}, cat).value == 260.0
    assert evaluate({"column": "ttft", "agg": "min"}, cat).value == 30.0
    assert evaluate({"column": "ttft", "agg": "count"}, cat).value == 4.0
    mean = evaluate({"column": "ttft", "agg": "mean"}, cat)
    assert abs(mean.value - (30 + 40 + 200 + 260) / 4) < 1e-9 and mean.ok
    # p50 over [30,40,200,260] linear-interpolated = 120.0
    assert evaluate({"column": "ttft", "agg": "p50"}, cat).value == 120.0
    # rate = count / total_duration_seconds(=1.0)
    assert evaluate({"column": "ttft", "agg": "rate"}, cat).value == 4.0


def test_evaluate_grouped_worst_per_tenant(tmp_path):
    # the H* metric: worst per-session p99 TTFT (group_by session, reduce max across tenants)
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    expr = {"column": "ttft", "agg": "p99", "group_by": "request_session_id",
            "group_reduce": "max", "units": "ms"}
    r = evaluate(expr, cat)
    # tenant 1 p99 ~= 40, tenant 2 p99 ~= 260 -> max across tenants ~= 260; n = 2 groups
    assert 255.0 <= r.value <= 260.0 and r.n == 2 and r.ok


def test_missing_column_is_explicit_none_never_invented(tmp_path):
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    r = evaluate({"column": "tpot", "agg": "p99"}, cat)            # Frontier didn't emit tpot here
    assert r.value is None and r.missing == ["tpot"] and not r.ok
    # a missing group_by column is also reported, never invented
    r2 = evaluate({"column": "ttft", "agg": "p99", "group_by": "cohort"}, cat)
    assert r2.value is None and "cohort" in r2.missing


def test_metric_expr_typed_api_is_shared_between_p1_and_p3(tmp_path):
    # AC1: a concrete MetricExpr is THE shared representation; a typed object and the equivalent
    # mapping evaluate identically, so P1 (experiment) and P3 (adjudication) cannot drift.
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    typed = MetricExpr(column="ttft", agg="p99", group_by="request_session_id",
                       group_reduce="max", units="ms")
    as_dict = {"column": "ttft", "agg": "p99", "group_by": "request_session_id",
               "group_reduce": "max", "units": "ms"}
    assert evaluate(typed, cat).value == evaluate(as_dict, cat).value      # same value both ways
    assert MetricExpr.from_obj(as_dict) == typed                            # normalization is exact
    assert typed.to_dict() == as_dict
    assert typed.is_valid()
    assert not MetricExpr(column="", agg="p99").is_valid()                  # bad column rejected
    assert not MetricExpr(column="ttft", agg="nope").is_valid()            # bad agg rejected
    assert not MetricExpr(column="ttft", agg="p99", group_reduce="median").is_valid()


def test_metrics_dir_selects_by_run_id_not_lexicographic(tmp_path):
    # AC1 run-selection: two sibling runs under one out_dir — run_id must pick its own run, not the
    # lexicographically-first one (else adjudication could cite a different experiment's numbers).
    for rid, ttft in (("run_a", "11.0"), ("run_b", "99.0")):
        d = tmp_path / "meta" / "online_serving" / rid
        d.mkdir(parents=True)
        (d / "system_metrics.json").write_text("{}", encoding="utf-8")
        (d / "request_metrics.csv").write_text(f"Request Id,ttft\n0,{ttft}\n", encoding="utf-8")
    assert metrics_dir(str(tmp_path), "run_b").name == "run_b"
    cat_b = load_catalog(str(tmp_path), "run_b")
    assert evaluate({"column": "ttft", "agg": "max"}, cat_b).value == 99.0   # run_b data, not run_a
    # ambiguous: a run_id that matches neither -> empty catalog (explicit missing), never a guess
    cat_x = load_catalog(str(tmp_path), "run_x")
    assert cat_x.rows == [] and evaluate({"column": "ttft", "agg": "max"}, cat_x).value is None


def test_bad_expr_and_empty_data(tmp_path):
    cat = load_catalog(str(_fixture(tmp_path)), "run0")
    assert evaluate({"column": "ttft", "agg": "nonsense"}, cat).value is None
    # a column present but no numeric rows -> None + missing (count is the documented exception)
    empty = tmp_path / "m" / "online_serving" / "r"
    empty.mkdir(parents=True)
    (empty / "request_metrics.csv").write_text("Request Id,ttft\n", encoding="utf-8")
    cat2 = load_catalog(str(tmp_path / "m"), "r")
    assert evaluate({"column": "ttft", "agg": "p99"}, cat2).value is None
    assert evaluate({"column": "ttft", "agg": "count"}, cat2).value == 0.0   # real count of zero
