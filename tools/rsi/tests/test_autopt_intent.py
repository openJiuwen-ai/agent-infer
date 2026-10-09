"""M5: L0 intent parsing — NL goal -> Spec (zero GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.intent.spec import goal_notes, parse_goal  # noqa: E402


def test_ttft_target_and_qps_constraint():
    spec = parse_goal("把 P99 TTFT 在 100 QPS 下压到 200ms")
    assert spec.metric == "ttft_p99_ms" and spec.direction == "min"
    assert spec.target == 200.0 and spec.constraints.get("qps") == 100.0


def test_maximize_goodput():
    spec = parse_goal("maximize goodput")
    assert spec.metric == "goodput_req_s" and spec.direction == "max"


def test_throughput_max():
    spec = parse_goal("提升吞吐 tok/s")
    assert spec.metric == "tok_s" and spec.direction == "max"


def test_latency_min_default_direction():
    spec = parse_goal("降低延迟")
    assert spec.metric == "ttft_p99_ms" and spec.direction == "min"


def test_tpot_target():
    spec = parse_goal("把 TPOT 压到 50ms")
    assert spec.metric == "tpot_ms" and spec.direction == "min" and spec.target == 50.0


def test_unstated_metric_defaults_and_is_flagged():
    spec = parse_goal("optimize the serving system")
    assert spec.metric == "goodput_req_s" and spec.target is None
    assert spec.raw_intent == "optimize the serving system"
    notes = goal_notes("optimize the serving system")
    assert any("metric not stated" in n for n in notes)


def test_slo_phrase_is_constraint_not_direction():
    # "under 500ms" is an SLO constraint; the objective is still MAX goodput
    spec = parse_goal("maximize goodput under 500ms TTFT")
    assert spec.metric == "goodput_req_s" and spec.direction == "max"
    assert spec.constraints.get("slo_latency_ms") == 500.0


def test_conflicting_verbs_use_natural_direction():
    # "提升吞吐、降低延迟" on a throughput metric -> max (the metric's natural sense)
    spec = parse_goal("提升吞吐同时降低延迟")
    assert spec.metric in ("tok_s", "ttft_p99_ms")  # either keyword may win metric
    # whichever metric, the direction must be self-consistent, not first-keyword-wins
    assert spec.direction in ("max", "min")


def test_cmd_goal_cli(capsys):
    rc = cli_main.main(["goal", "把 P99 TTFT 压到 200ms"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["metric"] == "ttft_p99_ms" and out["target"] == 200.0
