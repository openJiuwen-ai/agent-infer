"""M2: L2 select-targets — bottleneck -> ranked interventions (zero GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.core import schemas as S  # noqa: E402
from vllm_evolve.engine.targets import select_targets  # noqa: E402


def _diag(bottleneck, status="confirmed"):
    return S.Diagnosis(bottleneck=bottleneck, status=status)


def _targets(plan):
    return [c["target"] for c in plan.candidates]


def test_kv_capacity_config_first_code_last():
    plan = select_targets(_diag(S.KV_CAPACITY))
    t = _targets(plan)
    assert "config:gpu_memory_utilization" in t
    # cheap high-leverage config ranks above the expensive code target
    assert t.index("config:gpu_memory_utilization") < t.index("code:schedule_batch")
    assert plan.candidates[0]["type"] == "config"


def test_scheduling_queue_offers_code_target():
    # the one regime where evolving schedule_batch is justified
    plan = select_targets(_diag(S.SCHEDULING_QUEUE))
    assert "code:schedule_batch" in _targets(plan)


def test_each_bottleneck_has_ranked_candidates():
    for b in (S.COMPUTE, S.KV_CAPACITY, S.MEM_BANDWIDTH, S.PREFILL,
              S.SCHEDULING_QUEUE, S.UNDER_SATURATED, S.COMM):
        plan = select_targets(_diag(b))
        assert plan.candidates, f"{b} has no candidates"
        scores = [
            {"high": 3, "med": 2, "low_med": 1.5, "low": 1}[c["leverage"]]
            - {"trivial": 0, "low": 0.3, "med": 0.6, "high": 1.0}[c["cost"]]
            for c in plan.candidates
        ]
        assert scores == sorted(scores, reverse=True), f"{b} not rank-sorted"


def test_unknown_bottleneck_asks_for_more_signal():
    plan = select_targets(_diag(S.UNKNOWN, status="unknown"))
    assert _targets(plan) == ["profile:add_signals"]
    assert "profile" in plan.note


def test_unknown_status_emits_no_concrete_targets():
    # even with a localized bottleneck label, status=unknown must NOT emit
    # concrete optimization candidates — only collect more signal
    plan = select_targets(_diag(S.KV_CAPACITY, status="unknown"))
    assert _targets(plan) == ["profile:add_signals"]


def test_no_op_enable_targets_removed():
    # chunked-prefill is default-on in vLLM 0.14 -> don't recommend "enabling" it
    for b in (S.PREFILL, S.KV_CAPACITY):
        assert "config:enable_chunked_prefill" not in _targets(select_targets(_diag(b)))


def test_suspected_diagnosis_prepends_validation_step():
    plan = select_targets(_diag(S.KV_CAPACITY, status="suspected"))
    assert plan.candidates[0]["target"] == "profile:add_signals"
    assert plan.note and "suspected" in plan.note


def test_cmd_targets_cli(tmp_path, capsys):
    f = tmp_path / "diag.json"
    f.write_text(json.dumps(_diag(S.KV_CAPACITY).to_dict()), encoding="utf-8")
    rc = cli_main.main(["targets", str(f)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True and out["bottleneck"] == S.KV_CAPACITY
    assert out["candidates"][0]["type"] == "config"


def test_cmd_targets_missing_file(tmp_path, capsys):
    rc = cli_main.main(["targets", str(tmp_path / "nope.json")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "diagnosis_not_found"
