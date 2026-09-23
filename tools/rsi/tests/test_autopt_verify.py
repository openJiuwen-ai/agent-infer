"""M4: L4 verify-gain — bootstrap + holdout + bottleneck-shift (zero GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.core.schemas import Spec  # noqa: E402
from vllm_evolve.core.verify import extract_values, verify_gain  # noqa: E402

_SPEC = Spec(metric="goodput_req_s", direction="max")


def _bc(runner_kind="strong_baseline"):
    # same-caliber bench_config provenance the verify-gain adoption gate now requires
    from vllm_evolve.bench.config import build_bench_config
    return build_bench_config(runner_kind=runner_kind, model="facebook/opt-125m").to_dict()


# LOCK D: a real-vLLM clean run's metadata (verify-gain refuses anything else)
_REAL = {"source": "real_vllm", "outcome_class": "eval_result"}


def _hold(values, runner="strong_baseline"):
    # a real-vLLM holdout eval_result (verify-gain now applies real-source + same_caliber to it)
    return json.dumps({**_REAL, "values": values, "bench_config": _bc(runner)})


def test_better_and_stable_is_adopted():
    # candidate still beats baseline at the holdout config -> generalizes
    v = verify_gain([5.0, 5.1, 4.9], [7.0, 7.1, 6.9], _SPEC,
                    holdout_baseline_values=[5.0, 5.0, 4.9],
                    holdout_candidate_values=[6.8, 6.9, 7.0])
    assert v.verdict == "better" and v.holdout_ok is True and v.overfit is False
    assert "adopt" in v.next_action


def test_better_but_holdout_collapses_is_overfit():
    # at the holdout config the candidate falls below baseline -> gain didn't generalize
    v = verify_gain([5.0, 5.0, 5.0], [7.0, 7.0, 7.0], _SPEC,
                    holdout_baseline_values=[5.0, 5.0, 5.0],
                    holdout_candidate_values=[3.0, 3.0])
    assert v.overfit is True and v.verdict == "worse"
    assert "overfit" in v.next_action


def test_inconclusive_goes_back_to_l2():
    v = verify_gain([5.0, 5.0, 5.0], [5.02, 5.0, 5.01], _SPEC)
    assert v.verdict == "inconclusive" and "L2" in v.next_action


def test_high_variance_flagged():
    v = verify_gain([5.0, 5.0, 5.0], [3.0, 7.0, 5.0], _SPEC)
    assert v.verdict == "high_variance_inconclusive" and "noisy" in v.next_action


def test_unverified_candidate_rejected():
    v = verify_gain([5.0, 5.0], [99.0, 99.0], _SPEC, candidate_verified=False)
    assert v.verdict == "rejected_unverified" and "marker" in v.next_action


def test_better_without_holdout_needs_holdout():
    # positive verdict requires BOTH bootstrap AND holdout — no holdout != adopt
    v = verify_gain([5.0, 5.1, 4.9], [7.0, 7.1, 6.9], _SPEC)
    assert v.verdict == "needs_holdout" and "holdout" in v.next_action
    assert v.holdout_ok is None and v.overfit is False


def test_bottleneck_shift_loops_to_l1():
    v = verify_gain([5.0, 5.1, 4.9], [7.0, 7.1, 6.9], _SPEC,
                    holdout_baseline_values=[5.0, 5.0, 4.9],
                    holdout_candidate_values=[6.8, 6.9, 7.0],
                    baseline_bottleneck="compute", candidate_bottleneck="kv_capacity")
    assert v.verdict == "better" and v.bottleneck_shifted is True and "L1" in v.next_action


def test_min_direction_holdout():
    spec = Spec(metric="ttft_p99_ms", direction="min")
    # lower TTFT is better; candidate stays below baseline at the holdout config
    v = verify_gain([200.0, 200.0, 200.0], [120.0, 120.0, 120.0], spec,
                    holdout_baseline_values=[200.0, 205.0],
                    holdout_candidate_values=[130.0, 125.0])
    assert v.verdict == "better" and v.holdout_ok is True


def test_extract_values_from_eval_result_and_values():
    er = {"raw_per_seed_metrics": [{"metrics": {"goodput_req_s": 5.0}},
                                   {"metrics": {"goodput_req_s": 5.2}}]}
    assert extract_values(er, "goodput_req_s") == [5.0, 5.2]
    assert extract_values({"values": [1, 2, 3]}, "goodput_req_s") == [1.0, 2.0, 3.0]


def test_extract_values_from_primary_value():
    # headline metric in primary_value (only when primary_metric matches)
    er = {"primary_metric": "goodput_req_s",
          "raw_per_seed_metrics": [{"primary_value": 6.1}, {"primary_value": 6.3}]}
    assert extract_values(er, "goodput_req_s") == [6.1, 6.3]
    # wrong metric must NOT pick up primary_value
    assert extract_values(er, "tok_s") == []


def test_cmd_verify_gain_cli(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    hbase = tmp_path / "hbase.json"
    hcand = tmp_path / "hcand.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.1, 4.9],
                                "bottleneck": "compute", "bench_config": _bc()}), encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [7.0, 7.1, 6.9],
                                "bottleneck": "kv_capacity", "marker_verified": True,
                                "effective": True,    # invoked + reordered + no fallback
                                "quality_ok": True, "bench_config": _bc("candidate")}),
                    encoding="utf-8")
    hbase.write_text(_hold([5.0, 5.0, 4.9]), encoding="utf-8")
    hcand.write_text(_hold([6.8, 6.9, 7.0], "candidate"), encoding="utf-8")
    rc = cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s",
                      "--holdout-baseline", str(hbase), "--holdout-candidate", str(hcand)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True and out["verify_gain"]["verdict"] == "better"
    assert out["verify_gain"]["bottleneck_shifted"] is True
    # +40% gain, holdout +37%, effective + quality certified -> strict gate adopts
    assert out["strict_accept"]["accepted"] is True and out["adopt"] is True


def test_cmd_verify_gain_strict_gate_rejects_without_quality(tmp_path, capsys):
    # same big gain but NO quality evidence -> strict gate must refuse to adopt
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    hb = tmp_path / "hb.json"
    hc = tmp_path / "hc.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.1, 4.9],
                                "bench_config": _bc()}), encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [7.0, 7.1, 6.9],
                                "marker_verified": True, "bench_config": _bc("candidate")}),
                    encoding="utf-8")
    hb.write_text(_hold([5.0, 5.0, 4.9]), encoding="utf-8")
    hc.write_text(_hold([6.8, 6.9, 7.0], "candidate"), encoding="utf-8")
    rc = cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s",
                      "--holdout-baseline", str(hb), "--holdout-candidate", str(hc)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["adopt"] is False
    assert any("quality NOT certified" in r for r in out["strict_accept"]["reasons"])


def test_cmd_verify_gain_unverified_by_default(tmp_path, capsys):
    # a handwritten candidate without marker_verified must NOT pass the gate
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.0, 5.0],
                                "bench_config": _bc()}), encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [99.0, 99.0, 99.0],
                                "bench_config": _bc("candidate")}), encoding="utf-8")
    rc = cli_main.main(["verify-gain", str(base), str(cand)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["verify_gain"]["verdict"] == "rejected_unverified"
    assert out["adopt"] is False   # unverified + no holdout -> strict gate also refuses


def test_cmd_verify_gain_rejects_non_real_holdout(tmp_path, capsys):
    # Codex review P1: a synthetic/local_smoke holdout must NOT satisfy holdout_ok or enable adopt.
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    hb = tmp_path / "hb.json"
    hc = tmp_path / "hc.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.1, 4.9], "bench_config": _bc()}),
                    encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [7.0, 7.1, 6.9], "marker_verified": True,
                                "bench_config": _bc("candidate")}), encoding="utf-8")
    hb.write_text(_hold([5.0, 5.0, 4.9]), encoding="utf-8")
    hc.write_text(json.dumps({"source": "local_smoke",
                              "outcome_class": "local_smoke_nonqualifying",
                              "values": [9.9, 9.9, 9.9], "bench_config": _bc("candidate")}),
                  encoding="utf-8")
    rc = cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s",
                      "--holdout-baseline", str(hb), "--holdout-candidate", str(hc)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "non_real_source_blocked"
    assert out["adopt"] is False and out.get("which") == "holdout"


def test_cmd_verify_gain_invoked_but_not_effective_does_not_adopt(tmp_path, capsys):
    # Codex review P1: marker_verified (plugin merely LOADED) is NOT effectiveness. An invoked-only
    # candidate (no explicit effective=True) must NOT adopt even if gain + holdout + quality pass.
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    hb = tmp_path / "hb.json"
    hc = tmp_path / "hc.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.1, 4.9], "bench_config": _bc()}),
                    encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [7.0, 7.1, 6.9], "marker_verified": True,
                                "quality_ok": True, "bench_config": _bc("candidate")}),
                    encoding="utf-8")     # NOTE: marker_verified but NO effective
    hb.write_text(_hold([5.0, 5.0, 4.9]), encoding="utf-8")
    hc.write_text(_hold([6.8, 6.9, 7.0], "candidate"), encoding="utf-8")
    rc = cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s",
                      "--holdout-baseline", str(hb), "--holdout-candidate", str(hc)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["adopt"] is False
    assert any("effectiveness NOT proven" in r for r in out["strict_accept"]["reasons"])


def test_cmd_verify_gain_exempt_knob_allows_config_ab(tmp_path, capsys):
    # Codex review P2: a config-lever A/B (baseline+candidate differ only by a searched knob) must
    # be verifiable when the knob is declared with --exempt-knob; without it -> caliber mismatch.
    from vllm_evolve.bench.config import build_bench_config
    bc_b = build_bench_config(runner_kind="strong_baseline", model="m", max_num_seqs=128).to_dict()
    bc_c = build_bench_config(runner_kind="strong_baseline", model="m", max_num_seqs=256).to_dict()
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    base.write_text(json.dumps({**_REAL, "values": [5.0, 5.1, 4.9], "bench_config": bc_b}),
                    encoding="utf-8")
    cand.write_text(json.dumps({**_REAL, "values": [7.0, 7.1, 6.9], "bench_config": bc_c}),
                    encoding="utf-8")
    # WITHOUT --exempt-knob: rejected on the searched-knob diff before any gain math
    rc = cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "same_caliber_mismatch"
    # WITH --exempt-knob max_num_seqs: caliber passes -> reaches the gain verdict
    cli_main.main(["verify-gain", str(base), str(cand), "--metric", "goodput_req_s",
                 "--exempt-knob", "max_num_seqs"])
    out2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out2.get("outcome_class") != "same_caliber_mismatch" and "verify_gain" in out2
