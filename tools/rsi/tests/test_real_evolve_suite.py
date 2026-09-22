from __future__ import annotations

import json

import pytest

from vllm_evolve.bench.config import (
    CANDIDATE,
    STRONG_BASELINE,
    BenchConfig,
    same_caliber,
)
from vllm_evolve.engine.real_evolve_suite import (
    SCENARIOS,
    _bind_formal_gpu_selection,
    _config,
    evaluate_suite,
    load_frozen_bench_config,
    metric_summary,
    render_report,
)


def _eval(
    values,
    *,
    effective=True,
    completed=10,
    cv=0.01,
    mechanism_applicable=None,
):
    return {
        "source": "real_vllm",
        "outcome_class": "eval_result",
        "primary_metric": "goodput_req_s",
        "marker_verified": effective,
        "effective": effective,
        "mechanism_applicable": mechanism_applicable,
        "workload_validity": {
            "required": True,
            "valid": True,
            "verdict": "valid_saturated_real_vllm",
        },
        "aggregate_metrics": {"median": sorted(values)[1], "cv": cv},
        "raw_per_seed_metrics": [
            {
                "seed": seed,
                "primary_value": value,
                "metrics": {
                    "num_requests": completed,
                    "num_completed": completed,
                    "num_failed": 0,
                    "request_throughput_req_s": value + 1,
                    "output_throughput_tok_s": value * 10,
                    "ttft_ms": {"p50": 10, "p95": 20, "p99": 30},
                    "tpot_ms": {"p50": 1, "p95": 2, "p99": 3},
                    "e2e_ms": {"p50": 20, "p95": 40, "p99": 60},
                },
            }
            for seed, value in enumerate(values)
        ],
    }


def test_metric_summary_includes_full_latency_and_completion_metrics():
    summary = metric_summary(_eval([10, 11, 12]))
    assert summary["completed"] == 10
    assert summary["completion_rate"] == 1.0
    assert summary["error_rate"] == 0.0
    assert summary["ttft_ms"] == {"p50": 10, "p95": 20, "p99": 30}
    assert summary["tpot_ms"]["p99"] == 3
    assert summary["e2e_ms"]["p95"] == 40


def test_acceptance_requires_real_effective_stable_multi_scenario_gain_and_quality():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([104, 105, 106]),
        }
        for scenario in SCENARIOS
    }
    result = evaluate_suite(pairs, quality_ok=True)
    assert result["accepted"] is True
    assert result["positive_scenarios"] == 3

    pairs["burstgpt_severe_pressure"]["candidate"]["effective"] = False
    assert evaluate_suite(pairs, quality_ok=True)["accepted"] is False
    pairs["burstgpt_severe_pressure"]["candidate"]["effective"] = True
    assert evaluate_suite(pairs, quality_ok=False)["accepted"] is False


def test_inapplicable_scenario_needs_no_fake_reorder():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([104, 105, 106]),
        }
        for scenario in SCENARIOS
    }
    inactive = pairs["burstgpt_saturated"]["candidate"]
    inactive["effective"] = False
    inactive["mechanism_applicable"] = False
    assert evaluate_suite(pairs, quality_ok=True)["accepted"] is True

    inactive["mechanism_applicable"] = True
    assert evaluate_suite(pairs, quality_ok=True)["accepted"] is False


def test_real_mechanism_control_is_a_required_acceptance_gate():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([110, 110, 110]),
        }
        for scenario in SCENARIOS
    }
    controls = {
        scenario: _eval([106, 106, 106])
        for scenario in SCENARIOS
    }
    result = evaluate_suite(
        pairs,
        quality_ok=True,
        controls=controls,
        require_mechanism_control=True,
    )
    assert result["accepted"] is True
    assert result["mechanism_evidence"]["supported"] is True

    controls["burstgpt_high_pressure"] = _eval([111, 111, 111])
    controls["burstgpt_severe_pressure"] = _eval([111, 111, 111])
    result = evaluate_suite(
        pairs,
        quality_ok=True,
        controls=controls,
        require_mechanism_control=True,
    )
    assert result["accepted"] is False
    assert result["mechanism_ok"] is False


def test_high_variance_inactive_control_is_valid_execution_not_stable_evidence():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([110, 110, 110]),
        }
        for scenario in SCENARIOS
    }
    controls = {}
    for scenario in SCENARIOS:
        control = _eval(
            [106, 106, 106],
            effective=False,
            mechanism_applicable=False,
        )
        control["marker_verified"] = True
        controls[scenario] = control
    controls["burstgpt_saturated"]["outcome_class"] = "high_variance_inconclusive"
    controls["burstgpt_saturated"]["aggregate_metrics"]["cv"] = 0.2

    result = evaluate_suite(
        pairs,
        quality_ok=True,
        controls=controls,
        require_mechanism_control=True,
    )

    assert result["accepted"] is True
    burst = result["mechanism_evidence"]["rows"][0]
    assert burst["execution_ok"] is True


def test_paired_acceptance_aligns_by_seed_not_row_position():
    pairs = {
        scenario: {
            "baseline": _eval([100, 105, 110], effective=False),
            "candidate": _eval([104, 109.2, 114.4]),
        }
        for scenario in SCENARIOS
    }
    for pair in pairs.values():
        pair["candidate"]["raw_per_seed_metrics"].reverse()

    result = evaluate_suite(pairs, quality_ok=True)

    assert result["accepted"] is True
    assert result["median_gain_pct"] == pytest.approx(4.0)
    assert result["rows"][0]["comparison"]["seeds"] == [0, 1, 2]
    assert result["rows"][0]["comparison"]["ci_low_pct"] == pytest.approx(4.0)


def test_paired_acceptance_fails_closed_on_missing_or_duplicate_seed():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([104, 105, 106]),
        }
        for scenario in SCENARIOS
    }
    malformed = pairs["burstgpt_high_pressure"]["candidate"]
    malformed["raw_per_seed_metrics"][2]["seed"] = 1

    result = evaluate_suite(pairs, quality_ok=True)

    assert result["accepted"] is False
    row = result["rows"][1]
    assert row["paired_gate_ok"] is False
    assert "duplicate seed" in row["comparison"]["error"]


def test_paired_point_gain_without_ci_low_does_not_count_as_positive():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([100, 104, 108]),
        }
        for scenario in SCENARIOS
    }

    result = evaluate_suite(pairs, quality_ok=True)

    assert result["median_gain_pct"] == pytest.approx(4.0)
    assert result["accepted"] is False
    assert result["positive_scenarios"] == 0
    assert result["rows"][0]["comparison"]["point_gain_pct"] == pytest.approx(4.0)
    assert result["rows"][0]["comparison"]["ci_low_pct"] == pytest.approx(0.0)


def _frozen_config() -> BenchConfig:
    config = BenchConfig()
    config.engine.model = "/models/qwen32b"
    config.engine.tensor_parallel_size = 2
    config.engine.gpu_memory_utilization = 0.86
    config.engine.max_model_len = 4096
    config.engine.max_num_seqs = 150
    config.engine.max_num_batched_tokens = 256
    config.engine.enforce_eager = True
    config.environment.gpus = "0,1"
    config.runner.runner_kind = STRONG_BASELINE
    config.runner.remote = "gpu-host"
    config.statistical.primary_metric = "output_throughput_tok_s"
    config.statistical.seed_tiers = [0, 1, 2]
    config.workload.regime = "evolve_real"
    config.workload.workload_spec = {
        "validity": {
            "required": True,
            "plateau_reference": {
                "base_offered_qps": 100.0,
                "higher_offered_qps": 120.0,
                "base_achieved_throughput": 1000.0,
                "higher_achieved_throughput": 1030.0,
            },
        },
    }
    return config


def _formal_workload(scenario: str, *, duration_s: float = 121.0) -> dict:
    return {
        "source_kind": "official_burstgpt",
        "official_release": "v2.0",
        "split": "heldout",
        "scenario": scenario,
        "payload_sha256": "a" * 64,
        "requests": [
            {
                "request_id": "r0",
                "num_prompt_tokens": 100,
                "num_output_tokens": 100,
            },
        ],
        "replay": {
            "target_qps": 100.0,
            "measured_requests": 512,
            "nominal_arrival_span_s": duration_s,
            "arrival_scaling_only": True,
        },
    }


def test_formal_config_clones_frozen_engine_and_derives_replay_fields(tmp_path):
    frozen = _frozen_config()
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen.to_dict()), encoding="utf-8")
    loaded, provenance = load_frozen_bench_config(frozen_path)
    workload_path = tmp_path / "burstgpt_saturated.json"
    workload = _formal_workload("burstgpt_saturated")
    workload_path.write_text(json.dumps(workload), encoding="utf-8")

    baseline = _config(
        frozen=loaded,
        runner_kind=STRONG_BASELINE,
        policy_path=None,
        workload_path=workload_path,
        workload=workload,
        scenario="burstgpt_saturated",
        port=9000,
        max_seeds=3,
        artifact_root=tmp_path / "artifacts",
    )
    candidate = _config(
        frozen=loaded,
        runner_kind=CANDIDATE,
        policy_path="/tmp/candidate.py",
        workload_path=workload_path,
        workload=workload,
        scenario="burstgpt_saturated",
        port=9001,
        max_seeds=3,
        artifact_root=tmp_path / "artifacts",
    )

    assert baseline.engine.gpu_memory_utilization == 0.86
    assert baseline.engine.tensor_parallel_size == 2
    assert baseline.engine.max_num_batched_tokens == 256
    assert baseline.workload.n_requests == 512
    assert baseline.workload.concurrency == 512
    assert baseline.workload.arrival_rate_qps == 100.0
    assert baseline.workload.workload_spec["validity"]["offered_qps"] == 100.0
    assert baseline.statistical.max_seeds == 3
    assert provenance["bench_config_sha256"]
    assert same_caliber(baseline, candidate) == (True, [])


def test_formal_config_accepts_short_arrival_span_but_keeps_it_in_provenance(
    tmp_path,
):
    frozen = _frozen_config()
    path = tmp_path / "workload.json"
    workload = _formal_workload("burstgpt_saturated", duration_s=2.046)
    path.write_text(json.dumps(workload), encoding="utf-8")

    config = _config(
        frozen=frozen,
        runner_kind=STRONG_BASELINE,
        policy_path=None,
        workload_path=path,
        workload=workload,
        scenario="burstgpt_saturated",
        port=9000,
        max_seeds=3,
        artifact_root=tmp_path / "artifacts",
    )

    assert config.workload.workload_spec["nominal_arrival_span_s"] == 2.046


def test_formal_config_rejects_non_positive_arrival_span(tmp_path):
    frozen = _frozen_config()
    path = tmp_path / "workload.json"
    workload = _formal_workload("burstgpt_saturated", duration_s=0.0)
    path.write_text(json.dumps(workload), encoding="utf-8")

    with pytest.raises(ValueError, match="arrival span must be positive"):
        _config(
            frozen=frozen,
            runner_kind=STRONG_BASELINE,
            policy_path=None,
            workload_path=path,
            workload=workload,
            scenario="burstgpt_saturated",
            port=9000,
            max_seeds=3,
            artifact_root=tmp_path / "artifacts",
        )


def test_frozen_formal_config_accepts_explicit_dual_replica_mode(tmp_path):
    frozen = _frozen_config()
    frozen.engine.tensor_parallel_size = 1
    frozen.workload.workload_spec["parallel_mode"] = "dual_replica"
    frozen_path = tmp_path / "dual.json"
    frozen_path.write_text(json.dumps(frozen.to_dict()), encoding="utf-8")

    loaded, _ = load_frozen_bench_config(frozen_path)

    assert loaded.engine.tensor_parallel_size == 1
    assert loaded.workload.workload_spec["parallel_mode"] == "dual_replica"


def test_frozen_formal_config_rejects_ambiguous_two_visible_gpu_tp1(tmp_path):
    frozen = _frozen_config()
    frozen.engine.tensor_parallel_size = 1
    frozen.workload.workload_spec.pop("parallel_mode", None)
    frozen_path = tmp_path / "ambiguous.json"
    frozen_path.write_text(json.dumps(frozen.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="must select tp2 or dual_replica"):
        load_frozen_bench_config(frozen_path)


def test_frozen_formal_config_allows_deferred_auto_gpu_binding(tmp_path):
    frozen = _frozen_config()
    frozen.environment.gpus = "auto"
    frozen_path = tmp_path / "auto.json"
    frozen_path.write_text(json.dumps(frozen.to_dict()), encoding="utf-8")

    loaded, provenance = load_frozen_bench_config(frozen_path)

    assert loaded.environment.gpus == "auto"
    assert provenance["bench_config_sha256"]


def test_suite_auto_reuses_an_already_frozen_auto_pair():
    frozen = _frozen_config()
    frozen.environment.gpus = "2,3"
    frozen.environment.gpu_selection_mode = "auto"
    frozen.environment.gpu_selection_evidence = {
        "mode": "auto",
        "resolved": "2,3",
        "selected_gpus": ["2", "3"],
        "selected_uuids": ["GPU-2", "GPU-3"],
    }

    _bind_formal_gpu_selection(frozen, "auto")

    assert frozen.environment.gpus == "2,3"
    assert frozen.environment.gpu_selection_mode == "auto"
    assert frozen.environment.gpu_selection_evidence["resolved"] == "2,3"


def test_report_contains_full_metric_and_gpu_evidence():
    pairs = {
        scenario: {
            "baseline": _eval([100, 100, 100], effective=False),
            "candidate": _eval([110, 110, 110]),
        }
        for scenario in SCENARIOS
    }
    acceptance = evaluate_suite(pairs, quality_ok=True)
    run = {
        "local_run_dir": "/tmp/evidence",
        "status": {
            "gpu_summary": {
                "per_gpu": {
                    "3": {
                        "max_memory_used_mib": 1234,
                        "max_utilization_gpu_pct": 88,
                    }
                }
            }
        },
    }
    report = render_report({
        "acceptance": acceptance,
        "model": "model",
        "remote": "box",
        "gpus": "3",
        "policy_path": "candidate.py",
        "control_path": "control.py",
        "pairs": {
            scenario: {"baseline": run, "candidate": run}
            for scenario in SCENARIOS
        },
    })
    assert "TTFT p95" in report
    assert "GPU 3: 1234 MiB, 88%" in report
