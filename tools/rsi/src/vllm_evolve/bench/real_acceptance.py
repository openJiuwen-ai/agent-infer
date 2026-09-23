"""Frozen, deterministic judgment for the three-scenario real-vLLM suite."""
from __future__ import annotations

import statistics

from vllm_evolve.bench.compare import (
    coefficient_of_variation,
    paired_ci,
    paired_relative_improvement,
)

SCENARIOS = (
    "burstgpt_saturated",
    "burstgpt_high_pressure",
    "burstgpt_severe_pressure",
)
FORMAL_SEEDS = (0, 1, 2)


def _median(values) -> float | None:
    real = [float(value) for value in values if value is not None]
    return statistics.median(real) if real else None


def _nested(value: dict, *path):
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def metric_summary(eval_result: dict) -> dict:
    rows = list(eval_result.get("raw_per_seed_metrics") or [])

    def med(*path):
        return _median([_nested(row, *path) for row in rows])

    completed = med("metrics", "num_completed")
    failed = med("metrics", "num_failed")
    attempted = completed + failed if completed is not None and failed is not None else None
    return {
        "primary_metric": eval_result.get("primary_metric"),
        "primary_median": _nested(eval_result, "aggregate_metrics", "median"),
        "primary_cv": _nested(eval_result, "aggregate_metrics", "cv"),
        "completed": completed,
        "failed": failed,
        "completion_rate": completed / attempted if attempted else None,
        "error_rate": failed / attempted if attempted else None,
        "request_throughput_req_s": med("metrics", "request_throughput_req_s"),
        "output_throughput_tok_s": med("metrics", "output_throughput_tok_s"),
        "ttft_ms": {
            percentile: med("metrics", "ttft_ms", percentile)
            for percentile in ("p50", "p95", "p99")
        },
        "tpot_ms": {
            percentile: med("metrics", "tpot_ms", percentile)
            for percentile in ("p50", "p95", "p99")
        },
        "e2e_ms": {
            percentile: med("metrics", "e2e_ms", percentile)
            for percentile in ("p50", "p95", "p99")
        },
        "outcome_class": eval_result.get("outcome_class"),
        "effective": eval_result.get("effective"),
        "marker_verified": eval_result.get("marker_verified"),
        "plugin_provenance": eval_result.get("plugin_provenance"),
        "wall_time_s": eval_result.get("wall_time_s"),
        "remote_worker_duration_s": eval_result.get("remote_worker_duration_s"),
        "gpu_summary": eval_result.get("remote_resource_evidence"),
    }


def _seed_values(
    eval_result: dict,
    *,
    required_seeds: tuple[int, ...] = FORMAL_SEEDS,
) -> tuple[dict[int, float] | None, str | None]:
    """Return primary values by seed, rejecting missing or ambiguous identities."""
    values: dict[int, float] = {}
    for row in eval_result.get("raw_per_seed_metrics") or []:
        if "seed" not in row:
            return None, "raw_per_seed_metrics row is missing seed"
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            return None, f"invalid seed value {row.get('seed')!r}"
        if seed in values:
            return None, f"duplicate seed {seed}"
        value = row.get("primary_value")
        if not isinstance(value, (int, float)):
            return None, f"seed {seed} is missing a numeric primary_value"
        values[seed] = float(value)
    observed = tuple(sorted(values))
    expected = tuple(sorted(required_seeds))
    if observed != expected:
        return None, f"paired seeds must be exactly {list(expected)}, got {list(observed)}"
    return values, None


def _paired_comparison(
    baseline: dict,
    candidate: dict,
    *,
    threshold_pct: float,
    max_cv: float,
    n_boot: int,
    bootstrap_seed: int,
) -> dict:
    metric = str(candidate.get("primary_metric") or "")
    baseline_metric = str(baseline.get("primary_metric") or "")
    base_by_seed, base_error = _seed_values(baseline)
    cand_by_seed, cand_error = _seed_values(candidate)
    error = base_error or cand_error
    if baseline_metric != metric:
        error = error or (
            f"primary metric mismatch: baseline={baseline_metric!r}, candidate={metric!r}"
        )
    if not metric:
        error = error or "primary metric is missing"
    if error:
        return {
            "metric": metric or baseline_metric,
            "higher_is_better": True,
            "seeds": list(FORMAL_SEEDS),
            "point_gain_pct": None,
            "rel_improvement_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
            "candidate_cv": None,
            "threshold_pct": threshold_pct,
            "gate_ok": False,
            "error": error,
        }
    baseline_values = [base_by_seed[seed] for seed in FORMAL_SEEDS]
    candidate_values = [cand_by_seed[seed] for seed in FORMAL_SEEDS]
    point = paired_relative_improvement(baseline_values, candidate_values)
    lo, hi = paired_ci(
        baseline_values,
        candidate_values,
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    candidate_cv = coefficient_of_variation(candidate_values)
    gate_ok = point >= threshold_pct and lo >= threshold_pct and candidate_cv <= max_cv
    return {
        "metric": metric,
        "higher_is_better": True,
        "seeds": list(FORMAL_SEEDS),
        "baseline_values_by_seed": {
            str(seed): base_by_seed[seed] for seed in FORMAL_SEEDS
        },
        "candidate_values_by_seed": {
            str(seed): cand_by_seed[seed] for seed in FORMAL_SEEDS
        },
        "baseline_median": statistics.median(baseline_values),
        "candidate_median": statistics.median(candidate_values),
        "point_gain_pct": point,
        "rel_improvement_pct": point,
        "ci_low_pct": lo,
        "ci_high_pct": hi,
        "candidate_cv": candidate_cv,
        "threshold_pct": threshold_pct,
        "gate_ok": gate_ok,
        "error": None,
    }


def _completion_exact(eval_result: dict) -> bool:
    rows = list(eval_result.get("raw_per_seed_metrics") or [])
    return bool(rows) and all(
        isinstance((row.get("metrics") or {}).get("num_requests"), (int, float))
        and int((row.get("metrics") or {}).get("num_requests")) > 0
        and int((row.get("metrics") or {}).get("num_completed") or 0)
        == int((row.get("metrics") or {}).get("num_requests"))
        and int((row.get("metrics") or {}).get("num_failed") or 0) == 0
        for row in rows
    )


def _workload_valid(eval_result: dict) -> bool:
    validity = eval_result.get("workload_validity") or {}
    return (
        validity.get("required") is True
        and validity.get("valid") is True
        and validity.get("verdict") == "valid_saturated_real_vllm"
    )


def candidate_execution_ok(eval_result: dict) -> bool:
    """Return whether a candidate has valid real-vLLM execution evidence."""
    return (
        eval_result.get("source") == "real_vllm"
        and _workload_valid(eval_result)
        and eval_result.get("outcome_class")
        in {"eval_result", "high_variance_inconclusive"}
        and eval_result.get("marker_verified") is True
        and (
            eval_result.get("effective") is True
            or eval_result.get("mechanism_applicable") is False
        )
    )


def evaluate_suite(
    pairs: dict[str, dict],
    *,
    quality_ok: bool,
    controls: dict[str, dict] | None = None,
    require_mechanism_control: bool = False,
    min_median_gain_pct: float = 3.0,
    min_positive_scenarios: int = 2,
    regression_floor_pct: float = -2.0,
    max_cv: float = 0.10,
    min_mechanism_gain_pct: float = 3.0,
    mechanism_name: str = "candidate mechanism vs supplied control",
    n_boot: int = 2000,
    bootstrap_seed: int = 0,
) -> dict:
    """Recompute every performance and mechanism gate from raw paired evidence."""
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        pair = pairs[scenario]
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        comparison = _paired_comparison(
            baseline,
            candidate,
            threshold_pct=min_median_gain_pct,
            max_cv=max_cv,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed + scenario_index,
        )
        gain = comparison["point_gain_pct"]
        rows.append({
            "scenario": scenario,
            "gain_pct": gain,
            "comparison": comparison,
            "baseline": metric_summary(baseline),
            "candidate": metric_summary(candidate),
            "completion_ok": _completion_exact(baseline) and _completion_exact(candidate),
            "execution_ok": _workload_valid(baseline) and candidate_execution_ok(candidate),
            "paired_gate_ok": comparison["gate_ok"],
            "mechanism_applicable": candidate.get("mechanism_applicable"),
        })
    gains = [float(row["gain_pct"]) for row in rows if row["gain_pct"] is not None]
    complete_paired_evidence = len(gains) == len(SCENARIOS)
    median_gain = statistics.median(gains) if complete_paired_evidence else None
    minimum_gain = min(gains) if complete_paired_evidence else None
    positive = sum(row["paired_gate_ok"] for row in rows)
    performance_ok = (
        complete_paired_evidence
        and median_gain >= min_median_gain_pct
        and positive >= min_positive_scenarios
        and minimum_gain >= regression_floor_pct
        and all(row["completion_ok"] and row["execution_ok"] for row in rows)
    )

    mechanism = {
        "required": require_mechanism_control,
        "supported": None,
        "mechanism": mechanism_name,
        "median_gain_pct": None,
        "positive_scenarios": 0,
        "rows": [],
    }
    if controls is not None:
        control_rows = []
        for scenario_index, scenario in enumerate(SCENARIOS):
            candidate = pairs[scenario]["candidate"]
            control = controls[scenario]
            comparison = _paired_comparison(
                control,
                candidate,
                threshold_pct=min_mechanism_gain_pct,
                max_cv=max_cv,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed + len(SCENARIOS) + scenario_index,
            )
            control_rows.append({
                "scenario": scenario,
                "gain_pct": comparison["point_gain_pct"],
                "comparison": comparison,
                "control": metric_summary(control),
                "candidate": metric_summary(candidate),
                "completion_ok": _completion_exact(candidate) and _completion_exact(control),
                "execution_ok": all(
                    candidate_execution_ok(item) for item in (candidate, control)
                )
                and _workload_valid(pairs[scenario]["baseline"]),
                "paired_gate_ok": comparison["gate_ok"],
                "candidate_active": candidate.get("effective") is True,
            })
        control_gains = [
            float(row["gain_pct"])
            for row in control_rows
            if row["gain_pct"] is not None
        ]
        complete_control_evidence = len(control_gains) == len(SCENARIOS)
        mechanism.update({
            "supported": (
                complete_control_evidence
                and statistics.median(control_gains) >= min_mechanism_gain_pct
                and sum(row["paired_gate_ok"] for row in control_rows)
                >= min_positive_scenarios
                and min(control_gains) >= regression_floor_pct
                and all(
                    row["completion_ok"] and row["execution_ok"]
                    for row in control_rows
                )
                and sum(row["candidate_active"] for row in control_rows)
                >= min_positive_scenarios
            ),
            "median_gain_pct": (
                statistics.median(control_gains) if complete_control_evidence else None
            ),
            "positive_scenarios": sum(
                row["paired_gate_ok"] for row in control_rows
            ),
            "rows": control_rows,
        })
    mechanism_ok = not require_mechanism_control or mechanism["supported"] is True
    return {
        "accepted": performance_ok and quality_ok and mechanism_ok,
        "performance_ok": performance_ok,
        "quality_ok": quality_ok,
        "mechanism_ok": mechanism_ok,
        "mechanism_evidence": mechanism,
        "median_gain_pct": median_gain,
        "positive_scenarios": positive,
        "minimum_gain_pct": minimum_gain,
        "thresholds": {
            "min_median_gain_pct": min_median_gain_pct,
            "min_positive_scenarios": min_positive_scenarios,
            "regression_floor_pct": regression_floor_pct,
            "max_cv": max_cv,
            "min_mechanism_gain_pct": min_mechanism_gain_pct,
            "paired_seeds": list(FORMAL_SEEDS),
            "paired_bootstrap_samples": n_boot,
        },
        "rows": rows,
    }
