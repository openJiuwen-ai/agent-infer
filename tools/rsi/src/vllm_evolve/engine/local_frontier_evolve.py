"""Local-only Frontier evolution over real BurstGPT fragments and stress traces.

This module is the reproducible Codex-facing path for algorithm search without a GPU. It runs the
real Frontier subprocess through the existing ``frontier_catalog_eval`` seam; fixtures and
``local_smoke`` are never accepted here. Search sees train + validation scenarios. Held-out
scenarios are materialized and evaluated only after the winner is frozen.

The result is deliberately a ``sim_winner`` proposal. It preserves frontier_sim's quarantine and
never calls the remote backend or the real-vLLM adoption gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vllm_evolve.bench.datasets import burstgpt
from vllm_evolve.bench.frontier_catalog import FrontierCatalog
from vllm_evolve.core.schemas import Profile, Spec
from vllm_evolve.engine.evolve_loop import run_evolution
from vllm_evolve.engine.evolve_target import _build_policy, _parse_struct, template_author_fn
from vllm_evolve.engine.experiment import Arm, ExperimentSpec
from vllm_evolve.engine.experiment_chain import frontier_catalog_eval
from vllm_evolve.engine.workload_synth import (
    TraceRanges,
    WorkloadSpec,
    heterogeneous_bursty_multi_tenant,
    write_trace_csv,
)

BURSTGPT_SOURCE_URL = "https://github.com/HPMLL/BurstGPT"
BURSTGPT_SOURCE_COMMIT = "d895a53bb7b8ec137d0d2fe203b335835a78c10a"
BASELINE_RECIPES = {
    "fcfs": None,
    "sjf": {"order": "sjf"},
    "ljf": {"order": "ljf"},
    "lifo": {"order": "lifo"},
}
_SPLITS = ("train", "validation", "test")


@dataclass
class Scenario:
    name: str
    split: str
    source_kind: str
    rows: list
    slo_ttft_ms: float
    max_num_seqs: int
    enable_prefix_caching: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SeedMeasurement:
    seed: int
    goodput_req_s: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    throughput_req_s: float | None
    completed: int
    marker: dict | None
    marker_ok: bool
    metrics_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyScenarioResult:
    policy: str
    policy_sha256: str
    scenario: str
    split: str
    source_kind: str
    per_seed: list
    median_goodput_req_s: float | None
    median_ttft_p50_ms: float | None
    median_ttft_p99_ms: float | None
    median_throughput_req_s: float | None
    completed_min: int
    marker_ok: bool
    invocations: int
    fallbacks: int
    defers: int
    forced: int
    preemptions: int = 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["per_seed"] = [
            item if isinstance(item, dict) else item.to_dict() for item in self.per_seed
        ]
        return out


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rows_sha(rows: list[dict]) -> str:
    return _sha_text(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(xs):
        return xs[-1]
    return xs[lo] + (xs[lo + 1] - xs[lo]) * frac


def _median(values: list[float | None]) -> float | None:
    real = [float(value) for value in values if value is not None]
    return statistics.median(real) if real else None


def _marker_ok(marker: dict | None, policy_sha: str) -> bool:
    if not isinstance(marker, dict):
        return False
    invocations = int(marker.get("invocations") or 0)
    fallbacks = int(marker.get("fallbacks") or 0)
    return marker.get("policy_sha256") == policy_sha and invocations > 0 and fallbacks < invocations


def _measure_catalog(
    catalog: FrontierCatalog,
    *,
    seed: int,
    slo_ttft_ms: float,
    policy_sha: str,
) -> SeedMeasurement:
    ttfts = [
        value for value in (_num(row.get("ttft")) for row in catalog.rows) if value is not None
    ]
    completed = int(
        (catalog.system.get("simulation_metadata") or {}).get("completed_requests", len(ttfts))
    )
    throughput = _num((catalog.system.get("throughput_metrics") or {}).get("requests_per_second"))
    goodput = None
    if throughput is not None and ttfts:
        met = sum(1 for value in ttfts if value <= slo_ttft_ms)
        goodput = throughput * (met / len(ttfts))
    marker = catalog.marker
    marker_ok = True if not policy_sha else _marker_ok(marker, policy_sha)
    return SeedMeasurement(
        seed=seed,
        goodput_req_s=goodput,
        ttft_p50_ms=_percentile(ttfts, 50),
        ttft_p99_ms=_percentile(ttfts, 99),
        throughput_req_s=throughput,
        completed=completed,
        marker=marker,
        marker_ok=marker_ok,
        metrics_dir=str(catalog.dir),
    )


def _trace_ranges(rows: list[dict]) -> TraceRanges:
    max_prefill = max(int(row["num_prefill_tokens"]) for row in rows)
    max_decode = max(int(row["num_decode_tokens"]) for row in rows)
    max_arrival = max(float(row["arrived_at"]) for row in rows)
    return TraceRanges(
        max_rows=max(5000, len(rows)),
        prefill_range=(1, max(8192, max_prefill)),
        decode_range=(1, max(4096, max_decode)),
        max_duration_s=max(600.0, max_arrival + 1.0),
    )


def evaluate_policy(
    *,
    policy_name: str,
    policy_path: str | None,
    scenario: Scenario,
    seeds: list[int],
    out_dir: str | Path,
    dummy_execution_time_ms: float = 0.001,
) -> PolicyScenarioResult:
    """Run one policy/scenario over paired seeds using the real Frontier subprocess."""
    root = Path(out_dir).resolve() / scenario.split / scenario.name / policy_name
    root.mkdir(parents=True, exist_ok=True)
    trace_path = write_trace_csv(scenario.rows, root / "trace.csv")
    policy_sha = ""
    if policy_path:
        policy_sha = _sha_text(Path(policy_path).read_text(encoding="utf-8"))
    arm = Arm(policy_name, policy_path=policy_path)
    spec = ExperimentSpec(
        workload=WorkloadSpec(inline_rows=scenario.rows),
        arms=[arm],
        metrics=[{"column": "ttft", "agg": "p99"}],
        seeds=seeds,
        knobs={
            "max_num_seqs": scenario.max_num_seqs,
            "max_num_batched_tokens": 32768,
            "trace_max_tokens": max(
                int(row["num_prefill_tokens"]) + int(row["num_decode_tokens"])
                for row in scenario.rows
            ),
            "dummy_execution_time_ms": dummy_execution_time_ms,
            "enable_prefix_caching": scenario.enable_prefix_caching,
        },
        budget=len(seeds),
        ranges=_trace_ranges(scenario.rows),
    )
    per_seed = []
    for seed in seeds:
        catalog = frontier_catalog_eval(
            arm, seed, spec, str(trace_path), str(root / f"seed_{seed}")
        )
        per_seed.append(
            _measure_catalog(
                catalog,
                seed=seed,
                slo_ttft_ms=scenario.slo_ttft_ms,
                policy_sha=policy_sha,
            )
        )
    markers = [item.marker for item in per_seed if item.marker]
    return PolicyScenarioResult(
        policy=policy_name,
        policy_sha256=policy_sha,
        scenario=scenario.name,
        split=scenario.split,
        source_kind=scenario.source_kind,
        per_seed=per_seed,
        median_goodput_req_s=_median([item.goodput_req_s for item in per_seed]),
        median_ttft_p50_ms=_median([item.ttft_p50_ms for item in per_seed]),
        median_ttft_p99_ms=_median([item.ttft_p99_ms for item in per_seed]),
        median_throughput_req_s=_median([item.throughput_req_s for item in per_seed]),
        completed_min=min((item.completed for item in per_seed), default=0),
        marker_ok=all(item.marker_ok for item in per_seed),
        invocations=sum(int(marker.get("invocations") or 0) for marker in markers),
        fallbacks=sum(int(marker.get("fallbacks") or 0) for marker in markers),
        defers=sum(int(marker.get("defers") or 0) for marker in markers),
        forced=sum(int(marker.get("forced") or 0) for marker in markers),
        preemptions=sum(int(marker.get("preemptions") or 0) for marker in markers),
    )


def build_scenarios(
    burstgpt_path: str | Path,
    *,
    fragment_size: int = 64,
    slo_ttft_ms: float = 200.0,
    max_num_seqs: int = 4,
    stress_prefix_blocks: int = 2,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[list[Scenario], dict]:
    """Build disjoint train/validation/test BurstGPT fragments plus labeled synthetic stress."""
    if not splits or any(split not in _SPLITS for split in splits):
        raise ValueError(f"splits must be a non-empty subset of {_SPLITS}")
    if stress_prefix_blocks < 0:
        raise ValueError("stress_prefix_blocks must be non-negative")
    fragments = {}
    main_names = tuple(split for split in splits if split in ("train", "validation"))
    if main_names:
        fragments.update(
            burstgpt.extract_bursty_fragments(
                burstgpt_path,
                fragment_size=fragment_size,
                split_names=main_names,
                band_indices=tuple({"train": 0, "validation": 1}[name] for name in main_names),
                total_bands=3,
            )
        )
    tail_names = []
    tail_indices = []
    if "validation" in splits:
        tail_names.append("validation_tail")
        tail_indices.append(4)
    if "test" in splits:
        tail_names.append("test")
        tail_indices.append(5)
    if tail_names:
        fragments.update(
            burstgpt.extract_bursty_fragments(
                burstgpt_path,
                fragment_size=fragment_size,
                split_names=tuple(tail_names),
                band_indices=tuple(tail_indices),
                total_bands=6,
            )
        )
    target_qps = {
        "train": 18.0,
        "validation": 21.0,
        "validation_tail": 22.5,
        "test": 24.0,
    }
    scenarios: list[Scenario] = []
    fragment_meta = {}
    for fragment_name, dataset in fragments.items():
        rows, metadata = burstgpt.to_frontier_rows(
            dataset, target_qps=target_qps[fragment_name], session_id=0
        )
        metadata["frontier_rows_sha256"] = _rows_sha(rows)
        fragment_meta[fragment_name] = metadata
        split = "validation" if fragment_name == "validation_tail" else fragment_name
        scenarios.append(
            Scenario(
                name=f"burstgpt_{fragment_name}",
                split=split,
                source_kind="official_burstgpt_fragment",
                rows=rows,
                slo_ttft_ms=slo_ttft_ms,
                max_num_seqs=max_num_seqs,
                metadata=metadata,
            )
        )

    stress_specs = [
        (
            "stress_train",
            "train",
            3,
            4,
            0.030,
            ((192, 1536), (576, 48), (896, 80), (1280, 1536)),
        ),
        (
            "stress_validation",
            "validation",
            4,
            4,
            0.020,
            ((256, 2048), (768, 64), (1024, 96), (1536, 2048)),
        ),
        (
            "stress_test_moderate",
            "test",
            4,
            5,
            0.015,
            ((320, 1792), (704, 48), (1152, 128), (1664, 1792)),
        ),
        (
            "stress_test_severe",
            "test",
            5,
            5,
            0.010,
            ((224, 2560), (832, 80), (1280, 160), (1920, 2560)),
        ),
    ]
    for name, split, tenants, burst_size, gap, profiles in stress_specs:
        if split not in splits:
            continue
        rows = heterogeneous_bursty_multi_tenant(
            n_tenants=tenants,
            bursts_per_tenant=3,
            burst_size=burst_size,
            burst_period_s=0.75,
            intra_burst_gap_s=gap,
            length_profiles=profiles,
            prefix_blocks_per_tenant=stress_prefix_blocks,
        )
        scenarios.append(
            Scenario(
                name=name,
                split=split,
                source_kind="synthetic_multi_tenant_prefix_stress",
                rows=rows,
                slo_ttft_ms=slo_ttft_ms,
                max_num_seqs=max_num_seqs,
                enable_prefix_caching=True,
                metadata={
                    "synthetic": True,
                    "n_tenants": tenants,
                    "burst_size": burst_size,
                    "intra_burst_gap_s": gap,
                    "length_profiles": [list(profile) for profile in profiles],
                    "prefix_blocks_per_tenant": stress_prefix_blocks,
                    "frontier_rows_sha256": _rows_sha(rows),
                },
            )
        )
    manifest = {
        "source_url": BURSTGPT_SOURCE_URL,
        "source_commit": BURSTGPT_SOURCE_COMMIT,
        "source_path": str(Path(burstgpt_path).resolve()),
        "source_sha256": burstgpt.source_sha256(burstgpt_path),
        "fragment_size": fragment_size,
        "split_method": (
            "densest positive-duration contiguous windows: train=first third, "
            "validation=middle third plus first half of final third (regression), "
            "test=last sixth; no shuffle and no overlap"
        ),
        "materialized_splits": list(splits),
        "fragments": fragment_meta,
        "honesty": (
            "BurstGPT fragments use one neutral session and empty block hashes; synthetic "
            "multi-tenant/prefix workloads are separately labeled."
        ),
    }
    return scenarios, manifest


def _write_baseline_policies(root: Path) -> dict[str, str | None]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}
    for name, recipe in BASELINE_RECIPES.items():
        if recipe is None:
            paths[name] = None
            continue
        source = _build_policy(f"BASELINE-{name.upper()}", **recipe)
        path = root / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        paths[name] = str(path)
    current_seed = Path(__file__).resolve().parents[3] / "targets" / "scheduling" / "seed.py"
    if not current_seed.is_file():
        raise RuntimeError(f"current scheduling seed missing: {current_seed}")
    paths["current_seed"] = str(current_seed)
    return paths


def _scenario_baselines(results: list[PolicyScenarioResult]) -> dict[str, dict]:
    by_scenario: dict[str, list[PolicyScenarioResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario, []).append(result)
    best = {}
    for scenario, rows in by_scenario.items():
        scored = [row for row in rows if row.median_goodput_req_s is not None]
        if not scored:
            raise RuntimeError(f"no scoreable baseline for {scenario}")
        winner = max(scored, key=lambda row: row.median_goodput_req_s)
        best[scenario] = {
            "policy": winner.policy,
            "median_goodput_req_s": winner.median_goodput_req_s,
            "completed_min": winner.completed_min,
        }
    return best


def _objective(
    candidate: list[PolicyScenarioResult],
    strongest: dict[str, dict],
) -> float | None:
    improvements = []
    for result in candidate:
        baseline = strongest[result.scenario]["median_goodput_req_s"]
        if (
            result.median_goodput_req_s is None
            or not result.marker_ok
            or not baseline
            or result.completed_min < strongest[result.scenario]["completed_min"]
        ):
            return None
        improvements.append((result.median_goodput_req_s / baseline - 1.0) * 100.0)
    return statistics.median(improvements) if improvements else None


def _acceptance(
    candidate: list[PolicyScenarioResult],
    strongest: dict[str, dict],
) -> dict:
    rows = []
    for result in candidate:
        baseline = strongest[result.scenario]
        base_score = baseline["median_goodput_req_s"]
        gain = None
        if result.median_goodput_req_s is not None and base_score:
            gain = (result.median_goodput_req_s / base_score - 1.0) * 100.0
        rows.append(
            {
                "scenario": result.scenario,
                "source_kind": result.source_kind,
                "baseline_policy": baseline["policy"],
                "baseline_goodput_req_s": base_score,
                "candidate_goodput_req_s": result.median_goodput_req_s,
                "gain_pct": gain,
                "completed_baseline": baseline["completed_min"],
                "completed_candidate": result.completed_min,
                "marker_ok": result.marker_ok,
                "invocations": result.invocations,
                "fallbacks": result.fallbacks,
                "defers": result.defers,
                "forced": result.forced,
            }
        )
    gains = [row["gain_pct"] for row in rows if row["gain_pct"] is not None]
    aggregate = statistics.median(gains) if gains else None
    positive = sum(1 for gain in gains if gain > 0)
    no_large_regression = bool(gains) and min(gains) >= -2.0
    completion_ok = all(row["completed_candidate"] >= row["completed_baseline"] for row in rows)
    marker_ok = all(row["marker_ok"] for row in rows)
    passed = (
        aggregate is not None
        and aggregate >= 3.0
        and positive >= 2
        and len(rows) >= 3
        and no_large_regression
        and completion_ok
        and marker_ok
    )
    return {
        "passed": passed,
        "aggregate_median_gain_pct": aggregate,
        "positive_scenarios": positive,
        "scenario_count": len(rows),
        "no_regression_below_minus_2_pct": no_large_regression,
        "completion_ok": completion_ok,
        "marker_ok": marker_ok,
        "rows": rows,
    }


def _ablation_evidence(
    candidate: list[PolicyScenarioResult],
    control_results: list[PolicyScenarioResult],
    *,
    mechanism: str = "declared candidate contribution vs supplied control",
) -> dict:
    controls = {result.scenario: result for result in control_results}
    rows = []
    for result in candidate:
        control = controls[result.scenario]
        gain = None
        if result.median_goodput_req_s is not None and control.median_goodput_req_s:
            gain = (result.median_goodput_req_s / control.median_goodput_req_s - 1.0) * 100.0
        rows.append(
            {
                "scenario": result.scenario,
                "candidate_goodput_req_s": result.median_goodput_req_s,
                "control_goodput_req_s": control.median_goodput_req_s,
                "gain_vs_control_pct": gain,
                "candidate_completed": result.completed_min,
                "control_completed": control.completed_min,
                "candidate_marker_ok": result.marker_ok,
                "control_marker_ok": control.marker_ok,
                "control_fallbacks": control.fallbacks,
                "control_invocations": control.invocations,
            }
        )
    gains = [
        row["gain_vs_control_pct"]
        for row in rows
        if row["gain_vs_control_pct"] is not None
    ]
    valid_execution = all(
        row["candidate_marker_ok"]
        and row["control_marker_ok"]
        and row["control_invocations"] > 0
        and row["control_fallbacks"] < row["control_invocations"]
        and row["candidate_completed"] == row["control_completed"]
        for row in rows
    )
    return {
        "mechanism": mechanism,
        "median_gain_pct": statistics.median(gains) if gains else None,
        "positive_scenarios": sum(1 for gain in gains if gain > 0),
        "valid_execution": valid_execution,
        "supported": (
            len(gains) == len(rows)
            and len(rows) >= 3
            and valid_execution
            and statistics.median(gains) >= 3.0
            and sum(1 for gain in gains if gain > 0) >= 2
        )
        if gains
        else False,
        "rows": rows,
    }


def _multi_control_ablation_evidence(
    candidate: list[PolicyScenarioResult],
    controls: dict[str, list[PolicyScenarioResult]],
    *,
    control_mechanisms: dict[str, str] | None = None,
) -> dict:
    """Require every frozen parent/control policy to support the selected child mechanism."""
    labels = control_mechanisms or {}
    by_control = {
        name: _ablation_evidence(
            candidate,
            rows,
            mechanism=labels.get(name, f"declared contribution vs {name}"),
        )
        for name, rows in controls.items()
    }
    return {
        "mechanism": "manifest-declared parent contribution controls",
        "control_count": len(by_control),
        "supported": bool(by_control) and all(row["supported"] for row in by_control.values()),
        "valid_execution": bool(by_control)
        and all(row["valid_execution"] for row in by_control.values()),
        "controls": by_control,
    }


def _template_seed_ablation(source: str) -> tuple[str, str, str] | None:
    """Return a one-field structural control for a template seed.

    The control changes exactly one typed recipe dimension and labels that exact
    difference.  This prevents a generic FCFS/SJF policy from masquerading as a
    preemption/decode-awareness ablation.
    """
    recipe = _parse_struct(source)
    control = dict(recipe)
    if recipe["gate"] != "none":
        field, old, new = "gate", recipe["gate"], "none"
        control[field] = new
    elif recipe.get("dispersion_guard"):
        field, old, new = "dispersion_guard", "1", "0"
        control[field] = False
    elif recipe["switch"] and recipe["pressure_order"] != recipe["order"]:
        field, old, new = "switch", "1", "0"
        control[field] = False
    elif recipe["order"] != "fcfs":
        field, old, new = "order", recipe["order"], "fcfs"
        control[field] = new
    else:
        return None
    name = f"seed_remove_{field}"
    label = f"template structural ablation: {field}={old} -> {new}; all other fields frozen"
    rendered = _build_policy(f"SEED-ABLATE-{field.upper()}", **control)
    return name, rendered, label


def _render_research_evolution_report(result: dict, research) -> str:
    acceptance = result["acceptance"]
    lines = [
        "# vLLM-Evolve Auto Research + Frontier report",
        "",
        "## Verdict",
        "",
        (f"- Local acceptance thresholds: **{'PASS' if acceptance['passed'] else 'FAIL'}**"),
        "- Evidence class: `frontier_sim / simulator_nonqualifying`",
        "- This report is a search proposal; it is not a production vLLM gain or keep.",
        f"- Author kind: `{result['author_kind']}`",
        f"- Research snapshot: `{research.snapshot_id}` / `{research.snapshot_hash}`",
        f"- Research outcome: `{research.outcome_class}`",
        "",
        "## Expert brief",
        "",
        f"- Sources: {len(research.sources)}",
        f"- Mechanism cards: {len(research.mechanism_cards)}",
        f"- Internal lessons: {len(research.internal_lessons)}",
        "",
        "| Mechanism | Status | Structural change |",
        "|---|---|---|",
    ]
    portfolio = {
        tuple(item.get("mechanism_ids") or []): item
        for item in research.recommended_hypothesis_portfolio
    }
    for card in research.mechanism_cards:
        status = portfolio.get((card.mechanism_id,), {}).get("status", "context_only")
        lines.append(
            f"| `{card.mechanism_id}` | {status} | "
            f"{(card.structural_change or card.core_mechanism).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism lineage",
            "",
            "| Candidate | Operator | Parents | Mechanisms | Held-in score | Manifest |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in result["mechanism_lineage"]:
        mechanisms = ", ".join(row["mechanism_ids"]) or "uncited"
        score = row["score"]
        lines.append(
            f"| `{row['sha'][:12]}` | `{row.get('operator', 'legacy')}` | "
            f"`{','.join(parent[:12] for parent in row.get('parent_shas', [])) or '-'}` | "
            f"{mechanisms} | {score if score is not None else 'n/a'} | "
            f"`{row['candidate_manifest_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Held-out acceptance",
            "",
            f"- Aggregate median gain: {acceptance['aggregate_median_gain_pct']}",
            f"- Positive scenarios: {acceptance['positive_scenarios']}/"
            f"{acceptance['scenario_count']}",
            f"- Minimum-regression floor passed: {acceptance['no_regression_below_minus_2_pct']}",
            f"- Completion passed: {acceptance['completion_ok']}",
            f"- Marker passed: {acceptance['marker_ok']}",
            f"- Mechanism ablation supported: {acceptance['ablation_mechanism_supported']}",
            "",
            "| Scenario | Baseline | Candidate | Gain % | Completed | Marker |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in acceptance["rows"]:
        lines.append(
            f"| {row['scenario']} | {row['baseline_policy']} | "
            f"{row['candidate_goodput_req_s']} | {row['gain_pct']} | "
            f"{row['completed_candidate']} | {row['marker_ok']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction artifacts",
            "",
            f"- Research: `{result['research_artifact_dir']}`",
            f"- Winner: `{result['winner_path']}`",
            f"- Winner SHA: `{result['winner_sha256']}`",
            "- Knowledge writeback lesson IDs: "
            f"`{result['knowledge_writeback_lesson_ids']}`",
            "- Raw Frontier commands, trace hashes and metrics are under the run's `raw/` tree.",
            "",
        ]
    )
    return "\n".join(lines)


def run_local_frontier_evolution(
    *,
    burstgpt_path: str | Path,
    out_dir: str | Path,
    seeds: list[int] | None = None,
    fragment_size: int = 64,
    slo_ttft_ms: float = 200.0,
    max_num_seqs: int = 4,
    generations: int = 3,
    population: int = 8,
    max_total_evals: int = 24,
    author_fn=None,
    research_context: dict | None = None,
    research_snapshot: str | Path | None = None,
    research_mode: str = "offline",
    research_refresh: bool = False,
    store=None,
    goal: str = "maximize goodput while preserving completion and TTFT SLO",
    resume: bool = False,
) -> dict:
    """Run baseline search, generational evolution, then a once-only held-out audit."""
    seeds = list(seeds or [0, 1, 2])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("local Frontier acceptance requires at least 3 distinct paired seeds")
    search_seeds = [seeds[0]]
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from vllm_evolve.engine.scheduling_contract import load_scheduling_author_contract
    from vllm_evolve.engine.workflow_state import (
        TERMINAL_STAGES,
        WORKFLOW_STAGES,
        ArtifactRef,
        WorkflowLedger,
        WorkflowStage,
        file_sha256,
    )
    from vllm_evolve.intent.spec import parse_goal

    normalized_goal = parse_goal(goal)
    burstgpt_path = Path(burstgpt_path).resolve()
    scheduling_contract = load_scheduling_author_contract(backend="frontier")
    resolved_author = author_fn or template_author_fn
    author_identity = {
        "kind": getattr(resolved_author, "ve_author_kind", None) or "template",
        "command": list(getattr(resolved_author, "ve_author_command", []) or []),
    }
    workflow_request = {
        "goal": normalized_goal.to_dict(),
        "burstgpt_path": str(burstgpt_path),
        "burstgpt_sha256": file_sha256(burstgpt_path),
        "seeds": seeds,
        "fragment_size": fragment_size,
        "slo_ttft_ms": slo_ttft_ms,
        "max_num_seqs": max_num_seqs,
        "generations": generations,
        "population": population,
        "max_total_evals": max_total_evals,
        "author": author_identity,
        "research_mode": research_mode,
        "research_snapshot": str(Path(research_snapshot).resolve()) if research_snapshot else None,
        "research_refresh": research_refresh,
        "frontier_repo": os.environ.get("VE_FRONTIER_REPO", ""),
        "frontier_python": os.environ.get("VE_FRONTIER_PYTHON", ""),
        "scheduling_contract_sha256": scheduling_contract["contract_sha256"],
        "acceptance": {
            "median_gain_pct": 3.0,
            "positive_scenarios": 2,
            "minimum_scenario_gain_pct": -2.0,
            "completion_loss_allowed": False,
            "marker_and_ablation_required": True,
        },
    }
    state_path = root / "state.json"
    if state_path.is_file():
        if not resume:
            raise ValueError(
                f"workflow state already exists at {state_path}; pass resume=True/--resume"
            )
        workflow = WorkflowLedger.resume(root, workflow_request)
        if workflow.current_stage in TERMINAL_STAGES:
            terminal_dir = root / (
                "sim_winner" if workflow.current_stage == WorkflowStage.SIM_WINNER else "no_winner"
            )
            evidence_path = terminal_dir / "evidence.json"
            if not evidence_path.is_file():
                raise RuntimeError(
                    f"terminal workflow is missing evidence artifact: {evidence_path}"
                )
            return json.loads(evidence_path.read_text(encoding="utf-8"))
    else:
        if resume:
            raise ValueError(f"cannot resume: workflow state does not exist at {state_path}")
        workflow = WorkflowLedger.create(root, workflow_request, run_id=root.name)

    stage_order = {stage: index for index, stage in enumerate(WORKFLOW_STAGES)}

    def _ref(path: Path) -> ArtifactRef:
        return ArtifactRef(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))

    def _advance(stage: WorkflowStage, *paths: Path, diagnostics: dict | None = None):
        current = workflow.current_stage
        if current == stage or stage_order[current] > stage_order[stage]:
            return workflow.state
        return workflow.transition(
            stage,
            artifacts=[_ref(path) for path in paths],
            diagnostics=diagnostics,
        )

    request_path = root / "request.json"
    if not request_path.exists():
        request_path.write_text(json.dumps(workflow_request, indent=2), encoding="utf-8")
    context_path = root / "context.json"
    if not context_path.exists():
        context_path.write_text(
            json.dumps(
                {
                    "author_kind": author_identity["kind"],
                    "scheduling_contract": scheduling_contract,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _advance(WorkflowStage.CONTEXT_FROZEN, request_path, context_path)
    diagnosis_payload = {
        "bottleneck": "scheduling_queue",
        "status": "confirmed",
        "detail": "bursty arrivals plus heterogeneous request lengths",
        "normalized_goal": normalized_goal.to_dict(),
    }
    diagnosis_path = root / "diagnosis.json"
    if not diagnosis_path.exists():
        diagnosis_path.write_text(json.dumps(diagnosis_payload, indent=2), encoding="utf-8")
    _advance(WorkflowStage.DIAGNOSIS_FROZEN, diagnosis_path)
    research_artifact_dir = root / "research"
    if research_snapshot:
        from vllm_evolve.knowledge import (
            load_research_context,
            materialize_research_context,
        )

        compiled_research = load_research_context(research_snapshot)
        research_reused = True
        materialize_research_context(compiled_research, research_artifact_dir)
    elif research_context:
        from vllm_evolve.knowledge import materialize_research_context
        from vllm_evolve.knowledge.schemas import ResearchContext

        compiled_research = ResearchContext.from_dict(research_context)
        research_reused = True
        materialize_research_context(compiled_research, research_artifact_dir)
    else:
        from vllm_evolve.knowledge import ResearchCompiler, gather, offline_compiler

        try:
            internal_evidence = gather(
                "scheduling",
                n=5,
                regime="burstgpt_and_stress",
                metric="heldin_gain_pct",
            )
        except Exception as exc:  # noqa: BLE001 - empty/corrupt local KB is non-fatal context
            internal_evidence = {
                "lessons": [],
                "hypotheses": [],
                "knowledge_status": f"unavailable: {type(exc).__name__}: {exc}",
            }
        compiler = offline_compiler() if research_mode == "offline" else ResearchCompiler()
        research_result = compiler.compile(
            target="scheduling",
            spec={
                **normalized_goal.to_dict(),
                "evaluation_metric": "heldin_gain_pct",
            },
            diagnosis=diagnosis_payload,
            environment={
                "vllm_version": "0.21.0",
                "workloads": ["BurstGPT", "moderate stress", "severe stress"],
                "max_num_seqs": max_num_seqs,
                "slo_ttft_ms": slo_ttft_ms,
                "execution": "local Frontier simulator; real vLLM gate remains separate",
            },
            out_dir=research_artifact_dir,
            internal_evidence=internal_evidence,
            refresh=research_refresh,
        )
        compiled_research = research_result.context
        research_reused = research_result.reused_frozen_snapshot
    frozen_research_path = research_artifact_dir / "research_snapshot.json"
    _advance(
        WorkflowStage.RESEARCH_FROZEN,
        frozen_research_path,
        diagnostics={
            "snapshot_id": compiled_research.snapshot_id,
            "snapshot_hash": compiled_research.snapshot_hash,
            "outcome_class": compiled_research.outcome_class,
        },
    )
    _advance(WorkflowStage.SEARCHING)
    search_scenarios, search_manifest = build_scenarios(
        burstgpt_path,
        fragment_size=fragment_size,
        slo_ttft_ms=slo_ttft_ms,
        max_num_seqs=max_num_seqs,
        splits=("train", "validation"),
    )
    (root / "search_dataset_manifest.json").write_text(
        json.dumps(search_manifest, indent=2), encoding="utf-8"
    )

    baseline_paths = _write_baseline_policies(root / "baselines")
    baseline_search = []
    for name, path in baseline_paths.items():
        for scenario in search_scenarios:
            baseline_search.append(
                evaluate_policy(
                    policy_name=name,
                    policy_path=path,
                    scenario=scenario,
                    seeds=search_seeds,
                    out_dir=root / "raw" / "search_baselines",
                )
            )
    strongest_search = _scenario_baselines(baseline_search)
    (root / "search_baselines.json").write_text(
        json.dumps(
            {
                "strongest": strongest_search,
                "results": [result.to_dict() for result in baseline_search],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    evaluations: dict[str, list[PolicyScenarioResult]] = {}

    def eval_fn(config: dict) -> Profile:
        path = str(config["policy"])
        policy_sha = _sha_text(Path(path).read_text(encoding="utf-8"))
        if policy_sha not in evaluations:
            evaluations[policy_sha] = [
                evaluate_policy(
                    policy_name=policy_sha[:12],
                    policy_path=path,
                    scenario=scenario,
                    seeds=search_seeds,
                    out_dir=root / "raw" / "evolution",
                )
                for scenario in search_scenarios
            ]
        score = _objective(evaluations[policy_sha], strongest_search)
        return Profile(
            config=dict(config),
            metrics={"heldin_gain_pct": score} if score is not None else {},
            per_seed={
                "heldin_gain_pct": [
                    result.median_goodput_req_s
                    for result in evaluations[policy_sha]
                    if result.median_goodput_req_s is not None
                ]
            },
            source="frontier_sim",
            marker_verified=False,
            outcome_class="simulator_nonqualifying",
            evidence=[result.to_dict() for result in evaluations[policy_sha]],
        )

    eval_fn.ve_source = "frontier_sim"
    spec = Spec(
        metric="heldin_gain_pct",
        direction="max",
        constraints={"slo": {"ttft_ms": slo_ttft_ms}},
        workload={"train_validation_scenarios": [item.name for item in search_scenarios]},
        raw_intent=goal,
    )
    from vllm_evolve.engine.scheduling_contract import load_scheduling_author_contract

    scheduling_contract = load_scheduling_author_contract(backend="frontier")
    evolution = run_evolution(
        author_fn or template_author_fn,
        eval_fn,
        spec,
        {
            "profile": "burstgpt_and_stress",
            "max_num_seqs": max_num_seqs,
            "seeds": seeds,
        },
        {
            **diagnosis_payload,
            "evidence_refs": [item.name for item in search_scenarios],
        },
        generations=generations,
        population=population,
        repair_limit=1,
        max_total_evals=max_total_evals,
        variants_dir=str(root / "variants"),
        run_id=root.name,
        store=store,
        research_context=compiled_research.to_dict(),
        skeleton=scheduling_contract["rendered"],
        author_kind=(
            getattr(author_fn, "ve_author_kind", None) if author_fn is not None else "template"
        ),
    )
    if evolution.winner is None:
        no_winner_dir = root / "no_winner"
        no_winner_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "ok": False,
            "verdict": "NO_WINNER",
            "reason": "evolution produced no scoreable Frontier candidate",
            "source": "frontier_sim",
            "outcome_class": "simulator_nonqualifying",
            "remote_executed": False,
            "normalized_goal": normalized_goal.to_dict(),
            "winner_path": None,
            "winner_sha256": None,
            "author_kind": evolution.author_kind,
            "research_snapshot_id": compiled_research.snapshot_id,
            "research_snapshot_hash": compiled_research.snapshot_hash,
            "research_artifact_dir": str(research_artifact_dir),
            "evolution": evolution.to_dict(),
            "workflow_state_path": str(workflow.state_path),
            "workflow_events_path": str(workflow.events_path),
            "doctrine": (
                "No simulator winner was created. Frontier evidence is proposal-only and cannot "
                "enter real-vLLM keep/adoption."
            ),
        }
        evidence_path = no_winner_dir / "evidence.json"
        evidence_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _advance(
            WorkflowStage.NO_WINNER,
            evidence_path,
            diagnostics={"reason": result["reason"]},
        )
        report_path = root / "report.md"
        report_path.write_text(
            "# vLLM-Evolve Frontier report\n\n"
            "- Verdict: **NO_WINNER**\n"
            "- Reason: evolution produced no scoreable Frontier candidate.\n"
            "- Evidence: `frontier_sim / simulator_nonqualifying`.\n",
            encoding="utf-8",
        )
        result["report_path"] = str(report_path)
        result["evidence_path"] = str(evidence_path)
        return result
    winner_source = Path(evolution.winner.source_path).read_text(encoding="utf-8")
    selection_dir = root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "work_variant.py"
    selection_path.write_text(winner_source, encoding="utf-8")
    frozen_sha = _sha_text(winner_source)
    selection_manifest_path = selection_dir / "candidate_manifest.json"
    if evolution.winner.candidate_manifest_path:
        shutil.copyfile(
            evolution.winner.candidate_manifest_path,
            selection_manifest_path,
        )
    selection_record_path = selection_dir / "selection.json"
    selection_record_path.write_text(
        json.dumps(
            {
                "candidate_sha256": frozen_sha,
                "score": evolution.winner.score,
                "generation": evolution.winner.generation,
                "operator": evolution.winner.operator,
                "parent_shas": evolution.winner.parent_shas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selection_artifacts = [selection_path, selection_record_path]
    if selection_manifest_path.is_file():
        selection_artifacts.append(selection_manifest_path)
    _advance(
        WorkflowStage.SELECTION_FROZEN,
        *selection_artifacts,
        diagnostics={"candidate_sha256": frozen_sha},
    )
    winner_freeze_path = selection_dir / "winner.freeze.json"
    winner_freeze_path.write_text(
        json.dumps(
            {
                "source_sha256": frozen_sha,
                "candidate_manifest_path": (
                    str(selection_manifest_path) if selection_manifest_path.is_file() else None
                ),
                "research_snapshot_hash": compiled_research.snapshot_hash,
                "selection_score": evolution.winner.score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _advance(
        WorkflowStage.WINNER_FROZEN,
        winner_freeze_path,
        diagnostics={"candidate_sha256": frozen_sha},
    )
    frozen_parent_controls: dict[str, Path] = {}
    if evolution.winner.parent_shas:
        parent_dir = selection_dir / "parent_controls"
        parent_dir.mkdir(parents=True, exist_ok=True)
        canonical_by_sha = {
            entry.sha: entry
            for entry in evolution.archive
            if entry.verify_ok and not entry.deduplicated and entry.source_path
        }
        for parent_sha in evolution.winner.parent_shas:
            parent_entry = canonical_by_sha.get(parent_sha)
            if parent_entry is None:
                raise RuntimeError(
                    f"selected candidate references unavailable parent source: {parent_sha}"
                )
            parent_source = Path(parent_entry.source_path).read_text(encoding="utf-8")
            if _sha_text(parent_source) != parent_sha:
                raise RuntimeError(f"parent source SHA mismatch before held-out: {parent_sha}")
            parent_path = parent_dir / f"parent_{parent_sha[:12]}.py"
            parent_path.write_text(parent_source, encoding="utf-8")
            frozen_parent_controls[parent_sha] = parent_path

    # The winner is now frozen. Only after this point is the held-out third even materialized.
    heldout_scenarios, heldout_manifest = build_scenarios(
        burstgpt_path,
        fragment_size=fragment_size,
        slo_ttft_ms=slo_ttft_ms,
        max_num_seqs=max_num_seqs,
        splits=("test",),
    )
    data_manifest = {
        **search_manifest,
        "materialized_splits": ["train", "validation", "test"],
        "fragments": {
            **search_manifest["fragments"],
            **heldout_manifest["fragments"],
        },
        "heldout_materialized_after_winner_sha256": frozen_sha,
    }
    dataset_manifest_path = root / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(data_manifest, indent=2), encoding="utf-8"
    )
    _advance(
        WorkflowStage.HELDOUT_MATERIALIZED,
        dataset_manifest_path,
        diagnostics={"winner_sha256_before_materialization": frozen_sha},
    )

    baseline_heldout = []
    for name, path in baseline_paths.items():
        for scenario in heldout_scenarios:
            baseline_heldout.append(
                evaluate_policy(
                    policy_name=name,
                    policy_path=path,
                    scenario=scenario,
                    seeds=seeds,
                    out_dir=root / "raw" / "heldout_baselines",
                )
            )
    strongest_heldout = _scenario_baselines(baseline_heldout)
    heldout_baselines_path = root / "heldout_baselines.json"
    heldout_baselines_path.write_text(
        json.dumps(
            {
                "strongest": strongest_heldout,
                "results": [result.to_dict() for result in baseline_heldout],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _advance(WorkflowStage.BASELINES_EVALUATED, heldout_baselines_path)
    candidate_heldout = [
        evaluate_policy(
            policy_name=f"winner_{frozen_sha[:12]}",
            policy_path=str(selection_path),
            scenario=scenario,
            seeds=seeds,
            out_dir=root / "raw" / "heldout_candidate",
        )
        for scenario in heldout_scenarios
    ]
    heldout_candidate_path = root / "heldout_candidate.json"
    heldout_candidate_path.write_text(
        json.dumps([result.to_dict() for result in candidate_heldout], indent=2),
        encoding="utf-8",
    )
    _advance(WorkflowStage.HELDOUT_EVALUATED, heldout_candidate_path)
    acceptance = _acceptance(candidate_heldout, strongest_heldout)

    ablation_dir = root / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    selection_manifest = (
        json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        if selection_manifest_path.is_file()
        else {}
    )
    declared_ablation_plan = list(selection_manifest.get("ablation_plan") or [])
    ablation_sources: dict[str, str] = {}
    ablation_labels: dict[str, str] = {}
    if not frozen_parent_controls:
        # A template seed has no parent control.  Build one auditable control by changing exactly
        # one typed recipe field, and require that the manifest declared exactly that ablation.
        structural_control = _template_seed_ablation(winner_source)
        if structural_control is not None and len(declared_ablation_plan) == 1:
            name, source, delta = structural_control
            ablation_sources[name] = source
            ablation_labels[name] = f"{declared_ablation_plan[0]}; verified source delta: {delta}"
    ablation_results = {}
    for name, source in ablation_sources.items():
        path = ablation_dir / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        ablation_results[name] = [
            evaluate_policy(
                policy_name=name,
                policy_path=str(path),
                scenario=scenario,
                seeds=seeds,
                out_dir=root / "raw" / "heldout_ablations",
            )
            for scenario in heldout_scenarios
        ]
    parent_plan_complete = len(declared_ablation_plan) == len(frozen_parent_controls)
    for parent_sha, path in frozen_parent_controls.items():
        if not parent_plan_complete:
            continue
        name = f"parent_{parent_sha[:12]}"
        plan_index = list(frozen_parent_controls).index(parent_sha)
        ablation_labels[name] = declared_ablation_plan[plan_index]
        ablation_results[name] = [
            evaluate_policy(
                policy_name=name,
                policy_path=str(path),
                scenario=scenario,
                seeds=seeds,
                out_dir=root / "raw" / "heldout_ablations",
            )
            for scenario in heldout_scenarios
        ]
    ablation_evidence = _multi_control_ablation_evidence(
        candidate_heldout,
        ablation_results,
        control_mechanisms=ablation_labels,
    )
    ablation_evidence_path = root / "ablation_evidence.json"
    ablation_evidence_path.write_text(
        json.dumps(
            {
                "results": {
                    name: [result.to_dict() for result in rows]
                    for name, rows in ablation_results.items()
                },
                "evidence": ablation_evidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _advance(WorkflowStage.ABLATIONS_EVALUATED, ablation_evidence_path)
    acceptance["thresholds_passed"] = acceptance["passed"]
    acceptance["ablation_mechanism_supported"] = ablation_evidence["supported"]
    acceptance["passed"] = acceptance["thresholds_passed"] and ablation_evidence["supported"]
    knowledge_writeback = list(evolution.lessons_written)
    if store is not None:
        knowledge_writeback.append(
            store.put_lesson(
                run_id=root.name,
                source="frontier_sim",
                policy_sha=frozen_sha,
                regime="burstgpt_and_stress_heldout",
                metric="heldout_median_gain_pct",
                score=acceptance["aggregate_median_gain_pct"],
                conclusion=(
                    "held-out Frontier sim_winner "
                    f"{'passed' if acceptance['passed'] else 'failed'} local thresholds; "
                    f"positive={acceptance['positive_scenarios']}/"
                    f"{acceptance['scenario_count']}; "
                    f"ablation_supported={ablation_evidence['supported']}; "
                    "simulator_nonqualifying, not a real-vLLM performance verdict"
                ),
                eval_refs=[
                    str(
                        root
                        / ("sim_winner" if acceptance["passed"] else "no_winner")
                        / "evidence.json"
                    ),
                    str(root / "report.md"),
                ],
            )
        )
    terminal_dir = root / ("sim_winner" if acceptance["passed"] else "no_winner")
    terminal_dir.mkdir(parents=True, exist_ok=True)
    winner_path = terminal_dir / "work_variant.py" if acceptance["passed"] else None
    winner_manifest_path = terminal_dir / "candidate_manifest.json"
    if acceptance["passed"]:
        shutil.copyfile(selection_path, winner_path)
        if selection_manifest_path.is_file():
            shutil.copyfile(selection_manifest_path, winner_manifest_path)
    result = {
        "ok": acceptance["passed"],
        "verdict": "SIM_WINNER" if acceptance["passed"] else "NO_WINNER",
        "source": "frontier_sim",
        "outcome_class": "simulator_nonqualifying",
        "remote_executed": False,
        "normalized_goal": normalized_goal.to_dict(),
        "workflow_state_path": str(workflow.state_path),
        "workflow_events_path": str(workflow.events_path),
        "selection_path": str(selection_path),
        "winner_path": str(winner_path) if winner_path is not None else None,
        "winner_candidate_manifest": (
            str(winner_manifest_path)
            if acceptance["passed"] and winner_manifest_path.is_file()
            else None
        ),
        "winner_sha256": frozen_sha,
        "author_kind": evolution.author_kind,
        "research_snapshot_id": compiled_research.snapshot_id,
        "research_snapshot_hash": compiled_research.snapshot_hash,
        "research_outcome_class": compiled_research.outcome_class,
        "research_reused_frozen_snapshot": research_reused,
        "research_artifact_dir": str(research_artifact_dir),
        "mechanism_lineage": [
            {
                "sha": entry.sha,
                "parent_sha": entry.parent_sha,
                "parent_shas": entry.parent_shas,
                "operator": entry.operator,
                "mechanism_ids": entry.mechanism_ids,
                "candidate_manifest_path": entry.candidate_manifest_path,
                "score": entry.score,
            }
            for entry in evolution.archive
            if entry.verify_ok
        ],
        "knowledge_writeback_lesson_ids": knowledge_writeback,
        "evolution": evolution.to_dict(),
        "heldin_results": [result.to_dict() for result in evaluations.get(frozen_sha, [])],
        "heldout_baselines": {
            "strongest": strongest_heldout,
            "results": [result.to_dict() for result in baseline_heldout],
        },
        "heldout_candidate": [result.to_dict() for result in candidate_heldout],
        "ablations": {
            name: [result.to_dict() for result in rows] for name, rows in ablation_results.items()
        },
        "ablation_evidence": ablation_evidence,
        "acceptance": acceptance,
        "dataset_manifest": data_manifest,
        "doctrine": (
            "Frontier scores are simulator-only. A sim_winner exists only when every local gate "
            "passes; either verdict remains ineligible for real-vLLM keep/adoption."
        ),
    }
    evidence_path = terminal_dir / "evidence.json"
    report_path = root / "report.md"
    result["report_path"] = str(report_path)
    report_path.write_text(
        _render_research_evolution_report(result, compiled_research),
        encoding="utf-8",
    )
    evidence_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if acceptance["passed"]:
        (terminal_dir / "REQUIRES_REAL_VLLM_VERIFICATION").write_text(
            "Local-only goal result: Frontier simulator evidence, never a production vLLM claim.\n",
            encoding="utf-8",
        )
    shutil.copyfile(root / "dataset_manifest.json", terminal_dir / "dataset_manifest.json")
    _advance(
        WorkflowStage.SIM_WINNER if acceptance["passed"] else WorkflowStage.NO_WINNER,
        evidence_path,
        diagnostics={
            "acceptance_passed": acceptance["passed"],
            "winner_sha256": frozen_sha,
        },
    )
    return result


__all__ = [
    "Scenario",
    "build_scenarios",
    "evaluate_policy",
    "run_local_frontier_evolution",
]
