"""Pure real-vLLM generational evolution controller.

This module deliberately has no Frontier or simulator import.  Candidate fitness
exists only when the remote eval_result proves a valid saturated workload,
complete execution, a nonce-bound effective plugin, and the declared mechanism
action counters.  Invalid runs remain lineage evidence with no score.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vllm_evolve.bench.config import (
    CANDIDATE,
    STRONG_BASELINE,
    BenchConfig,
    same_caliber,
)
from vllm_evolve.bench.dispatch import run_remote_bench_config
from vllm_evolve.core.schemas import Profile, Spec
from vllm_evolve.engine.evolve_loop import run_evolution
from vllm_evolve.knowledge.schemas import ResearchContext

FITNESS_METRIC = "real_fitness_gain_pct"


def _primary_median(eval_result: dict) -> float | None:
    value = (eval_result.get("aggregate_metrics") or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _completion_is_exact(eval_result: dict) -> bool:
    rows = list(eval_result.get("raw_per_seed_metrics") or [])
    if not rows:
        return False
    return all(
        int((row.get("metrics") or {}).get("num_completed") or 0)
        > 0
        and int((row.get("metrics") or {}).get("num_failed") or 0) == 0
        and int((row.get("metrics") or {}).get("num_completed") or 0)
        == int((row.get("metrics") or {}).get("num_requests") or 0)
        for row in rows
    )


def _diagnostic_evidence(
    eval_result: dict,
    *,
    baseline_median: float | None,
    fitness_reasons: list[str],
) -> dict:
    """Preserve measurements from non-ranking real runs for the next author.

    This payload is deliberately diagnostic-only.  ``strict_real_fitness_reasons``
    remains the sole selection boundary and the caller still emits no fitness
    metric when any reason is present.
    """
    candidate_median = _primary_median(eval_result)
    raw_gain_pct = None
    if (
        candidate_median is not None
        and baseline_median not in {None, 0.0}
    ):
        raw_gain_pct = (
            float(candidate_median) / float(baseline_median) - 1.0
        ) * 100.0

    seed_measurements = []
    for row in list(eval_result.get("raw_per_seed_metrics") or []):
        metrics = dict(row.get("metrics") or {})
        slo = dict(row.get("slo_result") or {})
        seed_measurements.append(
            {
                "seed": row.get("seed"),
                "num_requests": metrics.get("num_requests"),
                "num_completed": metrics.get("num_completed"),
                "num_failed": metrics.get("num_failed"),
                "duration_s": metrics.get("duration_s"),
                "request_throughput_req_s": metrics.get(
                    "request_throughput_req_s"
                ),
                "output_throughput_tok_s": metrics.get(
                    "output_throughput_tok_s"
                ),
                "slo_attained": slo.get("attained"),
                "slo_attainment_rate": slo.get("attainment_rate"),
                "goodput_req_s": slo.get("goodput_req_s"),
            }
        )

    validity = dict(eval_result.get("workload_validity") or {})
    pressure = []
    for seed_row in list(validity.get("per_seed") or []):
        vllm_by_replica = dict(seed_row.get("vllm_per_replica") or {})
        for replica_index, (gpu_id, gpu_row) in enumerate(
            dict(seed_row.get("gpu") or {}).items()
        ):
            replica = str(replica_index)
            vllm = dict(vllm_by_replica.get(replica) or {})
            pressure.append(
                {
                    "seed": seed_row.get("seed"),
                    "replica_index": replica_index,
                    "gpu_id": str(gpu_id),
                    "utilization_p10_pct": (
                        gpu_row.get("utilization_pct") or {}
                    ).get("p10"),
                    "utilization_p50_pct": (
                        gpu_row.get("utilization_pct") or {}
                    ).get("p50"),
                    "active_duty_cycle": gpu_row.get("active_duty_cycle"),
                    "memory_ratio_p50": (
                        gpu_row.get("memory_ratio") or {}
                    ).get("p50"),
                    "running_requests_p50": (
                        vllm.get("running_requests") or {}
                    ).get("p50"),
                    "waiting_requests_p50": (
                        vllm.get("waiting_requests") or {}
                    ).get("p50"),
                    "kv_cache_occupancy_p50": (
                        vllm.get("kv_cache_occupancy") or {}
                    ).get("p50"),
                    "preemptions_delta": vllm.get("preemptions_delta"),
                    "scheduler_invocations_delta": vllm.get(
                        "scheduler_invocations_delta"
                    ),
                }
            )

    provenance = dict(eval_result.get("plugin_provenance") or {})
    action_counters = {
        key: value
        for key, value in provenance.items()
        if key.endswith("_count") or key.endswith("_actions")
        if isinstance(value, (int, float))
    }
    return {
        "evidence_role": "diagnostic_only_nonranking",
        "selection_eligible": not fitness_reasons,
        "source": eval_result.get("source"),
        "outcome_class": eval_result.get("outcome_class"),
        "workload_verdict": validity.get("verdict"),
        "fitness_reasons": list(fitness_reasons),
        "primary_metric": (
            (eval_result.get("aggregate_metrics") or {}).get("primary_metric")
        ),
        "candidate_primary_median": candidate_median,
        "baseline_primary_median": baseline_median,
        "raw_gain_pct": raw_gain_pct,
        "seed_measurements": seed_measurements,
        "gpu_pressure": pressure,
        "mechanism_effective": provenance.get("effective"),
        "mechanism_applicable": provenance.get("mechanism_applicable"),
        "action_counters": action_counters,
    }


def strict_real_fitness_reasons(
    eval_result: dict,
    *,
    candidate: bool,
    expected_action_counters: list[str] | None = None,
) -> list[str]:
    """Explain why an eval_result may not enter real-vLLM selection."""
    reasons = []
    validity = dict(eval_result.get("workload_validity") or {})
    if eval_result.get("source") != "real_vllm":
        reasons.append("source is not real_vllm")
    if eval_result.get("outcome_class") != "eval_result":
        reasons.append(
            f"outcome_class={eval_result.get('outcome_class')!r} is not a clean real run"
        )
    if not (
        validity.get("required") is True
        and validity.get("valid") is True
        and validity.get("verdict") == "valid_saturated_real_vllm"
    ):
        reasons.append("workload validity is not valid_saturated_real_vllm")
    if not _completion_is_exact(eval_result):
        reasons.append("completion is not exactly 100% with zero failures")
    if _primary_median(eval_result) is None:
        reasons.append("primary aggregate median is missing")
    if candidate:
        provenance = dict(eval_result.get("plugin_provenance") or {})
        if eval_result.get("marker_verified") is not True:
            reasons.append("candidate marker was not verified")
        if provenance.get("fallback"):
            reasons.append("candidate plugin fell back to stock scheduling")
        if provenance.get("effective") is not True:
            reasons.append("candidate mechanism had no runtime effect")
        for counter in expected_action_counters or []:
            value = provenance.get(counter)
            if not isinstance(value, (int, float)) or value <= 0:
                reasons.append(f"declared action counter {counter!r} did not fire")
    return reasons


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class RealEvolutionProgress:
    schema_version: int = 1
    source: str = "real_vllm"
    git_sha: str = ""
    research_snapshot_hash: str = ""
    baseline: dict = field(default_factory=dict)
    evaluations: list[dict] = field(default_factory=list)
    result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "git_sha": self.git_sha,
            "research_snapshot_hash": self.research_snapshot_hash,
            "baseline": self.baseline,
            "evaluations": self.evaluations,
            "result": self.result,
        }


class RealCandidateEvaluator:
    """Adapt immutable remote real-vLLM runs to Evolution Engine profiles."""

    ve_source = "real_vllm"

    def __init__(
        self,
        *,
        baseline_config: BenchConfig,
        baseline_eval: dict,
        research: ResearchContext,
        progress: RealEvolutionProgress,
        progress_path: Path,
        artifact_root: Path,
        git_sha: str,
        runner: Callable = run_remote_bench_config,
    ):
        self.baseline_config = baseline_config
        self.baseline_eval = baseline_eval
        self.baseline_median = _primary_median(baseline_eval)
        self.research = research
        self.progress = progress
        self.progress_path = progress_path
        self.artifact_root = artifact_root
        self.git_sha = git_sha
        self.runner = runner
        self.next_port = int(baseline_config.runner.port) + 1

    def _expected_counters(self, policy_path: Path) -> list[str]:
        manifest_path = policy_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mechanism_ids = set(manifest.get("mechanism_ids") or [])
        return list(
            dict.fromkeys(
                counter
                for card in self.research.mechanism_cards
                if card.mechanism_id in mechanism_ids
                for counter in card.expected_action_counters
            )
        )

    def __call__(self, config_value: dict) -> Profile:
        policy_path = Path(config_value["policy"]).resolve()
        config = BenchConfig.from_dict(self.baseline_config.to_dict())
        config.runner.runner_kind = CANDIDATE
        config.runner.policy_path = str(policy_path)
        config.runner.port = self.next_port
        config.runner.local_artifact_root = str(self.artifact_root)
        config.engine.scheduler_cls = "generated_scheduler.EvolvedScheduler"
        self.next_port += 1
        caliber_ok, caliber_diffs = same_caliber(self.baseline_config, config)
        expected = self._expected_counters(policy_path)
        event = {
            "source": "real_vllm",
            "policy_path": str(policy_path),
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "expected_action_counters": expected,
            "same_caliber": caliber_ok,
            "same_caliber_diffs": caliber_diffs,
            "eval_result_path": None,
            "remote_run_dir": None,
            "fitness": None,
            "fitness_reasons": [],
        }
        if not caliber_ok:
            event["fitness_reasons"] = [f"same-caliber mismatch: {caliber_diffs}"]
            self.progress.evaluations.append(event)
            _write_json(self.progress_path, self.progress.to_dict())
            return Profile(
                config=config.to_dict(),
                source="real_vllm",
                outcome_class="same_caliber_mismatch",
                evidence=[event],
            )
        try:
            remote = self.runner(config, git_sha=self.git_sha)
            eval_result = remote.eval_result
            event["eval_result_path"] = remote.local_eval_path
            event["remote_run_dir"] = remote.remote_run_dir
            reasons = strict_real_fitness_reasons(
                eval_result,
                candidate=True,
                expected_action_counters=expected,
            )
        except Exception as exc:  # failed real attempts remain negative lineage
            eval_result = {}
            reasons = [f"{type(exc).__name__}: {exc}"[:2000]]
        event["fitness_reasons"] = reasons
        event["diagnostics"] = _diagnostic_evidence(
            eval_result,
            baseline_median=self.baseline_median,
            fitness_reasons=reasons,
        )
        metrics = {}
        if not reasons and self.baseline_median not in {None, 0.0}:
            candidate_median = _primary_median(eval_result)
            gain = (
                float(candidate_median) / float(self.baseline_median) - 1.0
            ) * 100.0
            event["fitness"] = gain
            metrics[FITNESS_METRIC] = gain
        self.progress.evaluations.append(event)
        _write_json(self.progress_path, self.progress.to_dict())
        return Profile(
            config=config.to_dict(),
            metrics=metrics,
            source="real_vllm",
            marker_verified=eval_result.get("marker_verified") is True,
            saturated=not reasons,
            outcome_class=str(eval_result.get("outcome_class") or "real_candidate_invalid"),
            eval_result=eval_result or None,
            bench_config=config.to_dict(),
            remote_cmd=str(eval_result.get("remote_cmd") or ""),
            evidence=[event],
        )


def run_real_evolution(
    *,
    author_fn,
    baseline_config: BenchConfig,
    research: ResearchContext,
    out_dir: str | Path,
    generations: int = 2,
    population: int = 4,
    max_total_evals: int = 8,
    repair_limit: int = 1,
    runner: Callable = run_remote_bench_config,
) -> dict:
    """Run baseline then select every parent exclusively from real-vLLM fitness."""
    if int(generations) < 2:
        raise ValueError("real evolution requires at least two generations")
    if int(population) < 3:
        raise ValueError("real evolution requires at least three children per generation")
    if not 6 <= int(max_total_evals) <= 10:
        raise ValueError("real evolution requires a 6..10 real-evaluation budget")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    progress_path = out / "evolution_lineage.json"
    git_sha = _git_sha()
    progress = RealEvolutionProgress(
        git_sha=git_sha,
        research_snapshot_hash=research.snapshot_hash,
    )
    integrity = research.validate_snapshot_integrity()
    if not integrity["ok"]:
        raise ValueError(f"research snapshot integrity failed: {integrity}")
    if not research.live_source_manifest:
        raise ValueError("real evolution requires a non-empty live source manifest")
    if not research.mechanism_cards:
        raise ValueError("real evolution requires live-gap mechanism cards")
    invalid_cards = {}
    for card in research.mechanism_cards:
        missing = card.validate_live_gap()
        if missing:
            invalid_cards[card.mechanism_id] = missing
    if invalid_cards:
        raise ValueError(f"research contains non-live-gap mechanism cards: {invalid_cards}")
    validity = baseline_config.workload.workload_spec.get("validity") or {}
    if validity.get("required") is not True:
        raise ValueError("real evolution requires a formal saturated workload protocol")
    if baseline_config.runner.runner_kind != STRONG_BASELINE:
        raise ValueError("real evolution baseline must be runner_kind=strong_baseline")
    baseline_config.runner.local_artifact_root = str(out / "artifacts")
    baseline_remote = runner(baseline_config, git_sha=git_sha)
    baseline_eval = baseline_remote.eval_result
    baseline_reasons = strict_real_fitness_reasons(baseline_eval, candidate=False)
    progress.baseline = {
        "source": "real_vllm",
        "eval_result_path": baseline_remote.local_eval_path,
        "remote_run_dir": baseline_remote.remote_run_dir,
        "primary_median": _primary_median(baseline_eval),
        "fitness_reasons": baseline_reasons,
    }
    _write_json(progress_path, progress.to_dict())
    if baseline_reasons:
        result = {
            "source": "real_vllm",
            "status": "blocked_invalid_baseline_workload",
            "baseline_reasons": baseline_reasons,
            "lineage_path": str(progress_path),
        }
        progress.result = result
        _write_json(progress_path, progress.to_dict())
        return result

    evaluator = RealCandidateEvaluator(
        baseline_config=baseline_config,
        baseline_eval=baseline_eval,
        research=research,
        progress=progress,
        progress_path=progress_path,
        artifact_root=out / "artifacts",
        git_sha=git_sha,
        runner=runner,
    )
    skeleton = Path("targets/scheduling/skeleton.py").read_text(encoding="utf-8")
    evolution = run_evolution(
        author_fn,
        evaluator,
        Spec(
            metric=FITNESS_METRIC,
            direction="max",
            raw_intent=(
                "maximize paired gain over a valid saturated real-vLLM strong baseline"
            ),
        ),
        baseline_config.to_dict(),
        {
            **dict(research.diagnosis),
            "source": "real_vllm",
            "baseline_eval_result": baseline_remote.local_eval_path,
        },
        generations=generations,
        population=population,
        repair_limit=repair_limit,
        max_total_evals=max_total_evals,
        variants_dir=str(out / "candidates"),
        skeleton=skeleton,
        research_context=research.to_dict(),
        author_kind="codex",
        require_author_manifest=True,
    )
    result = {
        "source": "real_vllm",
        "status": (
            "real_winner_proposal"
            if evolution.winner is not None
            else "no_verified_real_candidate"
        ),
        "baseline": progress.baseline,
        "evolution": evolution.to_dict(),
        "lineage_path": str(progress_path),
    }
    progress.result = result
    _write_json(progress_path, progress.to_dict())
    _write_json(out / "result.json", result)
    return result


__all__ = [
    "FITNESS_METRIC",
    "RealCandidateEvaluator",
    "RealEvolutionProgress",
    "run_real_evolution",
    "strict_real_fitness_reasons",
]
