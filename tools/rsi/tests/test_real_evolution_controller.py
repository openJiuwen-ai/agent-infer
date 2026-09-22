from __future__ import annotations

import json

import pytest

from vllm_evolve.bench.config import STRONG_BASELINE, BenchConfig
from vllm_evolve.bench.dispatch import RemoteBench
from vllm_evolve.engine.real_evolution import (
    FITNESS_METRIC,
    RealCandidateEvaluator,
    RealEvolutionProgress,
    run_real_evolution,
    strict_real_fitness_reasons,
)
from vllm_evolve.knowledge.schemas import (
    MechanismCard,
    ResearchContext,
    ResearchSource,
)


def _valid_eval(*, candidate: bool = False, median: float = 100.0) -> dict:
    result = {
        "source": "real_vllm",
        "outcome_class": "eval_result",
        "aggregate_metrics": {"median": median},
        "raw_per_seed_metrics": [
            {
                "metrics": {
                    "num_requests": 512,
                    "num_completed": 512,
                    "num_failed": 0,
                }
            }
        ],
        "workload_validity": {
            "required": True,
            "valid": True,
            "verdict": "valid_saturated_real_vllm",
        },
        "marker_verified": candidate,
    }
    if candidate:
        result["plugin_provenance"] = {
            "effective": True,
            "fallback": False,
            "deferred_request_actions": 7,
        }
    return result


def _research() -> ResearchContext:
    source = ResearchSource(
        source_id="github:vllm-main:test",
        title="Pinned vLLM main scheduler source",
        authors=["vLLM"],
        year=2026,
        date="2026-07-28",
        source_kind="code",
        canonical_url="https://github.com/vllm-project/vllm",
        primary_source=True,
        citation_status="retrieved_primary",
    )
    card = MechanismCard(
        mechanism_id="bounded-kv-fit-bypass",
        name="Bounded KV-fit head bypass",
        source_ids=[source.source_id],
        problem="A temporarily unallocatable queue head can block a later fit request.",
        core_mechanism="Defer one unfit head for one scheduler event.",
        upstream_gap="Current main stops admission on the first allocation failure.",
        upstream_symbols_checked=["vllm.v1.core.sched.scheduler.Scheduler.schedule"],
        current_vllm_behavior="The allocation loop breaks when allocate_slots returns None.",
        candidate_delta="Try one later KV-fit request and restore the head next event.",
        why_not_already_integrated="No bounded head-bypass path exists at the pinned source.",
        required_runtime_signal="Waiting queue and high KV occupancy.",
        mechanism_trigger="Head allocation fails while a later request can fit.",
        expected_action_counters=["deferred_request_actions"],
        minimum_meaningful_ablation="Disable only the bounded bypass action.",
        version_constraints=["vLLM main pinned by live source manifest"],
    )
    return ResearchContext(
        target="scheduling",
        goal={"metric": FITNESS_METRIC, "direction": "max"},
        diagnosis={"bottleneck": "scheduling_queue_under_kv_pressure"},
        environment={"source": "real_vllm"},
        query_plan=["inspect current vLLM main scheduler"],
        sources=[source],
        mechanism_cards=[card],
        live_source_manifest=[source.to_dict()],
        citations=[f"source:{source.source_id}"],
    )


def _baseline_config(artifact_root: str) -> BenchConfig:
    config = BenchConfig()
    config.runner.runner_kind = STRONG_BASELINE
    config.runner.local_artifact_root = artifact_root
    config.workload.workload_spec = {
        "source": "real_vllm",
        "validity": {"required": True},
    }
    return config


def test_strict_real_fitness_accepts_only_valid_effective_candidate():
    valid = _valid_eval(candidate=True)
    assert strict_real_fitness_reasons(
        valid,
        candidate=True,
        expected_action_counters=["deferred_request_actions"],
    ) == []

    invalid = _valid_eval(candidate=True)
    invalid["source"] = "frontier_sim"
    invalid["raw_per_seed_metrics"][0]["metrics"]["num_failed"] = 1
    invalid["plugin_provenance"]["deferred_request_actions"] = 0
    reasons = strict_real_fitness_reasons(
        invalid,
        candidate=True,
        expected_action_counters=["deferred_request_actions"],
    )
    assert any("source is not real_vllm" in reason for reason in reasons)
    assert any("completion is not exactly 100%" in reason for reason in reasons)
    assert any("did not fire" in reason for reason in reasons)


def test_real_evaluator_scores_only_against_valid_real_baseline(tmp_path):
    artifact_root = tmp_path / "artifacts"
    config = _baseline_config(str(artifact_root))
    policy = tmp_path / "candidate.py"
    policy.write_text(
        "def schedule_batch(running, waiting, state):\n    return []\n",
        encoding="utf-8",
    )
    policy.with_suffix(".manifest.json").write_text(
        json.dumps({"mechanism_ids": ["bounded-kv-fit-bypass"]}),
        encoding="utf-8",
    )

    def runner(_config, *, git_sha):
        assert git_sha == "git"
        return RemoteBench(
            eval_result=_valid_eval(candidate=True, median=110.0),
            log="",
            local_eval_path=str(tmp_path / "candidate_eval.json"),
            local_log_path=str(tmp_path / "candidate.log"),
            remote_run_dir="/remote/run/candidate",
        )

    progress = RealEvolutionProgress()
    evaluator = RealCandidateEvaluator(
        baseline_config=config,
        baseline_eval=_valid_eval(median=100.0),
        research=_research(),
        progress=progress,
        progress_path=tmp_path / "lineage.json",
        artifact_root=artifact_root,
        git_sha="git",
        runner=runner,
    )
    profile = evaluator({"policy": str(policy)})
    assert profile.source == "real_vllm"
    assert profile.metrics[FITNESS_METRIC] == pytest.approx(10.0)
    assert progress.evaluations[0]["remote_run_dir"] == "/remote/run/candidate"


def test_real_evaluator_preserves_invalid_measurements_as_nonranking_diagnostics(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    config = _baseline_config(str(artifact_root))
    policy = tmp_path / "candidate.py"
    policy.write_text(
        "def schedule_batch(running, waiting, state):\n    return []\n",
        encoding="utf-8",
    )
    policy.with_suffix(".manifest.json").write_text(
        json.dumps({"mechanism_ids": ["bounded-kv-fit-bypass"]}),
        encoding="utf-8",
    )
    invalid = _valid_eval(candidate=True, median=110.0)
    invalid["workload_validity"] = {
        "required": True,
        "valid": False,
        "verdict": "invalid_underloaded_gpu",
        "per_seed": [
            {
                "gpu": {
                    "2": {
                        "utilization_pct": {"p10": 41.0, "p50": 68.0},
                        "active_duty_cycle": 0.81,
                        "memory_ratio": {"p50": 0.84},
                    }
                },
                "vllm_per_replica": {
                    "0": {
                        "running_requests": {"p50": 900.0},
                        "waiting_requests": {"p50": 700.0},
                        "kv_cache_occupancy": {"p50": 0.91},
                        "preemptions_delta": 12.0,
                        "scheduler_invocations_delta": 4000.0,
                    }
                },
            }
        ],
    }
    invalid["raw_per_seed_metrics"][0]["slo_result"] = {
        "attained": 400,
        "attainment_rate": 0.78125,
        "goodput_req_s": 110.0,
    }

    def runner(_config, *, git_sha):
        return RemoteBench(
            eval_result=invalid,
            log="",
            local_eval_path=str(tmp_path / "candidate_eval.json"),
            local_log_path=str(tmp_path / "candidate.log"),
            remote_run_dir="/remote/run/candidate",
        )

    progress = RealEvolutionProgress()
    evaluator = RealCandidateEvaluator(
        baseline_config=config,
        baseline_eval=_valid_eval(median=100.0),
        research=_research(),
        progress=progress,
        progress_path=tmp_path / "lineage.json",
        artifact_root=artifact_root,
        git_sha="git",
        runner=runner,
    )
    profile = evaluator({"policy": str(policy)})
    event = progress.evaluations[0]
    diagnostics = event["diagnostics"]
    assert profile.metrics == {}
    assert event["fitness"] is None
    assert diagnostics["evidence_role"] == "diagnostic_only_nonranking"
    assert diagnostics["selection_eligible"] is False
    assert diagnostics["raw_gain_pct"] == pytest.approx(10.0)
    assert diagnostics["seed_measurements"][0]["slo_attained"] == 400
    assert diagnostics["gpu_pressure"][0]["utilization_p10_pct"] == 41.0


def test_real_evolution_stops_before_authoring_on_invalid_baseline(tmp_path):
    config = _baseline_config(str(tmp_path / "unused"))
    invalid = _valid_eval()
    invalid["workload_validity"] = {
        "required": True,
        "valid": False,
        "verdict": "invalid_underloaded_gpu",
    }

    def runner(_config, *, git_sha):
        assert git_sha
        return RemoteBench(
            eval_result=invalid,
            log="",
            local_eval_path=str(tmp_path / "baseline_eval.json"),
            local_log_path=str(tmp_path / "baseline.log"),
            remote_run_dir="/remote/run/baseline",
        )

    result = run_real_evolution(
        author_fn=lambda _context: (_ for _ in ()).throw(
            AssertionError("author must not run")
        ),
        baseline_config=config,
        research=_research(),
        out_dir=tmp_path / "real-evolution",
        runner=runner,
    )
    assert result["status"] == "blocked_invalid_baseline_workload"
    assert "valid_saturated_real_vllm" in " ".join(result["baseline_reasons"])
    lineage = json.loads(
        (tmp_path / "real-evolution" / "evolution_lineage.json").read_text()
    )
    assert lineage["source"] == "real_vllm"
