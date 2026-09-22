"""Pure tests for the local Frontier evolution runner (the real subprocess has its own e2e)."""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_evolve.bench.frontier_catalog import FrontierCatalog
from vllm_evolve.engine.evolve_schemas import EvolutionResult, LineageEntry
from vllm_evolve.engine.local_frontier_evolve import (
    PolicyScenarioResult,
    Scenario,
    _ablation_evidence,
    _acceptance,
    _measure_catalog,
    _template_seed_ablation,
    build_scenarios,
    run_local_frontier_evolution,
)
from vllm_evolve.engine.workflow_state import WorkflowStage
from vllm_evolve.knowledge.compiler import offline_compiler

FIX = Path(__file__).parent / "bench" / "fixtures"


def _result(scenario: str, score: float, *, marker_ok: bool = True):
    return PolicyScenarioResult(
        policy="candidate",
        policy_sha256="abc",
        scenario=scenario,
        split="test",
        source_kind="fixture",
        per_seed=[],
        median_goodput_req_s=score,
        median_ttft_p50_ms=10.0,
        median_ttft_p99_ms=20.0,
        median_throughput_req_s=score,
        completed_min=10,
        marker_ok=marker_ok,
        invocations=10,
        fallbacks=0,
        defers=1,
        forced=0,
    )


def test_scenarios_separate_official_fragments_from_synthetic_stress():
    scenarios, manifest = build_scenarios(
        FIX / "burstgpt_fragments.csv", fragment_size=2
    )
    official = [item for item in scenarios if item.source_kind.startswith("official")]
    synthetic = [item for item in scenarios if item.source_kind.startswith("synthetic")]
    assert {item.split for item in official} == {"train", "validation", "test"}
    assert {item.name for item in official} == {
        "burstgpt_train",
        "burstgpt_validation",
        "burstgpt_validation_tail",
        "burstgpt_test",
    }
    assert len(synthetic) == 4
    assert all({row["session_id"] for row in item.rows} == {0} for item in official)
    assert all({row["block_hash_ids"] for row in item.rows} == {""} for item in official)
    assert manifest["split_method"].startswith("densest positive-duration")
    assert all(item.enable_prefix_caching is False for item in official)
    assert all(item.enable_prefix_caching is True for item in synthetic)


def test_search_can_be_built_without_materializing_heldout_data():
    scenarios, manifest = build_scenarios(
        FIX / "burstgpt_fragments.csv",
        fragment_size=2,
        splits=("train", "validation"),
    )
    assert {item.split for item in scenarios} == {"train", "validation"}
    assert manifest["materialized_splits"] == ["train", "validation"]
    assert set(manifest["fragments"]) == {
        "train",
        "validation",
        "validation_tail",
    }


def test_stress_prefix_depth_is_explicitly_configurable():
    scenarios, _ = build_scenarios(
        FIX / "burstgpt_fragments.csv",
        fragment_size=2,
        splits=("test",),
        stress_prefix_blocks=8,
    )
    stress = [item for item in scenarios if item.source_kind.startswith("synthetic")]
    assert all(item.metadata["prefix_blocks_per_tenant"] == 8 for item in stress)
    assert all(
        len(item.rows[0]["block_hash_ids"].split("|")) == 8
        for item in stress
    )


def test_catalog_measurement_computes_real_slo_goodput_and_marker():
    marker = {
        "policy_sha256": "abc",
        "invocations": 10,
        "fallbacks": 0,
        "defers": 3,
    }
    catalog = FrontierCatalog(
        out_dir="o",
        run_id="r",
        dir=Path("metrics"),
        system={
            "throughput_metrics": {"requests_per_second": 20.0},
            "simulation_metadata": {"completed_requests": 1},
        },
        rows=[{"ttft": "100"}, {"ttft": "900"}],
        columns=["ttft"],
        marker=marker,
    )
    measured = _measure_catalog(catalog, seed=0, slo_ttft_ms=800, policy_sha="abc")
    assert measured.goodput_req_s == 10.0
    assert measured.completed == 1
    assert measured.marker_ok is True


def test_acceptance_requires_three_percent_two_positive_and_no_large_regression():
    strongest = {
        "a": {"policy": "sjf", "median_goodput_req_s": 100.0, "completed_min": 10},
        "b": {"policy": "fcfs", "median_goodput_req_s": 100.0, "completed_min": 10},
        "c": {"policy": "lifo", "median_goodput_req_s": 100.0, "completed_min": 10},
    }
    passed = _acceptance(
        [_result("a", 105.0), _result("b", 104.0), _result("c", 99.0)],
        strongest,
    )
    assert passed["passed"] is True
    failed = _acceptance(
        [_result("a", 106.0), _result("b", 104.0), _result("c", 97.0)],
        strongest,
    )
    assert failed["passed"] is False


def test_ablation_requires_real_nonfallback_execution_and_equal_completion():
    candidate = [_result(name, score) for name, score in zip("abc", (10.0, 12.0, 14.0))]
    controls = [_result(name, 8.0) for name in "abc"]
    evidence = _ablation_evidence(candidate, controls)
    assert evidence["supported"] is True
    assert evidence["rows"][0]["control_goodput_req_s"] == 8.0
    assert "gain_vs_control_pct" in evidence["rows"][0]
    assert "gain_from_decode_awareness_pct" not in evidence["rows"][0]

    controls[1].fallbacks = controls[1].invocations
    evidence = _ablation_evidence(candidate, controls)
    assert evidence["valid_execution"] is False
    assert evidence["supported"] is False


def test_template_seed_ablation_changes_exactly_one_recipe_field():
    import vllm_evolve.engine.local_frontier_evolve as module

    source = module._build_policy(
        "SELECTED", order="cache_first", gate="none", switch=True,
        pressure_order="total_work",
    )
    name, control_source, label = _template_seed_ablation(source)
    original = module._parse_struct(source)
    control = module._parse_struct(control_source)
    changed = [field for field in original if original[field] != control[field]]
    assert name == "seed_remove_switch"
    assert changed == ["switch"]
    assert "all other fields frozen" in label
    assert "preemption" not in label


def _research(tmp_path):
    return offline_compiler().compile(
        target="scheduling",
        spec={"metric": "goodput_req_s", "direction": "max"},
        diagnosis={"bottleneck": "queue"},
        environment={"execution": "test"},
        out_dir=tmp_path / "compiled_research",
    ).context


def _fake_scenarios(_path, *, splits, **_kwargs):
    scenarios = [
        Scenario(
            name=f"scenario_{index}",
            split="test" if splits == ("test",) else "validation",
            source_kind="official_burstgpt",
            rows=[{"request_id": index}],
            slo_ttft_ms=200.0,
            max_num_seqs=4,
        )
        for index in range(3)
    ]
    names = ["test"] if splits == ("test",) else ["train", "validation"]
    return scenarios, {
        "materialized_splits": names,
        "fragments": {name: {"sha256": name} for name in names},
    }


def _fake_measurement(policy_name, policy_path, scenario, **_kwargs):
    score = 100.0
    if policy_path is not None and "selection" in Path(policy_path).parts:
        score = _fake_measurement.candidate_score
    return PolicyScenarioResult(
        policy=policy_name,
        policy_sha256="a" * 64,
        scenario=scenario.name,
        split=scenario.split,
        source_kind=scenario.source_kind,
        per_seed=[],
        median_goodput_req_s=score,
        median_ttft_p50_ms=10.0,
        median_ttft_p99_ms=20.0,
        median_throughput_req_s=score,
        completed_min=10,
        marker_ok=True,
        invocations=10,
        fallbacks=0,
        defers=1,
        forced=0,
    )


_fake_measurement.candidate_score = 105.0


@pytest.mark.parametrize(
    ("candidate_score", "expected_stage", "terminal_dir"),
    [
        (105.0, WorkflowStage.SIM_WINNER, "sim_winner"),
        (101.0, WorkflowStage.NO_WINNER, "no_winner"),
    ],
)
def test_terminal_directory_matches_acceptance(
    tmp_path, monkeypatch, candidate_score, expected_stage, terminal_dir
):
    import vllm_evolve.engine.local_frontier_evolve as module

    burst = tmp_path / "BurstGPT_1.csv"
    burst.write_text("timestamp,model\n0,0\n", encoding="utf-8")
    source = module._build_policy("SELECTED", order="cache_first", switch=True)
    source_path = tmp_path / "selected.py"
    source_path.write_text(source, encoding="utf-8")
    manifest_path = tmp_path / "selected.manifest.json"
    manifest_path.write_text(
        '{"ablation_plan":["disable only the queue-pressure regime switch"]}',
        encoding="utf-8",
    )
    sha = module._sha_text(source)
    winner = LineageEntry(
        sha=sha,
        generation=0,
        score=5.0,
        source_path=str(source_path),
        verify_ok=True,
        operator="seed",
        candidate_manifest_path=str(manifest_path),
    )
    monkeypatch.setattr(module, "build_scenarios", _fake_scenarios)
    _fake_measurement.candidate_score = candidate_score
    monkeypatch.setattr(module, "evaluate_policy", _fake_measurement)
    monkeypatch.setattr(
        module,
        "run_evolution",
        lambda *_args, **_kwargs: EvolutionResult(
            winner=winner,
            archive=[winner],
            generations_completed=1,
            evals_used=1,
            author_kind="template",
        ),
    )

    out = tmp_path / "run"
    result = run_local_frontier_evolution(
        burstgpt_path=burst,
        out_dir=out,
        seeds=[0, 1, 2],
        research_context=_research(tmp_path).to_dict(),
        generations=1,
        population=1,
        max_total_evals=1,
    )
    state = __import__("json").loads((out / "state.json").read_text(encoding="utf-8"))
    assert WorkflowStage(state["stage"]) == expected_stage
    assert (out / terminal_dir / "evidence.json").is_file()
    assert (out / "sim_winner").is_dir() is (expected_stage == WorkflowStage.SIM_WINNER)
    if expected_stage == WorkflowStage.SIM_WINNER:
        assert result["winner_path"] is not None
    else:
        assert result["winner_path"] is None

    resumed = run_local_frontier_evolution(
        burstgpt_path=burst,
        out_dir=out,
        seeds=[0, 1, 2],
        research_context=_research(tmp_path).to_dict(),
        generations=1,
        population=1,
        max_total_evals=1,
        resume=True,
    )
    assert resumed["verdict"] == result["verdict"]
