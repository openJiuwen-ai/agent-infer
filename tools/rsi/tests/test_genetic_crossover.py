"""Typed two-parent crossover and deterministic operator scheduling."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_evolve.core.schemas import Profile, Spec
from vllm_evolve.engine.authoring import synthesize_candidate_manifest
from vllm_evolve.engine.evolve_loop import _operator_plan, run_evolution
from vllm_evolve.engine.evolve_schemas import AuthorContext, LineageEntry
from vllm_evolve.engine.evolve_target import _build_policy, _parse_struct, template_author_fn
from vllm_evolve.knowledge.compiler import offline_compiler


def _eval(config):
    source = Path(config["policy"]).read_text(encoding="utf-8")
    recipe = _parse_struct(source)
    score = float(
        list(("fcfs", "sjf", "ljf", "lifo", "total_work", "cache_first", "starvation"))
        .index(recipe["order"])
        + (2 if recipe["gate"] != "none" else 0)
        + (1 if recipe["switch"] else 0)
    )
    return Profile(
        config=config,
        metrics={"score": score},
        marker_verified=False,
        source="frontier_sim",
        outcome_class="simulator_nonqualifying",
    )


def test_operator_plan_rejects_invalid_and_self_parents():
    valid_a = LineageEntry(sha="a", generation=0, score=2.0, verify_ok=True)
    valid_b = LineageEntry(sha="b", generation=0, score=1.0, verify_ok=True)
    invalid = LineageEntry(sha="c", generation=0, score=9.0, verify_ok=False)
    duplicate = LineageEntry(sha="a", generation=0, score=7.0, verify_ok=True)

    kind, selected = _operator_plan(1, 0, [valid_a, invalid, duplicate, valid_b])
    assert kind == "crossover"
    assert [parent.sha for parent in selected] == ["a", "b"]
    kind, selected = _operator_plan(1, 1, [valid_a, invalid, duplicate, valid_b])
    assert kind == "mutation" and [parent.sha for parent in selected] == ["b"]


def test_run_emits_real_crossover_and_mutation_children(tmp_path):
    seen = []

    def author(context):
        seen.append(context)
        if context.generation == 0:
            if len(context.peers) == 0:
                return _build_policy(
                    "PARENT-A", order="fcfs", gate="none", switch=True,
                    pressure_order="total_work",
                )
            return _build_policy(
                "PARENT-B", order="sjf", gate="burst_tail", switch=False,
                pressure_order="ljf", dispersion_guard=True,
            )
        return template_author_fn(context)

    author.ve_author_kind = "template"
    result = run_evolution(
        author,
        _eval,
        Spec(metric="score", direction="max"),
        {},
        {"bottleneck": "queue"},
        generations=2,
        population=2,
        repair_limit=1,
        max_total_evals=4,
        variants_dir=str(tmp_path / "variants"),
    )

    later = [context for context in seen if context.generation == 1 and not context.last_errors]
    assert [context.operator["kind"] for context in later] == ["crossover", "mutation"]
    crossover_context = later[0]
    assert len(crossover_context.parents) == 2
    assert len({parent["sha"] for parent in crossover_context.parents}) == 2
    assert all(
        parent["verify_ok"] and parent["score"] is not None
        for parent in crossover_context.parents
    )
    assert all(
        parent["source"] and "candidate_manifest" in parent
        for parent in crossover_context.parents
    )

    crossover = next(
        entry for entry in result.archive if entry.generation == 1 and entry.operator == "crossover"
    )
    manifest = json.loads(Path(crossover.candidate_manifest_path).read_text(encoding="utf-8"))
    assert manifest["operator"] == "crossover"
    assert manifest["parent_sha"] == manifest["parent_shas"][0]
    assert len(set(manifest["parent_shas"])) == 2
    assert set(manifest["inherited_mechanisms"]) == set(manifest["parent_shas"])
    assert set(manifest["inherited_components"]) == set(manifest["parent_shas"])
    assert all(manifest["inherited_components"][parent] for parent in manifest["parent_shas"])
    assert manifest["compatibility_reason"]
    assert manifest["new_control_flow"]
    assert len(manifest["ablation_plan"]) == 2
    assert crossover.sha not in set(crossover.parent_shas)
    assert crossover.author_context_paths
    assert crossover.verify_report_paths
    prompt = json.loads(Path(crossover.author_context_paths[0]).read_text(encoding="utf-8"))
    verify = json.loads(Path(crossover.verify_report_paths[0]).read_text(encoding="utf-8"))
    assert prompt["operator"]["kind"] == "crossover"
    assert verify["l1_l2_executed"] is True and verify["passed"] is True


def test_operator_schedule_is_reproducible(tmp_path):
    def run_once(root):
        counter = [0]

        def author(context):
            if context.generation == 0:
                counter[0] += 1
                return _build_policy(
                    f"SEED-{counter[0]}",
                    order=("fcfs", "sjf", "ljf")[len(context.peers) % 3],
                    gate=("none", "burst_tail", "seq_reserve")[len(context.peers) % 3],
                    switch=len(context.peers) % 2 == 0,
                )
            return template_author_fn(context)

        result = run_evolution(
            author,
            _eval,
            Spec(metric="score", direction="max"),
            {},
            {},
            generations=2,
            population=3,
            repair_limit=0,
            max_total_evals=6,
            variants_dir=str(root),
        )
        return [
            (entry.generation, entry.operator, entry.parent_shas, entry.sha)
            for entry in result.archive
        ]

    assert run_once(tmp_path / "a") == run_once(tmp_path / "b")


def test_template_records_research_as_inspiration_not_implemented_mechanism(tmp_path):
    research = offline_compiler().compile(
        target="scheduling",
        spec={"metric": "goodput_req_s", "direction": "max"},
        diagnosis={"bottleneck": "queue"},
        environment={"execution": "unit"},
        out_dir=tmp_path / "research",
    ).context
    context = AuthorContext(
        spec={"metric": "goodput_req_s", "direction": "max"},
        diagnosis={"bottleneck": "queue"},
        skeleton="def schedule_batch(...): ...",
        research_context=research.to_dict(),
        budget={"evals_remaining": 1, "repair_remaining": 0},
        operator={"kind": "seed", "parent_shas": []},
        author_kind="template",
    )
    source = template_author_fn(context)
    manifest = synthesize_candidate_manifest(
        source=source,
        parent_sha=None,
        context=context,
        research_context=research,
        author_kind="template",
    )
    assert "VE_RESEARCH_INSPIRATION:" in source
    assert "VE_MECHANISM:" not in source
    assert manifest.mechanism_ids == []
    assert manifest.research_inspirations
    assert manifest.required_controls == manifest.ablation_plan
