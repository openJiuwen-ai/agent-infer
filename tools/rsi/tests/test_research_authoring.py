"""Research context reaches authors and candidate manifests bind real lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_evolve.core.schemas import Profile, Spec
from vllm_evolve.engine.authoring import (
    export_author_bundle,
    import_author_submission,
)
from vllm_evolve.engine.evolve_loop import run_evolution
from vllm_evolve.engine.evolve_schemas import AuthorContext, prompt_source_author
from vllm_evolve.engine.evolve_target import template_author_fn
from vllm_evolve.knowledge.compiler import offline_compiler


def _research(tmp_path):
    return (
        offline_compiler()
        .compile(
            target="scheduling",
            spec={"metric": "tok_s", "direction": "max"},
            diagnosis={"bottleneck": "scheduling_queue"},
            environment={"vllm_version": "0.21.0", "workloads": ["BurstGPT"]},
            out_dir=tmp_path / "research",
        )
        .context
    )


def _eval(config):
    return Profile(
        config=config,
        metrics={"tok_s": 1.0},
        marker_verified=False,
        source="frontier_sim",
        outcome_class="simulator_nonqualifying",
    )


def test_author_prompt_contains_full_research_context(tmp_path):
    research = _research(tmp_path)
    seen = []

    def complete(prompt):
        seen.append(json.loads(prompt))
        return "source"

    author = prompt_source_author(complete, require_manifest=False)
    context = AuthorContext(
        research_context=research.to_dict(),
        author_kind="codex",
    )
    assert author(context) == "source"
    assert seen[0]["research_context"]["snapshot_hash"] == research.snapshot_hash
    assert seen[0]["research_context"]["mechanism_cards"]
    assert seen[0]["author_kind"] == "codex"


def test_explicit_codex_path_requires_manifest():
    author = prompt_source_author(lambda _prompt: "source", require_manifest=True)
    with pytest.raises(ValueError, match="manifest"):
        author(AuthorContext(author_kind="codex"))


def test_template_fallback_is_recorded_and_manifests_are_bound(tmp_path):
    research = _research(tmp_path)
    result = run_evolution(
        template_author_fn,
        _eval,
        Spec(metric="tok_s", direction="max"),
        {"profile": "throughput"},
        {"bottleneck": "scheduling_queue"},
        generations=1,
        population=2,
        repair_limit=0,
        variants_dir=str(tmp_path / "variants"),
        research_context=research.to_dict(),
    )
    assert result.author_kind == "template"
    assert result.research_snapshot_hash == research.snapshot_hash
    assert len(result.archive) == 2
    mechanism_families = set()
    for entry in result.archive:
        manifest = json.loads(Path(entry.candidate_manifest_path).read_text(encoding="utf-8"))
        source = Path(entry.source_path).read_text(encoding="utf-8")
        assert manifest["candidate_sha"] == hashlib.sha256(source.encode()).hexdigest()
        assert manifest["research_snapshot_hash"] == research.snapshot_hash
        assert manifest["author_kind"] == "template"
        assert manifest["proposal_only"] is True
        assert "fallback proxy" in manifest["structural_change"]
        assert "does not claim" in manifest["hypothesis"]
        assert entry.marker_verified is False
        assert manifest["mechanism_ids"] == []
        mechanism_families.update(manifest["research_inspirations"])
    assert len(mechanism_families) == 2


def test_export_import_seam_rejects_unknown_mechanism(tmp_path):
    research = _research(tmp_path)
    context = AuthorContext(
        spec={"metric": "tok_s"},
        diagnosis={"bottleneck": "queue"},
        skeleton="def schedule_batch(): ...",
        research_context=research.to_dict(),
        author_kind="codex",
    )
    bundle = export_author_bundle(context, tmp_path / "bundle")
    assert Path(bundle["prompt_path"]).is_file()
    source = "def schedule_batch():\n    return None\n"
    source_path = tmp_path / "candidate.py"
    source_path.write_text(source, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_sha": "",
                "parent_sha": None,
                "mechanism_ids": ["fabricated_mechanism"],
                "hypothesis": "test",
                "structural_change": "branch",
                "affected_symbols": [],
                "expected_gain_regimes": [],
                "expected_neutral_regimes": [],
                "expected_regression_regimes": [],
                "risks": [],
                "required_controls": [],
                "parameter_only": False,
                "research_snapshot_hash": "",
                "author_kind": "codex",
                "generation": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown mechanism"):
        import_author_submission(
            source_path=source_path,
            manifest_path=manifest_path,
            context=context,
        )
    proposed = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposed["mechanism_ids"] = [research.mechanism_cards[0].mechanism_id]
    proposed["source_citations"] = []
    manifest_path.write_text(json.dumps(proposed), encoding="utf-8")
    with pytest.raises(ValueError, match="missing source citations"):
        import_author_submission(
            source_path=source_path,
            manifest_path=manifest_path,
            context=context,
        )


def test_autopt_runs_research_after_diagnosis_and_before_author(tmp_path):
    from vllm_evolve.engine.orchestrate import run_autopt

    research = _research(tmp_path)
    events = []

    def research_fn(**kwargs):
        events.append(("research", kwargs["diagnosis"]["bottleneck"]))
        return research

    def author(ctx):
        events.append(("author", ctx.research_context["snapshot_hash"]))
        return template_author_fn(ctx)

    author.ve_author_kind = "template"

    def eval_fn(config):
        if config.get("policy"):
            return Profile(
                config=config,
                metrics={"tok_s": 10.0},
                per_seed={"tok_s": [10.0]},
                vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0},
                gpu={"duty_cycle": 0.7},
                marker_verified=False,
                source="frontier_sim",
                outcome_class="simulator_nonqualifying",
            )
        return Profile(
            config=config,
            metrics={"tok_s": 5.0},
            per_seed={"tok_s": [5.0]},
            vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0},
            gpu={"duty_cycle": 0.7},
            marker_verified=False,
            source="frontier_sim",
            outcome_class="simulator_nonqualifying",
        )

    result = run_autopt(
        Spec(metric="tok_s", direction="max"),
        eval_fn=eval_fn,
        author_fn=author,
        research_fn=research_fn,
        evolve_params={
            "generations": 1,
            "population": 1,
            "variants_dir": str(tmp_path / "variants"),
        },
        # The ranked plan deliberately tries the wired max_num_seqs knob before
        # escalating to code:schedule_batch.  The research hook belongs only to
        # that second, code-authoring round.
        max_rounds=2,
    )
    assert events[0] == ("research", "scheduling_queue")
    assert events[1] == ("author", research.snapshot_hash)
    code_round = next(
        item for item in result["rounds"] if item.get("searched_target") == "code:schedule_batch"
    )
    assert code_round["research"]["snapshot_hash"] == research.snapshot_hash
    assert code_round["evolution"]["research_snapshot_hash"] == research.snapshot_hash
    assert code_round["verdict"] == {"verdict": "no_verified_candidate"}
