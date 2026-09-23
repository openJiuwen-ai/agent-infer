"""Auto Research Expert Layer: offline, deterministic, proposal-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_evolve.knowledge.compiler import ResearchCompiler, load_research_context
from vllm_evolve.knowledge.providers import (
    CuratedCorpusProvider,
    ProviderResult,
    ResearchRequest,
    build_upstream_capability_matrix,
)
from vllm_evolve.knowledge.schemas import (
    MechanismCard,
    ResearchContext,
    ResearchSource,
)


def _source(source_id: str, *, year: int, freshness: str, title: str | None = None):
    return ResearchSource(
        source_id=source_id,
        title=title or source_id,
        authors=["A"],
        year=year,
        date=str(year),
        source_kind="paper",
        canonical_url=f"https://example.test/{source_id}",
        identifier=source_id,
        freshness=freshness,
        retrieval_time="2026-01-01T00:00:00Z",
        primary_source=True,
        summary="LLM scheduling TTFT KV cache goodput",
        citation_status="verified_primary",
        provider="test",
    )


def _card(mechanism_id: str, source_id: str):
    return MechanismCard(
        mechanism_id=mechanism_id,
        name=mechanism_id,
        source_ids=[source_id],
        problem="queueing delay",
        core_mechanism="change admission lifecycle",
        assumptions=["queue saturated"],
        applicable_workloads=["bursty"],
        expected_improved_metrics=["goodput_req_s"],
        implementation_hooks=["targets/mock/schedule"],
        required_controls=["disabled mechanism"],
        structural_change="add an admission branch",
        template_recipe={"order": "fcfs", "gate": "burst_tail"},
    )


class _Provider:
    name = "test"

    def __init__(self, result):
        self.result = result

    def fetch(self, request: ResearchRequest):
        return self.result


def _compile(tmp_path, providers, *, target="scheduling", internal=None):
    return ResearchCompiler(
        providers=providers,
        source_limit=4,
        min_classic=1,
        min_recent=1,
    ).compile(
        target=target,
        spec={"metric": "goodput_req_s", "raw_intent": "improve TTFT goodput"},
        diagnosis={"bottleneck": "scheduling_queue"},
        environment={"vllm_version": "0.21.0", "workloads": ["BurstGPT"]},
        out_dir=tmp_path,
        internal_evidence=internal or {},
    )


def test_research_schema_roundtrip_and_snapshot_hash():
    source = _source("paper:a", year=2022, freshness="classic")
    context = ResearchContext(
        target="scheduling",
        goal={"metric": "goodput_req_s"},
        diagnosis={"bottleneck": "queue"},
        environment={"gpu": "L20X"},
        query_plan=["query"],
        sources=[source],
        mechanism_cards=[_card("bounded_admission", source.source_id)],
        created_at="one wall clock",
    )
    same = ResearchContext.from_dict(context.to_dict())
    assert same.to_dict() == context.to_dict()
    assert same.snapshot_hash == context.snapshot_hash
    same.created_at = "another wall clock"
    assert same.seal() == context.snapshot_hash
    tampered = ResearchContext.from_dict(context.to_dict())
    tampered.goal["metric"] = "fabricated"
    assert not tampered.validate_snapshot_integrity()["ok"]


def test_dedup_and_classic_recent_quota(tmp_path):
    classic = _source("paper:classic", year=2022, freshness="classic")
    recent = _source("paper:recent", year=2026, freshness="recent")
    duplicate = _source("paper:recent-copy", year=2026, freshness="recent", title=recent.title)
    duplicate.identifier = recent.identifier
    result = _compile(
        tmp_path,
        [
            _Provider(
                ProviderResult(
                    provider="test",
                    sources=[classic, recent, duplicate],
                    mechanisms=[
                        _card("classic_mechanism", classic.source_id),
                        _card("recent_mechanism", recent.source_id),
                    ],
                )
            )
        ],
    )
    ids = [source.source_id for source in result.context.sources]
    assert len(ids) == 2
    assert {source.freshness for source in result.context.sources} == {"classic", "recent"}


def test_online_failure_is_explicit_curated_fallback(tmp_path):
    curated_source = _source("paper:curated", year=2022, freshness="classic")
    result = _compile(
        tmp_path,
        [
            _Provider(
                ProviderResult(
                    provider="curated_corpus",
                    sources=[curated_source],
                    mechanisms=[_card("curated_mechanism", curated_source.source_id)],
                )
            ),
            _Provider(
                ProviderResult(
                    provider="arxiv_primary",
                    status="online_unavailable_fallback_curated",
                    detail="network blocked",
                )
            ),
        ],
    )
    assert result.context.outcome_class == "research_compiled_offline_fallback"
    assert any(
        row["status"] == "online_unavailable_fallback_curated"
        for row in result.context.provider_status
    )


def test_snapshot_is_frozen_and_reused(tmp_path):
    source = _source("paper:a", year=2022, freshness="classic")
    compiler = ResearchCompiler(
        providers=[
            _Provider(
                ProviderResult(
                    provider="curated_corpus",
                    sources=[source],
                    mechanisms=[_card("bounded", source.source_id)],
                )
            )
        ],
        source_limit=3,
        min_classic=1,
        min_recent=0,
    )
    kwargs = dict(
        target="scheduling",
        spec={"metric": "goodput_req_s"},
        diagnosis={"bottleneck": "queue"},
        environment={"gpu": "test"},
        out_dir=tmp_path,
    )
    first = compiler.compile(**kwargs)
    second = compiler.compile(**kwargs)
    assert not first.reused_frozen_snapshot
    assert second.reused_frozen_snapshot
    assert second.context.snapshot_hash == first.context.snapshot_hash
    assert load_research_context(tmp_path / "research_snapshot.json").snapshot_hash == (
        first.context.snapshot_hash
    )


def test_invalid_source_citation_is_detected():
    source = _source("paper:a", year=2022, freshness="classic")
    context = ResearchContext(
        target="scheduling",
        goal={},
        diagnosis={},
        environment={},
        query_plan=[],
        sources=[source],
        mechanism_cards=[_card("bad", "paper:missing")],
    )
    assert not context.validate_citations()["ok"]
    assert context.validate_citations()["missing_source_ids"] == ["paper:missing"]


def test_falsified_hypothesis_reaches_next_portfolio(tmp_path):
    source = _source("paper:a", year=2022, freshness="classic")
    result = _compile(
        tmp_path,
        [
            _Provider(
                ProviderResult(
                    provider="curated_corpus",
                    sources=[source],
                    mechanisms=[_card("bounded_admission", source.source_id)],
                )
            )
        ],
        internal={
            "lessons": [{"lesson_id": 1, "conclusion": "prior attempt"}],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "bounded admission fails in this regime",
                    "verdict": "falsified",
                }
            ],
        },
    )
    assert result.context.hypotheses["falsified"][0]["hypothesis_id"] == "h1"
    assert (
        result.context.recommended_hypothesis_portfolio[0]["status"]
        == "previously_falsified_review_before_reuse"
    )


def test_non_scheduling_curated_target_is_not_hardcoded(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = _source("paper:mock", year=2026, freshness="recent")
    payload = {
        "metadata": {"target": "mock_inference"},
        "sources": [source.to_dict()],
        "mechanism_cards": [_card("mock_mechanism", source.source_id).to_dict()],
    }
    (corpus / "mock_inference.json").write_text(json.dumps(payload), encoding="utf-8")
    result = ResearchCompiler(
        providers=[CuratedCorpusProvider(corpus)],
        min_classic=0,
        min_recent=1,
    ).compile(
        target="mock_inference",
        spec={"metric": "latency"},
        diagnosis={"bottleneck": "mock"},
        environment={},
        out_dir=tmp_path / "run",
    )
    assert result.context.target == "mock_inference"
    assert result.context.mechanism_cards[0].mechanism_id == "mock_mechanism"


def test_all_required_research_artifacts_are_written(tmp_path):
    classic = _source("paper:a", year=2022, freshness="classic")
    recent = _source("paper:b", year=2026, freshness="recent")
    _compile(
        tmp_path,
        [
            _Provider(
                ProviderResult(
                    provider="curated_corpus",
                    sources=[classic, recent],
                    mechanisms=[
                        _card("m-classic", classic.source_id),
                        _card("m-recent", recent.source_id),
                    ],
                )
            )
        ],
    )
    assert {
        "research_query.json",
        "sources.json",
        "source_manifest.json",
        "mechanism_cards.json",
        "expert_brief.json",
        "research_snapshot.json",
    }.issubset(path.name for path in Path(tmp_path).iterdir())
    expert = json.loads((tmp_path / "expert_brief.json").read_text(encoding="utf-8"))
    assert [item["source_id"] for item in expert["selected_classic_sources"]] == ["paper:a"]
    assert [item["source_id"] for item in expert["selected_recent_sources"]] == ["paper:b"]


def test_live_required_mode_fails_closed_without_current_upstream(tmp_path):
    compiler = ResearchCompiler(
        providers=[
            _Provider(
                ProviderResult(
                    provider="github_vllm_upstream",
                    status="live_upstream_unavailable",
                    detail="network blocked",
                )
            )
        ],
        require_live=True,
        min_classic=0,
        min_recent=0,
    )
    with pytest.raises(RuntimeError, match="live-required"):
        compiler.compile(
            target="scheduling",
            spec={"metric": "goodput_req_s"},
            diagnosis={"bottleneck": "queue"},
            environment={},
            out_dir=tmp_path,
        )


def test_live_required_mode_records_live_manifest(tmp_path):
    source = _source("github:vllm-main:abc", year=2026, freshness="recent")
    source.provider = "github_vllm_upstream"
    result = ResearchCompiler(
        providers=[
            _Provider(
                ProviderResult(
                    provider="github_vllm_upstream",
                    sources=[source],
                    status="ok",
                    corpus_metadata={
                        "upstream_capability_matrix": {
                            "capabilities": [
                                {
                                    "mechanism_id": "continuous_batching",
                                    "status": "integrated",
                                }
                            ]
                        }
                    },
                )
            )
        ],
        require_live=True,
        min_classic=0,
        min_recent=1,
    ).compile(
        target="scheduling",
        spec={"metric": "goodput_req_s"},
        diagnosis={"bottleneck": "queue"},
        environment={},
        out_dir=tmp_path,
    )
    assert result.context.outcome_class == "research_compiled_live_required"
    manifest = json.loads(
        (tmp_path / "live_source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["required"] is True
    assert manifest["retrieved_primary_sources"][0]["source_id"] == source.source_id


def test_live_gap_card_requires_upstream_delta_fields():
    card = _card("gap", "paper:a")
    assert "upstream_gap" in card.validate_live_gap()
    card.upstream_gap = "release lacks pressure-aware admission"
    card.upstream_symbols_checked = ["vllm/v1/core/sched/scheduler.py:Scheduler.schedule"]
    card.current_vllm_behavior = "FCFS admission consumes the next fitting request"
    card.candidate_delta = "bounded defer lifecycle with a starvation cap"
    card.why_not_already_integrated = "no equivalent branch exists in the pinned release"
    card.required_runtime_signal = "non-zero waiting queue and KV pressure"
    card.mechanism_trigger = "queue pressure plus low free KV blocks"
    card.expected_action_counters = ["deferred_request_actions"]
    card.minimum_meaningful_ablation = "remove only the defer branch"
    card.version_constraints = ["vLLM==0.21.0"]
    assert card.validate_live_gap() == []


def test_upstream_matrix_distinguishes_installed_main_and_surface_gaps():
    installed = {
        "scheduler.py": "def schedule(self):\n self.waiting = []\n self.running = []\n"
    }
    main = {
        **installed,
        "kv_cache_manager.py": "watermark_blocks = 8\n",
    }
    result = build_upstream_capability_matrix(
        installed_ref="installed",
        main_ref="main",
        installed_files=installed,
        main_files=main,
        environment={
            "strong_baseline": {"continuous_batching": True},
            "workload_signals": [],
        },
    )
    by_id = {item["mechanism_id"]: item for item in result["capabilities"]}
    assert by_id["continuous_batching"]["status"] == "integrated"
    assert by_id["continuous_batching"]["strong_baseline_enabled"] is True
    assert by_id["kv_admission_watermark"]["status"] == "partially_integrated"
    assert by_id["allocation_failure_fit_bypass"]["status"] == "missing"
    assert (
        by_id["tenant_fair_scheduling"]["status"]
        == "incompatible_with_current_surface"
    )


def test_live_required_excludes_legacy_cards_without_gap_proof(tmp_path):
    source = _source("github:vllm-main:abc", year=2026, freshness="recent")
    source.provider = "github_vllm_upstream"
    result = ResearchCompiler(
        providers=[
            _Provider(
                ProviderResult(
                    provider="github_vllm_upstream",
                    sources=[source],
                    mechanisms=[_card("legacy_sjf", source.source_id)],
                    status="ok",
                    corpus_metadata={
                        "upstream_capability_matrix": {
                            "capabilities": [
                                {
                                    "mechanism_id": "continuous_batching",
                                    "status": "integrated",
                                }
                            ]
                        }
                    },
                )
            )
        ],
        require_live=True,
        min_classic=0,
        min_recent=1,
    ).compile(
        target="scheduling",
        spec={"metric": "goodput_req_s"},
        diagnosis={"bottleneck": "queue"},
        environment={},
        out_dir=tmp_path,
    )
    assert result.context.mechanism_cards == []
    assert result.context.integrated_mechanisms_excluded[-1] == {
        "mechanism_id": "legacy_sjf",
        "reason": "live gap proof is incomplete",
        "missing_fields": _card("legacy_sjf", source.source_id).validate_live_gap(),
    }


def test_cli_accepts_long_inline_environment_json():
    from vllm_evolve.cli.main import _json_object_arg

    payload = {"installed_vllm_commit": "a" * 40, "flags": list(range(100))}
    assert _json_object_arg(json.dumps(payload)) == payload
