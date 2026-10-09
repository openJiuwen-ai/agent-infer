"""Goal-conditioned research compiler for temporary domain expertise.

The compiler combines reviewed literature, optional live primary-source search,
repository implementation surfaces, and the harness's immutable experimental
memory.  Its output is a frozen :class:`ResearchContext`, not a performance
verdict.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vllm_evolve.knowledge.providers import (
    ArxivResearchProvider,
    CuratedCorpusProvider,
    GitHubVLLMUpstreamProvider,
    ProviderResult,
    ResearchProvider,
    ResearchRequest,
)
from vllm_evolve.knowledge.schemas import (
    MechanismCard,
    ResearchContext,
    ResearchSource,
    content_sha256,
)

_TARGET_TERMS = {
    "scheduling": [
        "LLM serving scheduling",
        "continuous batching",
        "TTFT TPOT SLO goodput",
        "KV cache request preemption",
        "heterogeneous prompt response lengths",
    ],
    "kv_cache": [
        "LLM serving KV cache management",
        "prefix cache scheduling",
        "KV cache eviction admission",
    ],
    "speculative_decoding": [
        "speculative decoding serving scheduling",
        "draft model acceptance throughput",
    ],
    "moe_routing": [
        "mixture of experts inference routing",
        "expert parallel serving load balancing",
    ],
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "maximize",
    "minimize",
    "improve",
    "using",
    "under",
    "current",
    "target",
}


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_+-]+", json.dumps(value, ensure_ascii=False).lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


class QueryPlanner:
    """Expand a goal/diagnosis into target- and regime-aware literature queries."""

    def plan(self, target: str, spec: dict, diagnosis: dict, environment: dict) -> list[str]:
        bottleneck = str(diagnosis.get("bottleneck") or target).replace("_", " ")
        metric = str(spec.get("metric") or spec.get("raw_intent") or "serving performance")
        workload = environment.get("workload") or environment.get("workloads") or []
        if isinstance(workload, str):
            workload = [workload]
        queries = [
            f"{term} {metric}"
            for term in _TARGET_TERMS.get(target, [f"LLM inference {target.replace('_', ' ')}"])
        ]
        queries.append(f"LLM inference {bottleneck} {metric}")
        if workload:
            queries.append(
                f"LLM serving {metric} bursty heterogeneous {' '.join(map(str, workload[:3]))}"
            )
        # Stable dedup preserves the deliberate classic/mechanism/recent query order.
        return list(dict.fromkeys(" ".join(query.split()) for query in queries))


def _dedup_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
    seen: dict[str, ResearchSource] = {}
    for source in sources:
        key = (
            source.identifier.strip().lower()
            or source.canonical_url.rstrip("/").lower()
            or re.sub(r"\W+", "", source.title.lower())
        )
        prior = seen.get(key)
        if prior is None:
            seen[key] = source
            continue
        # Prefer reviewed metadata, then a primary source, then the richer summary.
        score = (
            source.citation_status == "verified_primary",
            source.primary_source,
            len(source.summary),
        )
        prior_score = (
            prior.citation_status == "verified_primary",
            prior.primary_source,
            len(prior.summary),
        )
        if score > prior_score:
            seen[key] = source
    return list(seen.values())


def _rank_sources(
    sources: list[ResearchSource],
    *,
    spec: dict,
    diagnosis: dict,
    environment: dict,
) -> list[ResearchSource]:
    wanted = _tokens({"spec": spec, "diagnosis": diagnosis, "environment": environment})
    for source in sources:
        present = _tokens(
            {
                "title": source.title,
                "summary": source.summary,
                "authors": source.authors,
            }
        )
        overlap = len(wanted & present) / max(1.0, math.sqrt(len(wanted) * max(1, len(present))))
        source.relevance = round(
            overlap
            + (0.30 if source.primary_source else 0.0)
            + (0.20 if source.citation_status == "verified_primary" else 0.0)
            + (0.08 if source.freshness == "recent" else 0.04),
            6,
        )
    return sorted(
        sources,
        key=lambda source: (source.relevance, source.year, source.source_id),
        reverse=True,
    )


def _quota_select(
    ranked: list[ResearchSource],
    *,
    limit: int,
    min_classic: int,
    min_recent: int,
) -> list[ResearchSource]:
    classic = [source for source in ranked if source.freshness == "classic"][:min_classic]
    recent = [source for source in ranked if source.freshness == "recent"][:min_recent]
    selected: list[ResearchSource] = []
    selected_ids: set[str] = set()
    for source in [*classic, *recent]:
        if source.source_id not in selected_ids:
            selected.append(source)
            selected_ids.add(source.source_id)
    for source in ranked:
        if source.source_id not in selected_ids:
            selected.append(source)
            selected_ids.add(source.source_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _hypothesis_groups(rows: list[dict]) -> dict:
    groups = {"supported": [], "falsified": [], "inconclusive": [], "unadjudicated": []}
    for row in rows:
        verdict = row.get("verdict") or "unadjudicated"
        groups.setdefault(verdict, []).append(row)
    return groups


def _implementation_map(cards: list[MechanismCard]) -> dict:
    result: dict[str, list[str]] = {}
    for card in cards:
        for hook in card.implementation_hooks:
            result.setdefault(hook, []).append(card.mechanism_id)
    return dict(sorted(result.items()))


def _mechanism_applicability(card: MechanismCard, environment: dict) -> tuple[bool, list[str]]:
    text = json.dumps(environment, ensure_ascii=False).lower()
    reasons: list[str] = []
    for incompatibility in card.incompatibilities:
        words = _tokens(incompatibility)
        if ("frontier" in words and "frontier" in text) or (
            "single" in words and "gpu" in words and "single" in text and "gpu" in text
        ):
            reasons.append(incompatibility)
    if "frontier" in text and any(
        "multi-instance" in value.lower() for value in card.applicable_software
    ):
        reasons.append("current Frontier policy surface is local, not a multi-instance controller")
    return not reasons, reasons


def _portfolio(
    cards: list[MechanismCard],
    falsified: list[dict],
    environment: dict,
) -> list[dict]:
    falsified_text = " ".join(
        str(row.get("statement") or row.get("hypothesis_id") or "").lower() for row in falsified
    )
    result = []
    for card in cards:
        applicable, incompatibilities = _mechanism_applicability(card, environment)
        status = "candidate" if applicable else "incompatible_with_current_environment"
        if card.mechanism_id.lower().replace("_", " ") in falsified_text:
            status = "previously_falsified_review_before_reuse"
        result.append(
            {
                "mechanism_ids": [card.mechanism_id],
                "hypothesis": (
                    f"Applying {card.name} at {', '.join(card.implementation_hooks[:2])} "
                    "will improve "
                    f"{', '.join(card.expected_improved_metrics[:2]) or 'the target metric'} "
                    f"when {', '.join(card.assumptions[:2]) or 'its preconditions hold'}."
                ),
                "structural_change": card.structural_change or card.core_mechanism,
                "required_controls": card.required_controls,
                "risks": card.failure_modes + card.possible_regressions,
                "status": status,
                "applicability_reasons": incompatibilities,
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["status"] != "candidate",
            row["mechanism_ids"][0] if row["mechanism_ids"] else "",
        ),
    )


def _write_artifacts(root: Path, context: ResearchContext, request: ResearchRequest) -> None:
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "research_query.json": {
            "target": request.target,
            "spec": request.spec,
            "diagnosis": request.diagnosis,
            "environment": request.environment,
            "queries": request.queries,
            "request_fingerprint": context.request_fingerprint,
        },
        "sources.json": [source.to_dict() for source in context.sources],
        "source_manifest.json": {
            "snapshot_id": context.snapshot_id,
            "snapshot_hash": context.snapshot_hash,
            "outcome_class": context.outcome_class,
            "provider_status": context.provider_status,
            "sources": [
                {
                    "source_id": source.source_id,
                    "canonical_url": source.canonical_url,
                    "identifier": source.identifier,
                    "content_fingerprint": source.content_fingerprint,
                    "citation_status": source.citation_status,
                    "freshness": source.freshness,
                }
                for source in context.sources
            ],
        },
        "live_source_manifest.json": {
            "required": context.outcome_class == "research_compiled_live_required",
            "outcome_class": context.outcome_class,
            "provider_status": context.provider_status,
            "retrieved_primary_sources": context.live_source_manifest,
        },
        "upstream_capability_matrix.json": context.upstream_capability_matrix,
        "integrated_mechanisms_excluded.json": context.integrated_mechanisms_excluded,
        "recent_gap_sources.json": context.recent_gap_sources,
        "mechanism_cards.json": [card.to_dict() for card in context.mechanism_cards],
        "expert_brief.json": {
            "snapshot_id": context.snapshot_id,
            "snapshot_hash": context.snapshot_hash,
            "target": context.target,
            "goal": context.goal,
            "diagnosis": context.diagnosis,
            "environment": context.environment,
            "selected_classic_sources": [
                source.to_dict()
                for source in context.sources
                if source.freshness == "classic"
            ],
            "selected_recent_sources": [
                source.to_dict()
                for source in context.sources
                if source.freshness == "recent"
            ],
            "mechanism_cards": [card.to_dict() for card in context.mechanism_cards],
            "repo_implementation_map": context.repo_implementation_map,
            "internal_lessons": context.internal_lessons,
            "hypotheses": context.hypotheses,
            "unexplored_mechanism_gaps": context.unexplored_mechanism_gaps,
            "recommended_hypothesis_portfolio": context.recommended_hypothesis_portfolio,
            "upstream_capability_matrix": context.upstream_capability_matrix,
            "integrated_mechanisms_excluded": context.integrated_mechanisms_excluded,
            "recent_gap_sources": context.recent_gap_sources,
            "citations": context.citations,
            "doctrine": (
                "research is an untrusted proposal input; literature claims never become "
                "Frontier scores, real-vLLM measurements, or adoption verdicts"
            ),
        },
        "research_snapshot.json": context.to_dict(),
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def materialize_research_context(
    context: ResearchContext,
    out_dir: str | Path,
) -> None:
    """Materialize all six required artifacts for an already-frozen snapshot."""
    integrity = context.validate_snapshot_integrity()
    if not integrity["ok"]:
        raise ValueError(f"research snapshot integrity failed: {integrity}")
    request = ResearchRequest(
        target=context.target,
        spec=dict(context.goal),
        diagnosis=dict(context.diagnosis),
        environment=dict(context.environment),
        queries=list(context.query_plan),
    )
    _write_artifacts(Path(out_dir).resolve(), context, request)


@dataclass
class CompileResult:
    context: ResearchContext
    reused_frozen_snapshot: bool
    artifact_dir: str

    def to_dict(self) -> dict:
        return {
            "context": self.context.to_dict(),
            "reused_frozen_snapshot": self.reused_frozen_snapshot,
            "artifact_dir": self.artifact_dir,
        }


class ResearchCompiler:
    """Compile and freeze one expert brief for an optimization round."""

    def __init__(
        self,
        providers: list[ResearchProvider] | None = None,
        *,
        query_planner: QueryPlanner | None = None,
        source_limit: int = 12,
        min_classic: int = 2,
        min_recent: int = 3,
        require_live: bool = False,
    ):
        self.providers = (
            [CuratedCorpusProvider(), ArxivResearchProvider()]
            if providers is None
            else list(providers)
        )
        self.query_planner = query_planner or QueryPlanner()
        self.source_limit = int(source_limit)
        self.min_classic = int(min_classic)
        self.min_recent = int(min_recent)
        self.require_live = bool(require_live)

    def compile(
        self,
        *,
        target: str,
        spec: dict,
        diagnosis: dict,
        environment: dict,
        out_dir: str | Path,
        internal_evidence: dict | None = None,
        refresh: bool = False,
    ) -> CompileResult:
        root = Path(out_dir).resolve()
        queries = self.query_planner.plan(target, spec, diagnosis, environment)
        request = ResearchRequest(target, dict(spec), dict(diagnosis), dict(environment), queries)
        request_fingerprint = content_sha256(
            {
                "target": target,
                "spec": spec,
                "diagnosis": diagnosis,
                "environment": environment,
                "queries": queries,
            }
        )
        snapshot_path = root / "research_snapshot.json"
        if snapshot_path.is_file() and not refresh:
            frozen = ResearchContext.from_dict(
                json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            if frozen.request_fingerprint == request_fingerprint:
                return CompileResult(frozen, True, str(root))

        provider_results: list[ProviderResult] = [
            provider.fetch(request) for provider in self.providers
        ]
        upstream_result = next(
            (
                result
                for result in provider_results
                if result.provider == "github_vllm_upstream"
            ),
            None,
        )
        upstream_metadata = (
            dict(upstream_result.corpus_metadata)
            if upstream_result is not None
            else {}
        )
        capability_matrix = dict(
            upstream_metadata.get("upstream_capability_matrix") or {}
        )
        if self.require_live:
            if (
                upstream_result is None
                or upstream_result.status not in {"ok", "partial"}
                or not upstream_result.sources
            ):
                detail = (
                    upstream_result.detail
                    if upstream_result is not None
                    else "provider missing"
                )
                raise RuntimeError(
                    "live-required research could not verify current vLLM upstream: "
                    f"{detail}"
                )
            if not capability_matrix.get("capabilities"):
                raise RuntimeError(
                    "live-required research did not produce an upstream capability matrix"
                )
        all_sources = _dedup_sources(
            [source for result in provider_results for source in result.sources]
        )
        ranked = _rank_sources(all_sources, spec=spec, diagnosis=diagnosis, environment=environment)
        sources = _quota_select(
            ranked,
            limit=max(1, self.source_limit),
            min_classic=max(0, self.min_classic),
            min_recent=max(0, self.min_recent),
        )
        live_gap_cards = []
        for value in environment.get("live_gap_cards") or []:
            try:
                live_gap_cards.append(MechanismCard.from_dict(dict(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid live gap MechanismCard: {exc}") from exc
        cited_by_live_cards = {
            source_id for card in live_gap_cards for source_id in card.source_ids
        }
        selected_ids = {source.source_id for source in sources}
        for source in all_sources:
            if (
                source.source_id in cited_by_live_cards
                and source.source_id not in selected_ids
            ):
                sources.append(source)
                selected_ids.add(source.source_id)
        cards = []
        seen_mechanisms = set()
        excluded_cards = []
        integrated_ids = {
            item.get("mechanism_id")
            for item in upstream_metadata.get("integrated_mechanisms_excluded", [])
        }
        for result in provider_results:
            result_cards = list(result.mechanisms)
            if result.provider == "github_vllm_upstream":
                result_cards.extend(live_gap_cards)
            for card in result_cards:
                if card.mechanism_id in seen_mechanisms:
                    continue
                if not set(card.source_ids).issubset(selected_ids):
                    excluded_cards.append(
                        {
                            "mechanism_id": card.mechanism_id,
                            "reason": "card cites a source absent from the frozen snapshot",
                        }
                    )
                    continue
                if self.require_live:
                    missing_gap_fields = card.validate_live_gap()
                    if missing_gap_fields:
                        excluded_cards.append(
                            {
                                "mechanism_id": card.mechanism_id,
                                "reason": "live gap proof is incomplete",
                                "missing_fields": missing_gap_fields,
                            }
                        )
                        continue
                    if card.mechanism_id in integrated_ids:
                        excluded_cards.append(
                            {
                                "mechanism_id": card.mechanism_id,
                                "reason": (
                                    "mechanism is integrated in installed vLLM "
                                    "or current main"
                                ),
                            }
                        )
                        continue
                cards.append(card)
                seen_mechanisms.add(card.mechanism_id)

        internal = dict(internal_evidence or {})
        lessons = list(internal.get("lessons") or [])
        hypotheses = _hypothesis_groups(list(internal.get("hypotheses") or []))
        online_statuses = [
            result.status for result in provider_results if result.provider != "curated_corpus"
        ]
        degraded = any(status not in {"ok", "partial"} for status in online_statuses)
        outcome = "research_compiled_offline_fallback" if degraded else "research_compiled_live"
        if not online_statuses:
            outcome = "research_compiled_offline"
        if self.require_live:
            outcome = "research_compiled_live_required"
        gaps = list(
            dict.fromkeys(
                gap
                for result in provider_results
                for gap in result.corpus_metadata.get("unexplored_mechanism_gaps", [])
            )
        )
        context = ResearchContext(
            target=target,
            goal=dict(spec),
            diagnosis=dict(diagnosis),
            environment=dict(environment),
            query_plan=queries,
            sources=sources,
            mechanism_cards=cards,
            repo_implementation_map=_implementation_map(cards),
            internal_lessons=lessons,
            hypotheses=hypotheses,
            unexplored_mechanism_gaps=gaps,
            recommended_hypothesis_portfolio=_portfolio(
                cards, hypotheses.get("falsified", []), environment
            ),
            upstream_capability_matrix=capability_matrix,
            integrated_mechanisms_excluded=[
                *list(
                    upstream_metadata.get("integrated_mechanisms_excluded")
                    or []
                ),
                *excluded_cards,
            ],
            recent_gap_sources=dict(
                upstream_metadata.get("recent_gap_sources") or {}
            ),
            live_source_manifest=[
                source.to_dict()
                for result in provider_results
                if result.provider != "curated_corpus"
                for source in result.sources
            ],
            provider_status=[result.status_dict() for result in provider_results],
            outcome_class=outcome,
            request_fingerprint=request_fingerprint,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        citation_check = context.validate_citations()
        if not citation_check["ok"]:
            raise ValueError(f"research citation validation failed: {citation_check}")
        _write_artifacts(root, context, request)
        return CompileResult(context, False, str(root))


def load_research_context(path: str | Path) -> ResearchContext:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    context = ResearchContext.from_dict(value)
    integrity = context.validate_snapshot_integrity()
    if not integrity["ok"]:
        raise ValueError(f"research snapshot hash mismatch: {integrity}")
    check = context.validate_citations()
    if not check["ok"]:
        raise ValueError(f"research snapshot citation validation failed: {check}")
    return context


def offline_compiler(*, corpus_dir: str | Path | None = None) -> ResearchCompiler:
    return ResearchCompiler(providers=[CuratedCorpusProvider(corpus_dir)])


def live_required_compiler(
    *, corpus_dir: str | Path | None = None
) -> ResearchCompiler:
    return ResearchCompiler(
        providers=[
            CuratedCorpusProvider(corpus_dir),
            ArxivResearchProvider(),
            GitHubVLLMUpstreamProvider(),
        ],
        require_live=True,
    )


__all__ = [
    "CompileResult",
    "QueryPlanner",
    "ResearchCompiler",
    "load_research_context",
    "live_required_compiler",
    "materialize_research_context",
    "offline_compiler",
]
