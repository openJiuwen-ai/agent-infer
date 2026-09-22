"""Typed, auditable contracts for the Auto Research Expert Layer.

Research artifacts live outside the frozen measure/judge core.  They are proposal
inputs: a paper can suggest a mechanism, but only Frontier/real-vLLM measurement
can attach a performance score or an adoption verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON rendering used for hashes and snapshots."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass
class ResearchSource:
    """One primary or internal source; never a measured result by itself."""

    source_id: str
    title: str
    authors: list[str]
    year: int
    date: str
    source_kind: str
    canonical_url: str
    identifier: str = ""
    freshness: str = "recent"  # classic|recent
    retrieval_time: str = ""
    content_fingerprint: str = ""
    primary_source: bool = False
    summary: str = ""
    relevance: float = 0.0
    citation_status: str = "unverified"
    provider: str = ""

    def __post_init__(self) -> None:
        _require_text("source_id", self.source_id)
        _require_text("title", self.title)
        if self.source_kind not in {
            "paper",
            "framework_doc",
            "code",
            "internal_evidence",
        }:
            raise ValueError(f"unsupported source_kind: {self.source_kind}")
        if self.freshness not in {"classic", "recent"}:
            raise ValueError(f"unsupported freshness: {self.freshness}")
        if not self.content_fingerprint:
            self.content_fingerprint = content_sha256(
                {
                    "source_id": self.source_id,
                    "title": self.title,
                    "authors": self.authors,
                    "year": self.year,
                    "date": self.date,
                    "canonical_url": self.canonical_url,
                    "identifier": self.identifier,
                    "summary": self.summary,
                }
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ResearchSource:
        return cls(**dict(value))


@dataclass
class MechanismCard:
    """A source-grounded mechanism translated into this repository's code surface."""

    mechanism_id: str
    name: str
    source_ids: list[str]
    problem: str
    core_mechanism: str
    assumptions: list[str] = field(default_factory=list)
    applicable_workloads: list[str] = field(default_factory=list)
    applicable_hardware: list[str] = field(default_factory=list)
    applicable_software: list[str] = field(default_factory=list)
    expected_improved_metrics: list[str] = field(default_factory=list)
    possible_regressions: list[str] = field(default_factory=list)
    complexity_cost: str = ""
    resource_cost: str = ""
    failure_modes: list[str] = field(default_factory=list)
    implementation_hooks: list[str] = field(default_factory=list)
    required_controls: list[str] = field(default_factory=list)
    incompatibilities: list[str] = field(default_factory=list)
    confidence: str = "medium"
    evidence_type: str = "paper_mechanism"
    structural_change: str = ""
    template_recipe: dict = field(default_factory=dict)
    # Live upstream-gap proof. These stay optional for legacy curated cards;
    # live-required research validates them before a card enters a real-vLLM
    # candidate portfolio.
    upstream_gap: str = ""
    upstream_symbols_checked: list[str] = field(default_factory=list)
    current_vllm_behavior: str = ""
    candidate_delta: str = ""
    why_not_already_integrated: str = ""
    required_runtime_signal: str = ""
    mechanism_trigger: str = ""
    expected_action_counters: list[str] = field(default_factory=list)
    minimum_meaningful_ablation: str = ""
    version_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text("mechanism_id", self.mechanism_id)
        _require_text("name", self.name)
        _require_text("problem", self.problem)
        _require_text("core_mechanism", self.core_mechanism)
        if not self.source_ids:
            raise ValueError("MechanismCard.source_ids must not be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    def validate_live_gap(self) -> list[str]:
        """Reject cards that do not prove a structural delta over current vLLM."""
        required_text = {
            "upstream_gap": self.upstream_gap,
            "current_vllm_behavior": self.current_vllm_behavior,
            "candidate_delta": self.candidate_delta,
            "why_not_already_integrated": self.why_not_already_integrated,
            "required_runtime_signal": self.required_runtime_signal,
            "mechanism_trigger": self.mechanism_trigger,
            "minimum_meaningful_ablation": self.minimum_meaningful_ablation,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if not self.upstream_symbols_checked:
            missing.append("upstream_symbols_checked")
        if not self.expected_action_counters:
            missing.append("expected_action_counters")
        if not self.version_constraints:
            missing.append("version_constraints")
        return missing

    @classmethod
    def from_dict(cls, value: dict) -> MechanismCard:
        return cls(**dict(value))


@dataclass
class ResearchContext:
    """Frozen domain-expert brief consumed by every child in one evolution round."""

    target: str
    goal: dict
    diagnosis: dict
    environment: dict
    query_plan: list[str]
    sources: list[ResearchSource]
    mechanism_cards: list[MechanismCard]
    repo_implementation_map: dict = field(default_factory=dict)
    internal_lessons: list[dict] = field(default_factory=list)
    hypotheses: dict = field(default_factory=dict)
    unexplored_mechanism_gaps: list[str] = field(default_factory=list)
    recommended_hypothesis_portfolio: list[dict] = field(default_factory=list)
    upstream_capability_matrix: dict = field(default_factory=dict)
    integrated_mechanisms_excluded: list[dict] = field(default_factory=list)
    recent_gap_sources: dict = field(default_factory=dict)
    live_source_manifest: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    provider_status: list[dict] = field(default_factory=list)
    outcome_class: str = "research_compiled"
    request_fingerprint: str = ""
    created_at: str = ""
    snapshot_id: str = ""
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        _require_text("target", self.target)
        if not self.request_fingerprint:
            self.request_fingerprint = content_sha256(
                {
                    "target": self.target,
                    "goal": self.goal,
                    "diagnosis": self.diagnosis,
                    "environment": self.environment,
                    "query_plan": self.query_plan,
                }
            )
        if not self.citations:
            self.citations = [f"source:{source.source_id}" for source in self.sources]
        if not self.snapshot_hash:
            self.seal()

    def _hash_payload(self) -> dict:
        value = self.to_dict()
        value.pop("snapshot_hash", None)
        value.pop("snapshot_id", None)
        # Wall-clock time must not make the same frozen evidence non-reproducible.
        value.pop("created_at", None)
        return value

    def seal(self) -> str:
        self.snapshot_hash = content_sha256(self._hash_payload())
        self.snapshot_id = f"research-{self.snapshot_hash[:16]}"
        return self.snapshot_hash

    def validate_snapshot_integrity(self) -> dict:
        actual = content_sha256(self._hash_payload())
        expected_id = f"research-{actual[:16]}"
        return {
            "ok": self.snapshot_hash == actual and self.snapshot_id == expected_id,
            "recorded_hash": self.snapshot_hash,
            "computed_hash": actual,
            "recorded_id": self.snapshot_id,
            "computed_id": expected_id,
        }

    def validate_citations(self) -> dict:
        source_ids = {source.source_id for source in self.sources}
        cited = {
            citation.split(":", 1)[1]
            for citation in self.citations
            if citation.startswith("source:")
        }
        mechanism_refs = {
            source_id for card in self.mechanism_cards for source_id in card.source_ids
        }
        missing = sorted((cited | mechanism_refs) - source_ids)
        invalid_sources = sorted(
            source.source_id
            for source in self.sources
            if source.citation_status
            not in {
                "verified_primary",
                "retrieved_primary",
                "verified_internal",
            }
        )
        return {
            "ok": not missing and not invalid_sources,
            "missing_source_ids": missing,
            "invalid_source_ids": invalid_sources,
            "source_count": len(source_ids),
            "mechanism_count": len(self.mechanism_cards),
        }

    def to_dict(self) -> dict:
        value = asdict(self)
        value["sources"] = [source.to_dict() for source in self.sources]
        value["mechanism_cards"] = [card.to_dict() for card in self.mechanism_cards]
        return value

    @classmethod
    def from_dict(cls, value: dict) -> ResearchContext:
        data = dict(value)
        data["sources"] = [
            item if isinstance(item, ResearchSource) else ResearchSource.from_dict(item)
            for item in data.get("sources", [])
        ]
        data["mechanism_cards"] = [
            item if isinstance(item, MechanismCard) else MechanismCard.from_dict(item)
            for item in data.get("mechanism_cards", [])
        ]
        return cls(**data)


@dataclass
class CandidateManifest:
    """Auditable intent/lineage sidecar for one authored source file."""

    candidate_sha: str
    parent_sha: str | None
    mechanism_ids: list[str]
    hypothesis: str
    structural_change: str
    affected_symbols: list[str]
    expected_gain_regimes: list[str]
    expected_neutral_regimes: list[str]
    expected_regression_regimes: list[str]
    risks: list[str]
    required_controls: list[str]
    parameter_only: bool
    research_snapshot_hash: str
    author_kind: str
    generation: int
    source_citations: list[str] = field(default_factory=list)
    proposal_only: bool = True
    # ``parent_sha`` remains the backward-compatible primary-parent alias.  The ordered
    # ``parent_shas`` list is the authoritative genetic lineage for new artifacts.
    parent_shas: list[str] = field(default_factory=list)
    operator: str = "seed"  # seed|mutation|crossover|repair
    inherited_mechanisms: dict[str, list[str]] = field(default_factory=dict)
    inherited_components: dict[str, list[str]] = field(default_factory=dict)
    research_inspirations: list[str] = field(default_factory=list)
    compatibility_reason: str = ""
    new_control_flow: str = ""
    ablation_plan: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.parent_shas and self.parent_sha:
            self.parent_shas = [self.parent_sha]
        if self.parent_shas and self.parent_sha is None:
            self.parent_sha = self.parent_shas[0]
        if not self.operator:
            self.operator = (
                "seed" if not self.parent_shas else "crossover" if len(self.parent_shas) == 2
                else "mutation"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> CandidateManifest:
        return cls(**dict(value))

    def validate(
        self,
        *,
        source: str,
        parent_sha: str | None,
        parent_shas: list[str] | None = None,
        operator: str | None = None,
        research_context: ResearchContext | None,
        require_mechanism: bool = False,
    ) -> list[str]:
        issues: list[str] = []
        actual_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if self.candidate_sha != actual_sha:
            issues.append("candidate manifest SHA does not match source")
        if self.parent_sha != parent_sha:
            issues.append("candidate manifest parent SHA does not match selected parent")
        expected_parents = list(parent_shas) if parent_shas is not None else (
            [parent_sha] if parent_sha else []
        )
        if self.parent_shas != expected_parents:
            issues.append("candidate manifest parent SHAs do not match selected parents")
        if len(set(self.parent_shas)) != len(self.parent_shas):
            issues.append("candidate manifest parent SHAs must be distinct")
        expected_operator = operator or self.operator
        if self.operator != expected_operator:
            issues.append("candidate manifest operator does not match selected operator")
        if self.operator == "crossover":
            if len(self.parent_shas) != 2:
                issues.append("crossover requires exactly two distinct parents")
            if not self.compatibility_reason.strip():
                issues.append("crossover manifest compatibility_reason is empty")
            if not self.new_control_flow.strip():
                issues.append("crossover manifest new_control_flow is empty")
            if not self.ablation_plan:
                issues.append("crossover manifest ablation_plan is empty")
            missing_inheritance = sorted(
                parent for parent in self.parent_shas if parent not in self.inherited_mechanisms
            )
            if missing_inheritance:
                issues.append(
                    "crossover manifest is missing inherited mechanisms for parents: "
                    f"{missing_inheritance}"
                )
            missing_components = sorted(
                parent
                for parent in self.parent_shas
                if not self.inherited_components.get(parent)
            )
            if missing_components:
                issues.append(
                    "crossover manifest is missing concrete inherited components for parents: "
                    f"{missing_components}"
                )
            inherited_union = {
                mechanism
                for mechanisms in self.inherited_mechanisms.values()
                for mechanism in mechanisms
            }
            missing_child_mechanisms = sorted(inherited_union - set(self.mechanism_ids))
            if missing_child_mechanisms:
                issues.append(
                    "crossover child is missing inherited mechanism ids: "
                    f"{missing_child_mechanisms}"
                )
        elif self.operator in {"mutation", "repair"} and len(self.parent_shas) != 1:
            issues.append(f"{self.operator} requires exactly one parent")
        elif self.operator == "seed" and self.parent_shas:
            issues.append("seed candidates may not claim parents")
        elif self.operator not in {"seed", "mutation", "crossover", "repair"}:
            issues.append(f"unsupported genetic operator: {self.operator}")
        if research_context is not None:
            if self.research_snapshot_hash != research_context.snapshot_hash:
                issues.append("candidate manifest research snapshot hash mismatch")
            known = {card.mechanism_id for card in research_context.mechanism_cards}
            unknown = sorted(set(self.mechanism_ids) - known)
            if unknown:
                issues.append(f"candidate manifest cites unknown mechanism ids: {unknown}")
            unknown_inspirations = sorted(set(self.research_inspirations) - known)
            if unknown_inspirations:
                issues.append(
                    "candidate manifest cites unknown research inspirations: "
                    f"{unknown_inspirations}"
                )
            source_ids = {source.source_id for source in research_context.sources}
            cited_source_ids = {
                citation.removeprefix("source:")
                for citation in self.source_citations
                if citation.startswith("source:")
            }
            bad_citations = sorted(
                citation
                for citation in self.source_citations
                if not citation.startswith("source:")
                or citation.removeprefix("source:") not in source_ids
            )
            if bad_citations:
                issues.append(f"candidate manifest cites unknown sources: {bad_citations}")
            if require_mechanism and not unknown:
                card_sources = {
                    source_id
                    for card in research_context.mechanism_cards
                    if card.mechanism_id in self.mechanism_ids
                    for source_id in card.source_ids
                }
                missing_citations = sorted(card_sources - cited_source_ids)
                if missing_citations:
                    issues.append(
                        "candidate manifest is missing source citations for its mechanisms: "
                        f"{missing_citations}"
                    )
        if require_mechanism and not self.mechanism_ids:
            issues.append("Codex candidate manifest must cite at least one mechanism")
        tagged_mechanisms = set(re.findall(r"VE_MECHANISM:([A-Za-z0-9_.-]+)", source))
        if (require_mechanism or self.operator == "crossover") and tagged_mechanisms != set(
            self.mechanism_ids
        ):
            issues.append(
                "candidate manifest mechanism ids do not match VE_MECHANISM source tags"
            )
        if not self.hypothesis.strip():
            issues.append("candidate manifest hypothesis is empty")
        if not self.structural_change.strip():
            issues.append("candidate manifest structural_change is empty")
        if not self.proposal_only:
            issues.append("candidate manifest may not self-promote beyond proposal_only")
        return issues


__all__ = [
    "CandidateManifest",
    "MechanismCard",
    "ResearchContext",
    "ResearchSource",
    "canonical_json",
    "content_sha256",
]
