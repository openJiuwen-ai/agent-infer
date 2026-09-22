"""Data contracts for the generational evolution engine (Evolution Engine v2).

Lives in ``engine/`` — NOT ``core/`` — so the frozen judgment closure stays thin; if ``bench/**``
ever imported this module the frozen-core whitelist test would fail the build (the closure would
reach ``engine/*``), so no extra guard is needed.

``AuthorContext.to_prompt()`` is the SINGLE serialization source consumed by BOTH author forms
(the headless ``template_author_fn`` and the ve-author sub-agent): deterministic structured JSON
covering EVERY field plus the doctrine line, so neither form can silently drop the feedback that
makes evolution genetic (parent scores, failure reasons, lessons).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

# The doctrine line every author prompt carries, verbatim.
DOCTRINE = (
    "Only evidence from the declared evaluator may rank candidates. Simulator scores are "
    "proposal-only and can never be kept/adopted (DoD-B); a real-evolution score exists only "
    "for complete source=real_vllm runs that pass the saturated-workload and mechanism-action "
    "gates. Final adoption still requires paired held-out real-vLLM verification."
)


@dataclass
class AuthorContext:
    """Everything an author (template or agent) sees before writing the next child policy."""

    spec: dict = field(default_factory=dict)  # {metric, direction, raw_intent, ...}
    diagnosis: dict = field(default_factory=dict)  # {bottleneck, status, evidence_refs, ...}
    skeleton: str = ""  # the schedule_batch contract source
    parents: list = field(default_factory=list)  # [{sha, score, generation, source}]
    peers: list = field(default_factory=list)  # same-generation [{sha, score}]
    lessons: list = field(default_factory=list)  # [{lesson_id, conclusion, score, ...}]
    research_context: dict = field(default_factory=dict)  # frozen ResearchContext.to_dict()
    last_errors: list = field(default_factory=list)  # verify/marker errors for repair
    budget: dict = field(default_factory=dict)  # {evals_remaining, repair_remaining}
    operator: dict = field(default_factory=dict)  # {kind, parent_shas, schedule_slot, ...}
    generation: int = 0
    author_kind: str = "custom"  # template|codex|external|custom

    def to_prompt(self) -> str:
        """Deterministic structured JSON over EVERY field + the doctrine line. Both author forms
        consume exactly this rendering — the snapshot test pins all field names + the doctrine."""
        payload = {"doctrine": DOCTRINE, **asdict(self)}
        return json.dumps(payload, sort_keys=True, indent=2, default=str, ensure_ascii=False)


def prompt_source_author(
    completion_fn: Callable[[str], Any],
    *,
    author_kind: str = "codex",
    require_manifest: bool = False,
) -> Callable[[AuthorContext], Any]:
    """Adapt a Codex-style prompt completion into the engine's source-returning author contract.

    There is deliberately no model SDK dependency here. The client integration supplies one
    callable, receives exactly ``AuthorContext.to_prompt()``, and returns source. The evolution
    orchestrator remains the only file writer and applies verification after this boundary.
    """

    def author(context: AuthorContext) -> str:
        result = completion_fn(context.to_prompt())
        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "source" in parsed:
                result = parsed
        if isinstance(result, dict):
            if not isinstance(result.get("source"), str) or not result["source"].strip():
                raise ValueError("Codex author response must contain non-empty source")
            if require_manifest and not isinstance(result.get("manifest"), dict):
                raise ValueError("Codex author response must contain a candidate manifest")
            return result
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Codex author must return a non-empty policy source string")
        if require_manifest:
            raise ValueError("Codex author must return JSON with source and manifest")
        return result

    author.ve_author_kind = author_kind
    author.ve_require_manifest = require_manifest
    return author


@dataclass
class LineageEntry:
    """One evaluated child in the Archive (the run's family tree)."""

    sha: str
    generation: int
    parent_sha: str | None = None
    parent_shas: list[str] = field(default_factory=list)
    operator: str = "seed"
    score: float | None = None
    source_path: str = ""
    verify_ok: bool = False
    repair_count: int = 0
    marker_verified: bool = False  # from the BENCH profile only (False under any simulator)
    candidate_manifest_path: str = ""
    mechanism_ids: list[str] = field(default_factory=list)
    research_snapshot_hash: str = ""
    author_kind: str = "custom"
    outcome_class: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    author_context_paths: list[str] = field(default_factory=list)
    verify_report_paths: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    deduplicated: bool = False
    duplicate_of_sha: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Lesson:
    """An immutable evidence row written to the knowledge store at run END (never mid-run).
    ``source`` always names the true origin (e.g. ``frontier_sim``) — a lesson can never
    impersonate a real-vLLM result."""

    run_id: str
    source: str
    policy_sha: str
    regime: str
    metric: str
    score: float | None
    conclusion: str
    eval_refs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvolutionResult:
    """What ``run_evolution`` returns; the orchestrator maps the winner back to a ``Candidate``."""

    winner: LineageEntry | None = None
    archive: list = field(default_factory=list)  # [LineageEntry]
    generations_completed: int = 0
    evals_used: int = 0
    terminated: str = ""  # generations_exhausted|no_improvement|budget_exhausted|no_candidate
    history: list = field(default_factory=list)  # per-gen {generation, best_sha, best_score}
    lessons_written: list = field(default_factory=list)  # lesson ids persisted at run end
    author_kind: str = "custom"
    research_snapshot_hash: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
