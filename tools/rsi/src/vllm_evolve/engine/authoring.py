"""Model-independent author transport and candidate-manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_evolve.engine.evolve_schemas import AuthorContext
from vllm_evolve.knowledge.schemas import CandidateManifest, ResearchContext


@dataclass
class AuthorSubmission:
    source: str
    manifest: dict | CandidateManifest | None = None


def unpack_author_submission(value: Any) -> AuthorSubmission:
    if isinstance(value, AuthorSubmission):
        return value
    if isinstance(value, str):
        return AuthorSubmission(value)
    if isinstance(value, dict):
        return AuthorSubmission(str(value.get("source") or ""), value.get("manifest"))
    raise TypeError(f"unsupported author response type: {type(value).__name__}")


def _source_mechanisms(source: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"VE_MECHANISM:([A-Za-z0-9_.-]+)", source)))


def _source_research_inspirations(source: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(r"VE_RESEARCH_INSPIRATION:([A-Za-z0-9_.-]+)", source)
        )
    )


def _source_recipe(source: str) -> dict[str, str]:
    match = re.search(
        r"VE_STRUCT:order=(\w+);gate=(\w+);switch=([01])"
        r"(?:;dispersion=([01]))?(?:;pressure_order=(\w+))?",
        source,
    )
    if not match:
        return {}
    order, gate, switch, dispersion, pressure_order = match.groups()
    return {
        "order": order,
        "gate": gate,
        "switch": switch,
        "dispersion_guard": dispersion or "0",
        "pressure_order": pressure_order or "total_work",
    }


def _inherited_components(source: str, parents: list[dict]) -> dict[str, list[str]]:
    child = _source_recipe(source)
    parent_recipes = {
        str(parent.get("sha")): _source_recipe(str(parent.get("source") or ""))
        for parent in parents
        if parent.get("sha")
    }
    result: dict[str, list[str]] = {}
    for parent_sha, recipe in parent_recipes.items():
        siblings = [row for sha, row in parent_recipes.items() if sha != parent_sha]
        result[parent_sha] = [
            f"{field}={value}"
            for field, value in recipe.items()
            if child.get(field) == value
            and (not siblings or any(other.get(field) != value for other in siblings))
        ]
    return result


def _template_ablation_plan(source: str) -> list[str]:
    recipe = _source_recipe(source)
    if not recipe:
        return []
    if recipe["gate"] != "none":
        return [f"disable only admission gate {recipe['gate']} (set gate=none)"]
    if recipe["dispersion_guard"] == "1":
        return ["disable only the dispersion guard"]
    if recipe["switch"] == "1" and recipe["pressure_order"] != recipe["order"]:
        return ["disable only the queue-pressure regime switch"]
    if recipe["order"] != "fcfs":
        return [f"replace only request order {recipe['order']} with fcfs"]
    return []


def _card_for(mechanism_id: str, context: ResearchContext | None):
    if context is None:
        return None
    return next(
        (card for card in context.mechanism_cards if card.mechanism_id == mechanism_id),
        None,
    )


def _template_proxy_structure(source: str) -> str:
    match = re.search(
        r"VE_STRUCT:order=(\w+);gate=(\w+);switch=([01])"
        r"(?:;dispersion=([01]))?",
        source,
    )
    if not match:
        return (
            "Deterministic template fallback proxy with no machine-verified paper-mechanism "
            "equivalence"
        )
    order, gate, switch, dispersion = match.groups()
    return (
        "Deterministic template fallback proxy: "
        f"order={order}, admission_gate={gate}, "
        f"queue_regime_switch={switch == '1'}, dispersion_guard={dispersion == '1'}; "
        "the cited research mechanism is inspiration, not a claim of semantic equivalence"
    )


def synthesize_candidate_manifest(
    *,
    source: str,
    parent_sha: str | None,
    context: AuthorContext,
    research_context: ResearchContext | None,
    author_kind: str,
    parent_shas: list[str] | None = None,
) -> CandidateManifest:
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    mechanism_ids = _source_mechanisms(source)
    research_inspirations = _source_research_inspirations(source)
    if not mechanism_ids and author_kind != "template" and research_context is not None:
        portfolio = research_context.recommended_hypothesis_portfolio
        slot = len(context.peers)
        if portfolio:
            mechanism_ids = list(portfolio[slot % len(portfolio)].get("mechanism_ids") or [])
    cards = [
        card
        for mechanism_id in [*mechanism_ids, *research_inspirations]
        if (card := _card_for(mechanism_id, research_context)) is not None
    ]
    portfolio_by_mechanism = {
        tuple(row.get("mechanism_ids") or []): row
        for row in (
            research_context.recommended_hypothesis_portfolio
            if research_context is not None
            else []
        )
    }
    portfolio = portfolio_by_mechanism.get(
        tuple(mechanism_ids or research_inspirations), {}
    )
    source_ids = list(dict.fromkeys(source_id for card in cards for source_id in card.source_ids))
    structural = "; ".join(card.structural_change or card.core_mechanism for card in cards)
    if author_kind == "template":
        structural = _template_proxy_structure(source)
    if not structural:
        structural = "Complete schedule_batch source change; no research mechanism tag was emitted"
    hypothesis = str(portfolio.get("hypothesis") or "")
    if author_kind == "template" and research_inspirations:
        hypothesis = (
            "Evaluate whether the deterministic fallback proxy inspired by "
            f"{', '.join(research_inspirations)} moves the declared metric. "
            "The proxy does not claim "
            "to reproduce the cited paper mechanism; inspect source and runtime markers."
        )
    if not hypothesis:
        hypothesis = (
            f"Evaluate generation {context.generation} scheduling source against the declared "
            "metric; this proposal has no literature-backed performance claim."
        )
    selected_parent_shas = list(parent_shas) if parent_shas is not None else (
        [parent_sha] if parent_sha else []
    )
    operator = str(context.operator.get("kind") or (
        "seed" if not selected_parent_shas else
        "crossover" if len(selected_parent_shas) == 2 else "mutation"
    ))
    inherited_mechanisms = {
        str(parent.get("sha")): list(
            (parent.get("candidate_manifest") or {}).get("mechanism_ids") or []
        )
        for parent in context.parents
        if parent.get("sha") in selected_parent_shas
    }
    inherited_components = _inherited_components(
        source,
        [parent for parent in context.parents if parent.get("sha") in selected_parent_shas],
    )
    compatibility_reason = ""
    new_control_flow = ""
    ablation_plan: list[str] = []
    if operator == "crossover":
        compatibility_reason = (
            "The selected parents are independently verified and scored on the same evaluator, "
            "and contribute disjoint typed scheduling recipe dimensions."
        )
        new_control_flow = (
            "Compose the primary parent's ordering/regime branch with the secondary parent's "
            "admission/guard branch, then execute the complete child policy from scratch."
        )
        ablation_plan = [
            f"remove contribution inherited from parent {parent[:12]}"
            for parent in selected_parent_shas
        ]
    elif operator in {"mutation", "repair"} and selected_parent_shas:
        ablation_plan = [
            f"restore parent {selected_parent_shas[0][:12]} control flow"
        ]
    elif operator == "seed" and author_kind == "template":
        ablation_plan = _template_ablation_plan(source)
    return CandidateManifest(
        candidate_sha=sha,
        parent_sha=parent_sha,
        mechanism_ids=mechanism_ids,
        hypothesis=hypothesis,
        structural_change=structural,
        affected_symbols=(
            ["targets/scheduling/skeleton.py:schedule_batch"]
            if author_kind == "template"
            else list(dict.fromkeys(hook for card in cards for hook in card.implementation_hooks))
            or ["targets/scheduling/skeleton.py:schedule_batch"]
        ),
        expected_gain_regimes=list(
            dict.fromkeys(workload for card in cards for workload in card.applicable_workloads)
        ),
        expected_neutral_regimes=[],
        expected_regression_regimes=list(
            dict.fromkeys(metric for card in cards for metric in card.possible_regressions)
        ),
        risks=list(dict.fromkeys(risk for card in cards for risk in card.failure_modes)),
        required_controls=(
            list(ablation_plan)
            if author_kind == "template"
            else list(
                dict.fromkeys(
                    control for card in cards for control in card.required_controls
                )
            )
        ),
        parameter_only=False,
        research_snapshot_hash=(
            research_context.snapshot_hash if research_context is not None else ""
        ),
        author_kind=author_kind,
        generation=context.generation,
        source_citations=[f"source:{source_id}" for source_id in source_ids],
        parent_shas=selected_parent_shas,
        operator=operator,
        inherited_mechanisms=inherited_mechanisms,
        inherited_components=inherited_components,
        research_inspirations=research_inspirations,
        compatibility_reason=compatibility_reason,
        new_control_flow=new_control_flow or structural,
        ablation_plan=ablation_plan,
    )


def bind_candidate_manifest(
    proposed: dict | CandidateManifest | None,
    *,
    source: str,
    parent_sha: str | None,
    context: AuthorContext,
    research_context: ResearchContext | None,
    author_kind: str,
    parent_shas: list[str] | None = None,
) -> CandidateManifest:
    selected_parent_shas = list(parent_shas) if parent_shas is not None else (
        [parent_sha] if parent_sha else []
    )
    if proposed is None:
        return synthesize_candidate_manifest(
            source=source,
            parent_sha=parent_sha,
            context=context,
            research_context=research_context,
            author_kind=author_kind,
            parent_shas=selected_parent_shas,
        )
    manifest = (
        proposed
        if isinstance(proposed, CandidateManifest)
        else CandidateManifest.from_dict(proposed)
    )
    # The orchestrator, not the untrusted author, binds provenance fields.
    manifest.candidate_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    manifest.parent_sha = parent_sha
    manifest.parent_shas = selected_parent_shas
    manifest.operator = str(context.operator.get("kind") or (
        "seed" if not selected_parent_shas else
        "crossover" if len(selected_parent_shas) == 2 else "mutation"
    ))
    manifest.inherited_mechanisms = {
        str(parent.get("sha")): list(
            (parent.get("candidate_manifest") or {}).get("mechanism_ids") or []
        )
        for parent in context.parents
        if parent.get("sha") in selected_parent_shas
    }
    if author_kind == "template":
        manifest.inherited_components = _inherited_components(
            source,
            [parent for parent in context.parents if parent.get("sha") in selected_parent_shas],
        )
        manifest.research_inspirations = _source_research_inspirations(source)
    if manifest.operator == "crossover":
        if not manifest.compatibility_reason:
            manifest.compatibility_reason = (
                "Both parents were independently verified and scored on the same evaluator; "
                "the author supplied a complete child rather than a text splice."
            )
        if not manifest.new_control_flow:
            manifest.new_control_flow = manifest.structural_change
        if not manifest.ablation_plan:
            manifest.ablation_plan = [
                f"remove contribution inherited from parent {parent[:12]}"
                for parent in selected_parent_shas
            ]
    elif manifest.operator in {"mutation", "repair"} and not manifest.ablation_plan:
        manifest.ablation_plan = (
            [f"restore parent {parent_sha[:12]} control flow"] if parent_sha else []
        )
    elif manifest.operator == "seed" and author_kind == "template" and not manifest.ablation_plan:
        manifest.ablation_plan = _template_ablation_plan(source)
        manifest.required_controls = list(manifest.ablation_plan)
    manifest.research_snapshot_hash = (
        research_context.snapshot_hash if research_context is not None else ""
    )
    manifest.author_kind = author_kind
    manifest.generation = context.generation
    manifest.proposal_only = True
    return manifest


def command_source_author(
    command: str | list[str],
    *,
    timeout_s: float = 300.0,
    author_kind: str = "codex",
):
    """Adapt any stdin/stdout completion command to the author contract.

    The command receives ``AuthorContext.to_prompt()`` on stdin and must print a
    JSON object with ``source`` and ``manifest``.  ``shell=False`` is deliberate.
    """
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if not argv:
        raise ValueError("author command must not be empty")

    def author(context: AuthorContext):
        completed = subprocess.run(
            argv,
            input=context.to_prompt(),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"author command exited {completed.returncode}: {completed.stderr[-1000:]}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("author command stdout must be one JSON object") from exc
        if not isinstance(result, dict) or not isinstance(result.get("source"), str):
            raise ValueError("author command JSON must contain source")
        if not isinstance(result.get("manifest"), dict):
            raise ValueError("author command JSON must contain manifest")
        return result

    author.ve_author_kind = author_kind
    author.ve_require_manifest = True
    author.ve_author_command = list(argv)
    return author


def export_author_bundle(context: AuthorContext, out_dir: str | Path) -> dict:
    """Write the exact prompt/input bundle a local Codex session should consume."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompt = context.to_prompt()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_path = root / "author_prompt.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    contract = {
        "prompt_sha256": prompt_hash,
        "prompt_path": str(prompt_path),
        "response_contract": {
            "source": "complete Python source string",
            "manifest": "CandidateManifest object",
        },
        "doctrine": "agent output is proposal-only and must pass orchestrator verification",
    }
    contract_path = root / "author_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return {**contract, "contract_path": str(contract_path)}


def import_author_submission(
    *,
    source_path: str | Path,
    manifest_path: str | Path,
    context: AuthorContext,
) -> AuthorSubmission:
    source = Path(source_path).read_text(encoding="utf-8")
    proposed = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(proposed, dict):
        raise ValueError("candidate manifest file must contain one object")
    research_context = (
        ResearchContext.from_dict(context.research_context) if context.research_context else None
    )
    selected_parent_shas = list(context.operator.get("parent_shas") or [])
    if not selected_parent_shas:
        selected_parent_shas = [
            str(parent["sha"]) for parent in context.parents if parent.get("sha")
        ]
    parent_sha = selected_parent_shas[0] if selected_parent_shas else None
    manifest = bind_candidate_manifest(
        proposed,
        source=source,
        parent_sha=parent_sha,
        context=context,
        research_context=research_context,
        author_kind=context.author_kind,
        parent_shas=selected_parent_shas,
    )
    issues = manifest.validate(
        source=source,
        parent_sha=parent_sha,
        parent_shas=selected_parent_shas,
        operator=str(context.operator.get("kind") or manifest.operator),
        research_context=research_context,
        require_mechanism=context.author_kind == "codex",
    )
    if issues:
        raise ValueError(f"candidate import rejected: {issues}")
    return AuthorSubmission(source, manifest)


__all__ = [
    "AuthorSubmission",
    "bind_candidate_manifest",
    "command_source_author",
    "export_author_bundle",
    "import_author_submission",
    "synthesize_candidate_manifest",
    "unpack_author_submission",
]
