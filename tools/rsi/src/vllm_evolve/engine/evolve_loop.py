"""The generational evolution engine (Evolution Engine v2).

What makes it *genetic* rather than one-shot: every child is authored from an ``AuthorContext``
that carries the diagnosis, the parents' code+scores, same-generation peers, knowledge-store
lessons and the child's own last verify errors (repair). Selection keeps elites, sha-dedup never
re-evaluates the same code, and the budget is pinned at the ``eval_fn`` call boundary — repair
re-evaluations included — so no caller can exceed it.

Scores come only from the injected ``eval_fn``. A simulator evaluator can produce search-only
proposals; a strict real evaluator can score only complete, saturated, mechanism-effective
``source=real_vllm`` runs. Either way an evolved winner is still a PROPOSAL until held-out
adjudication. Lessons are written to the store ONLY at run end (in-run feedback flows through
the in-memory Archive, never the store).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from vllm_evolve.engine.evolve_schemas import (
    AuthorContext,
    EvolutionResult,
    Lesson,
    LineageEntry,
)

_ELITES = 2  # top-k parents carried into the next generation
_NO_IMPROVE_K = 2  # consecutive generations without improvement -> stop
_LESSON_TOP_K = 5  # entries persisted at run end (plus one run summary)
_HARD_EVAL_CAP = 24  # ceiling enforced HERE, not at the CLI — no caller can exceed it


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify(source: str) -> list[str]:
    """The SAME full static gate as ``ve verify`` — reuse ``cli.main._verify_code`` so an evolved
    variant gets L1 (safety + return annotation) + L2 (signature PARITY with the skeleton, nested
    loop, fabricated-id) exactly like a hand-authored policy, then add the marker-forgery check.
    Every authored source (including after repair) passes through here before any eval. (Lazy import
    avoids an import cycle and keeps this off the frozen-core closure.)"""
    from vllm_evolve.bench.runtime import contains_marker_forgery
    from vllm_evolve.cli.main import _verify_code

    issues = list(_verify_code(source, "scheduling"))
    if contains_marker_forgery(source):
        issues.append("L1: emits a reserved provenance marker (forgery)")
    return issues


def _score_of(prof, metric: str) -> float | None:
    md = getattr(prof, "metrics", None) or {}
    v = md.get(metric)
    if isinstance(v, (int, float)):
        return float(v)
    return next((float(x) for x in md.values() if isinstance(x, (int, float))), None)


def _operator_plan(
    generation: int,
    child_index: int,
    parents: list[LineageEntry],
) -> tuple[str, list[LineageEntry]]:
    """Return a deterministic typed genetic operation and its selected parents.

    Only independently verified, score-bearing, distinct candidates can become parents.  Once
    two exist, even child slots cross two adjacent parents and odd slots mutate one parent.  This
    lets even a one-child generation exercise crossover while a normal population of at least two
    exercises both operators without relying on prompt wording or random model behavior.
    """
    eligible: list[LineageEntry] = []
    seen: set[str] = set()
    for parent in parents:
        if not parent.verify_ok or parent.score is None or parent.sha in seen:
            continue
        eligible.append(parent)
        seen.add(parent.sha)
    if generation == 0 or not eligible:
        return "seed", []
    if len(eligible) >= 2 and child_index % 2 == 0:
        first = (child_index // 2) % len(eligible)
        return "crossover", [eligible[first], eligible[(first + 1) % len(eligible)]]
    return "mutation", [eligible[child_index % len(eligible)]]


def _parent_payload(
    parent: LineageEntry,
    sources: dict[str, str],
    manifests: dict[str, dict],
) -> dict:
    return {
        "sha": parent.sha,
        "score": parent.score,
        "generation": parent.generation,
        "source": sources.get(parent.sha, ""),
        "candidate_manifest": manifests.get(parent.sha, {}),
        "diagnostics": parent.diagnostics,
        "verify_ok": parent.verify_ok,
        "outcome_class": parent.outcome_class,
        "evidence_refs": parent.evidence_refs,
    }


def run_evolution(
    author_fn,
    eval_fn,
    spec,
    base_config: dict,
    diagnosis: dict,
    *,
    generations: int = 2,
    population: int = 3,
    repair_limit: int = 1,
    max_total_evals: int = 24,
    lessons: list | None = None,
    store=None,
    run_id: str = "evolve",
    variants_dir: str = "runs/evolve_variants",
    skeleton: str = "",
    research_context: dict | None = None,
    author_kind: str | None = None,
    require_author_manifest: bool | None = None,
) -> EvolutionResult:
    """Evolve ``schedule_batch`` over generations; budget enforced at every ``eval_fn`` call."""
    from vllm_evolve.engine.authoring import (
        bind_candidate_manifest,
        unpack_author_submission,
    )
    from vllm_evolve.knowledge.schemas import ResearchContext

    rng = random.Random(0)  # deterministic survivor pick (reproducible runs)
    out_dir = Path(variants_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_d = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    metric = spec_d.get("metric", "")
    minimize = spec_d.get("direction") == "min"

    def _better(a: float, b: float) -> bool:
        return a < b if minimize else a > b

    archive: list[LineageEntry] = []
    seen_sha: dict[str, LineageEntry] = {}
    sources: dict[str, str] = {}  # sha -> source (parent code fed to children)
    manifests: dict[str, dict] = {}  # sha -> audited CandidateManifest
    research = ResearchContext.from_dict(research_context) if research_context else None
    if research is not None:
        integrity = research.validate_snapshot_integrity()
        if not integrity["ok"]:
            raise ValueError(f"research context snapshot integrity failed: {integrity}")
    resolved_author_kind = (
        author_kind
        or getattr(author_fn, "ve_author_kind", None)
        or ("template" if getattr(author_fn, "__name__", "") == "template_author_fn" else "custom")
    )
    require_manifest = (
        bool(require_author_manifest)
        if require_author_manifest is not None
        else bool(getattr(author_fn, "ve_require_manifest", False))
    )
    budget = min(max(0, int(max_total_evals)), _HARD_EVAL_CAP)
    evals_used = 0
    history: list[dict] = []
    parents: list[LineageEntry] = []
    best_overall: float | None = None
    no_improve = 0
    terminated = "generations_exhausted"

    for gen in range(max(1, int(generations))):
        gen_entries: list[LineageEntry] = []
        for child_i in range(max(1, int(population))):
            if budget <= 0:
                terminated = "budget_exhausted"
                break
            planned_operator, planned_parents = _operator_plan(gen, child_i, parents)
            peer_view = [
                {
                    "sha": e.sha,
                    "score": e.score,
                    "selection_eligible": e.score is not None,
                    "source": sources.get(e.sha, ""),
                    "candidate_manifest": manifests.get(e.sha, {}),
                    "outcome_class": e.outcome_class,
                    "diagnostics": e.diagnostics,
                    "note": e.note,
                }
                for e in gen_entries
            ]
            archive_feedback = [
                {
                    "kind": "prior_candidate_evidence",
                    "sha": entry.sha,
                    "generation": entry.generation,
                    "score": entry.score,
                    "selection_eligible": entry.score is not None,
                    "outcome_class": entry.outcome_class,
                    "note": entry.note,
                    "mechanism_ids": entry.mechanism_ids,
                    "diagnostics": entry.diagnostics,
                    "source": sources.get(entry.sha, ""),
                    "candidate_manifest": manifests.get(entry.sha, {}),
                }
                for entry in archive[-8:]
            ]
            errors: list[str] = []
            source = ""
            candidate_manifest = None
            attempts = 0
            attempt_context_paths: list[str] = []
            attempt_verify_paths: list[str] = []
            bound_parent_shas: list[str] = []
            bound_operator = planned_operator
            while attempts <= max(0, int(repair_limit)):
                repair_attempt = bool(errors)
                effective_operator = (
                    "repair" if repair_attempt and planned_parents else planned_operator
                )
                effective_parents = (
                    planned_parents[:1]
                    if effective_operator == "repair"
                    else planned_parents
                )
                parent_view = [
                    _parent_payload(parent, sources, manifests) for parent in effective_parents
                ]
                selected_parent_shas = [parent.sha for parent in effective_parents]
                ctx = AuthorContext(
                    spec=spec_d,
                    diagnosis=dict(diagnosis or {}),
                    skeleton=skeleton,
                    parents=parent_view,
                    peers=peer_view,
                    lessons=[*list(lessons or []), *archive_feedback],
                    research_context=(research.to_dict() if research is not None else {}),
                    last_errors=list(errors),
                    budget={
                        "evals_remaining": budget,
                        "repair_remaining": max(0, repair_limit - attempts),
                    },
                    operator={
                        "kind": effective_operator,
                        "origin_kind": planned_operator,
                        "parent_shas": selected_parent_shas,
                        "schedule_slot": child_i,
                        "deterministic_seed": 0,
                        "repair_attempt": attempts,
                    },
                    generation=gen,
                    author_kind=resolved_author_kind,
                )
                attempt_no = attempts
                attempt_stem = out_dir / f"g{gen}_{child_i}_attempt{attempt_no}"
                context_path = attempt_stem.with_suffix(".author_context.json")
                context_prompt = ctx.to_prompt()
                context_path.write_text(context_prompt, encoding="utf-8")
                attempt_context_paths.append(str(context_path))
                verify_report_path = attempt_stem.with_suffix(".verify.json")

                def record_attempt(
                    *,
                    stage: str,
                    attempt_source: str = "",
                    l1_l2_issues: list[str] | None = None,
                    manifest_issues: list[str] | None = None,
                    all_issues: list[str] | None = None,
                ) -> None:
                    source_path = None
                    source_sha = None
                    if attempt_source:
                        attempt_source_path = attempt_stem.with_suffix(".source.py")
                        attempt_source_path.write_text(attempt_source, encoding="utf-8")
                        source_path = str(attempt_source_path)
                        source_sha = _sha(attempt_source)
                    verify_report_path.write_text(
                        json.dumps(
                            {
                                "generation": gen,
                                "child_index": child_i,
                                "attempt": attempt_no,
                                "stage": stage,
                                "operator": effective_operator,
                                "parent_shas": selected_parent_shas,
                                "author_context_path": str(context_path),
                                "author_context_sha256": _sha(context_prompt),
                                "source_path": source_path,
                                "source_sha256": source_sha,
                                "l1_l2_executed": bool(attempt_source),
                                "l1_l2_passed": bool(attempt_source)
                                and not (l1_l2_issues or []),
                                "l1_l2_issues": list(l1_l2_issues or []),
                                "manifest_passed": bool(attempt_source)
                                and not (manifest_issues or []),
                                "manifest_issues": list(manifest_issues or []),
                                "passed": not (all_issues or []),
                                "issues": list(all_issues or []),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    attempt_verify_paths.append(str(verify_report_path))

                try:
                    raw_submission = author_fn(ctx)
                except Exception as exc:  # noqa: BLE001 - one failed author attempt is repairable
                    attempts += 1
                    source = ""
                    errors = [f"author execution failed: {type(exc).__name__}: {exc}"]
                    record_attempt(stage="author_execution", all_issues=errors)
                    continue
                attempts += 1
                try:
                    submission = unpack_author_submission(raw_submission)
                except (TypeError, ValueError) as exc:
                    source = ""
                    errors = [f"author response invalid: {exc}"]
                    record_attempt(stage="author_response", all_issues=errors)
                    continue
                source = submission.source
                if not isinstance(source, str) or not source.strip():
                    errors = ["author produced no source"]
                    record_attempt(stage="author_response", all_issues=errors)
                    continue
                l1_l2_issues = _verify(source)
                errors = list(l1_l2_issues)
                if (
                    effective_operator == "crossover"
                    and _sha(source) in set(selected_parent_shas)
                ):
                    errors.append("crossover child is identical to one of its parents")
                parent_sha = selected_parent_shas[0] if selected_parent_shas else None
                manifest_issues: list[str] = []
                try:
                    candidate_manifest = bind_candidate_manifest(
                        submission.manifest,
                        source=source,
                        parent_sha=parent_sha,
                        context=ctx,
                        research_context=research,
                        author_kind=resolved_author_kind,
                        parent_shas=selected_parent_shas,
                    )
                    manifest_issues = candidate_manifest.validate(
                            source=source,
                            parent_sha=parent_sha,
                            parent_shas=selected_parent_shas,
                            operator=effective_operator,
                            research_context=research,
                            require_mechanism=require_manifest,
                        )
                    errors.extend(manifest_issues)
                except (TypeError, ValueError) as exc:
                    manifest_issues = [f"candidate manifest invalid: {exc}"]
                    errors.extend(manifest_issues)
                record_attempt(
                    stage="verified" if not errors else "verify_or_manifest_rejected",
                    attempt_source=source,
                    l1_l2_issues=l1_l2_issues,
                    manifest_issues=manifest_issues,
                    all_issues=errors,
                )
                if not errors:
                    bound_parent_shas = selected_parent_shas
                    bound_operator = effective_operator
                    break
            sha = _sha(source) if isinstance(source, str) and source.strip() else ""
            if errors or not sha:
                archive.append(
                    LineageEntry(
                        sha=sha or "invalid",
                        generation=gen,
                        parent_sha=(bound_parent_shas[0] if bound_parent_shas else (
                            planned_parents[0].sha if planned_parents else None
                        )),
                        parent_shas=(bound_parent_shas or [p.sha for p in planned_parents]),
                        operator=bound_operator,
                        verify_ok=False,
                        repair_count=attempts - 1,
                        author_context_paths=list(attempt_context_paths),
                        verify_report_paths=list(attempt_verify_paths),
                        note="; ".join(errors)[:300],
                    )
                )
                continue
            if sha in seen_sha:  # dedup: identical code is never re-evaluated
                prior = seen_sha[sha]
                duplicate_manifest_path = (
                    out_dir / f"g{gen}_{child_i}_{sha[:8]}.duplicate.manifest.json"
                )
                duplicate_manifest_path.write_text(
                    json.dumps(candidate_manifest.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                archive.append(
                    LineageEntry(
                        sha=sha,
                        generation=gen,
                        parent_sha=bound_parent_shas[0] if bound_parent_shas else None,
                        parent_shas=list(bound_parent_shas),
                        operator=bound_operator,
                        source_path=prior.source_path,
                        verify_ok=True,
                        repair_count=attempts - 1,
                        candidate_manifest_path=str(duplicate_manifest_path),
                        mechanism_ids=list(candidate_manifest.mechanism_ids),
                        research_snapshot_hash=(
                            research.snapshot_hash if research is not None else ""
                        ),
                        author_kind=resolved_author_kind,
                        outcome_class="deduplicated_nonranking",
                        author_context_paths=list(attempt_context_paths),
                        verify_report_paths=list(attempt_verify_paths),
                        diagnostics={"selection_eligible": False, "duplicate_of_sha": sha},
                        deduplicated=True,
                        duplicate_of_sha=sha,
                        note=(
                            "identical source reused cached evaluation; lineage edge is nonranking"
                        ),
                    )
                )
                gen_entries.append(prior)
                continue
            path = out_dir / f"g{gen}_{child_i}_{sha[:8]}.py"
            path.write_text(source, encoding="utf-8")
            manifest_path = path.with_suffix(".manifest.json")
            manifest_path.write_text(
                json.dumps(candidate_manifest.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            budget -= 1  # the budget boundary: charged per eval_fn call
            evals_used += 1
            try:
                prof = eval_fn({**base_config, "runner_kind": "candidate", "policy": str(path)})
            except Exception as exc:  # noqa: BLE001 - a failed candidate must not abort the goal
                entry = LineageEntry(
                    sha=sha,
                    generation=gen,
                    parent_sha=bound_parent_shas[0] if bound_parent_shas else None,
                    parent_shas=list(bound_parent_shas),
                    operator=bound_operator,
                    source_path=str(path),
                    verify_ok=True,
                    repair_count=attempts - 1,
                    candidate_manifest_path=str(manifest_path),
                    mechanism_ids=list(candidate_manifest.mechanism_ids),
                    research_snapshot_hash=(research.snapshot_hash if research is not None else ""),
                    author_kind=resolved_author_kind,
                    outcome_class="evaluation_failed",
                    author_context_paths=list(attempt_context_paths),
                    verify_report_paths=list(attempt_verify_paths),
                    diagnostics={
                        "exception_type": type(exc).__name__,
                        "error": str(exc)[-1000:],
                        "selection_eligible": False,
                    },
                    note=f"evaluation failed: {type(exc).__name__}: {exc}"[:1000],
                )
                sources[sha] = source
                manifests[sha] = candidate_manifest.to_dict()
                seen_sha[sha] = entry
                archive.append(entry)
                continue
            prof_evidence = list(getattr(prof, "evidence", None) or [])
            evidence_refs = [
                str(value)
                for row in prof_evidence
                if isinstance(row, dict)
                for key in ("eval_result_path", "remote_run_dir")
                if (value := row.get(key))
            ]
            invalid_reasons = [
                str(reason)
                for row in prof_evidence
                if isinstance(row, dict)
                for reason in (row.get("fitness_reasons") or [])
            ]
            diagnostics = next(
                (
                    dict(row["diagnostics"])
                    for row in reversed(prof_evidence)
                    if isinstance(row, dict)
                    and isinstance(row.get("diagnostics"), dict)
                ),
                {},
            )
            entry = LineageEntry(
                sha=sha,
                generation=gen,
                parent_sha=bound_parent_shas[0] if bound_parent_shas else None,
                parent_shas=list(bound_parent_shas),
                operator=bound_operator,
                score=_score_of(prof, metric),
                source_path=str(path),
                verify_ok=True,
                repair_count=attempts - 1,
                marker_verified=bool(getattr(prof, "marker_verified", False)),
                candidate_manifest_path=str(manifest_path),
                mechanism_ids=list(candidate_manifest.mechanism_ids),
                research_snapshot_hash=(research.snapshot_hash if research is not None else ""),
                author_kind=resolved_author_kind,
                outcome_class=str(getattr(prof, "outcome_class", "") or ""),
                evidence_refs=evidence_refs,
                author_context_paths=list(attempt_context_paths),
                verify_report_paths=list(attempt_verify_paths),
                diagnostics=diagnostics,
                note="; ".join(invalid_reasons)[:1000],
            )
            sources[sha] = source
            manifests[sha] = candidate_manifest.to_dict()
            seen_sha[sha] = entry
            archive.append(entry)
            gen_entries.append(entry)
        else:
            scored = [e for e in gen_entries if e.score is not None]
            if scored:
                gen_best = (min if minimize else max)(e.score for e in scored)
                history.append(
                    {
                        "generation": gen,
                        "best_sha": next(e.sha for e in scored if e.score == gen_best),
                        "best_score": gen_best,
                    }
                )
                if best_overall is None or _better(gen_best, best_overall):
                    best_overall = gen_best
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= _NO_IMPROVE_K:
                        terminated = "no_improvement"
                        break
            # Strict elitism selects from prior parents plus new scoreable candidates.  Stable SHA
            # tie-breaking makes survivor and therefore parent-pair order reproducible.
            selection_pool = {
                entry.sha: entry
                for entry in [*parents, *scored]
                if entry.score is not None and entry.verify_ok
            }
            ranked = sorted(
                selection_pool.values(),
                key=lambda e: ((e.score if minimize else -e.score), e.sha),
            )
            parents = ranked[:_ELITES]
            rest = ranked[_ELITES:]
            if rest:
                parents = parents + [rest[rng.randrange(len(rest))]]
            continue
        break  # inner loop hit the budget wall

    evaluated = [e for e in archive if e.score is not None]
    winner = (min if minimize else max)(evaluated, key=lambda e: e.score) if evaluated else None
    if winner is None and terminated == "generations_exhausted":
        terminated = "no_candidate"

    lessons_written: list[int] = []
    if store is not None and evaluated:  # run END only — never mid-run
        regime = str(base_config.get("profile", "throughput"))
        src_name = getattr(eval_fn, "ve_source", "frontier_sim")
        top = sorted(evaluated, key=lambda e: e.score, reverse=not minimize)[:_LESSON_TOP_K]
        for e in top:
            lessons_written.append(
                store.put_lesson(
                    run_id=run_id,
                    source=src_name,
                    policy_sha=e.sha,
                    regime=regime,
                    metric=metric,
                    score=e.score,
                    conclusion=f"gen{e.generation} variant scored {e.score} on {metric}",
                    eval_refs=[e.source_path],
                )
            )
        lessons_written.append(
            store.put_lesson(
                run_id=run_id,
                source=src_name,
                policy_sha=winner.sha,
                regime=regime,
                metric=metric,
                score=winner.score,
                conclusion=(
                    f"run summary: {len(evaluated)} variants over "
                    f"{len(history)} generation(s); winner gen{winner.generation} "
                    f"{metric}={winner.score}; terminated={terminated}"
                ),
                eval_refs=[winner.source_path],
            )
        )

    return EvolutionResult(
        winner=winner,
        archive=archive,
        generations_completed=len(history),
        evals_used=evals_used,
        terminated=terminated,
        history=history,
        lessons_written=lessons_written,
        author_kind=resolved_author_kind,
        research_snapshot_hash=(research.snapshot_hash if research is not None else ""),
    )


__all__ = ["run_evolution", "Lesson"]
