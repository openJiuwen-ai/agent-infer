"""Top orchestration — ``ve autopt``: run the L1->L4 loop toward a Spec.

Each round: L1 profile+diagnose the current config -> L2 rank targets -> pick the
top *wired config* target not yet tried -> L3 search it -> L4 verify (bootstrap +
holdout). A holdout-confirmed BETTER candidate is adopted; if the bottleneck then
shifted, the new config seeds another round (L1 again). Otherwise the next round
tries the next target. When rounds/targets/budget are exhausted with no adoption,
the honest outcome is **DoD-B**: no provable gain, with the full per-round trace
and a next-direction suggestion — never a fabricated win.

``eval_fn(config) -> Profile`` is injected (real = ``collect_profile``; tests pass
a mock). The Profile must carry ``per_seed[metric]`` so L4 can bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_evolve.bench.config import BenchConfig, same_caliber
from vllm_evolve.core.accept import _DEFAULT_ACCEPT_THRESHOLD_PCT, accept_vs_strong_baseline
from vllm_evolve.core.schemas import Candidate, Profile, Spec
from vllm_evolve.core.verify import verify_gain
from vllm_evolve.engine.diagnose import diagnose
from vllm_evolve.engine.optimize import optimize
from vllm_evolve.engine.profile import WIRED_KNOBS
from vllm_evolve.engine.quality_runner import is_quality_certified, measured_quality_verdict
from vllm_evolve.engine.targets import select_targets

EvalFn = Callable[[dict], Profile]

# _DEFAULT_ACCEPT_THRESHOLD_PCT is imported from the frozen core (core.accept) — re-exported here
# for backward-compat callers; the gate threshold is never taken from a Spec.


def _pick_target(plan, tried: set) -> dict | None:
    """Top-ranked actionable target not yet tried: a wired config knob, or a code
    target (evolving ``schedule_batch`` — only ranked in when scheduling is the
    bottleneck). deploy/profile targets are not auto-actioned in v1."""
    for c in plan.candidates:
        if c["target"] in tried:
            continue
        if c.get("type") == "config":
            knob = c["target"].split(":", 1)[1]
            if knob in WIRED_KNOBS and c.get("search_space"):
                return c
        elif c.get("type") == "code":
            return c
    return None


def _holdout_config(config: dict, avoid_knob: str) -> dict | None:
    """A genuinely different operating point via a WIRED knob (not the one under test).

    Returns ``None`` when no wired workload knob is available to vary — the
    orchestrator then runs no holdout (honest: verdict stays needs_holdout) rather
    than faking one with a key the real bench would ignore.
    """
    for knob in ("concurrency", "n_requests"):
        if knob == avoid_knob:
            continue
        if isinstance(config.get(knob), (int, float)):
            base_val = config[knob]
            scaled = int(base_val * 1.25)
            # +25% truncates to the SAME value for small workloads (1-3), which would record a
            # holdout at an UNCHANGED operating point -> a candidate could "pass holdout" without
            # generalizing. Bump by at least one so the holdout genuinely differs (Codex review P2).
            if scaled <= base_val:
                scaled = int(base_val) + 1
            h = dict(config)
            h[knob] = scaled
            return h
    return None


def _conclude(adopted, spec: Spec, rounds: list) -> dict:
    if adopted:
        return {
            "result": "gain",
            "summary": f"adopted {adopted['target']} -> {spec.metric} "
            f"{adopted['verdict'].get('improvement_pct')}% "
            f"(bootstrap better, holdout-confirmed)",
        }
    return {
        "result": "dod_b",
        "summary": "no provable gain over the baseline across the tried targets "
        f"({len(rounds)} round(s))",
        "why": "every tried target (config search and/or schedule_batch evolution) was "
        "inconclusive / unverified / overfit under bootstrap+holdout — the gate "
        "was never cleared",
        "next_directions": [
            "change the objective (priority / fairness / deadline) where FCFS-style "
            "defaults are no longer near-optimal",
            "a workload vLLM handles less well (heavy prefix sharing, bursty, multi-tenant)",
            "go deeper than config (KV allocation, speculative decoding, batching) or "
            "a larger model + TP",
        ],
    }


def run_autopt(
    spec: Spec,
    *,
    eval_fn: EvalFn,
    evolve_fn: Callable | None = None,
    author_fn: Callable | None = None,
    evolve_params: dict | None = None,
    research_fn: Callable | None = None,
    store=None,
    quality_measure_fn: Callable | None = None,
    base_config: dict | None = None,
    max_rounds: int = 3,
    max_evals: int = 8,
    run_holdout: bool = True,
    accept_threshold_pct: float | None = None,
) -> dict:
    """``quality_measure_fn(config) -> measure_fn`` (or None) supplies the FROZEN-set quality
    measurement for a config (real = served model, box-gated). With no quality_measure_fn,
    quality is NOT MEASURED and no candidate can be adopted (Codex R5: never assume quality).

    The accept-gate ``threshold_pct`` is a CONFIG/CLI decision, NEVER taken from the (possibly
    sub-agent-produced) ``spec`` — a ve-goal agent must not be able to lower the adoption bar.
    """
    base_config = dict(base_config or {})
    gate_threshold = (
        accept_threshold_pct if accept_threshold_pct is not None else _DEFAULT_ACCEPT_THRESHOLD_PCT
    )
    rounds: list = []
    tried: set = set()
    adopted: dict | None = None

    for rnd in range(max_rounds):
        base_prof = eval_fn(base_config)
        diag = diagnose(base_prof)
        plan = select_targets(diag)
        rec = {
            "round": rnd,
            "config": dict(base_config),
            "diagnosis": diag.to_dict(),
            "targets": [c["target"] for c in plan.candidates],
        }

        target = _pick_target(plan, tried)
        if target is None:
            rec["note"] = "no untried actionable target available -> stop searching"
            rounds.append(rec)
            break
        tried.add(target["target"])
        is_code = target.get("type") == "code"

        if is_code:
            # code:schedule_batch — only ranked in when scheduling IS the bottleneck.
            if author_fn is not None:
                # Evolution v2: the generational engine (Archive + repair + budget at the
                # eval boundary). The winner maps back to a Candidate so every existing
                # trajectory consumer + the adoption guard below read the same keys.
                from vllm_evolve.engine.evolve_loop import run_evolution

                round_evolve_params = dict(evolve_params or {})
                if research_fn is not None:
                    research_value = research_fn(
                        spec=spec,
                        diagnosis=diag.to_dict(),
                        base_config=dict(base_config),
                        target=target["target"].split(":", 1)[-1],
                        round_index=rnd,
                    )
                    if hasattr(research_value, "to_dict"):
                        research_value = research_value.to_dict()
                    if not isinstance(research_value, dict):
                        raise TypeError("research_fn must return ResearchContext or dict")
                    round_evolve_params["research_context"] = research_value
                    rec["research"] = {
                        "snapshot_id": research_value.get("snapshot_id"),
                        "snapshot_hash": research_value.get("snapshot_hash"),
                        "outcome_class": research_value.get("outcome_class"),
                        "provider_status": research_value.get("provider_status", []),
                    }
                er = run_evolution(
                    author_fn,
                    eval_fn,
                    spec,
                    base_config,
                    diag.to_dict(),
                    store=store,
                    **round_evolve_params,
                )
                rec["evolution"] = {
                    "generations_completed": er.generations_completed,
                    "evals_used": er.evals_used,
                    "terminated": er.terminated,
                    "history": er.history,
                    "lessons_written": er.lessons_written,
                    "author_kind": er.author_kind,
                    "research_snapshot_hash": er.research_snapshot_hash,
                }
                w = er.winner
                cand = Candidate(
                    target="code:schedule_batch",
                    kind="code",
                    value={"policy": w.source_path} if w else {},
                    score=w.score if w else None,
                    marker_verified=bool(w.marker_verified) if w else False,
                    evals_used=er.evals_used,
                    trials=[e.to_dict() for e in er.archive],
                    note=(
                        f"evolution winner gen{w.generation} {spec.metric}={w.score} "
                        f"(search-only; adoption requires real vLLM)"
                    )
                    if w
                    else "evolution produced no scoreable candidate",
                )
            elif evolve_fn is None:
                rec["note"] = (
                    "top target is code:schedule_batch but no evolve_fn wired "
                    "-> pass `ve autopt --evolve` to enable it; skipping"
                )
                rounds.append(rec)
                break
            else:
                cand = evolve_fn(base_config, spec)  # legacy single-shot (AC7 surface)
            avoid_knob = ""  # holdout may vary any wired workload knob
        else:
            knob = target["target"].split(":", 1)[1]
            cand = optimize(
                {knob: target["search_space"]}, base_config, spec, eval_fn, max_evals=max_evals
            )
            avoid_knob = knob
        rec["searched_target"] = target["target"]
        rec["candidate"] = cand.to_dict()

        if not (cand.value and cand.marker_verified):
            rec["verdict"] = {"verdict": "no_verified_candidate"}
            rounds.append(rec)
            continue  # search/evolution found nothing verified-improving -> next target

        winner_cfg = {**base_config, **cand.value}
        # The searched dimension(s) ARE the thing under test, so they may legitimately differ in the
        # A/B caliber check (a config win changes e.g. max_num_seqs; a code win changes the policy,
        # which same_caliber already exempts). Without this, every real config optimization would
        # falsely read as same_caliber_mismatch and could never adopt (Codex review P1).
        searched_knobs = set(cand.value)
        cand_prof = eval_fn(winner_cfg)
        # Honest holdout: re-measure BOTH baseline and winner at a different wired
        # operating point and require the candidate to still beat baseline there. The
        # holdout A/B must ALSO be same-caliber (Codex R6): if the holdout pair's provenance
        # is missing or mismatched, the holdout is unusable -> we do NOT feed its metrics.
        hbase_vals = hcand_vals = None
        holdout_cal = "skipped"
        if run_holdout:
            h_base = _holdout_config(base_config, avoid_knob)
            h_win = _holdout_config(winner_cfg, avoid_knob)
            if h_base is not None and h_win is not None:
                hbase_prof = eval_fn(h_base)
                hcand_prof = eval_fn(h_win)
                hbc_b, hbc_c = hbase_prof.bench_config, hcand_prof.bench_config
                if not hbc_b or not hbc_c:
                    holdout_cal = "holdout_same_caliber_unverifiable"
                else:
                    h_ok, h_diffs = same_caliber(
                        BenchConfig.from_dict(hbc_b),
                        BenchConfig.from_dict(hbc_c),
                        exempt=searched_knobs,
                    )
                    if not h_ok:
                        holdout_cal = "holdout_same_caliber_mismatch"
                        rec["holdout_same_caliber_diffs"] = h_diffs
                    else:
                        holdout_cal = "ok"
                        hbase_vals = hbase_prof.per_seed.get(spec.metric)
                        hcand_vals = hcand_prof.per_seed.get(spec.metric)
        rec["holdout_same_caliber"] = holdout_cal
        verdict = verify_gain(
            base_prof.per_seed.get(spec.metric, []),
            cand_prof.per_seed.get(spec.metric, []),
            spec,
            candidate_verified=cand_prof.marker_verified,
            holdout_baseline_values=hbase_vals or None,
            holdout_candidate_values=hcand_vals or None,
            baseline_bottleneck=diag.bottleneck,
            candidate_bottleneck=diagnose(cand_prof).bottleneck,
        )
        rec["verdict"] = verdict.to_dict()

        # AC1: enforce SAME CALIBER in the autopt loop too (Codex R5). Baseline + candidate
        # must have run the same口径; missing/mismatched provenance is a hard fail, never adopt.
        cal_status = "ok"
        bc_b, bc_c = base_prof.bench_config, cand_prof.bench_config
        if not bc_b or not bc_c:
            cal_status = "same_caliber_unverifiable"
        else:
            cal_ok, cal_diffs = same_caliber(
                BenchConfig.from_dict(bc_b), BenchConfig.from_dict(bc_c), exempt=searched_knobs
            )
            if not cal_ok:
                cal_status = "same_caliber_mismatch"
                rec["same_caliber_diffs"] = cal_diffs
        rec["same_caliber"] = cal_status

        # AC5: MEASURED quality (Codex R5) replaces inferred-by-construction. measured_quality_
        # verdict returns None when not measurable -> is_quality_certified False -> no adoption.
        q_verdict = None
        if quality_measure_fn is not None:
            q_verdict = measured_quality_verdict(
                quality_measure_fn(base_config), quality_measure_fn(winner_cfg)
            )
        quality_certified = is_quality_certified(q_verdict)
        rec["quality_verdict"] = q_verdict.to_dict() if q_verdict is not None else None

        # A CODE candidate must be proven EFFECTIVE by the REAL bench (invoked + reordered + no
        # fallback, recorded as eval_result["effective"] in native_bench), not merely have a
        # non-empty value — a policy that loads and falls back must NOT adopt (Codex review P1). A
        # CONFIG candidate's searched knob is effective by construction. Off-box / synthetic
        # profiles carry effective=False, so they can never clear this on top of LOCK D.
        if is_code:
            candidate_effective = bool((cand_prof.eval_result or {}).get("effective"))
        else:
            candidate_effective = bool(cand.value)
        # Strict strong-baseline gate BINDS adoption (Codex HIGH): hard X% on point + CI-low,
        # frozen holdout, explicit effective + MEASURED quality. A 'better' verify_gain under
        # the 2% epsilon path alone can NOT adopt.
        strict = accept_vs_strong_baseline(
            base_prof.per_seed.get(spec.metric, []),
            cand_prof.per_seed.get(spec.metric, []),
            threshold_pct=gate_threshold,
            higher_is_better=(spec.direction != "min"),
            candidate_effective=candidate_effective,
            quality_ok=quality_certified,
            holdout_baseline=hbase_vals or None,
            holdout_candidate=hcand_vals or None,
            # LOCK D at accept: a synthetic (local_smoke) profile is refused on metadata alone,
            # independent of the effective/quality booleans.
            candidate_meta={"source": cand_prof.source, "outcome_class": cand_prof.outcome_class},
            baseline_meta={"source": base_prof.source, "outcome_class": base_prof.outcome_class},
        )
        rec["strict_accept"] = strict.to_dict()
        rounds.append(rec)

        if (
            verdict.verdict == "better"
            and strict.accepted
            and cal_status == "ok"
            and holdout_cal in ("ok", "skipped")
        ):
            adopted = {
                "target": target["target"],
                "config": winner_cfg,
                "candidate": cand.to_dict(),
                "verdict": verdict.to_dict(),
                "strict_accept": strict.to_dict(),
            }
            if verdict.bottleneck_shifted:
                base_config = winner_cfg  # adopt and re-diagnose (bottleneck moved)
                continue
            break  # adopted, bottleneck stable -> done
        # needs_holdout / inconclusive / overfit / worse -> try the next target

    return {
        "outcome": "adopted" if adopted else "dod_b",
        "adopted": adopted,
        "rounds": rounds,
        "conclusion": _conclude(adopted, spec, rounds),
        "spec": spec.to_dict(),
    }
