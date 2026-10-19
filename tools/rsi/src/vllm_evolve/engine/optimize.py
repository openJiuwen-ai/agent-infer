"""L3 optimize — search a target's space on the real backend.

Given a target (a config knob + ``search_space``), propose configs, evaluate each
(real bench via ``collect_profile``, or an injected ``eval_fn`` for tests), and
return the best :class:`Candidate`. Two strategies: ``grid`` (exhaustive product
over the given knobs) and ``coordinate`` (descend one knob at a time — far fewer
evals when several knobs are open). Reuses an :class:`EvalBudget` (hard cap on
real evals) and a result cache (identical configs are evaluated once).

Anti-fabrication (the prime directive): an evaluation whose Profile is NOT
``marker_verified`` scores ``None`` and can never be selected as the winner — a
config only wins on a real, plugin-verified run. ``eval_fn`` raising (a failed
bench) is likewise a non-scoring trial, recorded honestly, never fatal.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import product

from vllm_evolve.core.schemas import Candidate, Profile, Spec

EvalFn = Callable[[dict], Profile]


class EvalBudget:
    """Hard cap on the number of (real) evaluations."""

    def __init__(self, max_evals: int = 8) -> None:
        self.max_evals = max_evals
        self.used = 0

    def exhausted(self) -> bool:
        return self.used >= self.max_evals

    def spend(self) -> None:
        self.used += 1


def _config_key(config: dict) -> str:
    return repr(sorted((k, repr(v)) for k, v in config.items()))


def _score(profile: Profile, spec: Spec) -> float | None:
    """Objective value (higher = better internally). None if unusable/unverified."""
    if profile is None or not profile.marker_verified:
        return None                       # anti-fabrication: unverified never scores
    v = (profile.metrics or {}).get(spec.metric)
    if not isinstance(v, (int, float)):
        return None
    return -float(v) if spec.direction == "min" else float(v)


def _evaluate(config: dict, eval_fn: EvalFn, budget: EvalBudget, cache: dict) -> Profile | None:
    key = _config_key(config)
    if key in cache:
        return cache[key]                 # cache hit does NOT spend budget
    if budget.exhausted():
        return None
    budget.spend()
    try:
        prof = eval_fn(config)
    except Exception:  # noqa: BLE001 - a failed bench is a non-scoring trial, not fatal
        prof = None
    cache[key] = prof
    return prof


def _better(score: float | None, best: float | None) -> bool:
    return score is not None and (best is None or score > best)


def grid_search(knobs: dict[str, list], base_config: dict, spec: Spec, eval_fn: EvalFn,
                budget: EvalBudget, cache: dict | None = None) -> Candidate:
    """Exhaustive product over ``knobs`` (cartesian), bounded by ``budget``."""
    cache = cache if cache is not None else {}
    keys = list(knobs)
    best_val: dict | None = None
    best_score: float | None = None
    best_prof: Profile | None = None
    trials: list = []

    for combo in product(*(knobs[k] for k in keys)):
        if budget.exhausted():
            break
        val = dict(zip(keys, combo))
        prof = _evaluate({**base_config, **val}, eval_fn, budget, cache)
        sc = _score(prof, spec)
        trials.append({"value": val, "score": sc,
                       "verified": bool(prof and prof.marker_verified)})
        if _better(sc, best_score):
            best_val, best_score, best_prof = val, sc, prof

    return _to_candidate(spec, base_config, best_val, best_score, best_prof, budget, trials,
                         strategy="grid")


def coordinate_descent(knobs: dict[str, list], base_config: dict, spec: Spec, eval_fn: EvalFn,
                       budget: EvalBudget, cache: dict | None = None) -> Candidate:
    """Optimize one knob at a time, holding the rest at their current best."""
    cache = cache if cache is not None else {}
    current = dict(base_config)
    best_val: dict = {}
    best_score: float | None = None
    best_prof: Profile | None = None
    trials: list = []

    for knob, space in knobs.items():
        for v in space:
            if budget.exhausted():
                break
            cfg = {**current, **best_val, knob: v}
            prof = _evaluate(cfg, eval_fn, budget, cache)
            sc = _score(prof, spec)
            trials.append({"value": {knob: v}, "score": sc,
                           "verified": bool(prof and prof.marker_verified)})
            if _better(sc, best_score):
                best_score, best_prof = sc, prof
                best_val = {**best_val, knob: v}

    return _to_candidate(spec, base_config, best_val or None, best_score, best_prof, budget,
                         trials, strategy="coordinate")


def _to_candidate(spec: Spec, base_config: dict, best_val: dict | None, best_score: float | None,
                  best_prof: Profile | None, budget: EvalBudget, trials: list,
                  strategy: str) -> Candidate:
    target = f"config:{','.join(best_val)}" if best_val else "config:none"
    # report the metric in its natural sign (we maximize -metric for 'min')
    raw = None
    if best_score is not None:
        raw = -best_score if spec.direction == "min" else best_score
    note = strategy
    if best_val is None:
        note = (f"{strategy}: no verified-improving config found over {budget.used} evals "
                f"(all trials unverified or non-scoring)")
    return Candidate(
        target=target, kind="config", value=best_val or {},
        config={**base_config, **(best_val or {})},
        metrics=(best_prof.metrics if best_prof else {}),
        score=raw, marker_verified=bool(best_prof and best_prof.marker_verified),
        evals_used=budget.used, trials=trials, note=note,
    )


def optimize(knobs: dict[str, list], base_config: dict, spec: Spec, eval_fn: EvalFn,
             strategy: str = "grid", max_evals: int = 8,
             cache: dict | None = None) -> Candidate:
    budget = EvalBudget(max_evals)
    fn = coordinate_descent if strategy == "coordinate" else grid_search
    return fn(knobs, base_config, spec, eval_fn, budget, cache)
