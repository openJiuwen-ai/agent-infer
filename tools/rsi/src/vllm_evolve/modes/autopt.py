"""autopt mode — goal-directed automatic optimization (the L1->L4 auto flow).

A thin wrapper over :func:`vllm_evolve.engine.orchestrate.run_autopt` (whose
internal *research* step uses ``ve-research`` + ``knowledge/``). It changes no
gate behavior: the orchestrator adopts a candidate only via the frozen
bootstrap+holdout accept gate, and a ``local_smoke`` backend runs the whole loop
off-box and can only conclude **DoD-B** (never a gain/keep/AC6). This is the
``modes/`` shape that ``tune`` / ``port`` follow.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_evolve.core.schemas import Spec
from vllm_evolve.engine.orchestrate import run_autopt


def run(
    spec: Spec,
    *,
    eval_fn: Callable,
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
    backend: str | None = None,
) -> dict:
    """Run the auto flow toward ``spec`` and return the orchestrator's result dict.

    ``eval_fn`` is the backend-selected profiler (real vs ``local_smoke``). The
    accept threshold is a config/CLI decision, never taken from ``spec`` (M4).
    ``author_fn``/``evolve_params``/``store`` drive the Evolution-v2 generational
    engine on the code target (search-only; the gate is unchanged).
    ``quality_measure_fn`` (real = box-gated served-model measure, off-box = None)
    is forwarded UNCHANGED to the frozen accept gate: with None, quality is NOT
    measured and no candidate can adopt — so off-box backends stay DoD-B."""
    res = run_autopt(
        spec,
        eval_fn=eval_fn,
        evolve_fn=evolve_fn,
        author_fn=author_fn,
        evolve_params=evolve_params,
        research_fn=research_fn,
        store=store,
        quality_measure_fn=quality_measure_fn,
        base_config=base_config or {},
        max_rounds=max_rounds,
        max_evals=max_evals,
        run_holdout=run_holdout,
        accept_threshold_pct=accept_threshold_pct,
    )
    if backend is not None:
        res["backend"] = backend
    return res
