"""tune mode — optimize / tune a GIVEN policy (its serving config).

Minimal viable: seed the auto loop with the provided policy and search its serving
config (no code evolution), reusing :func:`vllm_evolve.modes.autopt.run` →
``engine.orchestrate.run_autopt`` so every gate is preserved (bootstrap+holdout
accept, ``real_source_block``). A ``local_smoke`` backend runs plumbing only and
can never conclude a gain/keep/AC6. Deep policy-specific tuning is future work.
"""
from __future__ import annotations

from collections.abc import Callable

from vllm_evolve.core.schemas import Spec
from vllm_evolve.modes import autopt as autopt_mode


def run(
    policy_path: str,
    spec: Spec,
    *,
    eval_fn: Callable,
    quality_measure_fn: Callable | None = None,
    base_config: dict | None = None,
    max_rounds: int = 2,
    max_evals: int = 8,
    run_holdout: bool = True,
    accept_threshold_pct: float | None = None,
    backend: str | None = None,
) -> dict:
    """Tune the GIVEN policy's config toward ``spec``; returns the verdict dict.

    ``evolve_fn`` is None (tune does not evolve new code — it optimizes config around
    the supplied policy). Adoption stays in the frozen accept gate inside run_autopt;
    ``quality_measure_fn`` (None off-box) is forwarded there unchanged."""
    base = dict(base_config or {})
    base["policy"] = str(policy_path)
    res = autopt_mode.run(
        spec, eval_fn=eval_fn, quality_measure_fn=quality_measure_fn, base_config=base,
        max_rounds=max_rounds, max_evals=max_evals, run_holdout=run_holdout,
        accept_threshold_pct=accept_threshold_pct, backend=backend,
    )
    res["mode"] = "tune"
    res["policy"] = str(policy_path)
    return res
