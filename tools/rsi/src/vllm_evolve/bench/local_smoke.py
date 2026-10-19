"""In-process ``local_smoke`` bench backend — SYNTHETIC plumbing only, never a real run.

This lets the WHOLE ``ar`` flow (+ a full ``autopt`` loop) run end-to-end on a laptop with no
GPU, so cross-verb wiring is exercised off-box. Its result is quarantined by FOUR independent
locks, any ONE of which makes it impossible to ever become a real gain / keep / AC6:

* LOCK A — ``source='local_smoke'`` (a schema enum member; NEVER ``'real_vllm'``).
* LOCK B — ``outcome_class='local_smoke_nonqualifying'`` (a FAILURE class; ``is_success()`` False).
* LOCK C — ``marker_verified/effective=False``, ``quality_ok=None``, and the output NEVER contains
  the plugin marker namespace (so the effectiveness/quality gates have nothing to accept).
* LOCK D — a shared real-source guard at compare / verify-gain / accept / keep (see ar_cli /
  core.accept) HARD-REFUSES any non-``real_vllm`` input.

Selected ONLY by an explicit ``--backend local_smoke`` / ``VE_BENCH_BACKEND=local_smoke``; the
default backend is ``remote`` (the real SSH/vLLM path). Numbers are derived deterministically from
``hash(policy bytes + same-caliber knobs)`` so artifacts are stable and a candidate differs from a
baseline only when the policy bytes differ — but they are SYNTHETIC and carry no perf meaning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from vllm_evolve.bench.eval_result import (
    SCHEMA_VERSION,
    SCORE_AGG_FORMULA_VERSION,
    sha256_text,
)
from vllm_evolve.bench.outcome import OutcomeClass
from vllm_evolve.bench.runner import aggregate

LOCAL_SMOKE_SOURCE = "local_smoke"
SMOKE_BANNER = (
    "=== LOCAL SMOKE — synthetic numbers, NOT a real vLLM run; "
    "can never be a gain/keep/AC6 result. ==="
)


def selected_backend(args=None) -> str:
    """Resolve the bench backend (shared by ``cmd_bench`` and the ``autopt`` eval seam):
    an explicit ``--backend`` wins, else ``VE_BENCH_BACKEND``, else the real ``remote`` path."""
    import os
    b = getattr(args, "backend", None) if args is not None else None
    return b or os.environ.get("VE_BENCH_BACKEND") or "remote"


@dataclass
class LocalSmokeResult:
    """What the seam returns in place of a real ``RemoteBench`` (same ``to_dict`` shape)."""

    eval_result: dict
    banner: str = SMOKE_BANNER

    def to_dict(self) -> dict:
        # LOCK C surfaced on the result object too: never effective, never quality-certified.
        return {
            "banner": self.banner,
            "eval_result": self.eval_result,
            "marker_verified": False,
            "effective": False,
            "quality_ok": None,
        }


def _policy_source(config) -> str:
    from pathlib import Path
    p = getattr(config.runner, "policy_path", None)
    if p and Path(p).exists():
        return Path(p).read_text(encoding="utf-8")
    return ""


def run_local_smoke(config) -> LocalSmokeResult:
    """Build a schema-valid but structurally non-promotable smoke ``eval_result``."""
    prov = config.provenance()
    policy_sha = sha256_text(_policy_source(config))
    metric = config.statistical.primary_metric
    regime = config.workload.regime
    # determinism keyed on policy bytes + the SAME-CALIBER knobs (NOT runner_kind), so two
    # same-caliber runs differ only when their policy bytes differ.
    caliber = f"{prov['model_served']}|{regime}|{metric}|{'|'.join(prov['rendered_serve_args'])}"
    key = sha256_text(policy_sha + "::" + caliber)
    base = 100.0 + (int(key[:8], 16) % 5000) / 100.0

    seeds = [int(s) for s in (config.statistical.seed_tiers or [0, 1, 2])]
    per_seed = []
    for s in seeds:
        jitter = (int(sha256_text(f"{key}:{s}")[:4], 16) % 100) / 100.0
        per_seed.append({
            "seed": s,
            "primary_value": round(base + jitter, 4),
            "metrics": {metric: round(base + jitter, 4)},
            "source": LOCAL_SMOKE_SOURCE,
        })
    agg = aggregate([p["primary_value"] for p in per_seed])

    eval_result = {
        "schema_version": SCHEMA_VERSION,
        "source": LOCAL_SMOKE_SOURCE,                                  # LOCK A
        "policy_sha256": policy_sha,
        "config_sha256": sha256_text(json.dumps(prov["bench_config"], sort_keys=True)),
        "git_sha": "",
        "vllm_version": "local-smoke",
        "vllm_commit": None,
        "cuda_version": None,
        "gpu_driver_version": None,
        "hardware_profile": "local-smoke-cpu",
        "hardware_sku": None,
        "docker_image_digest": None,
        "profile": regime,
        "regime": regime,
        "primary_metric": metric,
        "seeds": seeds,
        "sample_count": len(per_seed),
        "raw_per_seed_metrics": per_seed,
        "aggregate_metrics": {"primary_metric": metric, **agg.to_dict()},
        "slo": dict(config.statistical.slo or {}),
        "goodput": None,
        "outcome_class": OutcomeClass.LOCAL_SMOKE_NONQUALIFYING.value,  # LOCK B
        "score_aggregation_formula_version": SCORE_AGG_FORMULA_VERSION,
        "warmup_policy_version": None,
        "command_line": "LOCAL_SMOKE (no real command executed)",      # LOCK C: no marker namespace
        "wall_time_s": 0.0,
        "error_text": None,
        # provenance augmentation — same keys dispatch adds, so compare reads it identically
        "bench_config": prov["bench_config"],
        "runner_kind": prov["runner_kind"],
        "model_served": prov["model_served"],
        "rendered_serve_args": prov["rendered_serve_args"],
    }
    return LocalSmokeResult(eval_result=eval_result)
