"""R6 — the research chain: register prediction -> run experiment -> adjudicate -> append ledger.

``run_hypothesis`` wires P1 (experiment) + P3 (adjudicate + ledger) into the one operation the
``ve experiment run`` admin verb performs. It is the EXACT chain the H* acceptance loop needs:
express (prediction registered FIRST) -> measure+execute (run_experiment) -> adjudicate (the
deterministic pure function) -> ledger (append-only execution + adjudication events). It NEVER
fabricates: a missing required measurement flows through ``per_arm=None`` to an ``inconclusive``
verdict, which is recorded honestly.

``eval_fn`` is injected (the production one drives ``frontier_sim``; tests stub it), so the chain is
testable offline. engine layer; ledger rows are PROPOSAL-layer and never feed the adoption gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from vllm_evolve.engine.adjudicate import adjudicate
from vllm_evolve.engine.experiment import ExperimentSpec, run_experiment

_API_REL = ("integrations", "frontier", "ve_policy_api.py")

# Serving-knob -> Frontier scheduler-config flag suffix (same mapping as bench.frontier_sim's
# build_frontier_argv). A knob absent on an arm simply yields no flag, so arms differ by exactly
# their declared per-arm deltas.
_KNOB_FLAGS = {
    "max_num_seqs": "batch_size_cap",
    "max_num_batched_tokens": "max_tokens_in_batch",
}


def _policy_root_for(policy_path: str) -> str:
    """Repo root a policy must see on sys.path (to import the sanctioned ve_policy_api AND
    targets.scheduling.skeleton). Resolved WITHOUT the caller's cwd — a wrong cwd is the exact
    fallback that lets the policy fail to import ve_policy_api so the bridge degrades to FCFS while
    still producing real metrics. Order: (1) walk UP from the (possibly out-of-tree) policy file;
    (2) else THIS vllm-evolve checkout, derived from this module; (3) else fail loudly rather than
    launch Frontier with an unknown import root."""
    p = Path(policy_path).resolve()
    for parent in p.parents:
        if parent.joinpath(*_API_REL).is_file():
            return str(parent)
    checkout = Path(__file__).resolve().parents[3]            # <repo>/src/vllm_evolve/engine/<here>
    if checkout.joinpath(*_API_REL).is_file():
        return str(checkout)
    raise RuntimeError(
        f"cannot locate {'/'.join(_API_REL)} for policy {policy_path!r} (searched the policy's "
        f"ancestors and the vllm-evolve checkout at {checkout}); refusing to launch Frontier with "
        "an unknown VE_POLICY_ROOT (a wrong import root would silently degrade the policy to FCFS)")


def frontier_catalog_eval(arm, seed, spec, trace_path, out_dir):
    """Production eval_fn: drive Frontier (frontier_sim) on the synthesized TRACE for one arm/seed
    and return its FrontierCatalog. The arm's policy (if any) runs inside Frontier via ve_policy.
    A failed sim yields an empty catalog (explicit missing) — never fabricated numbers. The exact
    trace_replay flag wiring is exercised live by the Phase-E e2e; offline tests stub this fn."""
    from vllm_evolve.bench.frontier_catalog import load_catalog
    from vllm_evolve.bench.frontier_sim import (
        _FIXED_QPS,
        _invoke_frontier,
        resolve_frontier_env,
    )
    repo, python_bin = resolve_frontier_env()
    run_id = f"{arm.name}_{seed}"
    arm_out = os.path.join(out_dir, "arms", run_id)
    # effective knobs = spec-wide knobs overlaid by THIS arm's per-arm delta, so an A/B that varies
    # a serving knob runs two DIFFERENT Frontier configs (not two identical ones it then judges).
    spec_knobs = spec.knobs if isinstance(getattr(spec, "knobs", None), dict) else {}
    arm_knobs = getattr(arm, "knobs", None) or {}
    knobs = {**spec_knobs, **arm_knobs}
    model = knobs.get("model") or "meta-llama/Llama-2-7b-hf"
    scheduler = "ve_policy" if arm.policy_path else "vllm_v1"
    argv = [
        python_bin, "-m", "frontier.main",
        "--simulation_mode", "online", "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical", "--cluster_config_num_replicas", "1",
        "--replica_config_model_name", model,
        "--replica_config_attn_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_config_attn_data_parallel_size", "1",
        "--replica_scheduler_config_type", scheduler,
        "--request_generator_config_type", "trace_replay",
        # the TRACE_REPLAY generator's config class is TraceRequestGeneratorConfig, so the flag
        # prefix is trace_request_generator_config (NOT trace_replay_*) — verified against the real
        # Frontier --help; the wrong prefix made Frontier exit 2 and produce an empty catalog.
        "--trace_request_generator_config_trace_file", trace_path,
        "--trace_request_generator_config_max_tokens",
        str(int(knobs.get("trace_max_tokens") or 32768)),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(_FIXED_QPS),
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        str(float(knobs.get("dummy_execution_time_ms") or 1.0)),
        "--metrics_config_output_dir", arm_out, "--metrics_config_run_id", run_id,
        "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics", "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace", "--no-metrics_config_write_json_trace",
        "--seed", str(seed),
    ]
    if knobs.get("enable_prefix_caching", True):
        # Same-caliber flag: ve_policy inherits vllm_v1's scheduler config, but Frontier exposes a
        # scheduler-specific CLI prefix. Using vllm_v1's flag for a ve_policy arm silently left
        # caching disabled on the candidate and invalidated prefix-stress comparisons.
        argv.append(f"--{scheduler}_scheduler_config_enable_prefix_caching")
    for knob, suffix in _KNOB_FLAGS.items():           # per-arm serving knobs -> scheduler config
        val = knobs.get(knob)
        if val is not None:
            argv += [f"--{scheduler}_scheduler_config_{suffix}", str(val)]
    extra_env = None
    if arm.policy_path:
        policy_abs = os.path.abspath(arm.policy_path)
        extra_env = {"VE_POLICY_PATH": policy_abs,
                     "VE_POLICY_ROOT": _policy_root_for(policy_abs), "VE_MARKER_DIR": arm_out}
    command_path = Path(arm_out) / "frontier_command.json"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(json.dumps({
        "cwd": repo,
        "argv": argv,
        "seed": seed,
        "policy_path": os.path.abspath(arm.policy_path) if arm.policy_path else None,
        "policy_root": (extra_env or {}).get("VE_POLICY_ROOT"),
    }, indent=2), encoding="utf-8")
    proc = _invoke_frontier(argv, repo, arm_out, extra_env=extra_env)
    if proc.returncode != 0:
        error_path = Path(arm_out) / "frontier_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps({
            "returncode": proc.returncode,
            "argv": argv,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }, indent=2), encoding="utf-8")
    return load_catalog(arm_out, run_id)


def run_hypothesis(store, *, hypothesis_id: str, statement: str, prediction: dict,
                   spec: ExperimentSpec, eval_fn, base_out_dir: str, run_id: str = "",
                   source: str = "frontier_sim", experiment_id: str | None = None) -> dict:
    """Run one hypothesis end-to-end and append its events to the ledger. Returns a summary dict
    (verdict + the three event ids + per-arm values + a verifiable ``hypothesis:<event_id>``
    citation). Prediction is registered BEFORE the experiment runs (prediction-first is structural
    in the store)."""
    experiment_id = experiment_id or hypothesis_id
    # Prediction-FIRST, but idempotent w.r.t. the documented two-step flow (`ve experiment register`
    # then `ve experiment run`): if this hypothesis already has a prediction, REUSE it (the store
    # enforces one per hypothesis_id, so re-registering would raise). A DIFFERENT prediction for the
    # same id is a real conflict -> fail clearly rather than judge against the wrong prediction.
    existing = store.get_hypothesis(hypothesis_id)
    if existing is not None and existing.get("prediction_event_id") is not None:
        if existing.get("prediction") != prediction:
            raise ValueError(
                f"hypothesis {hypothesis_id!r} is already registered with a different prediction; "
                "use a new hypothesis_id or run with the originally registered prediction")
        pred_event_id = existing["prediction_event_id"]
    else:
        pred_event_id = store.register_prediction(
            hypothesis_id=hypothesis_id, run_id=run_id, statement=statement,
            prediction=prediction, source=source)

    result = run_experiment(spec, eval_fn, base_out_dir=base_out_dir, experiment_id=experiment_id)

    exec_event_id = store.record_execution(
        hypothesis_id=hypothesis_id, run_id=run_id,
        experiment_ref={"spec_sha": result.spec_sha, "out_dir": result.out_dir,
                        "terminated": result.terminated})

    verdict = adjudicate(prediction, result.per_arm)
    adj_event_id = store.record_adjudication(
        hypothesis_id=hypothesis_id, experiment_event_id=exec_event_id,
        verdict=verdict.verdict, measured=verdict.measured)

    return {
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "verdict": verdict.verdict,
        "measured": verdict.measured,
        "reason": verdict.reason,
        "per_arm": result.per_arm,
        "per_arm_missing": result.per_arm_missing,
        "terminated": result.terminated,
        "evals_used": result.evals_used,
        "prediction_event_id": pred_event_id,
        "execution_event_id": exec_event_id,
        "adjudication_event_id": adj_event_id,
        "citations": [f"hypothesis:{adj_event_id}"],
    }
