"""Out-of-process ``frontier_sim`` bench backend — NON-REAL simulator plumbing, never a real run.

This lets the WHOLE ``ve`` flow (+ a full ``autopt`` loop) run end-to-end on a laptop with no GPU by
driving the **Frontier** discrete-event LLM-serving simulator (``NetX-lab/Frontier``) as a *local
subprocess* in its own interpreter. Baselines use Frontier's ``vllm_v1`` scheduler. Candidate runs
select the patched ``ve_policy`` scheduler, which genuinely calls the evolved
``targets/scheduling/work.py`` policy to reorder or explicitly defer waiting requests; the marker
proves the policy ran. Capacity, token-budget and KV admission remain enforced by Frontier's parent
vLLM-v1 scheduler. This makes the backend a real policy SEARCH evaluator, while its result remains
non-real and non-promotable.

Its result is quarantined by FOUR independent locks, any ONE of which makes it impossible to ever
become a real gain / keep / AC6 — identical in spirit to ``local_smoke``:

* LOCK A — ``source='frontier_sim'`` (a schema enum member; NEVER ``'real_vllm'``).
* LOCK B — ``outcome_class='simulator_nonqualifying'`` (a FAILURE class; ``is_success()`` False).
* LOCK C — ``marker_verified/effective=False``, ``quality_ok=None``, and the output NEVER contains
  the plugin marker namespace (so the effectiveness/quality gates have nothing to accept).
* LOCK D — the shared real-source guard at compare / verify-gain / accept / keep
  (``bench.eval_result.real_source_block``) HARD-REFUSES any non-``real_vllm`` input — UNCHANGED;
  a new non-real source costs the frozen judgment core zero changes.

Decoupling: Frontier is invoked OUT-OF-PROCESS via env-pointers (never a Python import dependency of
vllm-evolve). ``VE_FRONTIER_REPO`` = Frontier checkout, ``VE_FRONTIER_PYTHON`` = its interpreter.
Unset → a clean ``RuntimeError`` (this backend is opt-in; nothing else in the repo changes).

Honesty (no fabrication): every number comes from Frontier's real output files. ``primary_value`` is
the SLO-goodput ONLY when the real per-request latency columns + an SLO exist; otherwise a *named*
real aggregate from ``system_metrics.json`` (e.g. ``throughput_req_s``) + ``goodput=None`` — never a
goodput label slapped on something else. ``build_eval_result`` is deliberately NOT used (it stamps
``source='real_vllm'``); the dict is hand-built like ``local_smoke``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vllm_evolve.bench.eval_result import (
    SCHEMA_VERSION,
    SCORE_AGG_FORMULA_VERSION,
    sha256_text,
)
from vllm_evolve.bench.frontier_catalog import load_catalog
from vllm_evolve.bench.outcome import OutcomeClass
from vllm_evolve.bench.runner import aggregate

FRONTIER_SOURCE = "frontier_sim"
FRONTIER_BANNER = (
    "=== FRONTIER SIM — real simulator measurements for policy SEARCH, NOT a real vLLM run; "
    "can never be a production gain/keep/AC6. ==="
)

# Fixed single invocation (Q1): map only model + workload(n_requests) + regime; everything else is a
# constant so the backend stays decoupled and lightweight. CPU-only = co-location + analytical +
# dummy exec-time predictor + online (online+poisson gives arrival-based TTFT/e2e an SLO can read).
_FIXED_PREFILL_TOKENS = 512
_FIXED_DECODE_TOKENS = 128
_FIXED_QPS = 1.0
# 0.001ms per dummy op -> ~0.5ms/token step -> ~60ms engine time per 128-token request, so a
# CLI-reachable qps range (>=1) spans genuinely idle to genuinely overloaded regimes. At the old
# 1.0ms a single request kept the simulated engine busy ~58s and every run looked saturated.
_DUMMY_EXEC_TIME_MS = 0.001
_SUBPROC_TIMEOUT_S = 600.0


@dataclass
class FrontierSimResult:
    """What the seam returns in place of a real ``RemoteBench`` (same ``to_dict`` shape)."""

    eval_result: dict
    banner: str = FRONTIER_BANNER

    def to_dict(self) -> dict:
        # LOCK C surfaced on the result object too: never effective, never quality-certified.
        return {
            "banner": self.banner,
            "eval_result": self.eval_result,
            "marker_verified": False,
            "effective": False,
            "quality_ok": None,
        }


def resolve_frontier_env() -> tuple[str, str]:
    """Resolve (VE_FRONTIER_REPO, VE_FRONTIER_PYTHON); raise a clear error if either is unset.

    This backend is opt-in: with no Frontier configured there is nothing to run, so we refuse loudly
    rather than silently fabricate — like the ``remote`` dispatch on an unreachable box.
    """
    repo = os.environ.get("VE_FRONTIER_REPO")
    python_bin = os.environ.get("VE_FRONTIER_PYTHON")
    pairs = (("VE_FRONTIER_REPO", repo), ("VE_FRONTIER_PYTHON", python_bin))
    missing = [n for n, v in pairs if not v]
    if missing:
        raise RuntimeError(
            f"frontier_sim backend needs {' and '.join(missing)} set (the Frontier checkout + its "
            "interpreter). It runs Frontier as a local subprocess and is opt-in; configure the env "
            "vars or use --backend local_smoke / remote."
        )
    return repo, python_bin  # type: ignore[return-value]


def build_frontier_argv(
    python_bin: str, config, *, out_dir: str, run_id: str, seed: int
) -> list[str]:
    """Map a ``BenchConfig`` to the CPU-only Frontier CLI (one online co-location run).

    Knob mapping (gap-2) — each searched knob genuinely changes the simulation:
    * ``engine.max_num_seqs``      -> ``--vllm_v1_scheduler_config_batch_size_cap``
      (Frontier's own help text: "max_num_seqs in vLLM")
    * ``engine.max_num_batched_tokens`` -> ``--vllm_v1_scheduler_config_max_tokens_in_batch``
      ("max_num_batched_tokens in vLLM v1")
    * load -> ``--poisson_request_interval_generator_config_qps``: ``workload.arrival_rate_qps``
      when set; else ``workload.concurrency`` 1:1 as an open-loop pressure approximation
      (Frontier's synthetic generator is open-loop; the mapping is monotone and documented,
      which is what differential search needs); else the 1.0 default.
    Knobs with no Frontier analogue (gpu_memory_utilization, quantization, ...) are NOT mapped.

    Policy execution (Phase D): a candidate WITH a policy file selects the ``ve_policy``
    scheduler (the bridge patch in integrations/frontier/), so Frontier genuinely executes the
    evolved ``schedule_batch``; scheduler config flags switch prefix accordingly. Baselines run
    vanilla ``vllm_v1``.
    """
    model = config.model_to_serve()
    n_requests = int(config.workload.n_requests)
    qps = (config.workload.arrival_rate_qps
           or (float(config.workload.concurrency) if config.workload.concurrency else None)
           or _FIXED_QPS)
    sched = "ve_policy" if _policy_mode(config) else "vllm_v1"
    extra: list[str] = []
    if getattr(config.engine, "max_num_seqs", None):
        extra += [f"--{sched}_scheduler_config_batch_size_cap", str(config.engine.max_num_seqs)]
    if getattr(config.engine, "max_num_batched_tokens", None):
        extra += [f"--{sched}_scheduler_config_max_tokens_in_batch",
                  str(config.engine.max_num_batched_tokens)]
    return [
        python_bin, "-m", "frontier.main",
        "--simulation_mode", "online",
        "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical",
        "--cluster_config_num_replicas", "1",
        "--replica_config_model_name", model,
        "--replica_config_attn_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_config_attn_data_parallel_size", "1",
        "--replica_scheduler_config_type", sched,
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(n_requests),
        *_length_args(getattr(config.workload, "workload_spec", None) or {}),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(qps),
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        str(_DUMMY_EXEC_TIME_MS),
        "--metrics_config_output_dir", out_dir,
        "--metrics_config_run_id", run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--seed", str(seed),
    ] + extra


def _length_args(workload_spec: dict) -> list[str]:
    """Request-length distribution args. Default: the fixed lengths. ``workload_spec``
    {"length_dist": "uniform", "min_tokens": N, "max_tokens": M, "prefill_to_decode_ratio": R}
    selects Frontier's uniform generator (heterogeneous lengths — the regime where ordering
    policies genuinely differ)."""
    if workload_spec.get("length_dist") == "uniform":
        return [
            "--length_generator_config_type", "uniform",
            "--uniform_request_length_generator_config_min_tokens",
            str(int(workload_spec.get("min_tokens", 64))),
            "--uniform_request_length_generator_config_max_tokens",
            str(int(workload_spec.get("max_tokens", 1024))),
            "--uniform_request_length_generator_config_prefill_to_decode_ratio",
            str(float(workload_spec.get("prefill_to_decode_ratio", 4.0))),
        ]
    return [
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(_FIXED_PREFILL_TOKENS),
        "--fixed_request_length_generator_config_decode_tokens", str(_FIXED_DECODE_TOKENS),
    ]


def _policy_mode(config) -> bool:
    """True when this run must execute the candidate policy inside Frontier (ve_policy)."""
    p = getattr(config.runner, "policy_path", None)
    return bool(p) and getattr(config.runner, "runner_kind", "") == "candidate"


def _invoke_frontier(argv: list[str], repo: str, out_dir: str,
                     extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run Frontier as a local subprocess (its own interpreter). Single testable seam — unit tests
    monkeypatch THIS to drop fixture metrics into ``out_dir`` without a real Frontier install."""
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_DISABLED"] = "true"
    env["VIDUR_DISABLE_WANDB"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        argv, cwd=repo, env=env, capture_output=True, text=True, timeout=_SUBPROC_TIMEOUT_S
    )


def parse_frontier_metrics(out_dir: str, run_id: str) -> dict:
    """Read Frontier's real output files into the aggregates + bottleneck signals we consume.
    No fabrication: a missing file/field becomes ``None`` (callers decide), never an invented
    number. All latency columns/stats are in MILLISECONDS (confirmed against a real run)."""
    # P2: read every output file once via the shared catalog (frontier_sim reuses it; behavior
    # unchanged — same files, same parsing). The signals below are computed exactly as before.
    cat = load_catalog(out_dir, run_id)
    mdir = cat.dir
    sysm = cat.system
    per_request = cat.rows
    columns = cat.columns
    marker = cat.marker            # ve_policy honesty marker (bridge writes at exit), or None
    tput = sysm.get("throughput_metrics") or {}
    if not isinstance(tput, dict):
        tput = {}

    # real bottleneck signals (gap-1): memory pressure, preemption, queue depth, engine duty.
    mem_util = sysm.get("memory_utilization_percent")
    kv_util = None
    if isinstance(mem_util, dict) and mem_util:
        vals = [v for v in mem_util.values() if isinstance(v, (int, float))]
        if vals:
            # Frontier's per-cluster memory utilization (weights + KV), 0-100 -> 0-1. The closest
            # real memory-pressure signal it emits; labeled kv_util for the diagnose contract.
            kv_util = round(max(vals) / 100.0, 4)
    pre = sysm.get("preemption_statistics") or {}
    preempt = pre.get("total_preemption_events") if isinstance(pre, dict) else None
    duty, running = _ledger_signals(mdir)
    waiting = _waiting_depth(per_request)

    return {
        "requests_per_second": _num(tput.get("requests_per_second")),
        "tokens_per_second": _num(tput.get("tokens_per_second")),
        "e2e_stats": sysm.get("request_e2e_time_statistics"),
        "ttft_stats": sysm.get("ttft_statistics"),
        "per_request": per_request,
        "columns": columns,
        "kv_util": kv_util,
        "preempt": preempt if isinstance(preempt, int) else None,
        "waiting": waiting,
        "running": running,
        "engine_duty": duty,
        "policy_marker": marker,
    }


def _ledger_signals(mdir: Path) -> tuple[float | None, int | None]:
    """(engine busy fraction, max in-flight batch size) from the REAL per-batch ledger
    (``frontier_stage_batch_ledger.jsonl``: stage_start_ts/stage_end_ts seconds + request_ids).
    duty = union of busy intervals / span — the simulator's own engine utilization."""
    p = mdir / "frontier_stage_batch_ledger.jsonl"
    if not p.is_file():
        return None, None
    intervals: list[tuple[float, float]] = []
    running_max = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s, e = _num(r.get("stage_start_ts")), _num(r.get("stage_end_ts"))
        if s is not None and e is not None and e > s:
            intervals.append((s, e))
        running_max = max(running_max, len(r.get("request_ids") or []))
    if not intervals:
        return None, (running_max or None)
    intervals.sort()
    span = max(e for _, e in intervals) - min(s for s, _ in intervals)
    busy, cur_s, cur_e = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    duty = round(busy / span, 4) if span > 0 else None
    return duty, (running_max or None)


def _waiting_depth(per_request: list[dict]) -> int | None:
    """Max concurrent WAITING requests, reconstructed from real per-request columns: arrivals =
    cumsum(request_inter_arrival_delay seconds), waiting window = [arrival, arrival +
    request_first_scheduling_delay ms]. Plain interval-overlap math on real timestamps."""
    if not per_request:
        return None
    cols = per_request[0].keys()
    if "request_inter_arrival_delay" not in cols or "request_first_scheduling_delay" not in cols:
        return None
    events: list[tuple[float, int]] = []
    t_ms = 0.0
    for row in per_request:
        gap_s = _num(row.get("request_inter_arrival_delay"))
        t_ms += (gap_s or 0.0) * 1000.0
        delay_ms = _num(row.get("request_first_scheduling_delay"))
        if delay_ms is None:
            continue
        events.append((t_ms, +1))
        events.append((t_ms + delay_ms, -1))
    if not events:
        return None
    events.sort()
    depth = peak = 0
    for _, d in events:
        depth += d
        peak = max(peak, depth)
    return peak


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compute_goodput(parsed: dict, slo: dict) -> float | None:
    """Goodput (req/s) derived ENTIRELY from real Frontier per-request latencies + the SLO: the
    real throughput scaled by the fraction of requests meeting every SLO bound. Returns None
    if the needed real per-request columns or rps are absent (caller uses a named aggregate)."""
    rows = parsed["per_request"]
    rps = parsed["requests_per_second"]
    if not rows or rps is None:
        return None
    # SLO keys we know how to enforce, mapped to the real CSV column. The CSV latency columns are
    # in MILLISECONDS (same scale as the SLO; confirmed against a real run + system stats).
    bound_cols = {"ttft_ms": "ttft", "e2e_ms": "request_e2e_time", "tpot_ms": "tpot"}
    cols = parsed["columns"]
    active = {k: bound_cols[k] for k in slo if k in bound_cols and bound_cols[k] in cols}
    if not active:
        return None
    met = 0
    for row in rows:
        ok = True
        for slo_key, col in active.items():
            val_ms = _num(row.get(col))
            if val_ms is None or val_ms > float(slo[slo_key]):
                ok = False
                break
        met += 1 if ok else 0
    return round(rps * (met / len(rows)), 6)


# Latency aggregate metrics (e.g. ttft_p99_ms / ttft_mean_ms / e2e_p90_ms) read straight from
# Frontier's real *_statistics dicts (values already in ms); the metric's base picks the stats dict.
_LATENCY_METRIC = re.compile(r"^(ttft|e2e)_(p\d{1,3}|mean|median|min|max)_ms$")
_STATS_KEY = {"ttft": "ttft_stats", "e2e": "e2e_stats"}


def _primary_value(parsed: dict, config) -> tuple[str, float]:
    """Resolve (metric_name, value) honestly. goodput ONLY with real latency cols + SLO; a latency
    aggregate (ttft/e2e percentile|mean) from the real *_statistics dict; else a named real
    throughput aggregate from system_metrics.json. Raises if none is available (never fabricate)."""
    metric = config.statistical.primary_metric
    slo = config.statistical.slo or {}
    if metric.startswith("goodput"):
        gp = _compute_goodput(parsed, slo)
        if gp is not None:
            return metric, gp
        # fall through to a named aggregate; goodput stays None at the eval_result level.
    lat = _LATENCY_METRIC.match(metric)
    if lat:
        stats = parsed.get(_STATS_KEY[lat.group(1)])
        v = _num(stats.get(lat.group(2))) if isinstance(stats, dict) else None
        if v is not None:                       # the REAL p99/mean/... from Frontier's stats (ms)
            return metric, v
        raise RuntimeError(
            f"Frontier produced no real {metric} (missing {_STATS_KEY[lat.group(1)]}."
            f"{lat.group(2)} in system_metrics.json); refusing to fabricate a primary_value."
        )
    if "tok" in metric and parsed["tokens_per_second"] is not None:
        return metric, parsed["tokens_per_second"]
    if parsed["requests_per_second"] is not None:
        return "throughput_req_s", parsed["requests_per_second"]
    raise RuntimeError(
        "Frontier produced no usable real aggregate (no throughput in system_metrics.json and no "
        "SLO-computable goodput from request_metrics.csv); refusing to fabricate a primary_value."
    )


def _policy_source(config) -> str:
    p = getattr(config.runner, "policy_path", None)
    if p and Path(p).exists():
        return Path(p).read_text(encoding="utf-8")
    return ""


def _failed_eval_result(config, *, error_text: str, metric: str, seed: int) -> dict:
    """A schema-valid FAILED eval_result (clean Frontier non-zero exit / OOM): keeps the
    locks so the round still converges to DoD-B instead of crashing the loop. One per-seed entry
    records the failure (no fabricated metric — the outcome_class + error_text say it failed)."""
    prov = config.provenance()
    per_seed = [{"seed": seed, "error": error_text[:500], "source": FRONTIER_SOURCE}]
    return _base_eval_result(
        config, prov, metric=metric, seeds=[seed], per_seed=per_seed, agg=None,
        goodput=None, error_text=error_text,
    )


def _base_eval_result(config, prov, *, metric, seeds, per_seed, agg, goodput, error_text) -> dict:
    policy_sha = sha256_text(_policy_source(config))
    regime = config.workload.regime
    aggregate_metrics = {"primary_metric": metric}
    if agg is not None:
        aggregate_metrics.update(agg.to_dict())
    return {
        "schema_version": SCHEMA_VERSION,
        "source": FRONTIER_SOURCE,                                       # LOCK A
        "policy_sha256": policy_sha,
        "config_sha256": sha256_text(json.dumps(prov["bench_config"], sort_keys=True)),
        "git_sha": "",
        "vllm_version": "frontier-sim",
        "vllm_commit": None,
        "cuda_version": None,
        "gpu_driver_version": None,
        "hardware_profile": "frontier-sim-cpu",
        "hardware_sku": None,
        "docker_image_digest": None,
        "profile": regime,
        "regime": regime,
        "primary_metric": metric,
        "seeds": seeds,
        "sample_count": len(per_seed),
        "raw_per_seed_metrics": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "slo": dict(config.statistical.slo or {}),
        "goodput": goodput,
        "outcome_class": OutcomeClass.SIMULATOR_NONQUALIFYING.value,     # LOCK B
        "score_aggregation_formula_version": SCORE_AGG_FORMULA_VERSION,
        "warmup_policy_version": None,
        "command_line": "FRONTIER_SIM (out-of-process Frontier; no plugin marker)",  # LOCK C
        "wall_time_s": 0.0,
        "error_text": error_text,
        # provenance augmentation — same keys local_smoke adds, so compare reads it identically
        "bench_config": prov["bench_config"],
        "runner_kind": prov["runner_kind"],
        "model_served": prov["model_served"],
        "rendered_serve_args": prov["rendered_serve_args"],
    }


def run_frontier_sim(config) -> FrontierSimResult:
    """Drive Frontier (local subprocess) per seed, parse its REAL output, build a non-promotable
    frontier_sim ``eval_result``. env/timeout → raise; a clean Frontier non-zero exit
    → a schema-valid FAILED eval_result (still DoD-B)."""
    repo, python_bin = resolve_frontier_env()
    prov = config.provenance()
    metric = config.statistical.primary_metric
    # N4: normalize seeds at entry (seed_tiers may be None), then respect the max_seeds cap.
    seeds = [int(s) for s in (config.statistical.seed_tiers or [0, 1, 2])]
    if config.statistical.max_seeds:
        seeds = seeds[: config.statistical.max_seeds]

    policy_mode = _policy_mode(config)
    per_seed: list[dict] = []
    resolved_metric = metric
    seed_signals: list[dict] = []
    policy_marker: dict | None = None
    with tempfile.TemporaryDirectory(prefix="ve_frontier_") as tmp:
        for s in seeds:
            out_dir = str(Path(tmp) / f"seed_{s}")
            run_id = f"ve_frontier_{s}"
            argv = build_frontier_argv(python_bin, config, out_dir=out_dir, run_id=run_id, seed=s)
            extra_env = None
            if policy_mode:
                extra_env = {
                    "VE_POLICY_PATH": str(Path(config.runner.policy_path).resolve()),
                    "VE_POLICY_ROOT": _policy_root(),
                    "VE_MARKER_DIR": out_dir,
                }
            try:
                proc = _invoke_frontier(argv, repo, out_dir, extra_env=extra_env)
            except subprocess.TimeoutExpired as exc:  # operator/perf error → raise
                msg = f"frontier_sim subprocess timed out after {exc.timeout}s"
                raise RuntimeError(msg) from exc
            if proc.returncode != 0:  # clean Frontier failure → FAILED eval_result (still DoD-B)
                err = (proc.stderr or proc.stdout or "")[-1500:]
                return FrontierSimResult(
                    eval_result=_failed_eval_result(
                        config, error_text=f"Frontier exit {proc.returncode}: {err}",
                        metric=metric, seed=s)
                )
            parsed = parse_frontier_metrics(out_dir, run_id)
            if policy_mode:
                bad = _marker_violation(parsed.get("policy_marker"))
                if bad:  # honesty guard: never score baseline behavior as the candidate
                    return FrontierSimResult(
                        eval_result=_failed_eval_result(
                            config, error_text=f"ve_policy marker violation: {bad}",
                            metric=metric, seed=s)
                    )
                policy_marker = parsed.get("policy_marker")
            mname, val = _primary_value(parsed, config)
            resolved_metric = mname
            seed_signals.append({k: parsed.get(k) for k in
                                 ("kv_util", "preempt", "waiting", "running", "engine_duty")})
            per_seed.append({
                "seed": s,
                "primary_value": round(val, 6),
                "metrics": {mname: round(val, 6)},
                "source": FRONTIER_SOURCE,
            })

    agg = aggregate([p["primary_value"] for p in per_seed]) if per_seed else None
    # goodput field is an object|null (schema), derived from the aggregate — never a raw float.
    is_goodput = resolved_metric.startswith("goodput") and agg is not None
    goodput = {"median_req_s": agg.median} if is_goodput else None
    eval_result = _base_eval_result(
        config, prov, metric=resolved_metric, seeds=seeds, per_seed=per_seed, agg=agg,
        goodput=goodput, error_text=None,
    )
    # gap-1: surface the REAL simulator bottleneck signals (informational; every value comes from
    # Frontier's own output files). Pressure signals aggregate by peak, duty by mean.
    eval_result["sim_signals"] = _aggregate_signals(seed_signals)
    if policy_marker:
        # Phase D (informational, search-only): proof the evolved policy actually ran in-sim.
        # The quarantine locks are untouched — marker_verified/effective stay False.
        eval_result["sim_policy_sha"] = policy_marker.get("policy_sha256")
        eval_result["sim_marker"] = policy_marker
    return FrontierSimResult(eval_result=eval_result)


def _policy_root() -> str:
    """Repo root exported as VE_POLICY_ROOT so a policy can import targets.scheduling.skeleton.
    Prefer the cwd when it holds targets/ (normal `ve` usage); fall back to this checkout."""
    cwd = Path.cwd()
    if (cwd / "targets").is_dir():
        return str(cwd)
    return str(Path(__file__).resolve().parents[3])


def _marker_violation(marker: dict | None) -> str | None:
    """The honesty rule for policy-mode runs: the marker must prove the policy genuinely drove
    scheduling. Missing marker / zero invocations / all-fallback => violation string."""
    if not isinstance(marker, dict):
        return "marker file missing"
    inv = int(marker.get("invocations") or 0)
    fb = int(marker.get("fallbacks") or 0)
    if inv <= 0:
        return "policy was never invoked"
    if fb >= inv:
        return f"policy fell back on every invocation ({fb}/{inv})"
    return None


def _aggregate_signals(seed_signals: list[dict]) -> dict:
    def _peak(key):
        vals = [s.get(key) for s in seed_signals if s.get(key) is not None]
        return max(vals) if vals else None

    duties = [s.get("engine_duty") for s in seed_signals if s.get("engine_duty") is not None]
    return {
        "kv_util": _peak("kv_util"),
        "preempt": _peak("preempt"),
        "waiting": _peak("waiting"),
        "running": _peak("running"),
        "engine_duty": round(sum(duties) / len(duties), 4) if duties else None,
    }
