"""L1 profile — assemble a Profile from real signals.

Split into PURE assembly (parse a vLLM serve log + summarize nvidia-smi samples +
read an eval_result -> Profile) which is unit-tested off-GPU, and a thin
``collect_profile`` GPU wrapper that drives a real bench + a sampler. Every signal
is best-effort: absent -> ``None`` (never fabricated; the diagnoser degrades).
"""
from __future__ import annotations

import re
import statistics

from vllm_evolve.core.schemas import Profile

_PLUGIN_MARKER = "vllm-evolve: scheduler plugin invoked"

# Tunable knobs that ``collect_profile`` forwards to the real bench. Single-sourced
# from BenchConfig's executable lever set (Codex R1) PLUS the workload/meta search dims
# (concurrency / n_requests / max_seeds) that are not serve-args. Searching anything NOT
# here would silently run the SAME config every trial, so L3 rejects unwired knobs rather
# than report a fake "verified winner" for a change that was never applied.
from vllm_evolve.bench.config import WIRED_LEVERS  # noqa: E402

WIRED_KNOBS = WIRED_LEVERS | frozenset({"concurrency", "n_requests", "max_seeds"})

# vLLM logs lines like "GPU KV cache usage: 45.2%", "Running: 12 reqs, Waiting: 30 reqs".
_KV_RE = re.compile(r"KV cache usage:\s*([\d.]+)\s*%", re.I)
_RUN_RE = re.compile(r"Running:\s*(\d+)\s*reqs", re.I)
_WAIT_RE = re.compile(r"Waiting:\s*(\d+)\s*reqs", re.I)
_PROMPT_TP_RE = re.compile(r"prompt throughput:\s*([\d.]+)\s*tokens/s", re.I)
_GEN_TP_RE = re.compile(r"generation throughput:\s*([\d.]+)\s*tokens/s", re.I)


def parse_vllm_log(log: str) -> dict:
    """Extract KV util / queue / preemption / prefill-share from a vLLM serve log."""
    out: dict = {}
    if not log:
        return out
    kv = [float(x) for x in _KV_RE.findall(log)]
    if kv:
        out["kv_util"] = round(max(kv) / 100.0, 4)  # peak KV pressure
    runs = [int(x) for x in _RUN_RE.findall(log)]
    waits = [int(x) for x in _WAIT_RE.findall(log)]
    if runs:
        out["running"] = max(runs)
    if waits:
        out["waiting"] = max(waits)
    pre = len(re.findall(r"preempt", log, re.I))
    out["preempt"] = pre
    # COARSE prefill share by TOKEN RATE (prompt vs generation tokens/s) — NOT
    # elapsed prefill/decode time. A proxy only; the diagnoser treats it as
    # `suspected` and notes the limitation.
    pt = [float(x) for x in _PROMPT_TP_RE.findall(log)]
    gt = [float(x) for x in _GEN_TP_RE.findall(log)]
    if pt and gt:
        p, g = statistics.mean(pt), statistics.mean(gt)
        if p + g > 0:
            out["prefill_token_frac"] = round(p / (p + g), 3)
    return out


_BUSY = 20  # sm_util% above this counts the GPU as "doing work" this sample


def summarize_gpu(samples: list[dict]) -> dict:
    """Summarize nvidia-smi samples into a Profile.gpu.

    Reports BOTH the mean over ALL samples (so a bursty / under-saturated run
    stays low — never inflated by silently dropping idle samples) AND
    ``duty_cycle`` = fraction of samples the GPU was busy. The diagnoser uses
    duty_cycle to tell *sustained* compute from *bursty* idle. ``sm_util_busy`` /
    ``mem_bw_util`` are means over the BUSY window (when work was actually
    happening), which is the right context for a bandwidth reading.
    """
    out: dict = {}
    if not samples:
        return out
    sm = [s["sm_util"] for s in samples if s.get("sm_util") is not None]
    if sm:
        out["sm_util"] = round(statistics.mean(sm), 1)        # mean over ALL
        out["sm_util_max"] = max(sm)
        out["duty_cycle"] = round(sum(1 for x in sm if x > _BUSY) / len(sm), 2)
        busy = [x for x in sm if x > _BUSY]
        if busy:
            out["sm_util_busy"] = round(statistics.mean(busy), 1)
    bw = [s["mem_bw_util"] for s in samples
          if s.get("mem_bw_util") is not None and (s.get("sm_util") or 0) > _BUSY]
    if bw:
        out["mem_bw_util"] = round(statistics.mean(bw), 1)    # bandwidth while busy
    mem = [s["mem_used_mb"] for s in samples if s.get("mem_used_mb") is not None]
    if mem:
        out["mem_used_mb"] = max(mem)
    tot = next((s.get("mem_total_mb") for s in samples if s.get("mem_total_mb")), None)
    if tot:
        out["mem_total_mb"] = tot
    return out


def assemble_profile(config: dict, eval_result: dict | None, gpu_samples: list[dict],
                     serve_log: str) -> Profile:
    """Pure assembly of a Profile from the three real signal sources."""
    er = eval_result or {}
    agg = er.get("aggregate_metrics") or {}
    per = er.get("raw_per_seed_metrics") or []

    primary = er.get("primary_metric") or "primary"

    def _from_md(md, name):
        """Flat metrics[name], or nested percentile metrics[base].pXX / .mean."""
        v = md.get(name)
        if isinstance(v, (int, float)):
            return v
        m = re.fullmatch(r"(.+?)_(p\d+|mean|median)_ms", name)  # ttft_p99_ms -> ttft_ms.p99
        if m:
            nested = md.get(f"{m.group(1)}_ms")
            if isinstance(nested, dict) and isinstance(nested.get(m.group(2)), (int, float)):
                return nested[m.group(2)]
        return None

    def _seed_series(*names):
        """Per-seed values for a metric: metrics[name] -> nested pXX -> primary_value."""
        vals = []
        for s in per:
            if not isinstance(s, dict):
                continue
            md = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
            val = next((v for v in (_from_md(md, n) for n in names) if v is not None), None)
            if val is None and primary in names:
                pv = s.get("primary_value")
                val = pv if isinstance(pv, (int, float)) else None
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return vals

    # Label the aggregate by its actual primary_metric (not always goodput); keep
    # per-seed series for every metric so L4 can bootstrap without re-running.
    metrics: dict = {}
    per_seed: dict = {}
    if agg.get("median") is not None:
        metrics[primary] = agg["median"]
        ps = _seed_series(primary)
        if ps:
            per_seed[primary] = ps
    for key, names in (("ttft_p99_ms", ("ttft_p99_ms",)),
                       ("ttft_mean_ms", ("mean_ttft_ms", "ttft_ms")),
                       ("tpot_ms", ("tpot_ms", "mean_tpot_ms")),
                       ("tok_s", ("total_token_throughput_tok_s", "output_throughput_tok_s"))):
        series = _seed_series(*names)
        if series:
            metrics[key] = statistics.median(series)
            per_seed[key] = series

    gpu = summarize_gpu(gpu_samples)
    vllm = parse_vllm_log(serve_log)

    sm_max = gpu.get("sm_util_max")
    mem_used, mem_tot = gpu.get("mem_used_mb"), gpu.get("mem_total_mb")
    saturated = None
    if sm_max is not None:
        mem_full = (mem_used / mem_tot >= 0.92) if (mem_used and mem_tot) else False
        saturated = bool(sm_max >= 90 or mem_full)

    return Profile(
        config=dict(config), metrics=metrics, vllm=vllm, gpu=gpu, per_seed=per_seed,
        saturated=saturated, outcome_class=er.get("outcome_class", ""),
        source=er.get("source", ""),
        marker_verified=_PLUGIN_MARKER in (serve_log or ""),
        evidence=[f"gpu={gpu}", f"vllm={vllm}", f"metrics={metrics}"],
    )


def _sample_gpu(remote: str, card: str, tag: str, n: int = 300) -> None:
    import subprocess
    cmd = (f"nohup bash -c 'for i in $(seq 1 {n}); do "
           f"nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
           f"--format=csv,noheader,nounits -i {card}; sleep 2; done' "
           f"> /tmp/ve_smi_{tag}.log 2>&1 < /dev/null &")
    with __import__("contextlib").suppress(Exception):
        subprocess.run(["ssh", "-o", "BatchMode=yes", remote, cmd],
                       capture_output=True, text=True, timeout=20)


def _read_gpu_samples(remote: str, tag: str) -> list[dict]:
    import subprocess
    out = []
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", remote, f"cat /tmp/ve_smi_{tag}.log"],
                           capture_output=True, text=True, timeout=20)
        for line in r.stdout.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 4 and p[0].isdigit():
                out.append({"sm_util": int(p[0]), "mem_bw_util": int(p[1]),
                            "mem_used_mb": int(p[2]), "mem_total_mb": int(p[3])})
    except Exception:  # noqa: BLE001
        pass
    return out


def collect_profile(config: dict) -> Profile:
    """GPU wrapper: run a real bench while sampling nvidia-smi -> Profile.

    ``utilization.memory`` is used as a COARSE bandwidth proxy (it is %-of-time the
    memory controller was busy, not true GB/s) — recorded so the diagnoser treats
    bandwidth findings as ``suspected``.
    """
    import contextlib
    import hashlib
    import subprocess

    from vllm_evolve.bench.config import (
        EngineConfig,
        WorkloadConfig,
        build_bench_config,
    )
    from vllm_evolve.bench.dispatch import DEFAULTS, run_remote_bench_config
    remote = config.get("remote") or DEFAULTS["remote"]
    card = str(config.get("gpus", "0")).split(",")[0]
    # Unique-per-config sampler tag so a new profile can never pick up a prior
    # run's stale /tmp samples (the file is also truncated before sampling).
    h = hashlib.sha256(repr(sorted(config.items())).encode("utf-8")).hexdigest()[:8]
    tag = f"prof{config.get('port', 8260)}_{h}"

    # BenchConfig is the execution contract (Codex R2-R5): build it from the autopt config so
    # EVERY wired engine + workload lever the search may set actually reaches the command.
    _explicit = {"model", "gpus", "scheduler_cls", "runner_kind", "policy", "policy_path",
                 "profile", "remote", "port"}
    overrides = {k: v for k, v in config.items()
                 if k not in _explicit and (k in EngineConfig.__dataclass_fields__
                                            or k in WorkloadConfig.__dataclass_fields__
                                            or k == "max_seeds")}   # wired budget cap (Codex P2)
    runner_kind = config.get("runner_kind", "candidate")
    scheduler_cls = "generated_scheduler.EvolvedScheduler" if runner_kind == "candidate" else None
    bench_config = build_bench_config(
        runner_kind=runner_kind, policy_path=config.get("policy", "targets/scheduling/seed.py"),
        model=config.get("model") or "facebook/opt-125m", scheduler_cls=scheduler_cls,
        profile=config.get("profile", "throughput"), gpus=str(config.get("gpus", "0")),
        remote=remote, port=int(config.get("port", 8260)), **overrides)

    with contextlib.suppress(Exception):
        subprocess.run(["ssh", "-o", "BatchMode=yes", remote,
                        "pkill -f 've_smi_' 2>/dev/null; true"],
                       capture_output=True, text=True, timeout=15)
    _sample_gpu(remote, card, tag)
    try:
        rb = run_remote_bench_config(bench_config)
        er, log = rb.eval_result, rb.log
    finally:
        with contextlib.suppress(Exception):
            subprocess.run(["ssh", "-o", "BatchMode=yes", remote,
                            f"pkill -f 've_smi_{tag}' 2>/dev/null; true"],
                           capture_output=True, text=True, timeout=15)
    samples = _read_gpu_samples(remote, tag)
    prof = assemble_profile(config, er, samples, log)
    prof.bench_config = er.get("bench_config") or bench_config.to_dict()   # provenance for A/B
    prof.eval_result = er                       # raw eval_result (for calibrate freezing)
    prof.remote_cmd = rb.remote_cmd
    return prof


def collect_profile_local_smoke(config: dict) -> Profile:
    """Off-box autopt eval: run the synthetic in-process ``local_smoke`` bench and assemble a
    Profile from it — so the WHOLE L1->L4 loop (diagnose/optimize/verify/holdout/accept) runs on a
    laptop. SYNTHETIC, plumbing only: the serve log is empty, so ``marker_verified`` is False
    (LOCK C) and the candidate can never be effective/accepted — the loop can only conclude DoD-B.
    """
    from vllm_evolve.bench.config import (
        EngineConfig,
        WorkloadConfig,
        build_bench_config,
    )
    from vllm_evolve.bench.local_smoke import run_local_smoke
    _explicit = {"model", "gpus", "scheduler_cls", "runner_kind", "policy", "policy_path",
                 "profile", "remote", "port"}
    overrides = {k: v for k, v in config.items()
                 if k not in _explicit and (k in EngineConfig.__dataclass_fields__
                                            or k in WorkloadConfig.__dataclass_fields__
                                            or k == "max_seeds")}   # wired budget cap (Codex P2)
    runner_kind = config.get("runner_kind", "candidate")
    scheduler_cls = "generated_scheduler.EvolvedScheduler" if runner_kind == "candidate" else None
    bench_config = build_bench_config(
        runner_kind=runner_kind, policy_path=config.get("policy", "targets/scheduling/seed.py"),
        model=config.get("model") or "facebook/opt-125m", scheduler_cls=scheduler_cls,
        profile=config.get("profile", "throughput"), gpus=str(config.get("gpus", "0")),
        remote=config.get("remote") or "local", port=int(config.get("port", 8260)), **overrides)
    er = run_local_smoke(bench_config).eval_result
    # Synthetic, clearly-non-real UNDER-SATURATED signals so the loop actually exercises
    # diagnose -> optimize (-> a serve-arg target like max_num_seqs) -> accept. The serve log
    # carries KV/queue lines but NEVER the plugin marker, so assemble_profile sets
    # marker_verified=False (LOCK C) -> accept can only reject -> the loop concludes DoD-B.
    samples = [{"sm_util": v, "mem_bw_util": 12, "mem_used_mb": 6000, "mem_total_mb": 49140}
               for v in (5, 8, 35, 6, 7)]                      # sm_util_max=35, duty_cycle=0.2
    smoke_log = "GPU KV cache usage: 20.0%\nRunning: 2 reqs, Waiting: 0 reqs\n"  # no marker
    prof = assemble_profile(config, er, samples, smoke_log)
    prof.bench_config = er.get("bench_config") or bench_config.to_dict()
    return prof


def collect_profile_frontier(config: dict) -> Profile:
    """Off-box autopt eval: run the out-of-process ``frontier_sim`` bench and assemble a Profile
    from it — so the WHOLE L1->L4 loop runs with no GPU. The eval_result carries Frontier's REAL
    serving numbers but is quarantined (source=frontier_sim, never effective). With the ve_policy
    bridge installed, a candidate schedule_batch executes inside Frontier; without a candidate
    policy this path still exercises the stock Frontier scheduler.

    Signal honesty (gap-1): the bottleneck signals fed to diagnose are Frontier's OWN outputs —
    memory utilization, preemption events, reconstructed queue depth, and the per-batch-ledger
    engine-busy fraction (mapped to ``duty_cycle``). Signals Frontier does not emit (SM util,
    memory bandwidth, NCCL) stay ABSENT, so the rule engine degrades along its own honest
    limitation paths. No serve log exists -> ``marker_verified`` False (LOCK C) -> only DoD-B.
    """
    from vllm_evolve.bench.config import (
        EngineConfig,
        WorkloadConfig,
        build_bench_config,
    )
    from vllm_evolve.bench.frontier_sim import run_frontier_sim
    _explicit = {"model", "gpus", "scheduler_cls", "runner_kind", "policy", "policy_path",
                 "profile", "remote", "port"}
    overrides = {k: v for k, v in config.items()
                 if k not in _explicit and (k in EngineConfig.__dataclass_fields__
                                            or k in WorkloadConfig.__dataclass_fields__
                                            or k == "max_seeds")}
    runner_kind = config.get("runner_kind", "candidate")
    scheduler_cls = "generated_scheduler.EvolvedScheduler" if runner_kind == "candidate" else None
    bench_config = build_bench_config(
        runner_kind=runner_kind, policy_path=config.get("policy", "targets/scheduling/seed.py"),
        model=config.get("model") or "facebook/opt-125m", scheduler_cls=scheduler_cls,
        profile=config.get("profile", "throughput"), gpus=str(config.get("gpus", "0")),
        remote=config.get("remote") or "local", port=int(config.get("port", 8260)), **overrides)
    # the goal's metric + workload SLO reach the sim bench (goodput is SLO-derived from REAL
    # per-request latencies; without an SLO it honestly falls back to a named real aggregate)
    if config.get("primary_metric"):
        bench_config.statistical.primary_metric = str(config["primary_metric"])
    ws_slo = (config.get("workload_spec") or {}).get("slo")
    if ws_slo and not bench_config.statistical.slo:
        bench_config.statistical.slo = dict(ws_slo)
    er = run_frontier_sim(bench_config).eval_result
    prof = assemble_profile(config, er, [], "")     # no GPU samples, no serve log — nothing faked
    sig = er.get("sim_signals") or {}
    vllm_signals = {k: sig[k] for k in ("kv_util", "preempt", "waiting", "running")
                    if sig.get(k) is not None}
    if vllm_signals:
        prof.vllm = vllm_signals
    if sig.get("engine_duty") is not None:
        # the simulator's own engine-busy fraction is the duty-cycle analogue; sm/bw stay absent
        prof.gpu = {"duty_cycle": sig["engine_duty"]}
    prof.evidence = [f"sim_signals={sig}", f"metrics={prof.metrics}"]
    prof.bench_config = er.get("bench_config") or bench_config.to_dict()
    return prof
