"""SR E2E validation harness (Python port of run_office_e2e.sh).

Boots the full SR stack (two vLLM instances, router, envoy) on the GPU node,
verifies the thinking gate smoke, then runs the sr_baseline_eval concurrent
benchmark across a list of concurrency levels and prints per-concurrency
routing/domain-signal segment stats plus an acc/QPS/latency summary.

All paths come from run_config.yaml. This script is intended to execute on the
GPU node itself (launcher/pipeline.py handles dispatch); it manages the
vLLM/router/envoy lifecycle via subprocess.Popen and registers an atexit
cleanup so the stack is torn down on any exit path.
"""

import argparse
import atexit
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from common import (
    check_port,
    cleanup_vllm,
    generate_task_id,
    get_workspace,
    load_config,
)

# Benchmark / smoke knobs (match the historical shell defaults).
MAX_TOKENS = 12288
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 600.0
SMOKE_THINK_MIN = 250
SMOKE_NOTHINK_MAX = 100
HTTP_OK = 200
DEFAULT_CONCURRENCY = "4,8"
DEFAULT_READY_TIMEOUT = 1200.0
POLL_INTERVAL = 5
SMOKE_MAX_TOKENS = 4096
CONFIG_GEN_TIMEOUT = 90
ENVOY_STARTUP_WAIT = 8
ROUTER_STARTUP_WAIT = 5
TERM_GRACE = 2
PKILL_GRACE = 1
GEN_CPU_THREADS = 16

SMOKE_MATH_Q = (
    "Question: What is the sum of all positive integers n such that "
    "n^2 - 19n + 99 is a perfect square?\n"
    "Options:\n(A) 18\n(B) 22\n(C) 38\n(D) 57\n\n"
    "Answer with: Answer: X"
)
SMOKE_PHILO_Q = (
    "Question: Who wrote the Critique of Pure Reason?\n"
    "Options:\n(A) Hegel\n(B) Kant\n(C) Hume\n(D) Nietzsche\n\n"
    "Answer with: Answer: X"
)
ROUTER_PROBE_Q = (
    "Question: Which statement about Kantian ethics is most accurate?\n\nOptions:\n(A) a\n(B) b\n(C) c\n(D) d"
)
CONC_SEG_FIELDS = ("routing_latency_ms", "domain_signal_ms")


@dataclass
class E2EContext:
    """Resolved configuration and live process handles for one E2E run."""

    config: dict
    workspace: Path
    log_dir: Path
    results_dir: Path
    sr_root: Path
    envoy_bin: str
    binding_dir: str
    sr_base_url: str
    router_port: int
    envoy_port: int
    vllm_small_port: int
    vllm_large_port: int
    e2e_py: str
    vllm_bin: str
    bench_script: Path
    cfg_path: Path
    envoy_cfg_path: Path
    testset: Path
    conc_list: list
    ready_timeout: float
    large_model_dir: str
    small_model_dir: str
    large_gpus: str | None
    small_gpus: str | None
    procs: dict = field(default_factory=dict)
    log_files: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    stop_event: threading.Event | None = None
    cleanup_done: bool = False


def parse_args():
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description="SR E2E validation (Python port of run_office_e2e.sh)")
    ap.add_argument("--config", default="config/run_config.yaml", help="path to run_config.yaml")
    ap.add_argument("--task-id", default="", help="task ID (auto-generated if omitted)")
    ap.add_argument(
        "--testset",
        default="",
        help="bench testset JSONL (default: workspace/<task_id>/results/office_eval_testset.jsonl)",
    )
    ap.add_argument(
        "--concurrency",
        default=DEFAULT_CONCURRENCY,
        help="comma-separated concurrency levels (default: 4,8)",
    )
    return ap.parse_args()


def build_context(args):
    """Load config and resolve all paths/ports into an E2EContext."""
    config = load_config(args.config)
    task_id = args.task_id or generate_task_id("office-e2e")
    workspace = get_workspace(config, task_id)
    sr = config["sr_stack"]
    models = config["models"]
    e2e_py = config["python"]["e2e_venv"]
    vllm_bin = sr.get("vllm_bin") or str(Path(e2e_py).parent / "vllm")
    router_port = urlparse(sr["sr_base_url"]).port or 80
    if args.testset:
        testset = Path(args.testset)
    else:
        testset = workspace / "results" / "office_eval_testset.jsonl"
    conc_list = [int(c) for c in args.concurrency.split(",") if c.strip()]
    ready_timeout = float(config.get("vllm", {}).get("ready_timeout", DEFAULT_READY_TIMEOUT))
    large_gpus, small_gpus = _resolve_gpu_assignments(config)
    src_dir = Path(__file__).resolve().parent
    return E2EContext(
        config=config,
        workspace=workspace,
        log_dir=workspace / "logs",
        results_dir=workspace / "results",
        sr_root=Path(sr["sr_root"]),
        envoy_bin=sr["envoy_bin"],
        binding_dir=sr["binding_dir"],
        sr_base_url=sr["sr_base_url"],
        router_port=router_port,
        envoy_port=int(sr["envoy_port"]),
        vllm_small_port=int(sr["vllm_small_port"]),
        vllm_large_port=int(sr["vllm_large_port"]),
        e2e_py=e2e_py,
        vllm_bin=vllm_bin,
        bench_script=src_dir / "sr_baseline_eval.py",
        cfg_path=workspace / "config_output" / "config-office.yaml",
        envoy_cfg_path=workspace / "config_output" / "envoy-config.yaml",
        testset=testset,
        conc_list=conc_list,
        ready_timeout=ready_timeout,
        large_model_dir=models["large"]["model_dir"],
        small_model_dir=models["small"]["model_dir"],
        large_gpus=large_gpus,
        small_gpus=small_gpus,
    )


def _resolve_gpu_assignments(config):
    """Split visible GPUs into (large, small) subsets using models.n_gpu.

    Honors SLURM_JOB_GPUS / CUDA_VISIBLE_DEVICES (stripping node prefixes as
    the shell did). Falls back to nvidia-smi enumeration, then to None so each
    vLLM instance sees whatever the environment exposes.
    """
    large_n = int(config["models"]["large"].get("n_gpu", 1))
    small_n = int(config["models"]["small"].get("n_gpu", 1))
    raw = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    if not raw:
        raw = _enumerate_gpus()
    if not raw:
        return None, None
    ids = [part.split(":")[-1].strip() for part in raw.split(",") if part.strip()]
    if len(ids) < large_n + small_n:
        return None, None
    large_gpus = ",".join(ids[:large_n])
    small_gpus = ",".join(ids[large_n : large_n + small_n])
    return large_gpus, small_gpus


def _enumerate_gpus():
    """Return a comma-joined GPU index list from nvidia-smi, or ""."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    ids = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return ",".join(ids)


def _base_env(ctx):
    """Copy of os.environ with thread-limit knobs set (matches shell exports)."""
    env = os.environ.copy()
    env["MKL_NUM_THREADS"] = str(GEN_CPU_THREADS)
    env["OMP_NUM_THREADS"] = str(GEN_CPU_THREADS)
    env["RAYON_NUM_THREADS"] = str(GEN_CPU_THREADS)
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    return env


def _vllm_env(ctx, gpus):
    """Base env with CUDA_VISIBLE_DEVICES pinned to a GPU subset."""
    env = _base_env(ctx)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def _router_env(ctx):
    """Base env with LD_LIBRARY_PATH pointing at the SR binding trees."""
    env = _base_env(ctx)
    lib_paths = [
        ctx.binding_dir,
        f"{ctx.sr_root}/ml-binding/target/release",
        f"{ctx.sr_root}/nlp-binding/target/release",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        lib_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    return env


def _vllm_cmd(ctx, model_cfg, port):
    """Build the `vllm serve` argv for one model."""
    cmd = [
        ctx.vllm_bin,
        "serve",
        model_cfg["model_dir"],
        "--served-model-name",
        model_cfg["served_name"],
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(model_cfg["tensor_parallel"]),
        "--max-model-len",
        str(model_cfg["max_model_len"]),
        "--gpu-memory-utilization",
        str(model_cfg["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
    ]
    if model_cfg.get("kv_cache_dtype"):
        cmd += ["--kv-cache-dtype", model_cfg["kv_cache_dtype"]]
    return cmd


def _popen(cmd, log_file, env, cwd=None):
    """Launch a long-lived child, redirecting stdout+stderr to a log file."""
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=cwd)


def _http_get_ok(url, timeout=3.0):
    """Return True if a GET returns HTTP 200 within timeout."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == HTTP_OK
    except OSError:
        return False


def _http_post_json(url, payload, timeout=30.0):
    """POST JSON; return (status, body) or (None, "") on any failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except OSError:
        return None, ""


def _extract_completion_tokens(body):
    """Pull completion_tokens from a chat completion response body."""
    try:
        data = json.loads(body)
        usage = data.get("usage") or {}
        if "completion_tokens" in usage:
            return int(usage["completion_tokens"])
    except (ValueError, TypeError):
        pass
    m = re.search(r'"completion_tokens"\s*:\s*(\d+)', body)
    return int(m.group(1)) if m else 0


def _has_routing_decision(body):
    """True if the /api/v1/eval response carries a routing_decision."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return '"routing_decision"' in body
    return isinstance(data, dict) and "routing_decision" in data


def _read_lines(path):
    """Read a log file into a list of lines (empty list if unreadable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def _print_tail(path, num_lines):
    """Print the last num_lines of a log file."""
    lines = _read_lines(path)
    for line in lines[-num_lines:]:
        print(line)


def _fail(msg, log=None):
    """Print a fatal message (+ log tail) and exit (atexit cleanup runs)."""
    print(f"[FATAL] {msg}")
    if log is not None and log.exists():
        _print_tail(log, 20)
    sys.exit(1)


def preflight(ctx):
    """Verify required input files exist before launching anything."""
    for path in (ctx.cfg_path, ctx.testset):
        if not path.exists():
            print(f"[FATAL] missing required input: {path}")
            sys.exit(1)


def check_ports_free(ctx):
    """Abort if any stack port is already bound."""
    for port in (
        ctx.vllm_small_port,
        ctx.vllm_large_port,
        ctx.router_port,
        ctx.envoy_port,
    ):
        if not check_port(port):
            print(f"[FATAL] port {port} already in use")
            sys.exit(1)


def _start_cpu_monitor(ctx):
    """Background thread sampling the router process CPU% to e2e_threads.log."""
    stop = threading.Event()
    ctx.stop_event = stop
    log_path = ctx.log_dir / "e2e_threads.log"

    def loop():
        with open(log_path, "w", encoding="utf-8") as f:
            while not stop.is_set():
                router = ctx.procs.get("router")
                if router is not None and router.poll() is None:
                    cpu = _read_cpu(router.pid)
                    f.write(f"{time.strftime('%T')} cpu={cpu}\n")
                    f.flush()
                stop.wait(POLL_INTERVAL)

    threading.Thread(target=loop, daemon=True).start()


def _read_cpu(pid):
    """Read CPU% for a pid via ps; "" on failure."""
    try:
        out = subprocess.run(
            ["ps", "-o", "pcpu=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def launch_vllm_instances(ctx):
    """Start the large (TP) and small (TP1) vLLM backends."""
    large = ctx.config["models"]["large"]
    small = ctx.config["models"]["small"]
    big_log = open(ctx.log_dir / "e2e_big.log", "w", encoding="utf-8")  # noqa: SIM115
    small_log = open(ctx.log_dir / "e2e_small.log", "w", encoding="utf-8")  # noqa: SIM115
    ctx.log_files.extend((big_log, small_log))
    ctx.procs["vllm_large"] = _popen(
        _vllm_cmd(ctx, large, ctx.vllm_large_port),
        big_log,
        _vllm_env(ctx, ctx.large_gpus),
    )
    ctx.procs["vllm_small"] = _popen(
        _vllm_cmd(ctx, small, ctx.vllm_small_port),
        small_log,
        _vllm_env(ctx, ctx.small_gpus),
    )
    print(
        f"[info] vLLM launched: large(TP={large['tensor_parallel']}) "
        f"gpus={ctx.large_gpus} small(TP={small['tensor_parallel']}) "
        f"gpus={ctx.small_gpus}"
    )


def wait_vllm_ready(ctx):
    """Poll both vLLM /v1/models endpoints until both answer 200."""
    small_url = f"http://127.0.0.1:{ctx.vllm_small_port}/v1/models"
    large_url = f"http://127.0.0.1:{ctx.vllm_large_port}/v1/models"
    deadline = time.time() + ctx.ready_timeout
    while time.time() < deadline:
        if _http_get_ok(small_url) and _http_get_ok(large_url):
            print("[info] vLLM ready")
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _patch_envoy_access_log(ctx):
    """Rewrite the envoy access log path inside the generated config."""
    path = ctx.envoy_cfg_path
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    access_log = str(ctx.log_dir / "envoy_access.log")
    text = text.replace("path: /var/log/envoy_access.log", f"path: {access_log}")
    path.write_text(text, encoding="utf-8")


def generate_envoy_config(ctx):
    """Run SR's cli.config_generator to emit the envoy config (blocking)."""
    cmd = [
        ctx.e2e_py,
        "-m",
        "cli.config_generator",
        str(ctx.cfg_path),
        str(ctx.envoy_cfg_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ctx.sr_root / "src/vllm-sr"),
            timeout=CONFIG_GEN_TIMEOUT,
            capture_output=True,
            text=True,
            check=False,
            env=_base_env(ctx),
        )
    except subprocess.TimeoutExpired:
        print(f"[FATAL] config_generator timed out after {CONFIG_GEN_TIMEOUT}s")
        return False
    log_path = ctx.log_dir / "e2e_gen.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout or "")
        if result.stderr:
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)
    if result.returncode != 0:
        print("[FATAL] config_generator failed")
        print(result.stdout or "")
        print(result.stderr or "")
        return False
    _patch_envoy_access_log(ctx)
    return True


def launch_router(ctx):
    """Start the SR router binary against the office config."""
    log = open(ctx.log_dir / "e2e_router.log", "w", encoding="utf-8")  # noqa: SIM115
    ctx.log_files.append(log)
    cmd = [str(ctx.sr_root / "bin/router"), f"-config={ctx.cfg_path}"]
    ctx.procs["router"] = _popen(cmd, log, _router_env(ctx), cwd=str(ctx.sr_root))
    time.sleep(ROUTER_STARTUP_WAIT)


def wait_router_ready(ctx):
    """Wait for router /health, then for the classifier to return a decision."""
    health_url = f"{ctx.sr_base_url}/health"
    deadline = time.time() + ctx.ready_timeout
    ready = False
    while time.time() < deadline:
        if _http_get_ok(health_url, timeout=5):
            ready = True
            break
        time.sleep(POLL_INTERVAL)
    if not ready:
        return False
    eval_url = f"{ctx.sr_base_url}/api/v1/eval"
    deadline = time.time() + ctx.ready_timeout
    while time.time() < deadline:
        status, body = _http_post_json(eval_url, {"text": ROUTER_PROBE_Q}, timeout=30)
        if status == HTTP_OK and _has_routing_decision(body):
            print("[info] classification ready")
            return True
        time.sleep(POLL_INTERVAL)
    return False


def launch_envoy(ctx):
    """Start envoy against the generated config."""
    log = open(ctx.log_dir / "e2e_envoy.log", "w", encoding="utf-8")  # noqa: SIM115
    ctx.log_files.append(log)
    ctx.procs["envoy"] = _popen(
        [ctx.envoy_bin, "-c", str(ctx.envoy_cfg_path), "--log-level", "warning"],
        log,
        _router_env(ctx),
    )
    time.sleep(ENVOY_STARTUP_WAIT)


def _smoke(ctx, url, question):
    """Run one smoke completion through envoy and return completion_tokens."""
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": SMOKE_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    status, body = _http_post_json(url, payload, timeout=REQUEST_TIMEOUT)
    if status != HTTP_OK:
        return 0
    return _extract_completion_tokens(body)


def run_smoke_tests(ctx):
    """Verify the thinking gate: math thinks (>tokens), philosophy does not."""
    url = f"http://127.0.0.1:{ctx.envoy_port}/v1/chat/completions"
    think_tok = _smoke(ctx, url, SMOKE_MATH_Q)
    print(f"[smoke] math domain (expect thinking) completion_tokens={think_tok} (>{SMOKE_THINK_MIN})")
    nthink_tok = _smoke(ctx, url, SMOKE_PHILO_Q)
    print(f"[smoke] philosophy domain (expect no thinking) completion_tokens={nthink_tok} (<{SMOKE_NOTHINK_MAX})")
    if think_tok > SMOKE_THINK_MIN and nthink_tok < SMOKE_NOTHINK_MAX:
        print("[ok] use_reasoning injection verified")
        return True
    print(f"[FATAL] thinking gate mismatch (think={think_tok} nthink={nthink_tok})")
    return False


def _bench_base_args(ctx):
    """Build the shared sr_baseline_eval endpoint/model flags."""
    small = ctx.config["models"]["small"]
    large = ctx.config["models"]["large"]
    return [
        "--small-base-url",
        f"http://127.0.0.1:{ctx.vllm_small_port}/v1",
        "--large-base-url",
        f"http://127.0.0.1:{ctx.vllm_large_port}/v1",
        "--sr-base-url",
        f"http://127.0.0.1:{ctx.envoy_port}/v1",
        "--small-model",
        small["served_name"],
        "--large-model",
        large["served_name"],
        "--api-key",
        "EMPTY",
        "--max-tokens",
        str(MAX_TOKENS),
        "--temperature",
        str(TEMPERATURE),
        "--timeout",
        str(REQUEST_TIMEOUT),
        "--testset",
        str(ctx.testset),
    ]


def run_benchmarks(ctx):
    """Run sr_baseline_eval (sr-auto) for each concurrency level."""
    base = _bench_base_args(ctx)
    router_log = ctx.log_dir / "e2e_router.log"
    for conc in ctx.conc_list:
        out_dir = ctx.results_dir / f"e2e-c{conc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        line_before = len(_read_lines(router_log))
        print(f"================ sr-auto CONC={conc} {time.strftime('%T')} =================")
        cmd = [
            ctx.e2e_py,
            str(ctx.bench_script),
            "--mode",
            "sr-auto",
            "--runs",
            "1",
            "--concurrency",
            str(conc),
            "--output-dir",
            str(out_dir),
            *base,
        ]
        subprocess.run(cmd, check=False, env=_base_env(ctx))
        line_after = len(_read_lines(router_log))
        ctx.segments.append((str(conc), line_before, line_after))
        print(f"[info] c{conc} segment: router log {line_before} -> {line_after}")


def _field_values(lines_slice, field_name):
    """Collect all numeric values of field_name from the given log lines."""
    pat = re.compile(r'"' + field_name + r'":([0-9.]+)')
    vals = []
    for line in lines_slice:
        for m in pat.finditer(line):
            with contextlib.suppress(ValueError):
                vals.append(float(m.group(1)))
    vals.sort()
    return vals


def _percentile(sorted_vals, frac):
    """Truncation percentile mirroring the awk reference (1-indexed)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = int(n * frac)
    if idx < 1:
        idx = 1
    elif idx > n:
        idx = n
    return float(sorted_vals[idx - 1])


def _print_field_stats(tag, field_name, vals):
    """Print n/mean/p50/p90/p99/max for one field's value list."""
    n = len(vals)
    if n == 0:
        return
    mean = sum(vals) / n
    print(
        f"  CONC={tag} {field_name}: n={n} mean={mean:.0f} "
        f"p50={_percentile(vals, 0.5):.0f} p90={_percentile(vals, 0.9):.0f} "
        f"p99={_percentile(vals, 0.99):.0f} max={vals[-1]:.0f} ms"
    )


def print_segment_stats(ctx):
    """Per-concurrency routing/domain_signal stats sliced from the router log."""
    print("================ segment stats (routing / domain_signal per CONC) =================")
    lines = _read_lines(ctx.log_dir / "e2e_router.log")
    for tag, line_before, line_after in ctx.segments:
        seg = lines[line_before:line_after]
        for field_name in CONC_SEG_FIELDS:
            vals = _field_values(seg, field_name)
            _print_field_stats(tag, field_name, vals)


def _print_conc_summary(conc, summary_path):
    """Print the acc/QPS/latency line for one concurrency level."""
    if not summary_path.exists():
        print(f"CONC={conc} summary missing: {summary_path}")
        return
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    qps = data.get("request_throughput_per_second") or 0
    print(
        f"CONC={conc} wall={data.get('run_wall_time_seconds')} "
        f"QPS={qps:.2f} e2e_mean={data.get('latency_mean_seconds')} "
        f"e2e_p95={data.get('latency_p95_seconds')} "
        f"ttft_mean={data.get('ttft_mean_seconds')} "
        f"acc={data.get('accuracy')} "
        f"share_L={data.get('large_model_request_share')} "
        f"comp_tok={data.get('completion_tokens_total')} "
        f"noans={data.get('no_answer_requests')}"
    )


def print_summary(ctx):
    """Print the final per-concurrency summary from each summary.json."""
    print("================ summary =================")
    for conc in ctx.conc_list:
        summary = ctx.results_dir / f"e2e-c{conc}" / "sr-auto" / "summary.json"
        _print_conc_summary(conc, summary)


def _terminate_procs(ctx):
    """SIGTERM then SIGKILL any still-live stack processes."""
    names = ("envoy", "router", "vllm_large", "vllm_small")
    for name in names:
        proc = ctx.procs.get(name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
    time.sleep(TERM_GRACE)
    for name in names:
        proc = ctx.procs.get(name)
        if proc is not None and proc.poll() is None:
            proc.kill()


def _pkill_patterns(patterns):
    """pkill TERM then KILL fallback for stragglers by command-line pattern."""
    for pat in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pat], capture_output=True, check=False)
    time.sleep(PKILL_GRACE)
    for pat in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pat], capture_output=True, check=False)


def cleanup_all(ctx):
    """atexit handler: tear down the full SR stack idempotently."""
    if ctx.cleanup_done:
        return
    ctx.cleanup_done = True
    print("[cleanup] releasing SR stack")
    if ctx.stop_event is not None:
        ctx.stop_event.set()
    _terminate_procs(ctx)
    if ctx.procs.get("vllm_large") is not None:
        cleanup_vllm(ctx.large_model_dir)
    if ctx.procs.get("vllm_small") is not None:
        cleanup_vllm(ctx.small_model_dir)
    _pkill_patterns(("bin/router -config=", "/envoy -c "))
    for log_file in ctx.log_files:
        with contextlib.suppress(OSError):
            log_file.close()


def _handle_signal(_signum, _frame):
    """SIGTERM -> SystemExit so atexit cleanup runs."""
    sys.exit(1)


def run_e2e(ctx):
    """Orchestrate the full E2E flow end-to-end."""
    atexit.register(cleanup_all, ctx)
    signal.signal(signal.SIGTERM, _handle_signal)
    preflight(ctx)
    _start_cpu_monitor(ctx)
    launch_vllm_instances(ctx)
    if not wait_vllm_ready(ctx):
        _fail("vLLM not ready", ctx.log_dir / "e2e_big.log")
    if not generate_envoy_config(ctx):
        _fail("envoy config generation failed")
    launch_router(ctx)
    if not wait_router_ready(ctx):
        _fail("router not ready", ctx.log_dir / "e2e_router.log")
    launch_envoy(ctx)
    print("[info] SR stack ready")
    if not run_smoke_tests(ctx):
        _fail("thinking gate smoke failed")
    run_benchmarks(ctx)
    print_segment_stats(ctx)
    print_summary(ctx)
    print(f"[done] {time.strftime('%T')}")


def main():
    """Entry point."""
    args = parse_args()
    ctx = build_context(args)
    run_e2e(ctx)


if __name__ == "__main__":
    main()
