"""Native (no-Docker) real-vLLM backend.

Starts a real ``vllm serve`` with the rendered scheduler plugin on a GPU host
where vLLM is pip-installed (conda/venv), drives concurrent OpenAI-compatible
load, and records per-request timings. It is the real-vLLM bench backend; the
shared pure helpers (timing records, plugin-log markers, failure classification)
live in ``bench.runtime``.

GPU-gated by reality: if ``vllm`` is not importable / the server never becomes
healthy, the run raises rather than fabricating metrics.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import resource
import secrets
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from vllm_evolve.bench.gpu_guard import validate_gpu_budget
from vllm_evolve.bench.outcome import OutcomeClass
from vllm_evolve.bench.runtime import (
    PLUGIN_INVOKED_MARKER,
    PLUGIN_REORDERED_MARKER,
    StreamTiming,
    record_from_timing,
)

DEFAULT_MODEL = "facebook/opt-125m"
SERVED_NAME = "ve_bench"
SCHEDULER_QUALNAME = "generated_scheduler.EvolvedScheduler"
_NOFILE_RESERVE = 1024


def _wait_replica_seed_barrier(
    barrier_id: str,
    *,
    replica_index: int,
    replica_count: int,
    seed: int,
    timeout_s: float = 600.0,
    lead_s: float = 2.0,
    root: str | Path = "/tmp",
) -> float:
    """Synchronize independently started replica servers before each seed.

    Model loading and compilation can finish minutes apart. Without a barrier,
    two nominal replicas replay the same arrival trace at different wall-clock
    times, so they are not one two-GPU workload. The barrier uses a run-unique
    directory on the shared host and returns the common wall-clock start time.
    """
    if (
        not barrier_id
        or any(
            ch not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789_-"
            )
            for ch in barrier_id
        )
    ):
        raise ValueError("replica barrier id must be non-empty and filesystem-safe")
    if replica_count < 2 or not 0 <= replica_index < replica_count:
        raise ValueError("invalid replica barrier cardinality/index")
    seed_dir = (
        Path(root)
        / f"ve_replica_barrier_{barrier_id}"
        / f"seed_{int(seed)}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    ready = seed_dir / f"ready_{replica_index}.json"
    ready_tmp = seed_dir / f".ready_{replica_index}.{os.getpid()}.tmp"
    ready_tmp.write_text(
        json.dumps(
            {
                "replica_index": replica_index,
                "pid": os.getpid(),
                "ready_at_unix_s": time.time(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(ready_tmp, ready)
    deadline = time.monotonic() + float(timeout_s)
    expected = {f"ready_{index}.json" for index in range(replica_count)}
    while time.monotonic() < deadline:
        present = {path.name for path in seed_dir.glob("ready_*.json")}
        if expected <= present:
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(
            f"replica barrier {barrier_id} seed {seed} timed out waiting "
            f"for {replica_count} participants"
        )

    start_path = seed_dir / "start.json"
    if replica_index == 0 and not start_path.exists():
        start_tmp = seed_dir / f".start.{os.getpid()}.tmp"
        start_tmp.write_text(
            json.dumps(
                {"start_at_unix_s": time.time() + float(lead_s)},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(start_tmp, start_path)
    while time.monotonic() < deadline and not start_path.exists():
        time.sleep(0.05)
    if not start_path.exists():
        raise TimeoutError(
            f"replica barrier {barrier_id} seed {seed} has no start signal"
        )
    start_at = float(
        json.loads(start_path.read_text(encoding="utf-8"))["start_at_unix_s"]
    )
    remaining = start_at - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return start_at


def _hardware_name() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    names = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ", ".join(names) if names else "unknown"


@dataclass
class ServeHandle:
    proc: subprocess.Popen
    port: int
    log_path: str
    nonce: str = ""          # per-run marker nonce (candidate runs only); "" for vanilla

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _wait_healthy(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    import httpx

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False  # server died during startup
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


@contextlib.contextmanager
def serve_vllm(
    model: str,
    plugin_dir: str,
    *,
    gpus: str = "0",
    port: int = 8200,
    scheduler_cls: str | None = SCHEDULER_QUALNAME,
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 2048,
    max_num_seqs: int | None = None,
    tensor_parallel_size: int = 1,
    # match vLLM's own default (cudagraph, NOT eager): the enforce_eager LEVER is carried in
    # extra_args (--enforce-eager only when True), so this must NOT independently force eager —
    # otherwise a BenchConfig with enforce_eager=False would still run eager (Codex review P2).
    enforce_eager: bool = False,
    extra_args: list[str] | None = None,
    # A cold 32B/72B TP=2 start can spend several minutes loading weights and
    # compiling CUDA graphs.  This guards startup only; the remote worker still
    # owns the whole-run deadline and process cleanup.
    health_timeout_s: float = 900.0,
    log_path: str | None = None,
):
    """Start ``vllm serve`` with the rendered plugin; yield a ServeHandle.

    Candidate runner (``scheduler_cls`` set): the plugin module
    (``generated_scheduler.py``) must live in ``plugin_dir``; it is put on
    PYTHONPATH so ``--scheduler-cls`` can import it. **Vanilla runner**
    (``scheduler_cls=None``): NO ``--scheduler-cls`` flag and PYTHONPATH is left
    clean so no plugin can load — this is the true vLLM-default baseline (GAP-B).
    The server runs in its own process group so the whole tree is killed on exit.
    """
    devices = validate_gpu_budget(gpus, tensor_parallel_size)
    gpus = ",".join(devices)
    log_path = log_path or f"/tmp/ve_serve_{port}.log"
    env = dict(os.environ)
    nonce = ""
    if scheduler_cls is not None:               # candidate: expose the plugin + nonce its markers
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = plugin_dir + (os.pathsep + existing_pp if existing_pp else "")
        nonce = secrets.token_hex(8)            # per-run secret the policy source can't know
        env["VE_MARKER_NONCE"] = nonce
    # Exposes vLLM's local-only /reset_prefix_cache endpoint. Exact-trace
    # repetitions share prompt token ids, so resetting between measured seeds
    # is required to keep every repetition at the same cold-cache caliber.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpus
    cmd = [
        "vllm", "serve", model,
        "--served-model-name", SERVED_NAME,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    if scheduler_cls is not None:               # candidate: attach our scheduler plugin
        cmd.extend(["--scheduler-cls", scheduler_cls])
    if max_num_seqs:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if tensor_parallel_size and tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if enforce_eager:
        cmd.append("--enforce-eager")
    if extra_args:
        cmd.extend(extra_args)

    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    try:
        if not _wait_healthy(port, proc, health_timeout_s):
            tail = ""
            with contextlib.suppress(Exception):
                lines = Path(log_path).read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines(keepends=True)
                tail = "".join(lines[-25:])
            raise RuntimeError(
                f"vllm serve did not become healthy on :{port}\n"
                f"--- log tail ---\n{tail}"
            )
        yield ServeHandle(proc, port, log_path, nonce)
    finally:
        _terminate(proc)
        with contextlib.suppress(Exception):
            logf.close()


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except Exception:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        # Block until the process group is actually reaped so GPU memory + the
        # port are released before this card is reused by the next eval.
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)


async def _one_request(client, base_url: str, model: str, req: dict, sem) -> StreamTiming:
    prompt = (
        req.get("prompt_token_ids")
        or req.get("prompt")
        or ("hello " * max(1, int(req.get("num_prompt_tokens", 8) or 8)))
    )
    max_tokens = int(req.get("num_output_tokens", 16) or 16)
    rid = str(req.get("request_id", ""))
    n_prompt = int(req.get("num_prompt_tokens", 0) or 0)
    async with sem:
        submit = time.monotonic()
        first = end = None
        out_toks = 0
        ok = True
        err = None
        try:
            async with client.stream(
                "POST", f"{base_url}/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                      "temperature": 0.0, "ignore_eos": True, "stream": True,
                      "seed": int(req.get("seed", 0))},
                timeout=None,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    if first is None:
                        first = time.monotonic()
                    out_toks += 1
                end = time.monotonic()
        except Exception as e:  # noqa: BLE001 - record failure, never fabricate
            ok = False
            err = repr(e)
        return StreamTiming(
            request_id=rid, submit_time_s=submit, first_token_time_s=first,
            end_time_s=end, num_prompt_tokens=n_prompt, num_output_tokens=out_toks,
            success=ok and first is not None and end is not None, error=err,
        )


def _client_connection_limits(concurrency: int) -> tuple[int, int]:
    maximum = max(1, int(concurrency))
    return maximum, min(maximum, 512)


def _ensure_nofile_capacity(concurrency: int) -> int:
    """Give both the load client and its future vLLM child enough descriptors.

    A 1k default ``RLIMIT_NOFILE`` made a nominal 8k open-loop replay fail after
    exactly ~1k connections.  Silently capping the client would move queueing
    out of vLLM and invalidate scheduler experiments, so raise the process-local
    soft limit before spawning the server and fail closed when the hard limit
    cannot support the requested concurrency.
    """
    required = max(1, int(concurrency)) + _NOFILE_RESERVE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < required:
        target = required if hard == resource.RLIM_INFINITY else min(required, hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "open-loop replay requires at least "
                f"{required} file descriptors for concurrency={concurrency}; "
                f"soft={soft}, hard={hard}"
            ) from exc
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < required:
        raise RuntimeError(
            "open-loop replay file-descriptor capacity is insufficient: "
            f"required={required}, available={soft}"
        )
    return int(soft)


async def _drive(
    requests: list[dict],
    base_url: str,
    model: str,
    concurrency: int,
    *,
    start_at_s: float | None = None,
):
    import httpx

    # httpx defaults to only 100 connections.  That silently turns a nominal
    # 1k+ open-loop replay into a client-side queue and leaves vLLM's waiting
    # queue empty.  The remote worker owns the overall timeout, so individual
    # responses may wait behind a deliberately saturated server without a
    # client read timeout invalidating completion.
    max_connections, max_keepalive = _client_connection_limits(concurrency)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
    )
    sem = asyncio.Semaphore(concurrency)
    started = (
        float(start_at_s) if start_at_s is not None else time.monotonic()
    )

    async def at_arrival(request):
        target = max(0.0, float(request.get("arrival_s", 0.0) or 0.0))
        delay = target - (time.monotonic() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        return await _one_request(client, base_url, model, request, sem)

    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        tasks = [asyncio.create_task(at_arrival(request)) for request in requests]
        return await asyncio.gather(*tasks)


def _run_load_loop(awaitable):
    """Run the high-concurrency client on uvloop when it is available.

    The real-vLLM environment already carries uvloop as a vLLM dependency.  A
    7k+ stream replay makes the default selector loop spend enough time parsing
    response chunks that due arrival timers slip behind the frozen open-loop
    schedule.  That is client under-delivery, not server backpressure.  Keep a
    dependency-free fallback for local tests and minimal installations.
    """
    try:
        import uvloop
    except ImportError:
        return asyncio.run(awaitable)
    return uvloop.run(awaitable)


def _client_shard_count(concurrency: int) -> int:
    """Bound response parsing to roughly 1k streams per event-loop thread.

    Eight loops could sustain the 276 QPS calibration point but became the
    bottleneck before a strict +20% plateau probe.  The remote host has ample
    CPU cores; using at most sixteen bounded loops preserves the exact arrival
    schedule while keeping client-side response parsing out of the experiment.
    """
    return min(16, max(1, (max(1, int(concurrency)) + 1023) // 1024))


def _drive_load_sharded(
    requests: list[dict],
    base_url: str,
    model: str,
    concurrency: int,
) -> list[StreamTiming]:
    shard_count = _client_shard_count(concurrency)
    if shard_count == 1:
        return _run_load_loop(_drive(requests, base_url, model, concurrency))

    shards = [requests[index::shard_count] for index in range(shard_count)]
    concurrency_per_shard = max(
        1, (max(1, int(concurrency)) + shard_count - 1) // shard_count
    )
    # Give every worker time to create its loop/client before the common
    # open-loop origin.  Arrival offsets and request contents remain unchanged.
    start_at_s = time.monotonic() + 0.25

    def run_shard(shard: list[dict]) -> list[StreamTiming]:
        return _run_load_loop(
            _drive(
                shard,
                base_url,
                model,
                concurrency_per_shard,
                start_at_s=start_at_s,
            )
        )

    timings = []
    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        for shard_timings in pool.map(run_shard, shards):
            timings.extend(shard_timings)
    input_order = {
        str(request.get("request_id", "")): index
        for index, request in enumerate(requests)
    }
    timings.sort(key=lambda timing: input_order[timing.request_id])
    return timings


def drive_load(
    requests: list[dict], *, port: int, model: str = SERVED_NAME,
    concurrency: int = 32, warmup: int = 2,
) -> list[StreamTiming]:
    """Drive ``requests`` concurrently against a running vLLM server; return raw
    per-request timings. ``warmup`` leading requests are sent untimed first."""
    base = f"http://127.0.0.1:{port}"
    if warmup and requests:
        with contextlib.suppress(Exception):
            _run_load_loop(_drive(requests[:warmup], base, model, concurrency))
    return _drive_load_sharded(requests, base, model, concurrency)


def _warmup_subset(requests: list[dict], max_requests: int = 16) -> list[dict]:
    """Pick deterministic length quantiles without replaying a multi-minute trace.

    Prompt/output lengths remain exact; only warmup arrivals are collapsed to
    zero because warmup is explicitly outside the measured workload.
    """
    if not requests or max_requests < 1:
        return []
    ranked = sorted(
        enumerate(requests),
        key=lambda item: (
            int(item[1].get("num_prompt_tokens") or 0)
            + int(item[1].get("num_output_tokens") or 0),
            item[0],
        ),
    )
    count = min(max_requests, len(ranked))
    if count == 1:
        chosen = [ranked[-1]]
    else:
        chosen = [
            ranked[round(slot * (len(ranked) - 1) / (count - 1))]
            for slot in range(count)
        ]
    return [
        {
            **request,
            "request_id": f"{request['request_id']}-warmup",
            "arrival_s": 0.0,
            "seed": -1,
        }
        for slot, (_, request) in enumerate(chosen)
    ]


def reset_prefix_cache(port: int) -> None:
    """Reset server-side prefix state between exact-trace repetitions."""
    import httpx

    response = httpx.post(
        f"http://127.0.0.1:{int(port)}/reset_prefix_cache",
        timeout=30.0,
    )
    response.raise_for_status()


def run_profile_native(
    plugin_dir: str, model: str, requests: list[dict], *,
    gpus: str = "0", port: int = 8200, concurrency: int = 32, warmup: int = 2,
    **serve_kw,
):
    """Native end-to-end: serve + drive load. Returns
    ``(records, duration_s, outcome_class)``."""
    from vllm_evolve.bench.runtime import assert_plugin_invoked

    with serve_vllm(model, plugin_dir, gpus=gpus, port=port, **serve_kw) as h:
        start = time.monotonic()
        timings = drive_load(requests, port=port, concurrency=concurrency, warmup=warmup)
        duration = time.monotonic() - start
        log = Path(h.log_path).read_text(encoding="utf-8", errors="ignore")
    records = [record_from_timing(t) for t in timings]
    if not assert_plugin_invoked(log, h.nonce):     # nonce-aware (Codex HIGH#3)
        return records, duration, OutcomeClass.PLUGIN_LOAD_FAILURE
    return records, duration, OutcomeClass.EVAL_RESULT


def _build_requests(n: int, seed: int, regime: str) -> list[dict]:
    """Deterministic-by-seed synthetic workload.

    The throughput regime is **heterogeneous (bimodal)**: ~70% short requests and
    ~30% long ones (a ~10x prompt-size ratio) plus longer decode. That size
    variance is what makes scheduling matter — under FCFS a long request
    head-of-line-blocks the short ones and its prefill pressures KV, so
    order/admission-aware policies (SJF, KV-gated, packing) genuinely separate
    from FCFS. The earlier tiny-uniform workload (8-96 prompt tokens) finished so
    fast that no policy could differ. Latency stays small + uniform.
    """
    import random

    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        if regime == "throughput":
            # Long DECODE (output) is what grows KV after admission -> under high
            # concurrency the running set's KV exhausts mid-decode and vLLM must
            # reactively preempt. That is the regime where FCFS (no KV-gating) is
            # suboptimal, so outputs are long and ~40% of requests are heavy.
            if rng.random() < 0.6:                      # short: modest prompt + decode
                plen = rng.choice([16, 32, 48])
                olen = rng.choice([128, 256])
            else:                                       # long/heavy: big prompt + long decode
                plen = rng.choice([96, 128, 192])
                olen = rng.choice([768, 1024])
        else:                                           # latency: small + uniform
            plen = rng.choice([8, 16])
            olen = rng.choice([16, 32])
        prompt = "Summarize the following note in one sentence. " * plen + f"Note {i}."
        reqs.append({
            "request_id": f"s{seed}-r{i}",
            "arrival_s": 0.0,
            "prompt": prompt,
            "num_prompt_tokens": plen * 8,
            "num_output_tokens": olen,
        })
    return reqs


def native_bench(
    policy_source: str, profile, *,
    runner_kind: str = "candidate",
    model: str = DEFAULT_MODEL, gpus: str = "0", port: int = 8200,
    n_requests: int = 24, max_seeds: int | None = None, plugin_dir: str | None = None,
    concurrency: int | None = None,
    vllm_version: str = "", git_sha: str = "", hardware_profile: str = "",
    command_line: str = "",
    workload_requests: list[dict] | None = None,
    workload_provenance: dict | None = None,
    vllm_metrics_out: str | None = None,
    raw_requests_out: str | None = None,
    measurement_windows_out: str | None = None,
    replica_barrier_id: str | None = None,
    replica_index: int | None = None,
    replica_count: int | None = None,
    replica_barrier_timeout_s: float = 600.0,
    **serve_kw,
):
    """Full multi-seed native bench for one profile -> ``(eval_result, log)``.

    Starts the vLLM server ONCE and drives each seed's load against it (seeds
    vary only the workload, not the server), then aggregates via ``run_profile``
    and serializes to the schema-valid eval_result. ``runner_kind`` ``vanilla`` /
    ``strong_baseline`` serve the vLLM DEFAULT scheduler (no plugin, no policy, no
    ``--scheduler-cls``); ``candidate`` renders the policy + keeps the forgery check.
    """
    import dataclasses
    import tempfile

    from targets.scheduling.plugin_template import render_and_write
    from vllm_evolve.bench.eval_result import (
        build_eval_result,
        sha256_text,
        validate_eval_result,
    )
    from vllm_evolve.bench.profiles import primary_metric_fn
    from vllm_evolve.bench.runner import run_profile
    from vllm_evolve.bench.runtime import (
        contains_marker_forgery,
        plugin_provenance,
        record_from_timing,
    )
    from vllm_evolve.bench.vllm_metrics import VLLMMetricsSampler

    is_candidate = runner_kind == "candidate"
    plugin_dir = plugin_dir or tempfile.mkdtemp(prefix="ve_plugin_")
    if is_candidate:
        # Anti-fabrication (Codex HIGH#3): a candidate that prints reserved provenance
        # markers itself would forge the effective gate — refuse to render/run it.
        if contains_marker_forgery(policy_source):
            raise ValueError("policy source emits a reserved provenance marker "
                             "(effective-gate forgery) — rejected before render")
        render_and_write(plugin_dir, policy_source)
    else:
        # baseline: vLLM default scheduler, NO plugin (GAP-B). serve_vllm leaves
        # PYTHONPATH clean and omits --scheduler-cls when scheduler_cls is None.
        serve_kw["scheduler_cls"] = None
    seeds = list(profile.seeds[:max_seeds] if max_seeds else profile.seeds)
    measurement_windows: list[dict] = []
    # Client in-flight concurrency. Default scales with the server's max_num_seqs
    # (so high concurrency can actually saturate the GPU + fill KV), not a fixed
    # 32. An explicit --concurrency overrides. Latency regime stays single-stream.
    if concurrency is None:
        if workload_requests is not None:
            # Open-loop replay: all arrivals must reach vLLM at their recorded
            # times so queueing occurs in the server, not in the client.
            concurrency = len(workload_requests)
        else:
            concurrency = 1 if profile.regime == "latency" else min(
                n_requests, serve_kw.get("max_num_seqs") or 32
            )

    # Must run before ``serve_vllm``: the API server inherits this raised soft
    # limit and needs one server-side descriptor for every client connection.
    _ensure_nofile_capacity(concurrency)
    t0 = time.monotonic()
    with serve_vllm(model, plugin_dir, gpus=gpus, port=port, **serve_kw) as h:
        if workload_requests is not None:
            # Warm representative exact shapes without replaying a 120s+ trace.
            # This remains outside every measured window and is reported below.
            warmup_requests = _warmup_subset(workload_requests)
            drive_load(
                warmup_requests,
                port=port,
                concurrency=concurrency,
                warmup=0,
            )
            reset_prefix_cache(port)

        def run_one(seed):
            if workload_requests is None:
                reqs = _build_requests(n_requests, seed, profile.regime)
                warmup = 2
            else:
                # Repetitions must not inherit exact-prefix KV entries from the
                # previous seed. The reset is outside the measured duration.
                reset_prefix_cache(port)
                reqs = [
                    {
                        **request,
                        "request_id": f"{request['request_id']}-seed{seed}",
                        "seed": seed,
                    }
                    for request in workload_requests
                ]
                warmup = 0
            barrier_start = None
            if replica_barrier_id is not None:
                barrier_start = _wait_replica_seed_barrier(
                    replica_barrier_id,
                    replica_index=int(replica_index or 0),
                    replica_count=int(replica_count or 0),
                    seed=int(seed),
                    timeout_s=replica_barrier_timeout_s,
                )
            d0 = time.monotonic()
            started_at_unix_s = time.time()
            if vllm_metrics_out:
                sampler_context = VLLMMetricsSampler(
                    port=port,
                    path=vllm_metrics_out,
                    seed=seed,
                )
            else:
                sampler_context = contextlib.nullcontext()
            with sampler_context:
                timings = drive_load(
                    reqs, port=port, concurrency=concurrency, warmup=warmup)
            ended_at_unix_s = time.time()
            measurement_windows.append({
                "seed": int(seed),
                "started_at_unix_s": started_at_unix_s,
                "ended_at_unix_s": ended_at_unix_s,
                "duration_s": ended_at_unix_s - started_at_unix_s,
                "measured_requests": len(reqs),
                "replica_barrier_id": replica_barrier_id,
                "barrier_start_at_unix_s": barrier_start,
                "replica_index": replica_index,
                "replica_count": replica_count,
            })
            if raw_requests_out:
                raw_path = Path(raw_requests_out)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("a", encoding="utf-8", newline="\n") as fh:
                    for timing in timings:
                        row = dataclasses.asdict(timing)
                        row["seed"] = int(seed)
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
            return [record_from_timing(t) for t in timings], time.monotonic() - d0

        result = run_profile(
            profile.name, profile.primary_metric, run_one,
            primary_metric_fn(profile.primary_metric), profile.slo, seeds,
            seed_tiers=profile.seed_tiers, cv_threshold=profile.cv_threshold,
        )
        log = Path(h.log_path).read_text(encoding="utf-8", errors="ignore")
        # Anti-fabrication (Codex review P1): a CANDIDATE result is only real if the generated
        # scheduler plugin ACTUALLY ran. plugin_provenance is nonce-aware and distinguishes merely
        # INVOKED (loaded) from EFFECTIVE (invoked + reordered + no fallback).
        prov = plugin_provenance(log, h.nonce) if is_candidate else None
    wall = time.monotonic() - t0

    if is_candidate and not (prov and prov["invoked"]):
        # plugin never ran (e.g. --scheduler-cls ignored -> DEFAULT scheduler). Demote so LOCK D
        # (real_source_block requires outcome_class=='eval_result') blocks it — default-scheduler
        # metrics can never be promoted as a candidate gain/keep.
        result = dataclasses.replace(result, outcome_class=OutcomeClass.PLUGIN_LOAD_FAILURE)
    elif is_candidate and not prov["execution_valid"]:
        # plugin LOADED but never changed admission (no reorder / fell back) -> the metrics reflect
        # the DEFAULT order, not the candidate. Demote so compare/keep (LOCK D, source+outcome_class
        # only) also reject it, not just the accept gate's effective check (Codex review P1).
        result = dataclasses.replace(result, outcome_class=OutcomeClass.PLUGIN_INEFFECTIVE)

    er = build_eval_result(
        result, regime=profile.regime, slo=dataclasses.asdict(profile.slo),
        vllm_version=vllm_version, wall_time_s=wall,
        policy_sha256=sha256_text(policy_source), git_sha=git_sha,
        hardware_profile=hardware_profile, command_line=command_line,
    )
    er["measurement_protocol"] = {
        "full_trace_warmup_replays": (
            1
            if workload_requests is not None and len(workload_requests) <= 16
            else 0
        ),
        "warmup_request_count": (
            len(_warmup_subset(workload_requests))
            if workload_requests is not None else 0
        ),
        "warmup_is_outside_measured_window": True,
        "prefix_cache_reset_before_each_seed": workload_requests is not None,
    }
    er["measurement_windows"] = measurement_windows
    if measurement_windows_out:
        windows_path = Path(measurement_windows_out)
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        windows_path.write_text(
            json.dumps({
                "schema_version": 1,
                "source": "real_vllm",
                "windows": measurement_windows,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    # Record plugin provenance for the accept gate (Codex review P1): marker_verified == plugin
    # loaded (invoked); effective == invoked + reordered + no fallback. The accept boundary requires
    # effective (a merely-loaded or fallbacking policy must NOT adopt). Baselines: both False.
    er["marker_verified"] = bool(prov and prov["invoked"])
    er["effective"] = bool(prov and prov["effective"])
    er["mechanism_applicable"] = (
        prov.get("mechanism_applicable") if prov is not None else None
    )
    if prov is not None:
        er["plugin_provenance"] = prov
    if workload_provenance is not None:
        er["workload_provenance"] = workload_provenance
    validate_eval_result(er)
    return er, log


def main(argv=None) -> int:
    """Box-side CLI: run a native bench for one policy + profile, write eval_result.json."""
    import argparse
    import json
    import subprocess

    from vllm_evolve.bench.eval_result import write_eval_result
    from vllm_evolve.bench.profiles import load_profile

    ap = argparse.ArgumentParser(prog="python -m vllm_evolve.bench.native")
    ap.add_argument("--policy", default=None,
                    help="Policy .py (candidate runner). Omit for baselines.")
    ap.add_argument("--runner-kind", default="candidate", dest="runner_kind",
                    choices=["vanilla", "strong_baseline", "candidate"])
    ap.add_argument("--profile", required=True, help="profile yaml path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--n-requests", type=int, default=24, dest="n_requests")
    ap.add_argument("--max-seeds", type=int, default=None, dest="max_seeds")
    ap.add_argument(
        "--primary-metric",
        default=None,
        dest="primary_metric",
        help="Override the profile metric from frozen BenchConfig.",
    )
    ap.add_argument(
        "--slo-json",
        default=None,
        dest="slo_json",
        help="Override profile SLO with a JSON object from frozen BenchConfig.",
    )
    ap.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs",
                    help="vLLM max concurrent seqs (low value forces a waiting queue).")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Client in-flight requests (default scales with max-num-seqs).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    dest="gpu_memory_utilization",
                    help="vLLM GPU mem fraction (raises KV cache headroom).")
    ap.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    dest="tensor_parallel_size", help="Shard one model across N GPUs.")
    ap.add_argument("--out", required=True, help="eval_result.json output path")
    ap.add_argument("--log-out", default=None, dest="log_out")
    ap.add_argument("--vllm-metrics-out", default=None, dest="vllm_metrics_out")
    ap.add_argument("--raw-requests-out", default=None, dest="raw_requests_out")
    ap.add_argument(
        "--measurement-windows-out",
        default=None,
        dest="measurement_windows_out",
    )
    ap.add_argument("--replica-barrier-id", default=None)
    ap.add_argument("--replica-index", type=int, default=None)
    ap.add_argument("--replica-count", type=int, default=None)
    ap.add_argument(
        "--replica-barrier-timeout-s",
        type=float,
        default=600.0,
    )
    ap.add_argument("--extra-serve-args", dest="extra_serve_args", default=None,
                    help="Extra wired-lever flags forwarded verbatim to vllm serve "
                    "(quantization/kv dtype/prefix/chunked/eager), space-joined.")
    ap.add_argument(
        "--workload",
        default=None,
        help="Materialized exact-token replay JSON from engine.real_workloads.",
    )
    a = ap.parse_args(argv)

    if a.runner_kind == "candidate" and not a.policy:
        ap.error("--policy is required for --runner-kind candidate")
    policy_source = Path(a.policy).read_text(encoding="utf-8") if a.policy else ""
    profile = load_profile(a.profile)
    if a.primary_metric:
        profile.primary_metric = a.primary_metric
    if a.slo_json is not None:
        from vllm_evolve.bench.slo import SLO

        try:
            slo = json.loads(a.slo_json)
        except json.JSONDecodeError as exc:
            ap.error(f"--slo-json is invalid: {exc}")
        if not isinstance(slo, dict):
            ap.error("--slo-json must be a JSON object")
        unknown_slo = sorted(set(slo) - {"ttft_ms", "tpot_ms", "e2e_ms"})
        if unknown_slo:
            ap.error(f"--slo-json has unknown fields: {unknown_slo}")
        profile.slo = SLO(
            ttft_ms=slo.get("ttft_ms"),
            tpot_ms=slo.get("tpot_ms"),
            e2e_ms=slo.get("e2e_ms"),
        )
    workload_requests = None
    workload_provenance = None
    if a.workload:
        import hashlib

        payload = json.loads(Path(a.workload).read_text(encoding="utf-8"))
        expected = payload.get("payload_sha256")
        core = {key: value for key, value in payload.items() if key != "payload_sha256"}
        actual = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not expected or actual != expected:
            ap.error(f"workload payload SHA mismatch: expected={expected} actual={actual}")
        if (
            int(payload.get("schema_version") or 1) >= 2
            and payload.get("source_kind") == "official_burstgpt"
            and payload.get("replay")
        ):
            from vllm_evolve.bench.datasets.burstgpt import expand_scaled_payload

            workload_requests = expand_scaled_payload(payload)
        else:
            workload_requests = list(payload.get("requests") or [])
        if not workload_requests:
            ap.error("workload contains no requests")
        for request in workload_requests:
            if len(request.get("prompt_token_ids") or []) != int(
                request.get("num_prompt_tokens") or 0
            ):
                ap.error("workload prompt_token_ids length does not match num_prompt_tokens")
        workload_provenance = {
            key: payload.get(key) for key in (
                "scenario", "split", "source_kind", "slo_ttft_ms",
                "source_rows_sha256", "source_sha256", "payload_sha256",
                "replay", "token_generator_version",
            )
        }
        workload_provenance["expanded_requests_sha256"] = hashlib.sha256(
            json.dumps(
                workload_requests, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    try:
        vllm_version = __import__("vllm").__version__
    except Exception:
        vllm_version = "unknown"
    git_sha = ""
    with contextlib.suppress(Exception):
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
        ).stdout.strip()

    serve_extra = {}
    if a.gpu_memory_utilization is not None:
        serve_extra["gpu_memory_utilization"] = a.gpu_memory_utilization
    if a.max_model_len is not None:
        serve_extra["max_model_len"] = a.max_model_len
    if a.tensor_parallel_size is not None:
        serve_extra["tensor_parallel_size"] = a.tensor_parallel_size
    if a.extra_serve_args:
        serve_extra["extra_args"] = shlex.split(a.extra_serve_args)
    if a.log_out:
        # Persist startup failures inside the immutable run directory. Without
        # this, serve_vllm defaults to /tmp and dispatch cannot retrieve the
        # root cause when the server dies before native_bench returns.
        serve_extra["log_path"] = a.log_out
    er, log = native_bench(
        policy_source, profile, runner_kind=a.runner_kind,
        model=a.model, gpus=a.gpus, port=a.port,
        n_requests=a.n_requests, max_seeds=a.max_seeds, max_num_seqs=a.max_num_seqs,
        concurrency=a.concurrency,
        vllm_version=vllm_version, git_sha=git_sha, hardware_profile=_hardware_name(),
        command_line=f"native.py --runner-kind {a.runner_kind} --profile {a.profile}",
        workload_requests=workload_requests,
        workload_provenance=workload_provenance,
        vllm_metrics_out=a.vllm_metrics_out,
        raw_requests_out=a.raw_requests_out,
        measurement_windows_out=a.measurement_windows_out,
        replica_barrier_id=a.replica_barrier_id,
        replica_index=a.replica_index,
        replica_count=a.replica_count,
        replica_barrier_timeout_s=a.replica_barrier_timeout_s,
        **serve_extra,
    )
    write_eval_result(er, a.out)
    if a.log_out:
        Path(a.log_out).write_text(log, encoding="utf-8")
    print(json.dumps({
        "ok": True, "out": a.out, "outcome_class": er["outcome_class"],
        "primary_metric": er["primary_metric"],
        "median": er["aggregate_metrics"]["median"],
        "seeds": er["seeds"], "sample_count": er["sample_count"],
        "wall_time_s": round(er["wall_time_s"], 1),
    }))
    return 0


# Markers re-exported for convenience/tests.
__all__ = [
    "ServeHandle", "serve_vllm", "drive_load", "run_profile_native",
    "native_bench", "main", "_warmup_subset", "_ensure_nofile_capacity",
    "PLUGIN_INVOKED_MARKER", "PLUGIN_REORDERED_MARKER", "DEFAULT_MODEL",
]


if __name__ == "__main__":
    import sys
    sys.exit(main())
