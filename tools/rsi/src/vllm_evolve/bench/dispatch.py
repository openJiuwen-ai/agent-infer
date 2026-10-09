"""Remote bench dispatch (runs LOCALLY).

The machine driving ``ar`` has no GPU, so ``ve bench`` ships the policy to a
remote GPU host over ssh, runs ``python -m vllm_evolve.bench.native`` there
(real vLLM, native, no Docker), and pulls back the ``eval_result.json`` + server
log. Local only consumes the returned artifacts; all GPU work is remote.

Defaults are generic examples. Configure your own host and paths using flags or
``VE_*`` environment variables before running a remote benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vllm_evolve.bench.gpu_guard import (
    validate_gpu_budget,
    vllm_serve_port_pattern,
)
from vllm_evolve.bench.gpu_selection import bind_auto_gpu_selection

DEFAULTS = {
    "remote": os.environ.get("VE_REMOTE", "gpu-host"),
    "remote_repo": os.environ.get("VE_REMOTE_REPO", "/workspace/vllm-evolve"),
    "conda_env": os.environ.get("VE_CONDA_ENV", "vllm-evolve"),
    "conda_sh": os.environ.get(
        "VE_CONDA_SH", "/workspace/vllm-evolve/miniforge3/etc/profile.d/conda.sh"
    ),
    "model": os.environ.get("VE_MODEL", "facebook/opt-125m"),
    "gpus": os.environ.get("VE_GPUS", "0"),
    "hf_endpoint": os.environ.get("VE_HF_ENDPOINT", "https://hf-mirror.com"),
}


@dataclass
class RemoteBench:
    eval_result: dict
    log: str
    local_eval_path: str
    local_log_path: str
    remote_cmd: str = ""   # the ssh command line, for provenance/meta
    remote_run_dir: str = ""
    local_run_dir: str = ""
    status: dict | None = None

    def to_dict(self) -> dict:
        # emitted by `ve bench` on success so the next ve compare/keep has the JSON artifact + paths
        # (the serve log is on disk at local_log_path; not inlined here).
        return {
            "eval_result": self.eval_result,
            "local_eval_path": self.local_eval_path,
            "local_log_path": self.local_log_path,
            "remote_cmd": self.remote_cmd,
            "remote_run_dir": self.remote_run_dir,
            "local_run_dir": self.local_run_dir,
            "status": self.status,
        }


def _run(cmd: list[str], timeout_s: float):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def _scp(src: str, dst: str, timeout_s: float = 120.0) -> None:
    r = _run(["scp", "-q", "-o", "BatchMode=yes", src, dst], timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"scp {src} -> {dst} failed: {(r.stderr or r.stdout)[-500:]}")


def policy_remote_path(policy_path) -> tuple[str, str]:
    """Deterministic remote staging path for a local candidate policy:
    ``/tmp/ve_<content-sha>_policy.py``. Returns (remote_path, rid). Shared by the dispatch
    bench and the CUDA collector so both stage the SAME file to the SAME path (same caliber).
    Reads the policy source (raises if missing/unreadable)."""
    src = Path(policy_path).read_text(encoding="utf-8")
    rid = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/ve_{rid}_policy.py", rid


def stage_policy_to_remote(
    policy_path,
    remote,
    *,
    remote_path: str | None = None,
) -> tuple[str, str]:
    """scp the local candidate policy to its deterministic remote path; (remote_path, rid)."""
    r_policy, rid = policy_remote_path(policy_path)
    if remote_path is not None:
        r_policy = remote_path
    _scp(str(policy_path), f"{remote}:{r_policy}")
    return r_policy, rid


def _remote_cleanup(remote: str, port: int) -> None:
    """Stop every task-owned layer for a run-unique port.

    Losing or interrupting the local SSH client otherwise orphans the box-side
    worker, native driver, and detached vLLM process group.  Match only this
    user's Python module plus the run-unique port, then retain the historical
    vLLM-specific fallback for older remote releases. Best-effort; never raises.
    """
    import contextlib
    worker_pattern = (
        "python -m vllm_evolve.bench.remote_worker .*"
        f"--port {int(port)}([[:space:]]|$)"
    )
    native_pattern = (
        "python -m vllm_evolve.bench.native .*"
        f"--port {int(port)}([[:space:]]|$)"
    )
    serve_pattern = vllm_serve_port_pattern(port)
    kill = (
        "task_uid=\"$(id -u)\"; "
        f"pkill -TERM -u \"$task_uid\" -f -- {shlex.quote(worker_pattern)} "
        "2>/dev/null || true; "
        f"pkill -TERM -u \"$task_uid\" -f -- {shlex.quote(native_pattern)} "
        "2>/dev/null || true; "
        f"pkill -TERM -u \"$task_uid\" -f -- {shlex.quote(serve_pattern)} "
        "2>/dev/null || true; "
        "sleep 3; "
        f"pkill -KILL -u \"$task_uid\" -f -- {shlex.quote(native_pattern)} "
        "2>/dev/null || true; "
        f"pkill -KILL -u \"$task_uid\" -f -- {shlex.quote(serve_pattern)} "
        "2>/dev/null || true"
    )
    with contextlib.suppress(Exception):
        subprocess.run(["ssh", "-o", "BatchMode=yes", remote, kill],
                       capture_output=True, text=True, timeout=30)


def _ensure_remote_workspace(remote: str, workspace: str) -> None:
    """Require an existing writable workspace and create only owned subdirectories."""
    root = shlex.quote(workspace)
    command = (
        f"test -d {root} && test -w {root} && "
        f"mkdir -p {root}/staging {root}/runs {root}/locks "
        f"{root}/cache/huggingface {root}/cache/xdg"
    )
    result = _run(["ssh", "-o", "BatchMode=yes", remote, command], 30)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-500:]
        raise RuntimeError(
            f"remote workspace {workspace!r} is missing or not writable: {detail}"
        )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _compile_cache_namespace(
    *,
    model: str,
    gpus: str,
    tensor_parallel_size: int | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    extra_serve_args: list[str] | None,
) -> str:
    """Stable engine+physical-GPU cache key, independent of policy and workload.

    Two single-GPU replicas must never compile into the same Triton/Inductor
    directory concurrently (the Jiusi FUSE-backed cache produced stale file
    handles). Baseline/candidate/control on the same GPU and engine still reuse
    their compiled artifacts.
    """
    payload = {
        "schema_version": 1,
        "model": model,
        "gpus": gpus,
        "tensor_parallel_size": int(tensor_parallel_size or 1),
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "extra_serve_args": list(extra_serve_args or ()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"engine_{digest}"


def _pull_optional(remote: str, remote_path: str, local_path: Path) -> bool:
    # Remote run artifacts can live on a FUSE/NFS-backed workspace.  A single
    # transient read failure must not silently turn a complete raw timing file
    # into an apparent 0% client-delivery result.  Missing optional artifacts
    # still fail closed after the bounded retry budget.
    for _attempt in range(3):
        try:
            _scp(f"{remote}:{remote_path}", str(local_path))
            return True
        except Exception:
            continue
    return False


def run_remote_bench(
    policy_path: str | Path,
    profile: str,
    *,
    remote: str | None = None,
    remote_repo: str | None = None,
    conda_env: str | None = None,
    conda_sh: str | None = None,
    model: str | None = None,
    gpus: str | None = None,
    hf_endpoint: str | None = None,
    max_seeds: int | None = None,
    primary_metric: str | None = None,
    slo: dict | None = None,
    max_num_seqs: int | None = None,
    n_requests: int = 24,
    concurrency: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    extra_serve_args: list[str] | None = None,
    runner_kind: str = "candidate",
    port: int = 8260,
    timeout_s: float = 1200.0,
    remote_workspace: str | None = None,
    local_artifact_root: str | None = None,
    bench_config: dict | None = None,
    git_sha: str | None = None,
    workload_path: str | None = None,
) -> RemoteBench:
    """Ship ``policy_path`` to ``remote``, run the native bench for ``profile``
    there, and return the pulled-back eval_result + server log."""
    remote = remote or DEFAULTS["remote"]
    remote_repo = remote_repo or DEFAULTS["remote_repo"]
    conda_env = conda_env or DEFAULTS["conda_env"]
    conda_sh = conda_sh or DEFAULTS["conda_sh"]
    model = model or DEFAULTS["model"]
    gpus = gpus or DEFAULTS["gpus"]
    hf_endpoint = hf_endpoint or DEFAULTS["hf_endpoint"]
    gpus = ",".join(validate_gpu_budget(gpus, tensor_parallel_size))
    remote_workspace = remote_workspace or os.environ.get(
        "VE_REMOTE_WORKSPACE", "/workspace/vllm-evolve")
    local_artifact_root = local_artifact_root or os.environ.get(
        "VE_LOCAL_ARTIFACT_ROOT", "runs/real_vllm")
    git_sha = git_sha if git_sha is not None else _git_sha()

    is_candidate = runner_kind == "candidate"
    if is_candidate:
        policy_source = Path(policy_path).read_text(encoding="utf-8")
        policy_sha256 = hashlib.sha256(policy_source.encode("utf-8")).hexdigest()
        rid = policy_sha256[:12]
    else:
        # baseline: no policy to ship; rid is derived from the run identity
        rid = hashlib.sha256(
            f"{runner_kind}|{model}|{profile}".encode()).hexdigest()[:12]
        policy_sha256 = ""
    workload_sha256 = ""
    if workload_path:
        workload_sha256 = hashlib.sha256(Path(workload_path).read_bytes()).hexdigest()
    # The eval/log artifact name must be unique per (policy + bench config): rid alone is policy
    # bytes, so the SAME policy benched under different profiles/engine knobs would collide on one
    # /tmp path and a later run would overwrite the path an earlier `ve bench` emitted (Codex P2).
    run_sig = hashlib.sha256(repr((
        rid, runner_kind, model, profile, max_num_seqs, gpu_memory_utilization, max_model_len,
        tensor_parallel_size, concurrency, n_requests, max_seeds, workload_sha256,
        primary_metric, tuple(sorted((slo or {}).items())), tuple(extra_serve_args or ()),
    )).encode()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{run_sig}_{secrets.token_hex(3)}"
    remote_run_dir = f"{remote_workspace.rstrip('/')}/runs/{run_id}"
    r_eval = f"{remote_run_dir}/eval_result.json"
    r_log = f"{remote_run_dir}/serve.log"
    r_vllm_metrics = f"{remote_run_dir}/vllm_metrics.jsonl"
    r_raw_requests = f"{remote_run_dir}/raw_requests.jsonl"
    r_measurement_windows = f"{remote_run_dir}/measurement_windows.json"
    r_policy = f"{remote_run_dir}/policy.py" if is_candidate else None
    profile_rel = f"config/bench/profiles/{profile}.yaml"

    config_payload = bench_config or {
        "runner_kind": runner_kind,
        "model": model,
        "profile": profile,
        "gpus": gpus,
        "max_seeds": max_seeds,
        "max_num_seqs": max_num_seqs,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "extra_serve_args": list(extra_serve_args or ()),
        "workload_path": workload_path,
        "workload_sha256": workload_sha256,
        "port": port,
    }
    config_bytes = (
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    # Content-addressed staging is separate from the immutable per-run directory.
    # The box-side worker verifies both hashes before copying inputs into its run.
    _ensure_remote_workspace(remote, remote_workspace)
    staged_policy = None
    if is_candidate:
        staged_policy = f"{remote_workspace.rstrip('/')}/staging/policy_{policy_sha256}.py"
        stage_policy_to_remote(policy_path, remote, remote_path=staged_policy)
    staged_config = f"{remote_workspace.rstrip('/')}/staging/config_{config_sha256}.json"
    with tempfile.NamedTemporaryFile(prefix="ve_config_", suffix=".json", delete=False) as fh:
        fh.write(config_bytes)
        local_config_stage = Path(fh.name)
    try:
        _scp(str(local_config_stage), f"{remote}:{staged_config}")
    finally:
        local_config_stage.unlink(missing_ok=True)
    staged_workload = None
    if workload_path:
        staged_workload = (
            f"{remote_workspace.rstrip('/')}/staging/workload_{workload_sha256}.json"
        )
        _scp(str(workload_path), f"{remote}:{staged_workload}")

    # 2. run the native bench on the box
    native_argv = ["python", "-m", "vllm_evolve.bench.native"]
    if is_candidate:
        native_argv += ["--policy", r_policy]
    native_argv += [
        "--runner-kind", runner_kind,
        "--profile", profile_rel,
        "--model", model,
        "--gpus", gpus,
        "--port", str(port),
        "--n-requests", str(n_requests),
    ]
    if max_seeds:
        native_argv += ["--max-seeds", str(max_seeds)]
    if primary_metric:
        native_argv += ["--primary-metric", primary_metric]
    if slo is not None:
        native_argv += [
            "--slo-json",
            json.dumps(slo, sort_keys=True, separators=(",", ":")),
        ]
    if max_num_seqs:
        native_argv += ["--max-num-seqs", str(max_num_seqs)]
    if concurrency:
        native_argv += ["--concurrency", str(concurrency)]
    if gpu_memory_utilization is not None:
        native_argv += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len:
        native_argv += ["--max-model-len", str(max_model_len)]
    if tensor_parallel_size:
        native_argv += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if extra_serve_args:
        native_argv += ["--extra-serve-args", " ".join(extra_serve_args)]
    if staged_workload is not None:
        native_argv += ["--workload", f"{remote_run_dir}/workload.json"]
    replica_member = (
        config_payload.get("workload", {})
        .get("workload_spec", {})
        .get("replica_member", {})
        if isinstance(config_payload, dict) else {}
    )
    if replica_member:
        native_argv += [
            "--replica-barrier-id", str(replica_member["barrier_id"]),
            "--replica-index", str(int(replica_member["index"])),
            "--replica-count", str(int(replica_member["count"])),
            "--replica-barrier-timeout-s",
            str(float(replica_member.get("barrier_timeout_s") or 600.0)),
        ]
    native_argv += [
        "--out", r_eval,
        "--log-out", r_log,
        "--vllm-metrics-out", r_vllm_metrics,
        "--raw-requests-out", r_raw_requests,
        "--measurement-windows-out", r_measurement_windows,
    ]

    worker_argv = [
        "python", "-m", "vllm_evolve.bench.remote_worker",
        "--run-dir", remote_run_dir,
        "--workspace", remote_workspace,
        "--gpus", gpus,
        "--tensor-parallel-size", str(tensor_parallel_size or 1),
        "--port", str(port),
        "--timeout-s", str(timeout_s),
        "--candidate-id", rid,
        "--git-sha", git_sha or "unknown",
        "--config-sha256", config_sha256,
        "--staged-config", staged_config,
    ]
    if staged_policy is not None:
        worker_argv += [
            "--policy-sha256", policy_sha256,
            "--staged-policy", staged_policy,
        ]
    if staged_workload is not None:
        worker_argv += [
            "--workload-sha256", workload_sha256,
            "--staged-workload", staged_workload,
        ]
    worker_argv += ["--", *native_argv]
    cache = f"{remote_workspace.rstrip('/')}/cache"
    hf_home = cache + "/huggingface"
    hf_hub_cache = hf_home + "/hub"
    compile_namespace = _compile_cache_namespace(
        model=model,
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        extra_serve_args=extra_serve_args,
    )
    # Triton/Inductor cache files are mmap/read during worker startup.  Keeping
    # them on an NFS-backed remote workspace makes an interrupted compilation
    # capable of poisoning later runs with ESTALE.  The namespace already binds
    # every compilation-affecting engine/GPU field, so use node-local storage
    # while preserving safe reuse across same-caliber candidate/baseline arms.
    compile_cache = f"/tmp/vllm-evolve/compile/{compile_namespace}"
    pythonpath = f"{remote_repo.rstrip('/')}/src:{remote_repo.rstrip('/')}"
    remote_cmd = (
        f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && "
        f"cd {shlex.quote(remote_repo)} && "
        f"mkdir -p {shlex.quote(compile_cache + '/xdg')} "
        f"{shlex.quote(compile_cache + '/torchinductor')} "
        f"{shlex.quote(compile_cache + '/triton')} && "
        f"export HF_ENDPOINT={shlex.quote(hf_endpoint)} "
        f"HF_HOME={shlex.quote(hf_home)} "
        f"HF_HUB_CACHE={shlex.quote(hf_hub_cache)} "
        f"HUGGINGFACE_HUB_CACHE={shlex.quote(hf_hub_cache)} "
        f"TRANSFORMERS_CACHE={shlex.quote(hf_home + '/transformers')} "
        f"XDG_CACHE_HOME={shlex.quote(compile_cache + '/xdg')} "
        f"TORCHINDUCTOR_CACHE_DIR={shlex.quote(compile_cache + '/torchinductor')} "
        f"TRITON_CACHE_DIR={shlex.quote(compile_cache + '/triton')} "
        f"PYTHONPATH={shlex.quote(pythonpath)} PYTHONUNBUFFERED=1 && "
        f"{shlex.join(worker_argv)}"
    )

    local_run_dir = Path(local_artifact_root).expanduser().resolve() / run_id
    local_run_dir.mkdir(parents=True, exist_ok=False)
    try:
        # The box-side worker owns the experiment timeout and needs a grace
        # window to kill the process tree, sample final GPU state, and atomically
        # write status.json before the SSH transport is abandoned.
        r = _run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=40",
                remote,
                remote_cmd,
            ],
            timeout_s + 120,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _remote_cleanup(remote, port)  # don't leave a vLLM holding the card
        raise

    # 3. Pull provenance on success and failure.  A failed run remains a
    # machine-readable experiment rather than disappearing into SSH stderr.
    artifact_names = (
        "status.json", "manifest.json", "bench_config.json", "gpu_before.json",
        "gpu_after.json", "gpu_samples.jsonl", "gpu_summary.json",
        "native.stdout.log", "native.stderr.log", "serve.log",
        "eval_result.json", "policy.py",
        "workload.json", "vllm_metrics.jsonl", "raw_requests.jsonl",
        "measurement_windows.json",
    )
    for name in artifact_names:
        _pull_optional(remote, f"{remote_run_dir}/{name}", local_run_dir / name)
    local_status = local_run_dir / "status.json"
    status = (
        json.loads(local_status.read_text(encoding="utf-8"))
        if local_status.exists() else {
            "ok": False,
            "state": "ssh_failure",
            "exit_code": r.returncode,
            "reason": (r.stderr or r.stdout)[-1500:],
            "remote_run_dir": remote_run_dir,
        }
    )
    if not status.get("reason"):
        native_stderr = local_run_dir / "native.stderr.log"
        if native_stderr.exists():
            status["reason"] = native_stderr.read_text(
                encoding="utf-8", errors="ignore"
            )[-3000:]
    if r.returncode != 0 or not status.get("ok"):
        failure_path = local_run_dir / "status.json"
        if not failure_path.exists():
            failure_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"remote bench failed; structured status: {failure_path}\n"
            f"{status.get('reason') or (r.stderr or r.stdout)[-1000:]}"
        )

    local_eval = local_run_dir / "eval_result.json"
    local_log = local_run_dir / "serve.log"
    if not local_eval.exists():
        raise RuntimeError(f"remote bench succeeded without eval_result: {local_run_dir}")
    eval_result = json.loads(local_eval.read_text(encoding="utf-8"))
    from vllm_evolve.bench.validity_artifacts import attach_workload_validity

    eval_result = attach_workload_validity(
        eval_result,
        status=status,
        run_dir=local_run_dir,
        bench_config=config_payload,
    )
    local_eval.write_text(
        json.dumps(eval_result, indent=2) + "\n",
        encoding="utf-8",
    )
    log = local_log.read_text(encoding="utf-8", errors="ignore") if local_log.exists() else ""
    return RemoteBench(
        eval_result,
        log,
        str(local_eval),
        str(local_log),
        remote_cmd=remote_cmd,
        remote_run_dir=remote_run_dir,
        local_run_dir=str(local_run_dir),
        status=status,
    )


def _config_extra_serve_args(config) -> list[str]:
    """Wired levers beyond run_remote_bench's native kwargs, rendered FROM the config."""
    e = config.engine
    extra: list[str] = []
    if e.quantization:
        extra += ["--quantization", e.quantization]
    if e.kv_cache_dtype:
        extra += ["--kv-cache-dtype", e.kv_cache_dtype]
    if e.enable_prefix_caching is True:
        extra += ["--enable-prefix-caching"]
    elif e.enable_prefix_caching is False:
        extra += ["--no-enable-prefix-caching"]
    if e.enable_chunked_prefill is True:
        extra += ["--enable-chunked-prefill"]
    elif e.enable_chunked_prefill is False:
        extra += ["--no-enable-chunked-prefill"]    # explicit disable (False != None=vLLM default)
    if e.async_scheduling is True:
        extra += ["--async-scheduling"]
    elif e.async_scheduling is False:
        extra += ["--no-async-scheduling"]
    if e.enforce_eager:
        extra += ["--enforce-eager"]
    if e.max_num_batched_tokens:
        extra += ["--max-num-batched-tokens", str(e.max_num_batched_tokens)]
    return extra


def _run_remote_bench_config_single(
    config,
    *,
    git_sha: str | None = None,
) -> RemoteBench:
    """BenchConfig-consuming entry — THE execution contract. The config (not loose
    kwargs) is the single source of truth; this maps it to the remote native bench for
    ALL runner kinds: ``vanilla`` / ``strong_baseline`` serve the vLLM default scheduler
    (no policy / plugin / --scheduler-cls), ``candidate`` ships the policy + plugin. The
    quantized artifact is served when required and every wired lever is forwarded. Box
    unreachability is surfaced by the SSH call itself — never pre-empted by a guard.
    """
    e, w, r = config.engine, config.workload, config.runner
    rb = run_remote_bench(
        r.policy_path, w.regime, remote=r.remote, model=config.model_to_serve(),
        # forward the FULL remote execution environment so the SSH run matches the recorded
        # provenance — not the process-wide defaults (Codex review P2).
        remote_repo=r.remote_repo, conda_env=r.conda_env, conda_sh=r.conda_sh,
        remote_workspace=r.remote_workspace, local_artifact_root=r.local_artifact_root,
        hf_endpoint=r.hf_endpoint,
        gpus=config.environment.gpus, max_num_seqs=e.max_num_seqs,
        gpu_memory_utilization=e.gpu_memory_utilization, max_model_len=e.max_model_len,
        tensor_parallel_size=e.tensor_parallel_size,
        # honor the --max-seeds budget cap so smoke/budgeted runs don't run the full set (Codex P2)
        max_seeds=config.statistical.max_seeds,
        primary_metric=config.statistical.primary_metric,
        slo=config.statistical.slo,
        # workload fields must reach native so the sweep actually varies the load (Codex R2)
        concurrency=w.concurrency, n_requests=w.n_requests, workload_path=w.trace_path,
        # Large real-model calibration includes model loading, warmup, and a >=120 s
        # measured window.  Keep the execution deadline in the frozen BenchConfig
        # instead of silently falling back to run_remote_bench's 20 minute default.
        timeout_s=float(config.statistical.timeout_s),
        extra_serve_args=_config_extra_serve_args(config), bench_config=config.to_dict(),
        runner_kind=r.runner_kind, port=r.port, git_sha=git_sha)
    # Augment the pulled eval_result with BenchConfig provenance (Codex R3) so the A/B
    # caliber is enforceable end-to-end on REAL runs, not just synthetic fixtures.
    prov = config.provenance()
    if isinstance(rb.eval_result, dict):
        rb.eval_result.setdefault("bench_config", prov["bench_config"])
        rb.eval_result["runner_kind"] = prov["runner_kind"]
        rb.eval_result["model_served"] = prov["model_served"]
        rb.eval_result["rendered_serve_args"] = prov["rendered_serve_args"]
        rb.eval_result["remote_cmd"] = rb.remote_cmd
        rb.eval_result["remote_run_dir"] = rb.remote_run_dir
        rb.eval_result["remote_resource_evidence"] = (
            rb.status.get("gpu_summary") if isinstance(rb.status, dict) else None
        )
        rb.eval_result["remote_worker_duration_s"] = (
            rb.status.get("duration_s") if isinstance(rb.status, dict) else None
        )
        # Persist the augmented eval_result back to the advertised local_eval_path: `ve bench` emits
        # that path and the next `ve compare`/keep reads the FILE — which would otherwise be the raw
        # remote JSON without bench_config and fail same_caliber_unverifiable (Codex review P1).
        if rb.local_eval_path:
            try:
                Path(rb.local_eval_path).write_text(
                    json.dumps(rb.eval_result, indent=2) + "\n", encoding="utf-8")
            except OSError:
                pass
    return rb


def run_remote_bench_config(config, *, git_sha: str | None = None) -> RemoteBench:
    """Execute one frozen config as TP/one GPU or a synchronized replica group.

    ``environment.gpus=auto`` is intentionally bound in-place here, before
    topology dispatch.  A real-evolution baseline and all configs cloned from
    it therefore retain one concrete, provenance-recorded GPU set.
    """
    bind_auto_gpu_selection(config)
    parallel_mode = str(
        (config.workload.workload_spec or {}).get("parallel_mode") or ""
    )
    if parallel_mode == "dual_replica":
        from vllm_evolve.bench.replica_group import run_remote_replica_group

        result = run_remote_replica_group(
            config,
            git_sha=git_sha,
            single_runner=_run_remote_bench_config_single,
            cleanup=_remote_cleanup,
        )
    else:
        result = _run_remote_bench_config_single(config, git_sha=git_sha)
    evidence = dict(config.environment.gpu_selection_evidence or {})
    if evidence and isinstance(result.eval_result, dict):
        result.eval_result["gpu_selection"] = evidence
        if result.local_eval_path:
            try:
                Path(result.local_eval_path).write_text(
                    json.dumps(result.eval_result, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
    return result
