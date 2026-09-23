"""M-B4 (collector half) — CUDA profiling collection: capability probe + run window.

The PARSER (``autopt/cuda_parse.py``) is pure + unit-tested; this is the COLLECTOR. It
probes which profiler is available (``nsys`` / ``torch.profiler`` / ``ncu``), runs a SHORT
profiled diagnostic window against a real serve window (box-gated, injected as
``collect_fn``), stores the artifact, and parses it into a ``KernelBreakdown``. If the
requested profiler is unavailable or collection fails, it returns an HONEST degraded result
carrying the capability evidence + attempted command — never a fabricated breakdown.

Unprofiled benchmark metrics (real performance) are collected separately by the normal bench
path; this profiled window is for diagnosis only and is never used as a performance number.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field

_PROFILERS = ("nsys", "torch", "ncu")


def capability_probe(profiler: str, remote: str | None = None) -> dict:
    """Is ``profiler`` available where it will RUN? Collection-location aware (Codex R7):
    the CLI runs the profiler on the GPU box, so probing the local controller's PATH is
    wrong — pass ``remote`` to probe the box via SSH instead.
    """
    if remote:
        if profiler == "torch":
            cmd = "python -c 'import torch.profiler'"
        elif profiler in ("nsys", "ncu"):
            cmd = f"command -v {profiler}"
        else:
            return {"profiler": profiler, "available": False, "location": "remote",
                    "detail": "unknown profiler"}
        try:
            import subprocess
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", remote, cmd],
                               capture_output=True, text=True, timeout=20)
            ok = r.returncode == 0
            detail = ((r.stdout or r.stderr).strip()[-120:]) or ("ok" if ok else "not found")
            return {"profiler": profiler, "available": ok, "location": "remote", "detail": detail}
        except Exception as exc:  # noqa: BLE001 - honest remote probe failure
            return {"profiler": profiler, "available": False, "location": "remote",
                    "detail": f"remote probe failed: {exc}"}
    # local controller probe (used by unit tests / local profilers)
    if profiler == "torch":
        try:
            import torch.profiler  # noqa: F401
            return {"profiler": "torch", "available": True, "location": "local",
                    "detail": "torch.profiler importable"}
        except Exception as exc:  # noqa: BLE001 - honest capability result
            return {"profiler": "torch", "available": False, "location": "local",
                    "detail": f"import failed: {exc}"}
    if profiler in ("nsys", "ncu"):
        path = shutil.which(profiler)
        return {"profiler": profiler, "available": bool(path), "location": "local",
                "detail": path or f"{profiler} not on PATH"}
    return {"profiler": profiler, "available": False, "location": "local",
            "detail": "unknown profiler"}


def probe_all(remote: str | None = None) -> dict:
    return {p: capability_probe(p, remote) for p in _PROFILERS}


@dataclass
class CudaProfileResult:
    profiler: str
    status: str            # collected | degraded_unavailable | degraded_failed
    capability: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    breakdown: dict | None = None     # KernelBreakdown.to_dict() when parsed
    attempted_cmd: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def collect_cuda_profile(profiler: str, *, collect_fn=None, parse_fn=None,
                         attempted_cmd: str = "", remote: str | None = None) -> CudaProfileResult:
    """Probe capability (on ``remote`` if given), then run ``collect_fn`` (box-gated) and parse.

    ``collect_fn()`` -> ``(artifact_paths, attempted_cmd)``; ``parse_fn(artifact)`` ->
    KernelBreakdown. ``attempted_cmd`` (the command the collector WILL run) is recorded even
    when collection fails, so a degraded result still shows what was tried. Unavailable
    profiler or any failure yields an honest degraded result — never a fabricated breakdown.
    """
    cap = capability_probe(profiler, remote)
    if not cap["available"]:
        return CudaProfileResult(profiler, "degraded_unavailable", capability=cap,
                                 attempted_cmd=attempted_cmd)
    if collect_fn is None:
        return CudaProfileResult(profiler, "degraded_unavailable", capability=cap,
                                 attempted_cmd=attempted_cmd,
                                 error="no collector wired (box-gated real run)")
    try:
        artifacts, cmd = collect_fn()
    except Exception as exc:  # noqa: BLE001 - honest collection failure
        return CudaProfileResult(profiler, "degraded_failed", capability=cap,
                                 attempted_cmd=attempted_cmd, error=str(exc)[-300:])
    breakdown = None
    if parse_fn and artifacts:
        try:
            breakdown = parse_fn(artifacts[0]).to_dict()
        except Exception as exc:  # noqa: BLE001 - honest parse failure
            return CudaProfileResult(profiler, "degraded_failed", capability=cap,
                                     artifacts=list(artifacts), attempted_cmd=cmd or attempted_cmd,
                                     error=f"parse failed: {str(exc)[-200:]}")
    return CudaProfileResult(profiler, "collected", capability=cap, artifacts=list(artifacts),
                             breakdown=breakdown, attempted_cmd=cmd or attempted_cmd)


def make_cuda_collector(profiler: str, *, bench_config, run_dir: str, duration_s: int = 5):
    """Build a SAME-CALIBER, remote-correct CUDA collector (Codex R7/R8). Returns
    ``(collect_fn, attempted_cmd)``.

    The profiled window wraps the SAME canonical workload runner as the real bench —
    ``python -m vllm_evolve.bench.native`` — under the profiler, rendering the FULL BenchConfig:
    engine levers + WORKLOAD fields (regime / n_requests / concurrency) + runner kind + candidate
    policy + port, inside the SAME remote ENVIRONMENT prelude as ``run_remote_bench`` (conda
    activate, cd remote_repo, HF_ENDPOINT, CUDA_VISIBLE_DEVICES). Only the profiler wrapper +
    a short ``--max-seeds 1`` diagnostic window differ — so the trace captures the LOADED
    workload, not an idle server. The remote command uses a REMOTE POSIX artifact dir; the local
    ``run_dir/cuda/`` is a SEPARATE pull-back dir (a Windows path is NEVER interpolated into the
    remote command). Every subprocess boundary is checked: ssh / scp / no-artifact -> degraded.
    """
    import hashlib
    import os
    import shlex
    import subprocess

    from vllm_evolve.bench.dispatch import (
        DEFAULTS,
        _config_extra_serve_args,
        policy_remote_path,
        stage_policy_to_remote,
    )

    bc = bench_config
    r, e, w = bc.runner, bc.engine, bc.workload
    remote = r.remote
    run_id = hashlib.sha256(repr(bc.provenance()).encode("utf-8")).hexdigest()[:10]
    remote_dir = f"/tmp/vllm_evolve_cuda/{run_id}"          # POSIX, on the GPU box
    remote_art = f"{remote_dir}/{profiler}_window"
    remote_eval = f"{remote_dir}/eval.json"
    remote_log = f"{remote_dir}/serve.log"
    local_dir = os.path.join(run_dir, "cuda")               # local pull-back (may be Windows)
    gpus = bc.environment.gpus

    # the canonical same-caliber workload runner (NOT a bare `vllm serve`): full BenchConfig.
    # Candidate: stage the policy to its REMOTE path (same as run_remote_bench) — the command
    # references /tmp/ve_<hash>_policy.py, NOT the controller-local path (Codex R9). Reading the
    # policy here raises if it is missing -> the caller degrades instead of profiling a ghost file.
    staged_local = None
    remote_policy = None
    policy_arg = ""
    if r.runner_kind == "candidate" and r.policy_path:
        remote_policy, _ = policy_remote_path(r.policy_path)
        staged_local = r.policy_path
        policy_arg = f"--policy {remote_policy} "
    flags = ""
    if e.max_num_seqs:
        flags += f"--max-num-seqs {e.max_num_seqs} "
    if w.concurrency:
        flags += f"--concurrency {w.concurrency} "
    if e.gpu_memory_utilization is not None:
        flags += f"--gpu-memory-utilization {e.gpu_memory_utilization} "
    if e.max_model_len:
        flags += f"--max-model-len {e.max_model_len} "
    if e.tensor_parallel_size and e.tensor_parallel_size > 1:
        flags += f"--tensor-parallel-size {e.tensor_parallel_size} "
    extra_levers = _config_extra_serve_args(bc)
    if extra_levers:
        flags += "--extra-serve-args " + shlex.quote(" ".join(extra_levers)) + " "
    native = (
        f"python -m vllm_evolve.bench.native {policy_arg}--runner-kind {r.runner_kind} "
        f"--profile config/bench/profiles/{w.regime}.yaml --model {bc.model_to_serve()} "
        f"--gpus {gpus} --port {r.port} --n-requests {w.n_requests} --max-seeds 1 "
        f"{flags}--out {remote_eval} --log-out {remote_log}")
    if profiler == "nsys":
        wrapped = f"nsys profile --duration={duration_s} -o {remote_art} {native}"
    elif profiler == "ncu":
        wrapped = f"ncu --csv --launch-count 50 -o {remote_art} {native}"
    else:  # torch
        wrapped = f"VE_TORCH_PROFILE_OUT={remote_art}.json VE_TORCH_PROFILE_S={duration_s} {native}"
    prelude = (f"source {DEFAULTS['conda_sh']} && conda activate {r.conda_env} && "
               f"cd {r.remote_repo} && export HF_ENDPOINT={r.hf_endpoint} && "
               f"export CUDA_VISIBLE_DEVICES={gpus} && mkdir -p {remote_dir}")
    remote_cmd = f"{prelude} && {wrapped}"

    def collect_fn():
        os.makedirs(local_dir, exist_ok=True)
        # candidate: stage the policy to the box FIRST via the SHARED helper (the exact same
        # staging the unprofiled dispatch bench uses), so the two paths cannot drift. Guard that
        # the staged remote path matches the one rendered into the profiler command (same caliber).
        if staged_local:
            staged_remote, _ = stage_policy_to_remote(staged_local, remote)
            if staged_remote != remote_policy:
                raise RuntimeError(
                    f"policy staging mismatch: {staged_remote} != {remote_policy}")
        rr = subprocess.run(["ssh", "-o", "BatchMode=yes", remote, remote_cmd],
                            capture_output=True, text=True, timeout=240)
        if rr.returncode != 0:
            raise RuntimeError(f"remote cuda collect failed (exit {rr.returncode}): "
                               f"{(rr.stderr or rr.stdout)[-200:]}")
        s = subprocess.run(["scp", f"{remote}:{remote_art}*", local_dir],
                           capture_output=True, text=True, timeout=120)
        if s.returncode != 0:
            raise RuntimeError(f"scp pull-back failed: {(s.stderr or s.stdout)[-200:]}")
        pulled = [os.path.join(local_dir, f) for f in os.listdir(local_dir)
                  if f.startswith(f"{profiler}_window")]
        if not pulled:
            raise RuntimeError("no cuda artifact was pulled back from the box")
        return (pulled, remote_cmd)

    return collect_fn, remote_cmd
