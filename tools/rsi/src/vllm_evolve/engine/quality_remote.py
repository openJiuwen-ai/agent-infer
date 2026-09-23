"""Driver for the remote served-model quality measure used by ``ve run --backend remote``.

``run_autopt``'s accept gate certifies quality via an injected
``quality_measure_fn(config) -> measure_fn``, ``measure_fn(prompts) ->
{perplexity, task_em, output_agreement}``. The serve + calibration MUST run ON the GPU box — the
same SSH host ``collect_profile`` / ``run_remote_bench_config`` use — NOT the local no-GPU driver.
So this SSHes to ``bc.runner.remote`` and runs ``python -m vllm_evolve.engine.quality_probe`` there
(``engine.quality_probe.probe`` serves + queries on the box, where ``127.0.0.1`` is the GPU),
mirroring ``run_remote_bench``. Output agreement is candidate-vs-baseline, computed HERE from the
per-config continuations the box returns (the baseline is measured before the candidate).

Box-gated and honest: an SSH/serve failure, an unreachable box, a missing field, ``null`` probe
output, or no resolvable remote -> ``None``, which ``measure_quality`` treats as NOT MEASURED (the
gate hard-fails -> no false adopt). The live path needs the box; the driver logic is unit-tested by
mocking the SSH boundary.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import shlex
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from vllm_evolve.bench.gpu_guard import validate_gpu_budget
from vllm_evolve.engine.quality_probe import bench_config_for

_SSH_TIMEOUT_S = 1800.0


def _ssh_run(remote: str, remote_cmd: str, payload_json: str):
    """One SSH invocation of the box-side quality probe. Wrapped so tests can stub the boundary."""
    return subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            remote,
            remote_cmd,
        ],
        input=payload_json,
        capture_output=True,
        text=True,
        timeout=_SSH_TIMEOUT_S,
    )


def _remote_probe(bc, flat: dict, prompts: list[str]) -> dict | None:
    """Run the served-model quality probe ON ``bc.runner.remote``; return its dict or ``None``."""
    r = bc.runner
    if not r.remote:
        return None
    from vllm_evolve.bench.dispatch import (
        _pull_optional,
        _remote_cleanup,
    )
    from vllm_evolve.engine.quality_probe import _PORT

    tensor_parallel_size = int(bc.engine.tensor_parallel_size or 1)
    gpus = ",".join(validate_gpu_budget(
        bc.environment.gpus,
        tensor_parallel_size,
    ))
    payload = json.dumps(
        {"config": flat, "prompts": list(prompts)},
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_sha = hashlib.sha256(payload.encode()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"quality_{stamp}_{payload_sha[:12]}_{secrets.token_hex(3)}"
    remote_run_dir = f"{r.remote_workspace.rstrip('/')}/runs/{run_id}"
    local_run_dir = Path(r.local_artifact_root).expanduser().resolve() / run_id
    cache = f"{r.remote_workspace.rstrip('/')}/cache"
    worker = [
        "python", "-m", "vllm_evolve.bench.remote_worker",
        "--run-dir", remote_run_dir,
        "--workspace", r.remote_workspace,
        "--gpus", gpus,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--port", str(_PORT),
        "--timeout-s", str(_SSH_TIMEOUT_S - 120),
        "--candidate-id", f"quality-{payload_sha[:12]}",
        "--git-sha", "quality-probe",
        "--config-sha256", payload_sha,
        "--",
        "python", "-m", "vllm_evolve.engine.quality_probe",
    ]
    workspace = shlex.quote(r.remote_workspace)
    pythonpath = f"{r.remote_repo.rstrip('/')}/src:{r.remote_repo.rstrip('/')}"
    remote_cmd = (
        f"test -d {workspace} && test -w {workspace} && "
        f"mkdir -p {workspace}/runs {workspace}/locks "
        f"{workspace}/cache/huggingface {workspace}/cache/xdg && "
        f"source {shlex.quote(r.conda_sh)} && conda activate {shlex.quote(r.conda_env)} && "
        f"cd {shlex.quote(r.remote_repo)} && "
        f"export HF_ENDPOINT={shlex.quote(r.hf_endpoint)} "
        f"HF_HOME={shlex.quote(cache + '/huggingface')} "
        f"XDG_CACHE_HOME={shlex.quote(cache + '/xdg')} "
        f"PYTHONPATH={shlex.quote(pythonpath)} && "
        f"{shlex.join(worker)}; "
        "ve_quality_rc=$?; "
        f"cat {shlex.quote(remote_run_dir + '/native.stdout.log')} 2>/dev/null || true; "
        "exit ${ve_quality_rc}"
    )
    try:
        res = _ssh_run(r.remote, remote_cmd, payload)
    except subprocess.TimeoutExpired:
        # the box-side `vllm serve` (new session, fixed port) keeps running after our ssh client
        # gives up — kill it by port so the GPU/port is freed before the next probe/bench, mirroring
        # run_remote_bench's cleanup. Best-effort; still NOT MEASURED.
        _remote_cleanup(r.remote, _PORT)
        return None
    except Exception:        # noqa: BLE001 - box unreachable / ssh failure -> NOT MEASURED
        return None
    # Preserve the same pre/post/continuous GPU evidence contract as performance
    # runs. Unit-test stubs return SimpleNamespace; only a real CompletedProcess
    # triggers network artifact collection.
    if isinstance(res, subprocess.CompletedProcess):
        local_run_dir.mkdir(parents=True, exist_ok=False)
        for name in (
            "status.json", "manifest.json", "gpu_before.json",
            "gpu_after.json", "gpu_samples.jsonl", "gpu_summary.json",
            "native.stdout.log", "native.stderr.log",
        ):
            _pull_optional(
                r.remote,
                f"{remote_run_dir}/{name}",
                local_run_dir / name,
            )
    if res.returncode != 0 or not (res.stdout or "").strip():
        return None
    try:
        out = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:        # noqa: BLE001 - unparseable probe output -> NOT MEASURED
        return None
    if out and local_run_dir.exists():
        out["_ve_evidence_dir"] = str(local_run_dir)
    return out or None


def _agreement(cand: list[str], base: list[str] | None) -> float:
    if not base or len(cand) != len(base):
        return 0.0
    total = match = 0
    for c, b in zip(cand, base):
        ct, bt = (c or "").split(), (b or "").split()
        n = max(len(ct), len(bt))
        if n == 0:
            continue
        total += n
        match += sum(1 for i in range(min(len(ct), len(bt))) if ct[i] == bt[i])
    # No comparable tokens (the served model generated NOTHING on every prompt) is NOT "perfect
    # agreement" — it is no evidence, which must FAIL the gate (the harness's missing-signal=FAIL
    # invariant), so a degenerate/echo-only serve can never false-certify quality.
    return match / total if total else 0.0


def make_remote_quality_measure_fn(base_config: dict | None) -> Callable:
    """Return a ``quality_measure_fn(config) -> measure_fn`` that measures quality on the GPU box.

    The candidate is served (over SSH, on the box) with its OWN quality-affecting engine config, so
    a representation change is actually measured. Output agreement is candidate-vs-baseline and is
    re-established EACH round: ``measured_quality_verdict`` measures the round's baseline then its
    candidate (once each), so a call-parity counter makes every EVEN call the round's baseline and
    every ODD call its candidate. This matters for a shifted-bottleneck run where ``run_autopt``
    adopts and continues with the SAME closure — the follow-up candidate must be compared against
    the CURRENT adopted baseline, not the first baseline ever. The counter advances on every call
    (even a probe that returns None), so a failed measure cannot desync the parity (the model is
    constant, so the factory always yields a measure_fn -> exactly two measure calls per round).
    """
    base_config = dict(base_config or {})
    state: dict = {"calls": 0}     # call-parity + the current round's baseline continuations

    def quality_measure_fn(config: dict) -> Callable | None:
        flat = {**base_config, **dict(config or {})}
        if not flat.get("model"):
            return None                     # no model to serve -> NOT MEASURED (no false adopt)
        bc = bench_config_for(flat)

        def measure(prompts: list[str]) -> dict | None:
            idx = state["calls"]
            state["calls"] = idx + 1        # advance every call so a None probe can't desync parity
            out = _remote_probe(bc, flat, prompts)
            if not out:
                return None
            evidence_dir = out.pop("_ve_evidence_dir", None)
            if evidence_dir:
                state.setdefault("evidence_dirs", []).append(evidence_dir)
            conts = out.get("continuations")
            ppl, em = out.get("perplexity"), out.get("task_em")
            if conts is None or ppl is None or em is None:
                return None
            if idx % 2 == 0:                # the round's baseline -> (re)establish it
                state["baseline"] = conts
                return {"perplexity": ppl, "task_em": em, "output_agreement": 1.0}
            return {"perplexity": ppl, "task_em": em,
                    "output_agreement": _agreement(conts, state.get("baseline"))}

        return measure

    quality_measure_fn.evidence_dirs = state.setdefault("evidence_dirs", [])
    return quality_measure_fn
