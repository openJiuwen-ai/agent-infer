"""infra-doctor / preflight — read-only remote health probes (no script upload).

Each probe is a ``(command, pure-classifier)`` pair: the command is a single
read-only remote shell line; the classifier turns its ``(stdout, stderr,
returncode)`` into a :class:`ProbeResult`. The classifiers are pure, so the
whole thing is unit-testable with *constructed* outputs — no GPU, no live SSH.
Real execution goes through an injectable ``run`` callable; the default runs
``ssh`` in BatchMode and never uploads or executes a remote script.

This exists because the session's most painful failures were operational, not
algorithmic: the box's SSH closing at the KEX stage (sshd/container down), a
missing conda env, a vLLM version whose scheduler API drifted, a card with no
free memory. ``ar doctor --infra`` and ``ar preflight`` surface those before a
run is wasted.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass

from vllm_evolve.bench.dispatch import DEFAULTS

# A runner executes one remote command and returns (stdout, stderr, returncode).
Runner = Callable[[str], "tuple[str, str, int]"]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: str   # ok | warn | fail | unknown
    detail: str


@dataclass
class InfraContext:
    host: str
    conda_sh: str = DEFAULTS["conda_sh"]
    conda_env: str = DEFAULTS["conda_env"]
    model: str = DEFAULTS["model"]
    gpus: tuple[str, ...] = ("0", "1", "2", "3")
    remote_repo: str = DEFAULTS["remote_repo"]
    port: int = 8260
    min_gpu_free_mb: float = 2000.0
    min_disk_gb: float = 10.0


# ---------------------------------------------------------------------------
# pure classifiers (the testable core)
# ---------------------------------------------------------------------------

def classify_ssh(stdout: str, stderr: str, returncode: int) -> ProbeResult:
    """Classify an ``ssh -vvv ... 'printf VE_OK'`` liveness probe.

    Distinguishes the failure modes that look identical at a glance: a server
    that closes before key exchange (KEX) vs auth rejection vs timeout vs
    refused vs host-key mismatch.
    """
    out, err = stdout or "", stderr or ""
    if "VE_OK" in out:
        return ProbeResult("ssh", "ok", "reachable and authenticated")
    if "Permission denied" in err:
        return ProbeResult("ssh", "fail", "auth_fail: key/credentials rejected")
    if "Host key verification failed" in err:
        return ProbeResult("ssh", "fail", "hostkey_fail: known_hosts mismatch")
    if (
        "kex_exchange_identification" in err
        or "banner exchange" in err
        or "Connection reset by peer" in err
    ):
        return ProbeResult(
            "ssh", "fail",
            "kex_close: server closed before key exchange "
            "(sshd/container down or restarting?)",
        )
    if "Connection refused" in err:
        return ProbeResult("ssh", "fail", "refused: nothing listening on the ssh port")
    if (
        "No route to host" in err
        or "Network is unreachable" in err
        or "Could not resolve hostname" in err
        or "Name or service not known" in err
    ):
        return ProbeResult("ssh", "fail", "unreachable: no route / DNS / network down")
    if "timed out" in err.lower():
        return ProbeResult("ssh", "fail", "timeout: no response (host down / firewall?)")
    last = err.strip().splitlines()[-1] if err.strip() else f"rc={returncode}"
    return ProbeResult("ssh", "unknown", last[:200])


def classify_conda(stdout: str, target_env: str) -> ProbeResult:
    try:
        envs = json.loads(stdout or "{}").get("envs", [])
    except json.JSONDecodeError:
        return ProbeResult("conda_env", "unknown", "could not parse `conda env list --json`")
    names = [e.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] for e in envs]
    ok = target_env in names
    return ProbeResult(
        "conda_env", "ok" if ok else "fail",
        f"{target_env} {'present' if ok else 'NOT FOUND'} (envs: {names})",
    )


def classify_vllm_import(stdout: str, stderr: str, returncode: int) -> ProbeResult:
    out = (stdout or "").strip()
    if returncode == 0 and out:
        return ProbeResult("vllm_import", "ok", f"vllm {out.splitlines()[-1]}")
    return ProbeResult(
        "vllm_import", "fail",
        f"import vllm failed: {(stderr or '').strip()[-200:] or f'rc={returncode}'}",
    )


def classify_scheduler_cls_flag(stdout: str, stderr: str, returncode: int) -> ProbeResult:
    if "--scheduler-cls" in (stdout or ""):
        return ProbeResult("scheduler_cls_flag", "ok", "vllm serve supports --scheduler-cls")
    return ProbeResult(
        "scheduler_cls_flag", "fail",
        "vllm serve --help lacks --scheduler-cls (version drift?)",
    )


def classify_gpu(stdout: str, gpus: tuple[str, ...], min_free_mb: float) -> ProbeResult:
    """Parse nvidia-smi ``index,name,mem.total,mem.used,mem.free`` (csv,nounits)."""
    free: dict[str, float] = {}
    for line in (stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                free[parts[0]] = float(parts[4])
            except ValueError:
                pass
    if not free:
        return ProbeResult("gpu", "fail", "nvidia-smi returned no GPUs (no driver / no card?)")
    missing = [g for g in gpus if g not in free]
    if missing:
        return ProbeResult(
            "gpu", "fail",
            f"requested cards {missing} absent (present: {sorted(free)})",
        )
    low = {g: free[g] for g in gpus if free[g] < min_free_mb}
    if low:
        return ProbeResult("gpu", "warn", f"low free memory (MB) on {low} (< {min_free_mb})")
    return ProbeResult(
        "gpu", "ok", f"cards {list(gpus)} free MB: {[free[g] for g in gpus]}"
    )


def classify_disk(stdout: str, min_gb: float) -> ProbeResult:
    """Parse ``df -Pk`` output; warn when any listed mount is below ``min_gb``."""
    low = []
    parsed = 0
    for line in (stdout or "").strip().splitlines()[1:]:  # skip header
        cols = line.split()
        if len(cols) >= 4:
            try:
                avail_gb = float(cols[3]) / (1024 * 1024)  # KB -> GB
            except ValueError:
                continue
            parsed += 1
            if avail_gb < min_gb:
                low.append((cols[-1], round(avail_gb, 1)))
    if parsed == 0:
        return ProbeResult("disk", "unknown", "no parseable df output")
    if low:
        return ProbeResult("disk", "warn", f"low free space (GB) on {low} (< {min_gb})")
    return ProbeResult("disk", "ok", "sufficient free disk")


def classify_model_cache(stdout: str) -> ProbeResult:
    out = (stdout or "").strip()
    if "LOCAL" in out or "CACHED" in out:
        return ProbeResult("model_cache", "ok", "model present locally / in HF cache")
    return ProbeResult(
        "model_cache", "warn", "model not cached (will download at run time)"
    )


def classify_port(stdout: str, stderr: str, returncode: int) -> ProbeResult:
    if "FREE" in (stdout or ""):
        return ProbeResult("port", "ok", "port is free to bind")
    return ProbeResult(
        "port", "warn",
        f"port appears busy / unbindable: {(stderr or '').strip()[-150:] or f'rc={returncode}'}",
    )


# ---------------------------------------------------------------------------
# commands + orchestration
# ---------------------------------------------------------------------------

def probe_commands(ctx: InfraContext) -> dict[str, str]:
    """Read-only remote command per probe.

    Untrusted config values are ``shlex.quote``-d and each bash payload is
    wrapped with ``shlex.quote`` so a value containing spaces / quotes / ``;`` /
    ``$()`` cannot break or inject into the remote command. (We never upload a
    script — the bash payload interprets the ``&&`` / ``${}`` itself.)
    """
    sh = shlex.quote(ctx.conda_sh)
    env = shlex.quote(ctx.conda_env)
    repo = shlex.quote(ctx.remote_repo)
    model = shlex.quote(ctx.model)
    dashes = shlex.quote("models--" + ctx.model.replace("/", "--"))
    activate = f"source {sh} && conda activate {env}"

    def _bash(payload: str) -> str:
        return "bash -lc " + shlex.quote(payload)

    return {
        "conda_env": _bash(f"source {sh} && conda env list --json"),
        "vllm_import": _bash(
            f"{activate} && python -c 'import vllm; print(vllm.__version__)'"
        ),
        "scheduler_cls_flag": _bash(f"{activate} && vllm serve --help"),
        "gpu": (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free "
            "--format=csv,noheader,nounits"
        ),
        "model_cache": _bash(
            f"test -e {model} && echo LOCAL || "
            f'(find "${{HF_HOME:-$HOME/.cache/huggingface}}/hub" -maxdepth 1 '
            f"-type d -name {dashes} 2>/dev/null | grep -q . && echo CACHED || echo MISS)"
        ),
        "port": _bash(
            "python3 -c 'import socket; s=socket.socket(); "
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            f's.bind(("127.0.0.1", {ctx.port})); s.close(); print("FREE")\''
        ),
        "disk": f"df -Pk {repo} /tmp",
    }


def _classify(name: str, ctx: InfraContext, out: str, err: str, rc: int) -> ProbeResult:
    if name == "conda_env":
        return classify_conda(out, ctx.conda_env)
    if name == "vllm_import":
        return classify_vllm_import(out, err, rc)
    if name == "scheduler_cls_flag":
        return classify_scheduler_cls_flag(out, err, rc)
    if name == "gpu":
        return classify_gpu(out, ctx.gpus, ctx.min_gpu_free_mb)
    if name == "model_cache":
        return classify_model_cache(out)
    if name == "port":
        return classify_port(out, err, rc)
    if name == "disk":
        return classify_disk(out, ctx.min_disk_gb)
    return ProbeResult(name, "unknown", "no classifier")


def gather_infra(
    ctx: InfraContext,
    *,
    liveness: tuple[str, str, int],
    run: Runner,
) -> dict:
    """Build a structured infra report from a liveness result + a probe runner.

    ``liveness`` is the (stdout, stderr, rc) of the verbose SSH probe; ``run``
    executes each subsequent remote command. Both are injected so this is fully
    testable with constructed outputs. If SSH is not OK, no further probes run.
    """
    ssh = classify_ssh(*liveness)
    probes = [ssh]
    reachable = ssh.status == "ok"
    if reachable:
        for name, cmd in probe_commands(ctx).items():
            out, err, rc = run(cmd)
            if rc in (124, 255):
                # ssh/transport failure mid-probe — NOT a business result.
                probes.append(ProbeResult(
                    name, "unknown",
                    f"probe transport failure (rc={rc}): {(err or '').strip()[-120:]}",
                ))
            else:
                probes.append(_classify(name, ctx, out, err, rc))
    # Not reachable == can't run anything == fail (never report an unreachable
    # host as ok/ready just because the liveness class was 'unknown').
    if not reachable or any(p.status == "fail" for p in probes):
        overall = "fail"
    elif any(p.status in ("warn", "unknown") for p in probes):
        overall = "warn"
    else:
        overall = "ok"
    return {
        "host": ctx.host,
        "reachable": reachable,
        "overall": overall,
        "blockers": [p.name for p in probes if p.status == "fail"],
        "probes": [asdict(p) for p in probes],
    }


# ---------------------------------------------------------------------------
# default real SSH runners (only used live; never in tests)
# ---------------------------------------------------------------------------

def ssh_liveness(host: str, timeout_s: int = 8) -> tuple[str, str, int]:
    try:
        r = subprocess.run(
            ["ssh", "-vvv", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout_s}",
             "-o", "ConnectionAttempts=1", host, "printf VE_OK"],
            capture_output=True, text=True, timeout=timeout_s + 8,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "ssh liveness timed out", 124


def make_ssh_runner(host: str, timeout_s: int = 40) -> Runner:
    def run(cmd: str) -> tuple[str, str, int]:
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, cmd],
                capture_output=True, text=True, timeout=timeout_s,
            )
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", "probe timed out", 124
    return run


def run_infra_report(ctx: InfraContext) -> dict:
    """Live infra report: real SSH liveness + probe runner (used by the CLI)."""
    return gather_infra(
        ctx, liveness=ssh_liveness(ctx.host), run=make_ssh_runner(ctx.host)
    )
