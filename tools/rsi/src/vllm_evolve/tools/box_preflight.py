"""Box preflight for the real AC4->AC6 validation runbook (``docs/AC6_RUNBOOK.md``).

The AC4 / AC5 / AC6 work is GPU-gated — it must run on the real vLLM box. This preflight checks the
box is reachable and reports a GPU; when it is NOT, it prints an honest ``box_unreachable`` result
and points at the runbook. It NEVER fabricates a result. Run this before the runbook.
"""
from __future__ import annotations

import json
import subprocess
import sys

DEFAULT_REMOTE = "gpu-host"


def box_reachable(remote: str = DEFAULT_REMOTE, *, timeout_s: int = 20) -> tuple[bool, str]:
    """True iff the box answers SSH AND reports at least one GPU. Returns (ok, detail).

    Never raises — an unreachable box / closed connection / ssh error all return (False, reason).
    """
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(timeout_s, 30)}",
             remote, "nvidia-smi -L"],
            capture_output=True, text=True, timeout=timeout_s + 5)
    except Exception as exc:  # noqa: BLE001 - honest unreachable
        return False, f"ssh failed: {str(exc)[-160:]}"
    if r.returncode != 0:
        return False, f"ssh exit {r.returncode}: {(r.stderr or r.stdout).strip()[-160:]}"
    gpus = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return (bool(gpus), f"{len(gpus)} GPU(s)" if gpus else "ssh ok but no GPU reported")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m vllm_evolve.tools.box_preflight",
        description="Check the GPU box before the real AC4->AC6 validation (no fabrication).")
    ap.add_argument("--remote", default=DEFAULT_REMOTE, help="ssh host/alias of the GPU box.")
    a = ap.parse_args(argv)
    ok, detail = box_reachable(a.remote)
    if ok:
        print(json.dumps({
            "ok": True, "box": a.remote, "detail": detail,
            "next": "run docs/AC6_RUNBOOK.md — the real AC4->AC6 validation sequence"}))
        return 0
    print(json.dumps({
        "ok": False, "outcome_class": "box_unreachable", "box": a.remote, "detail": detail,
        "hint": "AC4/AC5/AC6 require a configured GPU host; check your SSH configuration, then "
                "re-run. NEVER fabricate a result — see docs/AC6_RUNBOOK.md."}))
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
