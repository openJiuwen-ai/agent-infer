"""port mode — adapt / regression-check one policy across vLLM versions x hardware.

Minimal viable: iterate the requested ``versions x hardware`` matrix, evaluate each
cell via the backend profiler, and emit a ``port_report.json``. It is a REPORT, not
an adoption path — it never keeps/adopts, and each cell carries its eval ``source``
(``local_smoke`` cells are non-promotable by construction; a real per-cell regression
verdict + adaptation diff are box-gated future work). Off-box runs use the
``local_smoke`` backend (plumbing only); real cells are ``@requires_gpu``.

Untrusted L2 layer — MUST NOT appear in the frozen closure (forbidden by
``tests/test_frozen_core.py``). See DESIGN.md for the mode boundaries.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence


def run(
    policy_path: str,
    *,
    versions: Sequence[str],
    hardware: Sequence[str],
    eval_fn: Callable,
    target: str = "scheduling",
    base_config: dict | None = None,
    backend: str | None = None,
) -> dict:
    """Evaluate ``policy_path`` over the version x hardware matrix; return a port report dict."""
    versions = list(versions) or ["unspecified"]
    hardware = list(hardware) or ["unspecified"]
    base = dict(base_config or {})
    base["policy"] = str(policy_path)

    matrix: list[dict] = []
    for v in versions:
        for h in hardware:
            # deep per-cell regression verdict + adaptation diff are box-gated future work
            cell = {"vllm_version": v, "hardware_id": h, "backend": backend or "remote",
                    "regression_verdict": "not_evaluated", "adapt_diff_path": None}
            try:
                prof = eval_fn(dict(base))
                cell["status"] = "ok"
                cell["source"] = getattr(prof, "source", "") or ""
                cell["outcome_class"] = getattr(prof, "outcome_class", "") or ""
            except Exception as exc:  # noqa: BLE001 - record the cell failure, don't abort the matrix
                cell["status"] = "error"
                cell["source"] = ""
                cell["outcome_class"] = "cell_error"
                cell["summary"] = f"{type(exc).__name__}: {exc}"
            cell.setdefault("summary", f"{v}/{h}: {cell['outcome_class'] or cell['status']}")
            matrix.append(cell)

    run_id = hashlib.sha256(json.dumps(
        {"policy": str(policy_path), "target": target,
         "versions": versions, "hardware": hardware}, sort_keys=True,
    ).encode("utf-8")).hexdigest()[:12]
    return {
        "mode": "port",
        "target": target,
        "input_policy": str(policy_path),
        "run_id": run_id,
        "matrix": matrix,
        "summary": {
            "cells": len(matrix),
            "real_cells": sum(1 for c in matrix if c.get("source") == "real_vllm"),
            "local_smoke_cells": sum(1 for c in matrix if c.get("source") == "local_smoke"),
            "errors": sum(1 for c in matrix if c.get("status") == "error"),
        },
    }
