"""Deterministic remote GPU discovery for real-vLLM runs.

``gpus=auto`` is resolved exactly once, before the first benchmark in a round.
The resulting concrete device list and the discovery evidence are written back
into :class:`EnvironmentConfig`; every candidate cloned from that config then
uses the same physical GPUs.
"""
from __future__ import annotations

import csv
import io
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

from vllm_evolve.bench.gpu_guard import (
    MAX_VISIBLE_GPUS,
    validate_gpu_budget,
)

AUTO_GPUS = "auto"
DEFAULT_AUTO_BUSY_MEMORY_MIB = 1024


def is_auto_gpu_selection(value: str) -> bool:
    return str(value).strip().lower() == AUTO_GPUS


def required_gpu_count(config) -> int:
    """Infer physical-card count from the frozen execution topology."""
    parallel_mode = str(
        (config.workload.workload_spec or {}).get("parallel_mode") or ""
    )
    tp = int(config.engine.tensor_parallel_size or 1)
    if tp < 1 or tp > MAX_VISIBLE_GPUS:
        raise ValueError(
            f"tensor_parallel_size={tp} is outside the 1..{MAX_VISIBLE_GPUS} GPU budget"
        )
    if parallel_mode == "dual_replica":
        if tp != 1:
            raise ValueError("dual_replica auto selection requires tensor_parallel_size=1")
        return 2
    return tp


def _ssh_nvidia_query(remote: str, query_flag: str, *, timeout_s: float) -> str:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "nvidia-smi",
            query_flag,
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-800:]
        raise RuntimeError(
            f"remote GPU discovery failed on {remote!r}: {detail}"
        )
    return result.stdout


def _ssh_gpu_service_query(remote: str, *, timeout_s: float) -> list[dict]:
    """Find live services that may allocate GPUs after ``nvidia-smi`` is sampled.

    A vLLM parent can exist for more than a minute before its workers create a
    CUDA context.  Treating that initialization gap as an idle GPU caused a
    cross-user race on a shared host, so production callers may require a
    server-wide quiescent window before freezing devices.
    """
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "ps",
            "-eo",
            "user=,pid=,args=",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-800:]
        raise RuntimeError(
            f"remote GPU service discovery failed on {remote!r}: {detail}"
        )
    services = []
    for raw_line in result.stdout.splitlines():
        fields = raw_line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        user, raw_pid, command = fields
        lowered = command.lower()
        if not (
            "vllm serve" in lowered
            or "vllm.entrypoints.openai.api_server" in lowered
            or "vllm_evolve.bench.remote_worker" in lowered
            or command.startswith("VLLM::")
        ):
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        services.append({
            "user": user,
            "pid": pid,
            "command": command[:1000],
        })
    return services


def _csv_rows(raw: str) -> list[list[str]]:
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(raw))
        if row and any(field.strip() for field in row)
    ]


def query_remote_gpu_inventory(
    remote: str,
    *,
    timeout_s: float = 30.0,
) -> dict:
    """Read physical GPU and compute-process state from the selected SSH host."""
    gpu_raw = _ssh_nvidia_query(
        remote,
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        timeout_s=timeout_s,
    )
    app_raw = _ssh_nvidia_query(
        remote,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        timeout_s=timeout_s,
    )
    gpu_service_processes = _ssh_gpu_service_query(
        remote,
        timeout_s=timeout_s,
    )
    gpus = []
    for row in _csv_rows(gpu_raw):
        if len(row) != 5:
            continue
        index, uuid, name, total, used = row
        try:
            gpus.append({
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            })
        except ValueError:
            continue
    apps = []
    for row in _csv_rows(app_raw):
        if len(row) != 4:
            continue
        uuid, pid, process_name, used = row
        try:
            apps.append({
                "gpu_uuid": uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used),
            })
        except ValueError:
            continue
    if not gpus:
        raise RuntimeError(
            f"remote GPU discovery returned no parseable GPUs on {remote!r}"
        )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "compute_apps": apps,
        "gpu_service_processes": gpu_service_processes,
    }


def select_same_caliber_gpus(
    inventory: dict,
    required_count: int,
    *,
    busy_memory_mib: int = DEFAULT_AUTO_BUSY_MEMORY_MIB,
) -> dict:
    """Select idle, same-model, same-capacity devices deterministically."""
    if required_count < 1 or required_count > MAX_VISIBLE_GPUS:
        raise ValueError(
            f"required GPU count {required_count} exceeds the {MAX_VISIBLE_GPUS}-GPU budget"
        )
    apps_by_uuid: dict[str, list[dict]] = defaultdict(list)
    for app in inventory.get("compute_apps") or []:
        apps_by_uuid[str(app.get("gpu_uuid") or "")].append(dict(app))

    examined = []
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for raw_gpu in inventory.get("gpus") or []:
        gpu = dict(raw_gpu)
        reasons = []
        used = int(gpu.get("memory_used_mib") or 0)
        if used > busy_memory_mib:
            reasons.append(
                f"memory_used_mib={used} exceeds {busy_memory_mib}"
            )
        apps = apps_by_uuid.get(str(gpu.get("uuid") or ""), [])
        if apps:
            reasons.append(
                "compute_apps=" + ",".join(str(app.get("pid")) for app in apps)
            )
        row = {
            "index": str(gpu.get("index")),
            "uuid": str(gpu.get("uuid") or ""),
            "name": str(gpu.get("name") or ""),
            "memory_total_mib": int(gpu.get("memory_total_mib") or 0),
            "memory_used_mib": used,
            "eligible": not reasons,
            "excluded_reasons": reasons,
        }
        examined.append(row)
        if not reasons:
            groups[(row["name"], row["memory_total_mib"])].append(row)

    eligible_groups = [
        (caliber, rows)
        for caliber, rows in groups.items()
        if len(rows) >= required_count
    ]
    if not eligible_groups:
        summary = "; ".join(
            f"GPU {row['index']}: "
            + (", ".join(row["excluded_reasons"]) or "caliber has too few idle peers")
            for row in examined
        )
        raise RuntimeError(
            f"no {required_count}-GPU idle same-caliber set is available ({summary})"
        )

    def index_key(row: dict) -> tuple[int, str]:
        index = str(row["index"])
        return (int(index), index) if index.isdigit() else (10**9, index)

    # Prefer the largest-memory viable caliber, then the lowest stable device IDs.
    eligible_groups.sort(
        key=lambda item: (
            -item[0][1],
            item[0][0],
            tuple(index_key(row) for row in sorted(item[1], key=index_key)),
        )
    )
    caliber, rows = eligible_groups[0]
    selected = sorted(rows, key=index_key)[:required_count]
    return {
        "selected_gpus": [row["index"] for row in selected],
        "selected_uuids": [row["uuid"] for row in selected],
        "caliber": {
            "name": caliber[0],
            "memory_total_mib": caliber[1],
        },
        "examined_gpus": examined,
    }


def bind_auto_gpu_selection(
    config,
    *,
    probe=None,
    wait_timeout_s: float = 0.0,
    poll_interval_s: float = 30.0,
    stable_poll_count: int = 1,
    require_server_quiescence: bool = False,
    sleep=time.sleep,
) -> dict | None:
    """Resolve ``auto`` in-place so the caller can freeze one pair for a round.

    Concrete selections are validated but never silently changed.  A config that
    was already auto-bound carries its original evidence through candidate clones.
    """
    requested = str(config.environment.gpus).strip()
    required_count = required_gpu_count(config)
    if not is_auto_gpu_selection(requested):
        devices = validate_gpu_budget(requested, config.engine.tensor_parallel_size)
        parallel_mode = str(
            (config.workload.workload_spec or {}).get("parallel_mode") or ""
        )
        if parallel_mode == "dual_replica" and len(devices) != 2:
            raise ValueError("dual_replica mode requires exactly two distinct GPUs")
        return None

    wait_timeout_s = max(0.0, float(wait_timeout_s))
    poll_interval_s = max(0.1, float(poll_interval_s))
    stable_poll_count = max(1, int(stable_poll_count))
    deadline = time.monotonic() + wait_timeout_s
    attempts = 0
    stable_observations = 0
    stable_selection = None
    while True:
        attempts += 1
        inventory = (probe or query_remote_gpu_inventory)(config.runner.remote)
        if require_server_quiescence and inventory.get("gpu_service_processes"):
            stable_observations = 0
            stable_selection = None
            remaining = deadline - time.monotonic()
            if wait_timeout_s <= 0.0 or remaining <= 0.0:
                raise RuntimeError(
                    "remote GPU services are still active; server quiescence "
                    "was required before automatic GPU binding"
                )
            sleep(min(poll_interval_s, remaining))
            continue
        try:
            chosen = select_same_caliber_gpus(inventory, required_count)
            selection = tuple(chosen["selected_uuids"])
            if selection == stable_selection:
                stable_observations += 1
            else:
                stable_selection = selection
                stable_observations = 1
            if stable_observations >= stable_poll_count:
                break
        except RuntimeError:
            stable_observations = 0
            stable_selection = None
            if wait_timeout_s <= 0.0:
                raise
        remaining = deadline - time.monotonic()
        if wait_timeout_s <= 0.0 or remaining <= 0.0:
            raise RuntimeError(
                f"GPU selection did not remain stable for "
                f"{stable_poll_count} consecutive probes"
            )
        sleep(min(poll_interval_s, remaining))
    selected = tuple(chosen["selected_gpus"])
    validate_gpu_budget(
        ",".join(selected), config.engine.tensor_parallel_size
    )
    evidence = {
        "schema_version": 1,
        "mode": AUTO_GPUS,
        "requested": AUTO_GPUS,
        "resolved": ",".join(selected),
        "required_count": required_count,
        "remote": config.runner.remote,
        "captured_at": inventory.get("captured_at"),
        "selection_attempts": attempts,
        "stable_poll_count": stable_poll_count,
        "server_quiescence_required": bool(require_server_quiescence),
        "busy_memory_limit_mib": DEFAULT_AUTO_BUSY_MEMORY_MIB,
        **chosen,
    }
    config.environment.gpus = evidence["resolved"]
    config.environment.gpu_selection_mode = AUTO_GPUS
    config.environment.gpu_selection_evidence = evidence
    return evidence
