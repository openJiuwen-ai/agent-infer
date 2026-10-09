"""Box-side guard for one real-vLLM benchmark run.

The local driver invokes this module over SSH.  It owns a single immutable run
directory, refuses busy GPUs, records pre/post ``nvidia-smi`` evidence, bounds
runtime, and writes ``status.json`` on every exit path.  The actual measurement
still happens in :mod:`vllm_evolve.bench.native`; this module is the resource
and failure-isolation boundary around it.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from vllm_evolve.bench.gpu_guard import (
    validate_gpu_budget,
    vllm_serve_port_pattern,
)

BUSY_EXIT = 75
CONTAMINATED_EXIT = 76
DEFAULT_BUSY_MEMORY_MIB = 1024


def _run_text(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout


def gpu_snapshot() -> dict:
    """Return raw and structured NVIDIA state without importing CUDA libraries."""
    query = (
        "index,uuid,name,driver_version,memory.total,memory.used,"
        "utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,"
        "temperature.gpu,pstate,clocks_event_reasons.sw_power_cap,"
        "clocks_event_reasons.sw_thermal_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.hw_power_brake_slowdown,compute_mode"
    )
    rc, raw = _run_text([
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
    ])
    app_rc, app_raw = _run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if rc == 0:
        for line in raw.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 18:
                continue
            (
                index,
                uuid,
                name,
                driver,
                total,
                used,
                util,
                memory_util,
                power,
                sm_clock,
                memory_clock,
                temperature,
                pstate,
                sw_power_cap,
                sw_thermal,
                hw_thermal,
                hw_power_brake,
                mode,
            ) = fields
            try:
                total_mib = int(total)
                used_mib = int(used)
                util_pct = int(util)
                memory_util_pct = int(memory_util)
                power_w = float(power)
                sm_clock_mhz = int(sm_clock)
                memory_clock_mhz = int(memory_clock)
                temperature_c = int(temperature)
            except ValueError:
                continue
            gpus.append({
                "index": index,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "utilization_gpu_pct": util_pct,
                "utilization_memory_pct": memory_util_pct,
                "power_draw_w": power_w,
                "clocks_sm_mhz": sm_clock_mhz,
                "clocks_memory_mhz": memory_clock_mhz,
                "temperature_gpu_c": temperature_c,
                "pstate": pstate,
                "power_throttle_active": sw_power_cap.lower() == "active",
                "thermal_throttle_active": any(
                    value.lower() == "active"
                    for value in (sw_thermal, hw_thermal, hw_power_brake)
                ),
                "clock_event_reasons": {
                    "software_power_cap": sw_power_cap,
                    "software_thermal": sw_thermal,
                    "hardware_thermal": hw_thermal,
                    "hardware_power_brake": hw_power_brake,
                },
                "compute_mode": mode,
            })
    apps = []
    if app_rc == 0:
        for line in app_raw.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                continue
            uuid, pid, process_name, used = fields
            with contextlib.suppress(ValueError):
                apps.append({
                    "gpu_uuid": uuid,
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(used),
                })
    return {
        "captured_at_unix_s": time.time(),
        "nvidia_smi_exit": rc,
        "compute_apps_exit": app_rc,
        "gpus": gpus,
        "compute_apps": apps,
        "raw_gpu_csv": raw,
        "raw_compute_apps_csv": app_raw,
    }


def selected_gpu_state(snapshot: dict, devices: tuple[str, ...]) -> list[dict]:
    """Resolve CUDA indices/UUIDs against a snapshot."""
    by_index = {gpu["index"]: gpu for gpu in snapshot.get("gpus", [])}
    by_uuid = {gpu["uuid"]: gpu for gpu in snapshot.get("gpus", [])}
    resolved = []
    for device in devices:
        gpu = by_index.get(device) or by_uuid.get(device)
        if gpu is None:
            raise RuntimeError(f"selected GPU {device!r} is absent from nvidia-smi")
        resolved.append(gpu)
    return resolved


def busy_reasons(
    snapshot: dict,
    devices: tuple[str, ...],
    *,
    memory_limit_mib: int = DEFAULT_BUSY_MEMORY_MIB,
) -> list[str]:
    selected = selected_gpu_state(snapshot, devices)
    uuids = {gpu["uuid"] for gpu in selected}
    reasons = []
    for gpu in selected:
        if gpu["memory_used_mib"] > memory_limit_mib:
            reasons.append(
                f"GPU {gpu['index']} uses {gpu['memory_used_mib']} MiB "
                f"(limit {memory_limit_mib} MiB)"
            )
    for app in snapshot.get("compute_apps", []):
        if app.get("gpu_uuid") in uuids:
            reasons.append(
                f"GPU process pid={app['pid']} name={app['process_name']} "
                f"memory={app['used_memory_mib']} MiB"
            )
    return reasons


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _acquire_gpu_locks(lock_root: Path, devices: tuple[str, ...]) -> list[object]:
    lock_root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for device in sorted(devices):
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in device)
            handle = (lock_root / f"gpu_{safe}.lock").open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        return handles
    except (BlockingIOError, OSError):
        for handle in handles:
            handle.close()
        raise RuntimeError("one or more selected GPUs are locked by another vllm-evolve run")


def _cleanup_vllm_port(port: int) -> None:
    """Kill only this user's vLLM process carrying the run-unique port."""
    uid = str(os.getuid())
    pattern = vllm_serve_port_pattern(port)
    subprocess.run(
        ["pkill", "-9", "-u", uid, "-f", pattern],
        capture_output=True,
        text=True,
        check=False,
    )


def _sample_gpus(
    stop: threading.Event,
    path: Path,
    devices: tuple[str, ...],
    root_pid: int,
    foreign_seen: dict[tuple[str, int], dict],
    *,
    interval_s: float = 1.0,
) -> None:
    # vLLM multiprocessing workers may be re-parented during orderly server
    # shutdown before NVML drops their compute-app row. Once a PID has been
    # positively tied to this run's process tree, retain that ownership for the
    # rest of the immutable run; otherwise the final sampler tick falsely calls
    # our own EngineCore an external contaminant.
    owned_pids: set[int] = set()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        while True:
            snapshot = gpu_snapshot()
            selected = selected_gpu_state(snapshot, devices)
            selected_uuids = {gpu["uuid"] for gpu in selected}
            apps = []
            for raw_app in snapshot.get("compute_apps", []):
                if raw_app.get("gpu_uuid") not in selected_uuids:
                    continue
                app = dict(raw_app)
                app["owned_by_run"] = _owned_by_run(
                    int(app["pid"]),
                    root_pid,
                    owned_pids,
                )
                apps.append(app)
                if not app["owned_by_run"]:
                    foreign_seen[(str(app["gpu_uuid"]), int(app["pid"]))] = app
            row = {
                "captured_at_unix_s": snapshot["captured_at_unix_s"],
                "gpus": selected,
                "compute_apps": apps,
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            if stop.wait(interval_s):
                break


def _owned_by_run(pid: int, root_pid: int, owned_pids: set[int]) -> bool:
    """Sticky positive process ownership across vLLM teardown re-parenting."""
    pid = int(pid)
    if pid in owned_pids:
        return True
    if _is_descendant_process(pid, root_pid):
        owned_pids.add(pid)
        return True
    return False


def _parent_pid(pid: int) -> int | None:
    """Read one Linux process parent without shelling out.

    ``/proc/<pid>/stat`` may contain spaces inside the parenthesized command
    name, so fields are parsed only after the final ``)``.
    """
    try:
        rest = Path(f"/proc/{int(pid)}/stat").read_text(
            encoding="utf-8"
        ).rsplit(")", 1)[1].strip().split()
        return int(rest[1])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _is_descendant_process(pid: int, root_pid: int) -> bool:
    """Whether ``pid`` belongs to the benchmark process tree.

    vLLM starts its server in a new session, so process-group equality is not
    sufficient. Parent ancestry remains the reliable ownership signal while
    the guarded native command is alive.
    """
    current = int(pid)
    root = int(root_pid)
    seen = set()
    for _ in range(128):
        if current == root:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        parent = _parent_pid(current)
        if parent is None:
            return False
        current = parent
    return False


def _gpu_summary(path: Path, *, windows_path: Path | None = None) -> dict:
    from vllm_evolve.bench.workload_validity import summarize_gpu_samples

    per_gpu: dict[str, dict] = {}
    foreign_apps: dict[tuple[str, int], dict] = {}
    samples = 0
    sample_rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_rows.append(row)
            samples += 1
            for gpu in row.get("gpus", []):
                item = per_gpu.setdefault(gpu["index"], {
                    "uuid": gpu["uuid"],
                    "name": gpu["name"],
                    "max_memory_used_mib": 0,
                    "max_utilization_gpu_pct": 0,
                })
                item["max_memory_used_mib"] = max(
                    item["max_memory_used_mib"], gpu["memory_used_mib"])
                item["max_utilization_gpu_pct"] = max(
                    item["max_utilization_gpu_pct"], gpu["utilization_gpu_pct"])
            for app in row.get("compute_apps", []):
                if app.get("owned_by_run") is False:
                    key = (str(app.get("gpu_uuid")), int(app.get("pid")))
                    foreign_apps[key] = app
    windows = []
    if windows_path is not None and windows_path.exists():
        with contextlib.suppress(Exception):
            payload = json.loads(windows_path.read_text(encoding="utf-8"))
            windows = list(payload.get("windows") or [])
    measured_rows = [
        row
        for row in sample_rows
        if any(
            float(window["started_at_unix_s"])
            <= float(row.get("captured_at_unix_s") or 0.0)
            <= float(window["ended_at_unix_s"])
            for window in windows
        )
    ]
    return {
        "sample_count": samples,
        "per_gpu": per_gpu,
        "distribution_all_process_phases": summarize_gpu_samples(sample_rows),
        "measurement_windows": windows,
        "measured_sample_count": len(measured_rows),
        "distribution_measured_window": summarize_gpu_samples(measured_rows),
        "contaminated": bool(foreign_apps),
        "foreign_compute_apps": list(foreign_apps.values()),
    }


def run_guarded(
    command: list[str],
    *,
    run_dir: Path,
    workspace: Path,
    gpus: str,
    tensor_parallel_size: int,
    port: int,
    timeout_s: float,
    candidate_id: str,
    git_sha: str,
    config_sha256: str,
    policy_sha256: str = "",
    workload_sha256: str = "",
    staged_policy: Path | None = None,
    staged_config: Path | None = None,
    staged_workload: Path | None = None,
    busy_memory_mib: int = DEFAULT_BUSY_MEMORY_MIB,
) -> dict:
    devices = validate_gpu_budget(gpus, tensor_parallel_size)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {
        "schema_version": 1,
        "ok": False,
        "state": "initializing",
        "candidate_id": candidate_id,
        "policy_sha256": policy_sha256 or None,
        "workload_sha256": workload_sha256 or None,
        "git_sha": git_sha,
        "config_sha256": config_sha256,
        "run_dir": str(run_dir),
        "gpus": list(devices),
        "tensor_parallel_size": tensor_parallel_size,
        "port": port,
        "command": command,
        "started_at_unix_s": started,
        "ended_at_unix_s": None,
        "duration_s": None,
        "exit_code": None,
        "reason": None,
    }
    before_path = run_dir / "gpu_before.json"
    after_path = run_dir / "gpu_after.json"
    status_path = run_dir / "status.json"
    stdout_path = run_dir / "native.stdout.log"
    stderr_path = run_dir / "native.stderr.log"
    samples_path = run_dir / "gpu_samples.jsonl"
    summary_path = run_dir / "gpu_summary.json"
    locks: list[object] = []
    proc: subprocess.Popen | None = None
    sampler_stop = threading.Event()
    sampler: threading.Thread | None = None
    foreign_seen: dict[tuple[str, int], dict] = {}
    previous_signal_handlers: dict[int, object] = {}

    def _interrupt(signum, _frame) -> None:
        raise InterruptedError(f"remote worker received signal {signum}")

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _interrupt)

    try:
        if staged_policy is not None:
            source = staged_policy.read_bytes()
            actual = hashlib.sha256(source).hexdigest()
            if policy_sha256 and actual != policy_sha256:
                raise RuntimeError(
                    f"staged policy SHA mismatch: expected {policy_sha256}, got {actual}"
                )
            (run_dir / "policy.py").write_bytes(source)
        if staged_config is not None:
            source = staged_config.read_bytes()
            actual = hashlib.sha256(source).hexdigest()
            if actual != config_sha256:
                raise RuntimeError(
                    f"staged config SHA mismatch: expected {config_sha256}, got {actual}"
                )
            (run_dir / "bench_config.json").write_bytes(source)
        if staged_workload is not None:
            source = staged_workload.read_bytes()
            actual = hashlib.sha256(source).hexdigest()
            if actual != workload_sha256:
                raise RuntimeError(
                    f"staged workload SHA mismatch: expected {workload_sha256}, got {actual}"
                )
            (run_dir / "workload.json").write_bytes(source)
        _atomic_json(run_dir / "manifest.json", {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "policy_sha256": policy_sha256 or None,
            "workload_sha256": workload_sha256 or None,
            "git_sha": git_sha,
            "config_sha256": config_sha256,
            "command": command,
            "gpus": list(devices),
            "tensor_parallel_size": tensor_parallel_size,
            "port": port,
        })
        locks = _acquire_gpu_locks(workspace / "locks", devices)
        before = gpu_snapshot()
        _atomic_json(before_path, before)
        reasons = busy_reasons(before, devices, memory_limit_mib=busy_memory_mib)
        if reasons:
            status.update(state="resource_busy", exit_code=BUSY_EXIT, reason="; ".join(reasons))
            return status

        status["state"] = "running"
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
        ):
            proc = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                cwd=str(Path.cwd()),
                start_new_session=True,
            )
            sampler = threading.Thread(
                target=_sample_gpus,
                args=(
                    sampler_stop,
                    samples_path,
                    devices,
                    proc.pid,
                    foreign_seen,
                ),
                daemon=True,
            )
            sampler.start()
            try:
                exit_code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=30)
                _cleanup_vllm_port(port)
                status.update(
                    state="timeout",
                    exit_code=124,
                    reason=f"remote command exceeded {timeout_s:.1f}s",
                )
                return status
        if exit_code != 0:
            _cleanup_vllm_port(port)
            status.update(
                state="failed",
                exit_code=exit_code,
                reason=f"native benchmark exited {exit_code}",
            )
            return status
        status.update(state="succeeded", ok=True, exit_code=0)
        return status
    except Exception as exc:  # noqa: BLE001 - status is the failure contract
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        _cleanup_vllm_port(port)
        status.update(state="failed", exit_code=1, reason=f"{type(exc).__name__}: {exc}")
        return status
    finally:
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=10)
        with contextlib.suppress(Exception):
            summary = _gpu_summary(
                samples_path,
                windows_path=run_dir / "measurement_windows.json",
            )
            _atomic_json(summary_path, summary)
            status["gpu_summary"] = summary
            if status["state"] == "succeeded" and foreign_seen:
                offenders = "; ".join(
                    f"pid={app['pid']} name={app['process_name']} "
                    f"memory={app['used_memory_mib']} MiB"
                    for app in foreign_seen.values()
                )
                status.update(
                    state="resource_contaminated",
                    ok=False,
                    exit_code=CONTAMINATED_EXIT,
                    reason=(
                        "foreign GPU process appeared during measurement: "
                        f"{offenders}"
                    ),
                )
        with contextlib.suppress(Exception):
            _atomic_json(after_path, gpu_snapshot())
        ended = time.time()
        status["ended_at_unix_s"] = ended
        status["duration_s"] = ended - started
        _atomic_json(status_path, status)
        for handle in locks:
            with contextlib.suppress(Exception):
                handle.close()
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vllm_evolve.bench.remote_worker")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--policy-sha256", default="")
    parser.add_argument("--workload-sha256", default="")
    parser.add_argument("--staged-policy", default=None)
    parser.add_argument("--staged-config", default=None)
    parser.add_argument("--staged-workload", default=None)
    parser.add_argument("--busy-memory-mib", type=int, default=DEFAULT_BUSY_MEMORY_MIB)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    status = run_guarded(
        command,
        run_dir=Path(args.run_dir),
        workspace=Path(args.workspace),
        gpus=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        port=args.port,
        timeout_s=args.timeout_s,
        candidate_id=args.candidate_id,
        git_sha=args.git_sha,
        config_sha256=args.config_sha256,
        policy_sha256=args.policy_sha256,
        workload_sha256=args.workload_sha256,
        staged_policy=Path(args.staged_policy) if args.staged_policy else None,
        staged_config=Path(args.staged_config) if args.staged_config else None,
        staged_workload=Path(args.staged_workload) if args.staged_workload else None,
        busy_memory_mib=args.busy_memory_mib,
    )
    print(json.dumps(status, sort_keys=True))
    return int(status["exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
