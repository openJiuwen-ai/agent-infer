# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Run BenchKit benchmarks, prepare datasets, and validate completed runs."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from agentinfer.agentbench.benchkit.compare import load_summary

from .case_loader import (
    E2EPerfConfig,
    build_benchmark_argv,
    build_benchmark_env,
    build_prepare_argv,
    ensure_directory,
    grant_benchmark_user_dataset_access,
    prepare_runs_as_root,
    resolve_benchmark_run_as_user,
)
from .server import managed_vllm


def run_benchmark(
    config: E2EPerfConfig,
    *,
    result_dir: Path | None = None,
) -> Path:
    """Execute one benchmark case and return the finalized run directory."""

    blockers = config.validate_prerequisites()
    if blockers:
        raise RuntimeError("; ".join(blockers))

    ensure_benchmark_inputs_prepared(config)
    hardware_slug = config.resolve_run_hardware_slug()
    print(f"[benchmark:{config.scenario.value}] hardware slug: {hardware_slug}", flush=True)
    output_dir = result_dir or config.make_result_dir(hardware_slug=hardware_slug)
    if output_dir.exists():
        raise FileExistsError(f"result directory already exists: {output_dir}")

    argv = build_benchmark_argv(config, output_dir)
    with managed_vllm(config):
        _invoke_benchmark(config, argv)

    return output_dir


def _invoke_benchmark(config: E2EPerfConfig, argv: list[str]) -> None:
    prefix = f"[benchmark:{config.scenario.value}] "
    if len(argv) >= 2 and argv[0] == "sudo":
        print(f"{prefix}running bench via sudo -u {argv[2]}", flush=True)
    else:
        print(f"{prefix}running: {' '.join(argv)}", flush=True)
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        capture_output=True,
        env=build_benchmark_env(),
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, flush=True)
        if completed.stderr:
            print(completed.stderr, flush=True)
        raise RuntimeError(f"benchmark exited with code {completed.returncode}")


def dataset_marker_paths(output_dir: Path) -> tuple[Path, ...]:
    """Return local files that must exist after a BenchKit prepare command.

    Only ``task-lists/default.txt`` is required here; other datasets may omit
    ``instances.jsonl`` and ``manifest.json`` while still sharing this layout.
    """

    return (output_dir / "task-lists" / "default.txt",)


def dataset_is_ready(output_dir: Path) -> bool:
    """Return whether local dataset inputs are present."""

    return all(path.exists() for path in dataset_marker_paths(output_dir))


def ensure_benchmark_inputs_prepared(config: E2EPerfConfig) -> Path:
    """Prepare dataset metadata and repo-cache inputs before the benchmark run."""

    output_dir = ensure_dataset_prepared(config)
    repo_cache_dir = ensure_repo_cache_prepared(config)
    grant_benchmark_user_dataset_access(config, repo_cache_dir=repo_cache_dir)
    return output_dir


def ensure_dataset_prepared(config: E2EPerfConfig) -> Path:
    """Prepare dataset inputs when missing, using the case-configured prepare dataset."""

    output_dir = config.dataset_output_dir.resolve()
    if dataset_is_ready(output_dir):
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"dataset directory exists but is incomplete: {output_dir}; "
            "remove it or run prepare manually:\n"
            f"  vllm bench serve --agentinfer prepare {config.prepare_dataset} "
            f"--output-dir {output_dir}"
        )
    directory_owner = (
        None
        if prepare_runs_as_root(config)
        else resolve_benchmark_run_as_user(
            agent_profile=config.agent_profile,
            benchmark_run_as_user=config.benchmark_run_as_user,
        )
    )
    ensure_directory(output_dir.parent, run_as_user=directory_owner)
    prefix = f"[benchmark:{config.scenario.value}] "
    if prepare_runs_as_root(config):
        print(f"{prefix}preparing dataset as root", flush=True)
    completed = subprocess.run(
        build_prepare_argv(config),
        check=False,
        text=True,
        capture_output=True,
        env=build_benchmark_env(),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "dataset prepare failed"
        raise RuntimeError(f"prepare {config.prepare_dataset} exited with code {completed.returncode}: {message}")
    if not dataset_is_ready(output_dir):
        raise RuntimeError(f"prepare {config.prepare_dataset} did not create expected files under {output_dir}")
    return output_dir


def ensure_repo_cache_prepared(config: E2EPerfConfig) -> Path | None:
    """Warm the shared SWE-bench repo cache as root before a sudo-wrapped benchmark run."""

    if not prepare_runs_as_root(config):
        return None

    from agentinfer.agentbench.benchkit.config import load_config
    from agentinfer.agentbench.benchkit.dataset import load_tasks
    from agentinfer.agentbench.benchkit.workspace import WorkspaceProcessOwner, ensure_repo_cache

    if not config.bench_config.exists():
        raise RuntimeError(f"bench config does not exist: {config.bench_config}")

    bench = load_config(config.bench_config)
    tasks = load_tasks(
        bench.dataset.index_path,
        bench.dataset.selection_path,
        config.task_num,
    )
    cache_dir = bench.dataset.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"[benchmark:{config.scenario.value}] "
    print(f"{prefix}warming repo-cache as root ({len(tasks)} tasks)", flush=True)
    owner = WorkspaceProcessOwner(time.monotonic() + config.task_timeout_seconds * max(len(tasks), 1))
    for task in tasks:
        try:
            ensure_repo_cache(task, cache_dir, owner)
        except TimeoutError as exc:
            raise RuntimeError(f"repo-cache warmup timed out for {task.repo}@{task.base_commit}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"repo-cache warmup failed for {task.repo}@{task.base_commit}: {exc}") from exc
    return cache_dir


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    """Return summary JSON when a run directory contains a completed benchmark."""

    summary = load_summary(run_dir)
    status = summary["lifecycle"]["status"]
    if status != "completed":
        raise RuntimeError(f"benchmark run {run_dir} did not complete (status={status!r})")
    completed = summary["tasks"]["completed"]
    if completed <= 0:
        raise RuntimeError(f"benchmark run {run_dir} has no completed tasks")
    requests = summary["requests"]["requests"]
    if requests <= 0:
        raise RuntimeError(f"benchmark run {run_dir} has no requests")
    return summary
