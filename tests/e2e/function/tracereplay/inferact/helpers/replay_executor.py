# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Cold vLLM + replay bench cycles for Inferact functional E2E."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .....helpers.benchmark import validate_completed_run
from .....helpers.server import managed_vllm
from .replay_config import ReplayE2EConfig, build_replay_argv, build_replay_env


def run_replay_case(config: ReplayE2EConfig) -> Path:
    """One cold vLLM start, one replay bench run, validated result directory."""

    blockers = config.validate_prerequisites()
    if blockers:
        raise RuntimeError("; ".join(blockers))

    output_dir = config.make_result_dir(hardware_slug="unknown")
    if output_dir.exists():
        raise FileExistsError(f"result directory already exists: {output_dir}")

    argv = build_replay_argv(config, output_dir)
    e2e = config.to_e2e_config()
    with managed_vllm(e2e):
        completed = subprocess.run(
            argv,
            check=False,
            text=True,
            capture_output=True,
            env=build_replay_env(),
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, flush=True)
            if completed.stderr:
                print(completed.stderr, flush=True)
            raise RuntimeError(f"replay bench exited with code {completed.returncode}")

    validate_completed_run(output_dir)
    return output_dir


def run_replay_cold_starts(config: ReplayE2EConfig, *, cold_runs: int) -> tuple[Path, ...]:
    """Run the same case ``cold_runs`` times; stop vLLM between runs for a cold start each time."""

    if cold_runs < 1:
        raise ValueError("cold_runs must be >= 1")

    result_dirs: list[Path] = []
    current = config
    for index in range(cold_runs):
        run_dir = run_replay_case(current)
        print(f"[replay] cold run {index + 1}/{cold_runs} completed: dir={run_dir}", flush=True)
        result_dirs.append(run_dir)
        if index + 1 < cold_runs:
            current = current.with_fresh_run()
    return tuple(result_dirs)
