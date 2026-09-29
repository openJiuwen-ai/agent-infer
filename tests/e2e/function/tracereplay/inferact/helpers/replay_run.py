# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Execute one Inferact replay E2E case."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .....helpers.benchmark import validate_completed_run
from .....helpers.server import managed_vllm
from .replay_config import ReplayE2EConfig, build_replay_argv, build_replay_env


def run_replay_case(config: ReplayE2EConfig) -> Path:
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


def run_replay_two_cold_starts(config: ReplayE2EConfig) -> tuple[Path, Path]:
    """Run the same case twice; stop vLLM between runs so each replay starts cold."""

    baseline_dir = run_replay_case(config)
    print(f"[replay] cold run 1/2 completed: dir={baseline_dir}", flush=True)
    candidate_dir = run_replay_case(config.with_fresh_run())
    print(f"[replay] cold run 2/2 completed: dir={candidate_dir}", flush=True)
    return baseline_dir, candidate_dir
