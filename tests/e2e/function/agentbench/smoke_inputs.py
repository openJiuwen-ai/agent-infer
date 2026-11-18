# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Materialize a small local repository and metadata for a real BenchKit run."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from agentinfer.agentbench.benchkit.config import load_config
from tests.e2e.helpers.case_loader import E2EConfig

TASK_ID = "agentinfer-smoke-add"


def prepare_smoke_case(config: E2EConfig) -> E2EConfig:
    """Seed a local repo cache so normal BenchKit dispatch never needs GitHub."""
    if config.task_num != 1 or config.max_concurrency != 1:
        raise ValueError("the local smoke suite requires task-num=1 and max-concurrency=1")
    inputs = config.result_root / "smoke-inputs" / config.run_tag
    repository = inputs / "repo-cache" / "agentinfer__smoke"
    repository.mkdir(parents=True, exist_ok=False)
    for name in ("arithmetic.py", "check_arithmetic.py"):
        shutil.copyfile(Path(__file__).with_name("fixtures") / name, repository / name)
    git = [
        "git",
        "-c",
        "user.name=AgentInfer Smoke",
        "-c",
        "user.email=smoke@agentinfer.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    for args in (["init"], ["add", "."], ["commit", "--signoff", "-m", "Initial smoke task"]):
        subprocess.run([*git, *args], cwd=repository, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    task = {
        "instance_id": TASK_ID,
        "repo": "agentinfer/smoke",
        "base_commit": commit,
        "problem_statement": (
            "The add(a, b) function in arithmetic.py returns the wrong result. "
            "Read arithmetic.py and check_arithmetic.py, make the smallest fix to arithmetic.py, "
            "and run python check_arithmetic.py to verify it. "
            "Do not modify check_arithmetic.py, install dependencies, or use the network. "
            "This repository uses only the Python standard library. "
            "After verification, provide a concise final report and stop."
        ),
    }
    (inputs / "instances.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    selection = inputs / "task-lists" / "default.txt"
    selection.parent.mkdir()
    selection.write_text(TASK_ID + "\n", encoding="utf-8")
    bench = load_config(config.bench_config).model_dump(mode="json")
    bench["dataset"] = {
        "name": "smoke",
        "index_path": str(inputs / "instances.jsonl"),
        "selection_path": str(selection),
        "cache_dir": str(inputs / "repo-cache"),
    }
    generated_config = inputs / "benchmark.yaml"
    generated_config.write_text(yaml.safe_dump(bench, sort_keys=False), encoding="utf-8")
    params = {
        **config.case.benchmark_params,
        "config": str(generated_config),
        "prepare_dataset": "smoke",
        "prepare_output_dir": str(inputs),
    }
    return replace(
        config,
        case=replace(config.case, benchmark_params=params),
        bench_config=generated_config,
        prepare_dataset="smoke",
        dataset_output_dir=inputs,
    )
