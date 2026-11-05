# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Parametrized Trace Replay functional E2E (cold starts + repeatability compare)."""

from __future__ import annotations

import pytest

from .helpers.replay_e2e_config import ReplaySuiteSpec
from .helpers.replay_repeatability import run_replay_repeatability_e2e

INFERACT_REPLAY_SUITE = ReplaySuiteSpec(
    workload="inferact-replay",
    default_bench_config="agentinfer/agentbench/configs/replay_inferact.yaml",
    trace_path_mode="required",
    missing_trace_path_hint=(
        "benchmark_params.trace-path is required "
        "(e.g. downloaded codex_swebenchpro.json from "
        "https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)"
    ),
)

TRACELAB_REPLAY_SUITE = ReplaySuiteSpec(
    workload="tracelab-replay",
    default_bench_config="agentinfer/agentbench/configs/replay_tracelab.yaml",
    trace_path_mode="optional",
    missing_trace_path_hint="benchmark_params.trace-path is required for this TraceLab case",
    optional_trace_download_hint=("UW-SyFI/TraceLab (or set benchmark_params.trace-path to a normalized JSONL file)"),
)

AGENTX_REPLAY_SUITE = ReplaySuiteSpec(
    workload="agentx-replay",
    default_bench_config="agentinfer/agentbench/configs/replay_agentX.yaml",
    trace_path_mode="required",
    missing_trace_path_hint=(
        "benchmark_params.trace-path is required "
        "(AgentX full hash-block snapshot JSONL, e.g. traces-shortest-16-sessions.jsonl)"
    ),
)

_REPLAY_SUITES: tuple[tuple[ReplaySuiteSpec, str], ...] = (
    (INFERACT_REPLAY_SUITE, "Inferact"),
    (TRACELAB_REPLAY_SUITE, "TraceLab"),
    (AGENTX_REPLAY_SUITE, "AgentX"),
)


@pytest.mark.parametrize(
    ("suite", "suite_label"),
    _REPLAY_SUITES,
    ids=["inferact", "tracelab", "agentx"],
)
def test_trace_replay(
    pytestconfig: pytest.Config,
    suite: ReplaySuiteSpec,
    suite_label: str,
) -> None:
    """Run a case N times (cold vLLM each time), then compare run 1 vs run 2 for repeatability."""

    run_replay_repeatability_e2e(pytestconfig, suite=suite, suite_label=suite_label)
