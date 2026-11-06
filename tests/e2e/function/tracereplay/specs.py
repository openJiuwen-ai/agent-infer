# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Per-trace-family Replay E2E suite specifications."""

from __future__ import annotations

from .helpers.replay_e2e_config import ReplaySuiteSpec

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
