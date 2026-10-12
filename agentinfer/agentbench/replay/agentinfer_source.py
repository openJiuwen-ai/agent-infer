# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Resolve the packaged AgentInfer Replay source."""

from __future__ import annotations

from pathlib import Path


def builtin_agentinfer_source() -> Path:
    """Return the packaged eight-session AgentInfer trace."""

    source = Path(__file__).resolve().parents[1] / "data" / "agentinfer_trace_requests.jsonl"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"packaged AgentInfer Replay dataset is missing or empty: {source}")
    return source
