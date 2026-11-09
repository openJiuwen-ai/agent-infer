# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Replay-specific pytest options for functional E2E."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup(
        "tracereplay_e2e",
        "Trace Replay functional E2E options (inferact, tracelab, agentx)",
    )
    group.addoption(
        "--bench-config",
        action="store",
        default=None,
        metavar="PATH",
        help="Override benchmark_params config (replay bench YAML for --config)",
    )
    group.addoption(
        "--cold-runs",
        action="store",
        type=int,
        default=2,
        help="Number of cold vLLM + replay cycles (default: 2; compare uses runs 1 and 2)",
    )
    group.addoption(
        "--stability-tolerance",
        action="store",
        type=float,
        default=0.05,
        help="Maximum relative metric gap for repeatability checks (default: 0.05)",
    )
