# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for Trace Replay functional E2E."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup(
        "tracereplay_e2e",
        "Trace Replay functional E2E options (inferact, tracelab, agentx)",
    )
    group.addoption(
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help="Replay case JSON under tests/e2e/function/tracereplay/cases/<suite>/",
    )
    group.addoption(
        "--task-num",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params task-num",
    )
    group.addoption(
        "--max-concurrency",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params max-concurrency",
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
