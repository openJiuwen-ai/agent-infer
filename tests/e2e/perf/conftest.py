# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest options for E2E benchmark runs."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e", "E2E performance benchmark options")
    group.addoption(
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help=('Case JSON path; scenario from JSON "scenario" (or legacy "arm") or cases/baseline|agentinfer/ path.'),
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
        "--benchmark-run-as-user",
        action="store",
        default=None,
        metavar="USER",
        help="For plan-subagent only: run bench subprocess as USER via sudo when pytest runs as root",
    )
