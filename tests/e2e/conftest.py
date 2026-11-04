# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for functional and performance E2E tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e_shared", "Shared E2E options")
    group.addoption(
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help="Case JSON path for functional or performance E2E tests",
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
