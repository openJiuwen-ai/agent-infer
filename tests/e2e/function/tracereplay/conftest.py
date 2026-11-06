# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for all Trace Replay functional E2E suites."""

from __future__ import annotations

import pytest

from .helpers.conftest_options import register_replay_e2e_options


def pytest_addoption(parser: pytest.Parser) -> None:
    register_replay_e2e_options(
        parser,
        group_name="tracereplay_e2e",
        group_description="Trace Replay functional E2E options (inferact, tracelab, agentx)",
        cases_help="Replay case JSON under tests/e2e/function/tracereplay/<suite>/cases/",
    )
