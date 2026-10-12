# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest: Inferact replay functional E2E (cold starts + repeatability compare)."""

from __future__ import annotations

import pytest

from ..helpers.replay_repeatability import run_replay_repeatability_e2e
from ..specs import INFERACT_REPLAY_SUITE


def test_inferact_replay(pytestconfig: pytest.Config) -> None:
    """Run a case N times (cold vLLM each time), then compare run 1 vs run 2 for repeatability."""

    run_replay_repeatability_e2e(pytestconfig, suite=INFERACT_REPLAY_SUITE, suite_label="Inferact")
