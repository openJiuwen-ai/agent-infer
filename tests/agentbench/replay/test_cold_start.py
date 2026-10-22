# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Real CLI, adapters and HTTP across two completely new pairs of processes."""

import json

import pytest

from .helpers.cold_start import run_round
from .helpers.fixtures import SOURCES, build_fixture
from .helpers.validation import compare_rounds, validate_round

pytestmark = [pytest.mark.cpu_test, pytest.mark.replay_smoke]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("concurrency", [1, 4])
def test_two_cold_starts_preserve_workload_and_own_round_history(tmp_path, source, concurrency):
    path, expected = build_fixture(tmp_path / "fixture", source)
    rounds = [
        validate_round(expected, run_round(tmp_path / f"round-{i}", path, expected, concurrency, i)) for i in range(2)
    ]
    try:
        compare_rounds(expected, *rounds)
    except AssertionError as exc:
        (tmp_path / "comparison.json").write_text(json.dumps({"passed": False, "error": str(exc)}))
        raise
    (tmp_path / "comparison.json").write_text(
        json.dumps({"passed": True, "source": source, "concurrency": concurrency})
    )
