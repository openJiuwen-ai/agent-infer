# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""A passing baseline plus deliberately corrupted evidence tests the comparator."""

import json
import shutil

import pytest

from tests.agentbench.helpers.mock_server.process import read_rows
from tests.agentbench.replay.helpers.cold_start import run_round
from tests.agentbench.replay.helpers.fixtures import build_fixture
from tests.agentbench.replay.helpers.validation import validate_round

pytestmark = [pytest.mark.cpu_test, pytest.mark.replay_smoke]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("validator-baseline")
    path, expected = build_fixture(root / "fixture", "agentinfer")
    expected["task_num"] = 2
    run = run_round(root / "round", path, expected, 2, 0)
    validate_round(expected, run)
    return expected, run


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("none", None),
        ("parameter", "max_tokens"),
        ("delete", "request_counts"),
        ("duplicate", "request_counts"),
        ("history", "history"),
        ("early", "early_send"),
        ("missing_events", "client_events"),
        ("unfinished", "observer_closed"),
    ],
)
def test_validator_finds_specific_corruption_without_rejecting_good_evidence(tmp_path, baseline, mutation, diagnostic):
    expected, baseline_root = baseline
    root = tmp_path / "evidence"
    shutil.copytree(baseline_root, root)
    plan = json.loads((root / "result/replay-plan.json").read_text())
    target = plan["tasks"][0]["requests"][1]["runtime_request_id"]
    path = root / "server/backend-requests.jsonl"
    rows = read_rows(path)
    index = next(i for i, r in enumerate(rows) if r["identity"].get("request_id") == target)
    if mutation == "parameter":
        rows[index]["body"]["max_tokens"] = 9
    elif mutation == "delete":
        rows.pop(index)
    elif mutation == "duplicate":
        rows.append(rows[index])
    elif mutation == "history":
        rows[index]["body"]["prompt"][64] += 1  # Same length, wrong actual predecessor answer.
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = root / "client-events.jsonl"
    events = read_rows(path)
    if mutation == "early":
        event = next(e for e in events if e.get("runtime_request_id") == target and e["event"] == "transport_entered")
        event["t_monotonic"] -= 10
    elif mutation == "missing_events":
        events = []
    elif mutation == "unfinished":
        events.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    if diagnostic is None:
        validate_round(expected, root)
        assert json.loads((root / "comparison.json").read_text())["passed"]
    else:
        with pytest.raises(AssertionError, match=diagnostic):
            validate_round(expected, root)
        report = json.loads((root / "comparison.json").read_text())
        assert report["passed"] is False and diagnostic in report["error"]
        if mutation in {"parameter", "history", "early"}:
            assert "task[0]" in report["error"]
