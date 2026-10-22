# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise dependency barriers against actual HTTP arrivals, never server scheduling."""

import json
import time

import pytest

from tests.agentbench.helpers.mock_server.process import read_rows
from tests.agentbench.replay.helpers.cold_start import run_round
from tests.agentbench.replay.helpers.fixtures import build_fixture
from tests.agentbench.replay.helpers.validation import validate_round

pytestmark = [pytest.mark.cpu_test, pytest.mark.replay_smoke]


def client_event(root, event, request_id=None, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for row in read_rows(root / "client-events.jsonl"):
            if row["event"] == event and (request_id is None or row.get("runtime_request_id") == request_id):
                return row
        time.sleep(0.01)
    raise AssertionError(f"missing client event {event} for {request_id}")


def captured(server):
    return {row.get("request_id") for row in server.get("/__mock__/events")["events"] if row["event"] == "received"}


def no_arrival(server, ids, duration=0.12):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        assert not captured(server).intersection(ids), f"premature requests: {ids}"
        time.sleep(0.01)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_serial_context_wait_and_session_admission(tmp_path, concurrency):
    path, expected = build_fixture(tmp_path / "fixture", "agentinfer")

    def control(server, plan, root):
        first, second = plan["tasks"][:2]
        a, b, c = [n["runtime_request_id"] for n in first["requests"]]
        server.post("/__mock__/arm", {"rules": [{"request_id": a, "gate": {"id": "main"}}]})
        server.wait_event("gate_waiting", gate_id="main")
        client_event(root, "dependency_wait_started", b)
        no_arrival(server, {b, c})
        if concurrency == 1:
            no_arrival(server, {n["runtime_request_id"] for n in second["requests"]})
            events = read_rows(root / "client-events.jsonl")
            assert len([e for e in events if e["event"] == "task_admitted"]) == 1
        else:
            client_event(root, "node_finished", second["requests"][-1]["runtime_request_id"])
        server.post("/__mock__/gates/main/release")

    root = run_round(tmp_path / "round", path, expected, concurrency, 0, control=control)
    validate_round(expected, root)


def test_title_can_finish_while_main_is_blocked_and_next_inherits_main(tmp_path):
    path, expected = build_fixture(tmp_path / "fixture", "agentinfer", title=True)
    expected["task_num"] = 2

    def control(server, plan, root):
        main, title, next_node = plan["tasks"][0]["requests"]
        server.post("/__mock__/arm", {"rules": [{"request_id": main["runtime_request_id"], "gate": {"id": "main"}}]})
        server.wait_event("gate_waiting", gate_id="main")
        client_event(root, "node_finished", title["runtime_request_id"])
        assert title["runtime_request_id"] in captured(server)
        no_arrival(server, {next_node["runtime_request_id"]})
        server.post("/__mock__/gates/main/release")

    root = run_round(tmp_path / "round", path, expected, 2, 0, control=control)
    validate_round(expected, root)


@pytest.mark.parametrize(
    "source,fault",
    [
        ("agentinfer", "http_error"),
        ("agentX", "http_error"),
        ("agentinfer", "omit_done"),
        ("agentinfer", "truncate_stream"),
        ("agentinfer", "usage_mismatch"),
        ("agentinfer", "token_ids_missing"),
        ("agentinfer", "token_ids_short"),
        ("agentinfer", "timeout"),
    ],
)
def test_failure_is_recorded_and_only_context_descendants_are_skipped(tmp_path, source, fault):
    path, expected = build_fixture(tmp_path / "fixture", source)
    expected["task_num"] = 2
    target = {}

    def control(server, plan, root):
        target.update(plan["tasks"][0])
        rid = target["requests"][0]["runtime_request_id"]
        rule = {"request_id": rid}
        rule.update({"gate": {"id": "never"}} if fault == "timeout" else {"fault": fault})
        server.post("/__mock__/arm", {"rules": [rule]})
        server.wait_event("rule_matched", request_id=rid)
        if fault == "timeout":
            server.wait_event("gate_waiting", gate_id="never")

    root = run_round(
        tmp_path / "round",
        path,
        expected,
        2,
        0,
        control=control,
        expect_failure=True,
        replay_options={"request_timeout_seconds": 1},
    )
    execution = json.loads((root / "result/replay-execution.json").read_text())
    tasks = execution["tasks"]
    bad = next(t for t in tasks if t["runtime_session_id"] == target["runtime_session_id"])
    other = next(t for t in tasks if t["runtime_session_id"] != target["runtime_session_id"])
    assert other["status"] == "completed"
    ordered = {n["runtime_request_id"]: n for n in bad["nodes"]}
    statuses = [ordered[n["runtime_request_id"]]["status"] for n in target["requests"]]
    assert statuses == ["failed"] + (["success"] * 2 if source == "agentX" else ["skipped_dependency_failed"] * 2)
    rows = [r for r in read_rows(root / "server/backend-requests.jsonl") if r["kind"] == "generation"]
    assert len(rows) == (6 if source == "agentX" else 4)
    summary = execution["summary"]
    assert summary["attempted_requests"] == len(rows)
    assert summary["failed_requests"] == 1
    assert summary["dependency_skipped_requests"] == (0 if source == "agentX" else 2)
    assert summary["pre_send_failed_requests"] == 0
    assert execution["trace_writer"]["pending"] == 0
    assert json.loads((root / "server/exit.json").read_text())["complete"]
    if source == "agentX":
        events = read_rows(root / "client-events.jsonl")
        times = {(e.get("runtime_request_id"), e["event"]): e["t_monotonic"] for e in events}
        a, b = [n["runtime_request_id"] for n in target["requests"][:2]]
        assert times[b, "http_dispatch"] >= times[a, "node_finished"] + 0.05 - 1e-6


@pytest.mark.parametrize("limit", [1, 2])
def test_agentx_overlapping_snapshots_obey_request_limit(tmp_path, limit):
    path, expected = build_fixture(tmp_path / "fixture", "agentX")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        for request, (start, duration) in zip(row["requests"], [(0, 0.1), (0.02, 0.03), (0.12, 0.03)], strict=True):
            request.update(t=start, api_time=duration)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected.update(task_num=1, overlap=True)

    def control(server, plan, root):
        a, b, c = [n["runtime_request_id"] for n in plan["tasks"][0]["requests"]]
        server.post("/__mock__/arm", {"rules": [{"request_id": a, "gate": {"id": "a"}}]})
        server.wait_event("gate_waiting", gate_id="a")
        client_event(root, "release_scheduled", b)
        if limit == 1:
            no_arrival(server, {b, c})
        else:
            client_event(root, "node_finished", b)
            no_arrival(server, {c})
        server.post("/__mock__/gates/a/release")

    root = run_round(
        tmp_path / "round", path, expected, 1, 0, control=control, replay_options={"max_inflight_requests": limit}
    )
    validate_round(expected, root)
    active = peak = 0
    for event in read_rows(root / "client-events.jsonl"):
        if event["event"] == "transport_entered":
            active += 1
            peak = max(peak, active)
        elif event["event"] == "node_finished":
            active -= 1
        assert 0 <= active <= limit
    assert peak == limit and active == 0


def test_task_timeout_cancels_held_http_stream_and_reaps_both_processes(tmp_path):
    path, expected = build_fixture(tmp_path / "fixture", "agentinfer")
    expected["task_num"] = 2
    target = {}

    def control(server, plan, root):
        target.update(plan["tasks"][0])
        rid = target["requests"][0]["runtime_request_id"]
        server.post("/__mock__/arm", {"rules": [{"request_id": rid, "gate": {"id": "never"}}]})
        server.wait_event("gate_waiting", gate_id="never")
        client_event(root, "node_finished", plan["tasks"][1]["requests"][-1]["runtime_request_id"])

    root = run_round(
        tmp_path / "round",
        path,
        expected,
        2,
        0,
        control=control,
        expect_failure=True,
        experiment_options={"task_timeout_seconds": 2},
    )
    execution = json.loads((root / "result/replay-execution.json").read_text())
    failed = next(t for t in execution["tasks"] if t["runtime_session_id"] == target["runtime_session_id"])
    assert failed["status"] == "failed" and "TimeoutError" in failed["error"]
    assert sum(t["status"] == "completed" for t in execution["tasks"]) == 1
    rows = read_rows(root / "server/backend-requests.jsonl")
    actual = {r["identity"].get("request_id") for r in rows}
    assert all(n["runtime_request_id"] not in actual for n in target["requests"][1:])
    first = next(r for r in rows if r["identity"].get("request_id") == target["requests"][0]["runtime_request_id"])
    events = read_rows(root / "server/server-events.jsonl")
    assert any(e["event"] == "response_aborted" and e.get("capture_id") == first["capture_id"] for e in events)
    assert json.loads((root / "replay-exit.json").read_text())["forced"] is False
    assert json.loads((root / "server/exit.json").read_text())["complete"] is True
