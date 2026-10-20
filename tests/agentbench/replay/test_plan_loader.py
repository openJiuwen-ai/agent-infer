# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Round-trip and fail-closed tests for loading a Replay plan from disk.

A ``replay-plan.json`` is currently a write-only diagnostic artifact; these
tests pin the new ``ReplayPlan.from_dict`` / ``ReplayPlan.load`` /
``validate_loaded_plan`` contract: a serialized plan reconstructs into a
structurally-equivalent plan, and a corrupted artifact raises instead of
round-tripping silently.
"""

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.planner import ReplayPlan, validate_loaded_plan
from agentinfer.agentbench.replay.unified_trace_ir import PromptReference


def _row(
    request_id,
    session_id,
    actor_id,
    started,
    finished,
    input_tokens=10,
    output_tokens=2,
    status="success",
    cached_tokens=None,
):
    row = {
        "request_id": request_id,
        "session_id": session_id,
        "actor_id": actor_id,
        "started_at": f"2026-01-01T00:00:{started:02d}+00:00",
        "finished_at": f"2026-01-01T00:00:{finished:02d}+00:00",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status,
    }
    if cached_tokens is not None:
        row["cached_tokens"] = cached_tokens
    return row


def _source(path: Path) -> None:
    rows = [
        _row("s1-lead", "s1", "lead", 0, 1, cached_tokens=6),
        _row("s1-failed", "s1", "child", 2, 3, None, None, "failed"),
        _row("s1-retry", "s1", "child", 4, 5, 20, 4),
        _row("s2-lead", "s2", "lead", 0, 1, 12, 3),
        _row("s2-timing", "s2", "orphan", 2, 3, None, None, "failed"),
        _row("s2-next", "s2", "lead", 4, 5, 16, 3),
        _row("s2-terminal", "s2", "tail", 6, 7, None, None, "failed"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(source: Path, result_dir: Path) -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 5, "result_dir": result_dir, "max_concurrency": 2},
            "backend": {"endpoint": "/v1/chat/completions"},
            "replay": {"trace_path": source, "sample_seed": 9},
        }
    )


def _build_plan(tmp_path: Path) -> ReplayPlan:
    from agentinfer.agentbench.replay.planner import build_replay_plan

    source = tmp_path / "requests.jsonl"
    _source(source)
    return build_replay_plan(_config(source, tmp_path / "result"), analyze_replay_trace(source))


def _roundtrip(plan: ReplayPlan) -> ReplayPlan:
    return ReplayPlan.from_dict(json.loads(json.dumps(plan.to_dict())))


def test_plan_roundtrip_is_structurally_equal(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    reloaded = _roundtrip(plan)
    validate_loaded_plan(reloaded)
    assert json.dumps(plan.to_dict(), sort_keys=True) == json.dumps(reloaded.to_dict(), sort_keys=True)
    assert reloaded.workload_fingerprint == plan.workload_fingerprint
    assert reloaded.plan_namespace == plan.plan_namespace
    assert reloaded.sample_seed == plan.sample_seed
    assert [task.task_id for task in reloaded.tasks] == [task.task_id for task in plan.tasks]


def test_plan_load_from_disk_roundtrips(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    path = tmp_path / "replay-plan.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    loaded = ReplayPlan.load(path)
    validate_loaded_plan(loaded)
    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(plan.to_dict(), sort_keys=True)


def test_timing_dependency_node_roundtrip_preserves_duration(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    reloaded = _roundtrip(plan)
    original_timing = [node for task in plan.tasks for node in task.requests if node.node_type == "timing_dependency"]
    reloaded_timing = {
        node.source_key: node
        for task in reloaded.tasks
        for node in task.requests
        if node.node_type == "timing_dependency"
    }
    assert original_timing, "fixture should contain a timing_dependency node"
    for node in original_timing:
        serialized = node.to_dict()
        # the early-return branch writes only the clock/duration surface
        assert set(serialized) == {
            "runtime_request_id",
            "source_key",
            "node_type",
            "send_after",
            "effective_interval_seconds",
            "effective_duration_seconds",
        }
        loaded = reloaded_timing[node.source_key]
        assert loaded.effective_duration_seconds == node.effective_duration_seconds
        assert loaded.effective_interval_seconds == node.effective_interval_seconds


def test_request_node_optional_fields_roundtrip(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    reloaded = _roundtrip(plan)
    original = {node.source_key: node for task in plan.tasks for node in task.requests if node.node_type == "request"}
    reloaded_by_key = {
        node.source_key: node for task in reloaded.tasks for node in task.requests if node.node_type == "request"
    }
    assert original
    for key, node in original.items():
        loaded = reloaded_by_key[key]
        assert loaded.prompt_ref == node.prompt_ref
        assert loaded.planned_input_tokens == node.planned_input_tokens
        assert loaded.planned_output_tokens == node.planned_output_tokens
        assert loaded.source_prefix_tokens == node.source_prefix_tokens
        assert loaded.context_after == node.context_after
        assert loaded.context_mode == node.context_mode
        assert loaded.response_validation == node.response_validation


def test_prompt_reference_from_dict_roundtrips() -> None:
    ref = PromptReference("session-1", 3)
    assert PromptReference.from_dict(ref.to_dict()) == ref


@pytest.mark.parametrize(
    "corrupt",
    [
        "not-an-object",
        "lead",
    ],
)
def test_plan_from_dict_rejects_non_object(corrupt: object) -> None:
    with pytest.raises(ValueError, match="must be an object"):
        ReplayPlan.from_dict(corrupt)


def test_plan_from_dict_rejects_bad_schema_version() -> None:
    data = _minimal_plan_dict()
    data["schema_version"] = "99"
    with pytest.raises(ValueError, match="schema_version"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_bad_plan_kind() -> None:
    data = _minimal_plan_dict()
    data["plan_kind"] = "experimental"
    with pytest.raises(ValueError, match="plan_kind"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_non_bool_execution_ready() -> None:
    data = _minimal_plan_dict()
    data["execution_ready"] = "yes"
    with pytest.raises(ValueError, match="execution_ready"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_unknown_interval_model_field() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_bad_prompt_ref_turn_index() -> None:
    data = _minimal_plan_dict()
    # first request node of the only task carries prompt_ref only if present; craft one explicitly
    node = data["tasks"][0]["requests"][0]
    if node["node_type"] == "request":
        node["prompt_ref"] = {"session_id": "s", "turn_index": -1}
    with pytest.raises(ValueError, match="turn_index"):
        ReplayPlan.from_dict(data)


def test_validate_loaded_plan_rejects_dangling_context_reference() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["context_after"] = "does-not-exist"
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="context_after reference .* is missing"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_dangling_send_after_reference() -> None:
    data = _minimal_plan_dict()
    _first_request_node(data)["send_after"] = "ghost"
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="send_after reference .* is missing"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_duplicate_runtime_session() -> None:
    data = _minimal_plan_dict()
    data["tasks"].append(json.loads(json.dumps(data["tasks"][0])))
    # task_index must stay distinct or the duplicate-session check still fires first
    data["tasks"][1]["task_index"] = 1
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="runtime session ids must be unique"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_request_without_positive_input() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["planned_input_tokens"] = 0
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="no positive input target"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_negative_interval() -> None:
    data = _minimal_plan_dict()
    data["tasks"][0]["requests"][0]["effective_interval_seconds"] = -1
    with pytest.raises(ValueError, match="effective_interval_seconds"):
        ReplayPlan.from_dict(data)


def _first_request_node(data: dict) -> dict:
    for task in data["tasks"]:
        for node in task["requests"]:
            if node["node_type"] == "request":
                return node
    raise AssertionError("expected at least one request node")


def _minimal_plan_dict() -> dict:
    """A hand-built, structurally-valid plan dict (no source needed)."""

    node = {
        "runtime_request_id": "rid-1",
        "source_key": "k1",
        "node_type": "request",
        "send_after": None,
        "effective_interval_seconds": 0.0,
        "prompt_recipe_key": "k1",
        "prompt_kind": "lead_main",
        "actor_id": "lead",
        "actor_role": "lead",
        "parent_actor_id": None,
        "context_after": None,
        "context_mode": "independent",
        "planned_input_tokens": 10,
        "planned_output_tokens": 2,
        "source_output_tokens": 2,
        "backend_sampling_seed": None,
    }
    return {
        "schema_version": "1",
        "plan_kind": "structural",
        "planner_version": "agentinfer-replay-structural",
        "execution_ready": False,
        "execution_unavailable_reason": "structural plan requires runtime Prompt calibration",
        "source": "/tmp/requests.jsonl",
        "source_sha256": "a" * 64,
        "source_bundle_sha256": None,
        "source_sessions": 1,
        "replayable_source_sessions": 1,
        "sample_seed": 0,
        "plan_namespace": "n" * 64,
        "workload_fingerprint": "f" * 64,
        "prompt_calibration": {"status": "pending", "model": "m", "tokenizer_base_url": None},
        "interval_model": {"mode": "trace", "fit_version": "agentinfer-replay-trace"},
        "tasks": [
            {
                "task_id": "replay-000000",
                "task_index": 0,
                "sample_ordinal": 0,
                "runtime_session_id": "sess-1",
                "source_session_id": "s1",
                "source_task_id": None,
                "requests": [node],
            }
        ],
    }


# --- review fixes (2026-10-16): required-field presence, IntervalModel validation,
# --- negative seed round-trip, context pairing, self-loop, empty tasks, str paths.


def test_plan_from_dict_rejects_missing_planned_output_tokens() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    del node["planned_output_tokens"]
    with pytest.raises(ValueError, match="planned_output_tokens"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_null_planned_output_tokens() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["planned_output_tokens"] = None
    with pytest.raises(ValueError, match="planned_output_tokens"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_rejects_missing_send_after_key() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    del node["send_after"]
    with pytest.raises(ValueError, match="send_after"):
        ReplayPlan.from_dict(data)


def test_plan_from_dict_accepts_null_send_after_key() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["send_after"] = None
    plan = ReplayPlan.from_dict(data)
    assert plan.tasks[0].requests[0].send_after is None


def test_interval_model_rejects_bad_mode() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["mode"] = "garbage"
    with pytest.raises(ValueError, match="mode"):
        ReplayPlan.from_dict(data)


def test_interval_model_rejects_bad_mu() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["mu"] = "banana"
    with pytest.raises(ValueError, match="mu"):
        ReplayPlan.from_dict(data)


def test_interval_model_rejects_nonfinite_mu() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["mu"] = float("nan")
    with pytest.raises(ValueError, match="mu"):
        ReplayPlan.from_dict(data)


def test_interval_model_rejects_bad_sigma() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["sigma"] = -1
    with pytest.raises(ValueError, match="sigma"):
        ReplayPlan.from_dict(data)


def test_interval_model_rejects_bad_anchors() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["anchors"] = []
    with pytest.raises(ValueError, match="anchors"):
        ReplayPlan.from_dict(data)


def test_interval_model_rejects_bad_cdf_version() -> None:
    data = _minimal_plan_dict()
    data["interval_model"]["cdf_version"] = 123
    with pytest.raises(ValueError, match="cdf_version"):
        ReplayPlan.from_dict(data)


def test_negative_sample_seed_roundtrips(tmp_path: Path) -> None:
    data = _minimal_plan_dict()
    data["sample_seed"] = -1
    plan = ReplayPlan.from_dict(data)
    assert plan.sample_seed == -1
    path = tmp_path / "replay-plan.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    loaded = ReplayPlan.load(path)
    assert loaded.sample_seed == -1


def test_validate_loaded_plan_rejects_empty_tasks() -> None:
    data = _minimal_plan_dict()
    data["tasks"] = []
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="at least one task"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_independent_with_context_after() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["context_after"] = node["source_key"]
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="independent context must not reference"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_context_after_self_loop() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["context_after"] = node["source_key"]
    node["context_mode"] = "append"
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="cycle"):
        validate_loaded_plan(plan)


def test_load_accepts_str_path(tmp_path: Path) -> None:
    data = _minimal_plan_dict()
    path = tmp_path / "replay-plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    plan = ReplayPlan.load(str(path))
    assert plan.schema_version == "1"


def test_interval_model_rejects_negative_sigma_roundtrip(tmp_path: Path) -> None:
    """The reviewer's exact repro: sigma=-1 previously loaded silently."""
    data = _minimal_plan_dict()
    data["interval_model"]["sigma"] = -1.0
    data["interval_model"]["mu"] = 1.0
    with pytest.raises(ValueError, match="sigma"):
        ReplayPlan.from_dict(data)
