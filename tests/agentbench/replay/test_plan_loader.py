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
from agentinfer.agentbench.replay.planner import ReplayPlan, ReplayPlanNode, validate_loaded_plan
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
        # the early-return branch writes the clock/duration surface plus
        # prompt_kind, which is carried so the node reloads unchanged
        assert set(serialized) == {
            "runtime_request_id",
            "source_key",
            "node_type",
            "send_after",
            "effective_interval_seconds",
            "effective_duration_seconds",
            "prompt_kind",
            "actor_id",
            "actor_role",
            "parent_actor_id",
        }
        # timing nodes never issue a request, so the carried kind is neutral
        loaded = reloaded_timing[node.source_key]
        assert loaded.effective_duration_seconds == node.effective_duration_seconds
        assert loaded.effective_interval_seconds == node.effective_interval_seconds
        assert loaded.prompt_kind == node.prompt_kind


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


def _timing_node_dict() -> dict:
    """A hand-built timing_dependency node as ``to_dict`` now emits it."""

    return {
        "runtime_request_id": "rid-timing",
        "source_key": "t1",
        "node_type": "timing_dependency",
        "send_after": None,
        "effective_interval_seconds": 0.0,
        "effective_duration_seconds": 1.5,
        "prompt_kind": "none",
        "actor_id": "sub",
        "actor_role": "subagent",
        "parent_actor_id": "lead",
    }


def test_timing_node_prompt_kind_roundtrips_losslessly() -> None:
    """A timing node reloads with the kind it was serialized with.

    Regression for the reviewer's warning that a reloaded timing node reported
    ``"none"`` regardless of its build-time value: ``to_dict`` used to drop the
    field in its early-return branch, so any aggregate over a reloaded plan
    diverged from the run that wrote it.
    """

    node = ReplayPlanNode.from_dict(_timing_node_dict())
    assert node.prompt_kind == "none"
    assert node.to_dict()["prompt_kind"] == "none"
    assert ReplayPlanNode.from_dict(node.to_dict()) == node


def test_timing_node_rejects_non_neutral_prompt_kind() -> None:
    """A request-shaped kind on a timing node fails closed instead of being
    silently rewritten to ``"none"``."""

    data = _timing_node_dict()
    data["prompt_kind"] = "lead_title"
    with pytest.raises(ValueError, match="prompt_kind"):
        ReplayPlanNode.from_dict(data)


def test_timing_node_absent_prompt_kind_loads_as_neutral() -> None:
    """An artifact written before the field was carried still loads.

    Timing nodes from an earlier release omit ``prompt_kind``. The only value a
    timing node can hold is ``none``, so the default is exact rather than a
    guess -- unlike a missing required request field, which still fails closed.
    """

    data = _timing_node_dict()
    del data["prompt_kind"]
    node = ReplayPlanNode.from_dict(data)
    assert node.prompt_kind == "none"
    # once written out again the field is present, so the artifact converges
    assert node.to_dict()["prompt_kind"] == "none"


def test_legacy_artifact_without_timing_prompt_kind_loads(tmp_path: Path) -> None:
    """A whole ``replay-plan.json`` written by the previous release loads.

    The previous ``to_dict`` omitted ``prompt_kind`` on timing nodes, so such a
    file has no such key. It must load, and re-serializing it must add the key
    without changing anything else.
    """

    data = _minimal_plan_dict()
    data["tasks"][0]["requests"].append(_timing_node_dict())
    del data["tasks"][0]["requests"][-1]["prompt_kind"]

    path = tmp_path / "replay-plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = ReplayPlan.load(path)

    timing = loaded.tasks[0].requests[-1]
    assert timing.node_type == "timing_dependency"
    assert timing.prompt_kind == "none"
    assert loaded.to_dict()["tasks"][0]["requests"][-1]["prompt_kind"] == "none"


def test_timing_and_request_node_aggregates_agree_after_reload(tmp_path: Path) -> None:
    """Coverage counts computed from a reloaded plan match the original.

    This is the property the reviewer's warning was about: the executor's audit
    summary sums ``prompt_kind`` over nodes, and it must produce the same
    numbers for ``plan`` and ``ReplayPlan.from_dict(plan.to_dict())``.
    """

    plan = _build_plan(tmp_path)
    reloaded = _roundtrip(plan)

    def counts(target: ReplayPlan) -> dict[str, int]:
        nodes = [node for task in target.tasks for node in task.requests]
        observed = {node.prompt_kind for node in nodes}
        return {kind: sum(node.prompt_kind == kind for node in nodes) for kind in sorted(observed)}

    assert counts(reloaded) == counts(plan)
    assert counts(plan), "fixture should have at least one counted prompt kind"


@pytest.mark.parametrize(
    "bad_kind",
    ["lead_title", "lead_name", "lead_main", "subagent_first", "continuation", "", "NONE", None, 1],
)
def test_timing_node_rejects_every_non_neutral_prompt_kind(bad_kind: object) -> None:
    """Only the exact string ``none`` is accepted on a timing node.

    Covers the whole request-kind vocabulary plus near-misses (empty string,
    wrong case, null, non-string) so a widened allow-list cannot slip in.
    """

    data = _timing_node_dict()
    data["prompt_kind"] = bad_kind
    with pytest.raises(ValueError, match="prompt_kind"):
        ReplayPlanNode.from_dict(data)


def test_timing_node_prompt_kind_survives_node_level_roundtrip() -> None:
    """Node-level round-trip keeps every serialized field, not just the clock.

    Guards the specific regression: ``to_dict`` used to early-return before
    ``prompt_kind``, so ``from_dict(to_dict(node))`` silently changed the node.
    """

    node = ReplayPlanNode.from_dict(_timing_node_dict())
    assert ReplayPlanNode.from_dict(node.to_dict()) == node
    assert json.dumps(node.to_dict(), sort_keys=True) == json.dumps(
        ReplayPlanNode.from_dict(node.to_dict()).to_dict(), sort_keys=True
    )


def test_timing_node_serialized_shape_is_exactly_the_contract() -> None:
    """Pin the timing node's serialized key set.

    ``to_dict`` gained ``prompt_kind``; this pins the full surface so an
    accidental extra field is a visible test failure rather than a silent
    fingerprint change.
    """

    node = ReplayPlanNode.from_dict(_timing_node_dict())
    assert set(node.to_dict()) == {
        "runtime_request_id",
        "source_key",
        "node_type",
        "send_after",
        "effective_interval_seconds",
        "effective_duration_seconds",
        "prompt_kind",
        "actor_id",
        "actor_role",
        "parent_actor_id",
    }


def test_legacy_artifact_reserializes_to_new_shape(tmp_path: Path) -> None:
    """Loading a legacy file and writing it back upgrades it exactly once.

    The first re-serialization adds ``prompt_kind``; a second load must then
    produce an identical file, so repeat upgrades are idempotent.
    """

    data = _minimal_plan_dict()
    data["tasks"][0]["requests"].append(_timing_node_dict())
    del data["tasks"][0]["requests"][-1]["prompt_kind"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    once = ReplayPlan.load(path).to_dict()
    second_path = tmp_path / "upgraded.json"
    second_path.write_text(json.dumps(once), encoding="utf-8")
    twice = ReplayPlan.load(second_path).to_dict()

    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)


def test_legacy_and_modern_artifacts_load_to_equal_plans(tmp_path: Path) -> None:
    """An artifact with and without the timing ``prompt_kind`` key loads equal.

    The two differ only in whether the legacy-redundant key was written, so the
    reconstructed plans must be identical.
    """

    legacy = _minimal_plan_dict()
    legacy["tasks"][0]["requests"].append(_timing_node_dict())
    del legacy["tasks"][0]["requests"][-1]["prompt_kind"]

    modern = json.loads(json.dumps(legacy))
    modern["tasks"][0]["requests"][-1]["prompt_kind"] = "none"

    legacy_path = tmp_path / "legacy.json"
    modern_path = tmp_path / "modern.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    modern_path.write_text(json.dumps(modern), encoding="utf-8")

    assert ReplayPlan.load(legacy_path) == ReplayPlan.load(modern_path)


def test_timing_node_prompt_kind_absent_and_null_differ() -> None:
    """Absent defaults to ``none``; an explicit ``null`` is still an error.

    The backward-compatible default applies only to a missing key. A written
    ``null`` is a malformed value and must not be rescued by the same default.
    """

    absent = _timing_node_dict()
    del absent["prompt_kind"]
    assert ReplayPlanNode.from_dict(absent).prompt_kind == "none"

    nulled = _timing_node_dict()
    nulled["prompt_kind"] = None
    with pytest.raises(ValueError, match="prompt_kind"):
        ReplayPlanNode.from_dict(nulled)


def test_built_plan_timing_nodes_carry_neutral_kind(tmp_path: Path) -> None:
    """Plans straight from the builder already serialize the neutral kind.

    Guards against the builder emitting a request-shaped kind on a timing node,
    which would then fail its own loader.
    """

    plan = _build_plan(tmp_path)
    timing = [node for task in plan.tasks for node in task.requests if node.node_type == "timing_dependency"]
    assert timing, "fixture should contain a timing_dependency node"
    for node in timing:
        assert node.prompt_kind == "none"
        assert node.to_dict()["prompt_kind"] == "none"
        assert ReplayPlanNode.from_dict(node.to_dict()).prompt_kind == "none"


def test_timing_node_is_not_rejected_by_validate_loaded_plan(tmp_path: Path) -> None:
    """A plan containing timing nodes still passes cross-field validation.

    Timing nodes have no input target and no context; the request-only checks
    in ``validate_loaded_plan`` must not leak onto them.
    """

    data = _minimal_plan_dict()
    data["tasks"][0]["requests"].append(_timing_node_dict())
    plan = ReplayPlan.from_dict(data)
    validate_loaded_plan(plan)
    assert len(plan.tasks[0].requests) == 2


def test_timing_node_preserves_identity_but_drops_source_only_fields(tmp_path: Path) -> None:
    """A built timing node reloads with its identity intact.

    Actor identity (``actor_id``/``actor_role``/``parent_actor_id``) and
    ``prompt_kind`` are carried, so a reloaded plan can still say whose wait a
    timing node is. Only the two build-time-only fields are dropped:
    ``source_status`` (a planning input for the context-success check) and
    ``prompt_recipe_key`` (locates prompt content a timing node never renders).
    Pinning both halves keeps the carried set from silently shrinking or growing.
    """

    plan = _build_plan(tmp_path)
    timing = [node for task in plan.tasks for node in task.requests if node.node_type == "timing_dependency"]
    assert timing, "fixture should contain a timing_dependency node"

    carried = ("prompt_kind", "actor_id", "actor_role", "parent_actor_id")
    dropped = ("source_status", "prompt_recipe_key")
    for node in timing:
        restored = ReplayPlanNode.from_dict(node.to_dict())
        assert restored.node_type == "timing_dependency"
        for field in carried:
            assert getattr(restored, field) == getattr(node, field)
        for field in dropped:
            assert getattr(restored, field) != getattr(node, field) or getattr(node, field) in {"", "unknown"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_status", "failed"),
        ("prompt_recipe_key", "k1"),
        ("prompt_ref", {"session_id": "s", "turn_index": 0}),
        ("context_after", "k1"),
        ("context_mode", "append"),
        ("planned_input_tokens", 10),
        ("planned_output_tokens", 2),
        ("source_output_tokens", 2),
        ("backend_sampling_seed", 1),
        ("response_validation", "exact_tokens"),
        ("effective_output_tokens", 0),
        ("source_prefix_tokens", 1),
    ],
)
def test_timing_node_rejects_request_only_fields(field: str, value: object) -> None:
    """A timing node carrying a request-only field is rejected, not silently
    discarded.

    ``to_dict`` never writes these for a timing node, so a well-formed artifact
    cannot contain them. Accepting and dropping them would load an artifact
    whose content disagrees with the loaded plan -- including a malformed
    ``actor_role`` that no other check would ever see.
    """

    data = _timing_node_dict()
    data[field] = value
    with pytest.raises(ValueError, match="must not carry request fields"):
        ReplayPlanNode.from_dict(data)


def test_timing_node_accepts_only_its_own_fields() -> None:
    """The positive side of the same contract: the timing surface loads."""

    data = _timing_node_dict()
    node = ReplayPlanNode.from_dict(data)
    # serialized back out, the node carries exactly the timing surface
    assert set(node.to_dict()) == set(data)


def test_timing_node_actor_identity_roundtrips() -> None:
    """Actor identity survives the node round-trip.

    Without it a reloaded plan cannot answer "whose wait is this?", which is
    the structural information the build path records on timing nodes.
    """

    node = ReplayPlanNode.from_dict(_timing_node_dict())
    assert (node.actor_id, node.actor_role, node.parent_actor_id) == ("sub", "subagent", "lead")
    restored = ReplayPlanNode.from_dict(node.to_dict())
    assert (restored.actor_id, restored.actor_role, restored.parent_actor_id) == ("sub", "subagent", "lead")


def test_timing_node_actor_fields_default_when_absent() -> None:
    """Legacy timing nodes without actor fields load with neutral defaults."""

    data = _timing_node_dict()
    for field in ("actor_id", "actor_role", "parent_actor_id"):
        del data[field]
    node = ReplayPlanNode.from_dict(data)
    assert (node.actor_id, node.actor_role, node.parent_actor_id) == ("", "unknown", None)
    # re-serializing materializes the fields so the artifact converges
    assert node.to_dict()["actor_id"] == ""
    assert node.to_dict()["actor_role"] == "unknown"
    assert node.to_dict()["parent_actor_id"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_id", 1),
        ("actor_id", None),
        ("actor_role", "garbage"),
        ("actor_role", None),
        ("parent_actor_id", 1),
    ],
)
def test_timing_node_rejects_malformed_actor_fields(field: str, value: object) -> None:
    """A present-but-malformed identity fails closed instead of being defaulted."""

    data = _timing_node_dict()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        ReplayPlanNode.from_dict(data)


def test_timing_node_actor_defaults_do_not_mask_null() -> None:
    """The absent-key default must not rescue an explicit null ``actor_id``."""

    data = _timing_node_dict()
    data["actor_id"] = None
    with pytest.raises(ValueError, match="actor_id"):
        ReplayPlanNode.from_dict(data)


def test_legacy_timing_node_without_actor_fields_loads(tmp_path: Path) -> None:
    """A whole plan written before actor fields were carried still loads."""

    data = _minimal_plan_dict()
    timing = _timing_node_dict()
    for field in ("prompt_kind", "actor_id", "actor_role", "parent_actor_id"):
        del timing[field]
    data["tasks"][0]["requests"].append(timing)

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = ReplayPlan.load(path)

    node = loaded.tasks[0].requests[-1]
    assert node.node_type == "timing_dependency"
    assert node.prompt_kind == "none"
    assert (node.actor_id, node.actor_role, node.parent_actor_id) == ("", "unknown", None)
