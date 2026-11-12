# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Round-trip and fail-closed tests for loading a Replay plan from disk.

A ``replay-plan.json`` is currently a write-only diagnostic artifact; these
tests pin the new ``ReplayPlan.from_dict`` / ``ReplayPlan.load`` /
``validate_loaded_plan`` contract: a serialized plan reconstructs into a
structurally-equivalent plan, and a corrupted artifact raises instead of
round-tripping silently.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentbench.replay.planner import (
    ReplayPlan,
    ReplayPlanNode,
    build_replay_plan,
    validate_loaded_plan,
)
from agentinfer.agentbench.replay.prompt import PromptBuilder, TokenizerClient
from agentinfer.agentbench.replay.transport import TransportResult
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


@pytest.mark.parametrize("language", ["en", "zh"])
def test_documented_plan_example_roundtrips(language: str) -> None:
    # The reference docs claim the examples re-serialize unchanged; keep that
    # claim true whenever a field is added to to_dict().
    path = Path(__file__).resolve().parents[3] / "docs" / language / "reference" / "replay-plan.example.json"
    plan = ReplayPlan.load(path)
    validate_loaded_plan(plan)
    assert json.dumps(plan.to_dict(), sort_keys=True) == json.dumps(
        json.loads(path.read_text(encoding="utf-8")), sort_keys=True
    )


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
    # task_index and task_id must stay distinct or their own uniqueness checks
    # fire before the duplicate-session one this test is about.
    data["tasks"][1]["task_index"] = 1
    data["tasks"][1]["task_id"] = "replay-000001"
    data["tasks"][1]["requests"][0]["runtime_request_id"] = "rid-2"
    data["tasks"][1]["requests"][0]["source_key"] = "k2"
    data["tasks"][1]["requests"][0]["prompt_recipe_key"] = "k2"
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


@pytest.mark.parametrize(
    "session_id",
    [
        "../../outside",
        "/absolute",
        "nested/dir",
        ".",
        "..",
    ],
)
def test_prompt_reference_rejects_unsafe_session_id(session_id: str) -> None:
    with pytest.raises(ValueError, match="one safe path component"):
        PromptReference.from_dict({"session_id": session_id, "turn_index": 0})


def test_plan_from_dict_rejects_prompt_ref_path_traversal() -> None:
    data = _minimal_plan_dict()
    # A reference escaping the text root would let TraceTextStore.read() open a
    # file outside the bundle once the loaded plan is executed.
    _first_request_node(data)["prompt_ref"] = {"session_id": "../../outside", "turn_index": 0}
    with pytest.raises(ValueError, match="one safe path component"):
        ReplayPlan.from_dict(data)


def test_validate_loaded_plan_accepts_byte_offset_prompt_recipe_key() -> None:
    data = _minimal_plan_dict()
    # AgentX hash-snapshot plans store a byte offset here, not a source key, and
    # the artifact records no prompt shape; the executor picks the reader from
    # the run-time config. A load-time membership check would reject every such
    # plan, so this stays a build-time rule.
    _first_request_node(data)["prompt_recipe_key"] = "4096"
    validate_loaded_plan(ReplayPlan.from_dict(data))


@pytest.mark.parametrize("field", ["source_key", "runtime_request_id", "prompt_recipe_key", "actor_id"])
def test_validate_loaded_plan_rejects_empty_request_identity(field: str) -> None:
    """Identity fields key the executor's prompt namespace and result rows.

    The namespace is ``f"{runtime_session_id}:{prompt_recipe_key}"``, so two
    requests sharing ``""`` would share one tokenizer cache entry and one
    calibrated prompt, and an empty ``actor_id`` makes the same-actor context
    check compare "" against "". Every planning path writes them non-empty, so
    this only rejects hand-built or corrupted artifacts.
    """

    data = _minimal_plan_dict()
    _first_request_node(data)[field] = ""
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


@pytest.mark.parametrize("field", ["source_key", "runtime_request_id"])
def test_validate_loaded_plan_rejects_empty_timing_identity(field: str) -> None:
    data = _minimal_plan_dict()
    data["tasks"][0]["requests"].append(_timing_node("gap"))
    data["tasks"][0]["requests"][-1][field] = ""
    with pytest.raises(ValueError, match=f"{field} must be non-empty"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


@pytest.mark.parametrize("malformed", [["structural"], {"kind": "structural"}])
@pytest.mark.parametrize(
    ("where", "field"),
    [
        ("plan", "schema_version"),
        ("plan", "plan_kind"),
        ("node", "node_type"),
        ("node", "response_validation"),
    ],
)
def test_enum_membership_rejects_unhashable_values(where: str, field: str, malformed: object) -> None:
    """A list or object in an enum field must fail closed as ValueError.

    ``x not in frozenset`` raises TypeError for an unhashable ``x``, and a
    caller documenting "loading raises ValueError" would not catch it.
    """

    data = _minimal_plan_dict()
    target = data if where == "plan" else _first_request_node(data)
    target[field] = malformed
    with pytest.raises(ValueError, match=field):
        ReplayPlan.from_dict(data)


def test_validate_loaded_plan_rejects_duplicate_task_index() -> None:
    data = _minimal_plan_dict()
    second = json.loads(json.dumps(data["tasks"][0]))
    second["runtime_session_id"] = "another-session"
    data["tasks"].append(second)
    # task_id is derived from task_index and becomes the result instance_id, so
    # two tasks sharing an index rewrite each other's results.
    with pytest.raises(ValueError, match="task_index 0 must be unique"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_duplicate_task_id() -> None:
    data = _minimal_plan_dict()
    second = json.loads(json.dumps(data["tasks"][0]))
    second["task_index"] = 1
    second["runtime_session_id"] = "another-session"
    second["requests"][0]["runtime_request_id"] = "rid-2"
    second["requests"][0]["source_key"] = "k2"
    second["requests"][0]["prompt_recipe_key"] = "k2"
    # task_id and task_index are read independently from the artifact, so a
    # distinct index does not imply a distinct id.
    data["tasks"].append(second)
    with pytest.raises(ValueError, match="task_id 'replay-000000' must be unique"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_empty_task_id() -> None:
    data = _minimal_plan_dict()
    data["tasks"][0]["task_id"] = ""
    with pytest.raises(ValueError, match="task_id must be non-empty"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_runtime_request_id_reused_by_another_task() -> None:
    data = _minimal_plan_dict()
    second = json.loads(json.dumps(data["tasks"][0]))
    second["task_id"] = "replay-000001"
    second["task_index"] = 1
    second["runtime_session_id"] = "another-session"
    second["sample_ordinal"] = 1
    second["requests"][0]["source_key"] = "k2"
    second["requests"][0]["prompt_recipe_key"] = "k2"
    # runtime_request_id stays "rid-1"; the per-task check cannot see this.
    data["tasks"].append(second)
    with pytest.raises(ValueError, match="runtime_request_id rid-1 must be unique"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_prefix_exceeding_input() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["planned_input_tokens"] = 1024
    node["source_prefix_tokens"] = 2048
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="source_prefix_tokens must fit the input target"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_accepts_prefix_equal_to_input() -> None:
    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node["planned_input_tokens"] = 1024
    node["source_prefix_tokens"] = 1024
    validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_context_targeting_timing_node() -> None:
    data = _minimal_plan_dict()
    requests = data["tasks"][0]["requests"]
    requests.append(_timing_node("s1-gap"))
    node = _first_request_node(data)
    node["context_mode"] = "append"
    node["context_after"] = "s1-gap"
    plan = ReplayPlan.from_dict(data)
    # A timing node carries no prompt exchange, so this only fails later in the
    # executor; reject it while loading instead.
    with pytest.raises(ValueError, match="context_after must reference a request node"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_rejects_context_targeting_other_actor() -> None:
    data = _minimal_plan_dict()
    requests = data["tasks"][0]["requests"]
    requests.append(_request_node("k2", actor_id="other"))
    node = requests[0]
    node["context_mode"] = "append"
    node["context_after"] = "k2"
    plan = ReplayPlan.from_dict(data)
    with pytest.raises(ValueError, match="context_after must reference the same actor"):
        validate_loaded_plan(plan)


def test_validate_loaded_plan_accepts_same_actor_append_context() -> None:
    data = _minimal_plan_dict()
    requests = data["tasks"][0]["requests"]
    requests.append(_request_node("k2"))
    node = requests[0]
    node["context_mode"] = "append"
    node["context_after"] = "k2"
    validate_loaded_plan(ReplayPlan.from_dict(data))


def _request_node(key: str, *, actor_id: str = "lead") -> dict:
    return {
        "runtime_request_id": f"rid-{key}",
        "source_key": key,
        "node_type": "request",
        "send_after": None,
        "effective_interval_seconds": 0.0,
        "prompt_recipe_key": key,
        "prompt_kind": "lead_main",
        "actor_id": actor_id,
        "actor_role": "lead",
        "parent_actor_id": None,
        "context_after": None,
        "context_mode": "independent",
        "planned_input_tokens": 10,
        "planned_output_tokens": 2,
        "source_output_tokens": 2,
        "backend_sampling_seed": None,
    }


def _timing_node(key: str, *, actor_id: str = "lead") -> dict:
    return {
        "runtime_request_id": f"rid-{key}",
        "source_key": key,
        "node_type": "timing_dependency",
        "send_after": None,
        "effective_interval_seconds": 0.0,
        "effective_duration_seconds": 1.0,
        "prompt_kind": "none",
        "actor_id": actor_id,
        "actor_role": "unknown",
        "parent_actor_id": None,
    }


def test_loaded_plan_drives_the_executor(tmp_path: Path) -> None:
    """A plan written by the runner and reloaded from disk still executes.

    The runner currently executes the in-memory plan and only *writes*
    ``replay-plan.json``, so nothing else proves the loader's output is a valid
    executor input. This drives the real :class:`ReplayExecutor` with a stub
    transport: no backend and no accelerator are involved, but the loaded plan
    goes through prompt preparation, scheduling, interval release and result
    collection exactly as a live run would.
    """

    async def exercise() -> None:
        source = tmp_path / "requests.jsonl"
        _source(source)
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 5, "result_dir": tmp_path / "result", "max_concurrency": 2},
                "backend": {"endpoint": "/v1/chat/completions"},
                "replay": {"trace_path": source, "sample_seed": 9, "prompt_transport": "completion_token_ids"},
            }
        )
        plan = build_replay_plan(config, analyze_replay_trace(source))

        path = tmp_path / "replay-plan.json"
        path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
        loaded = ReplayPlan.load(path)
        validate_loaded_plan(loaded)

        tokenizer = TokenizerClient(config)
        await tokenizer.client.aclose()
        tokenizer.client = httpx.AsyncClient(transport=httpx.MockTransport(_tokens))
        try:
            prompts = PromptBuilder(config, tokenizer)
            await prompts.prepare_tasks(loaded)
            results = await ReplayExecutor(config, loaded, prompts, _StubTransport()).execute()
        finally:
            await tokenizer.close()

        assert len(results) == len(loaded.tasks)
        assert all(result.status == "completed" for result in results)
        expected = {task.runtime_session_id: len(task.requests) for task in loaded.tasks}
        for result in results:
            assert len(result.nodes) == expected[result.runtime_session_id]

    asyncio.run(exercise())


async def _tokens(request) -> httpx.Response:
    return httpx.Response(200, json={"tokens": [10, 20, 30]})


class _StubTransport:
    """Records a successful completion without contacting a backend."""

    async def send(self, task, node, prompt) -> TransportResult:
        return TransportResult(
            True,
            "answer",
            time.monotonic(),
            200,
            None,
            input_tokens=len(prompt.token_ids),
            assistant_token_ids=(42,),
            output_tokens=1,
        )


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


# --- second self-audit round (2026-11-07): the required-key rule, applied to
# --- every nullable field ``to_dict`` writes unconditionally, plus the interval
# --- model's per-mode required fields. B6/B7 named the rule; only the token
# --- targets and ``send_after`` were actually covered.


@pytest.mark.parametrize(
    "field",
    [
        "prompt_recipe_key",
        "prompt_kind",
        "actor_id",
        "actor_role",
        "parent_actor_id",
        "context_after",
        "context_mode",
        "source_output_tokens",
        "backend_sampling_seed",
    ],
)
def test_request_node_rejects_missing_unconditionally_written_key(field: str) -> None:
    """A nullable field written by every ``to_dict`` still needs its key.

    Dropping ``context_after`` or ``send_after`` turns a dependent request into a
    root one, and dropping ``source_output_tokens`` changes the zero-output
    accounting in the run summary. Absence and an explicit null are different
    statements, so only the latter is allowed.
    """

    data = _minimal_plan_dict()
    node = _first_request_node(data)
    del node[field]
    with pytest.raises(ValueError, match=field):
        ReplayPlan.from_dict(data)


@pytest.mark.parametrize("field", ["parent_actor_id", "context_after", "source_output_tokens", "backend_sampling_seed"])
def test_request_node_still_accepts_the_key_present_as_null(field: str) -> None:
    """Requiring presence must not also require a value.

    These four are nullable on the dataclass and null on most real nodes, so the
    guard has to accept an explicit null for all of them. The non-nullable
    fields above are covered by their own value checks and are not affected.
    """

    data = _minimal_plan_dict()
    node = _first_request_node(data)
    node[field] = None
    plan = ReplayPlan.from_dict(data)  # must not raise "key must be present"
    verify = validate_loaded_plan if field == "context_after" else (lambda _plan: None)
    verify(plan)
    assert getattr(plan.tasks[0].requests[0], field) is None


def test_task_rejects_missing_nullable_source_task_id() -> None:
    data = _minimal_plan_dict()
    del data["tasks"][0]["source_task_id"]
    with pytest.raises(ValueError, match="source_task_id"):
        ReplayPlan.from_dict(data)


def test_task_accepts_null_source_task_id() -> None:
    data = _minimal_plan_dict()
    data["tasks"][0]["source_task_id"] = None
    assert ReplayPlan.from_dict(data).tasks[0].source_task_id is None


@pytest.mark.parametrize("field", ["execution_unavailable_reason", "source_bundle_sha256"])
def test_plan_rejects_missing_nullable_plan_field(field: str) -> None:
    data = _minimal_plan_dict()
    del data[field]
    with pytest.raises(ValueError, match=field):
        ReplayPlan.from_dict(data)


def test_interval_model_lognormal_requires_mu_and_sigma() -> None:
    """Mode and parameters must agree, or execution fails mid-run.

    ``timing.effective_interval_seconds`` asserts ``mu``/``sigma`` are not None
    when ``mode == "lognormal"``, so a lognormal artifact without them loaded
    cleanly and then blew up inside the executor instead of at load.
    """

    data = _minimal_plan_dict()
    data["interval_model"] = {"mode": "lognormal", "fit_version": "lognormal-three-quantile-ols"}
    with pytest.raises(ValueError, match="requires both mu and sigma"):
        ReplayPlan.from_dict(data)


@pytest.mark.parametrize("present", [{"mu": 1.0}, {"sigma": 0.5}])
def test_interval_model_lognormal_rejects_half_a_fit(present: dict) -> None:
    data = _minimal_plan_dict()
    data["interval_model"] = {"mode": "lognormal", "fit_version": "v", **present}
    with pytest.raises(ValueError, match="requires both mu and sigma"):
        ReplayPlan.from_dict(data)


def test_interval_model_trace_rejects_mu_and_sigma() -> None:
    """``trace`` samples the recorded gap; carrying a fit means the mode is wrong."""

    data = _minimal_plan_dict()
    data["interval_model"] = {"mode": "trace", "fit_version": "trace-v1", "mu": 0.5, "sigma": 2.0}
    with pytest.raises(ValueError, match="must not carry mu or sigma"):
        ReplayPlan.from_dict(data)


def test_interval_model_lognormal_roundtrips(tmp_path: Path) -> None:
    """The rejection must not also reject a real lognormal artifact."""

    from agentinfer.agentbench.replay.config import ReplayConfig
    from agentinfer.agentbench.replay.timing import build_interval_model

    model = build_interval_model(
        ReplayConfig.model_validate(
            {
                "interval_mode": "lognormal",
                # Exactly consistent with a lognormal (sigma = 0.5), so the
                # builder's 2% anchor-residual guard accepts them.
                "interval_lognormal": {
                    "p50_seconds": 1.0,
                    "p95_seconds": 2.276016608514316,
                    "p99_seconds": 3.200074007942962,
                },
            }
        )
    )
    data = _minimal_plan_dict()
    data["interval_model"] = model.to_dict()
    loaded = ReplayPlan.from_dict(data).interval_model
    assert loaded == model
    assert loaded.mu is not None and loaded.sigma is not None


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


# --- Field contract -------------------------------------------------------
#
# Every review round so far found the same shape of defect: a field that
# ``to_dict`` writes, that a consumer reads, and that the loader therefore has
# to constrain -- but nobody enumerated the fields, so each round fixed the one
# the reviewer named. The tables below are that enumeration, and the tests keep
# them honest in both directions:
#
#   * a field added to ``to_dict`` without a classification fails the coverage
#     test, so it cannot slip in unexamined;
#   * a field classified as "mutating it must be rejected" that stops being
#     rejected fails the mutation test, so a check cannot be lost silently.
#
# The classification is deliberately not "reject everything". Fields the loader
# does not constrain are split by *why*, because "no check" means three
# different things and collapsing them hides the two that still matter at run
# time:
#
#   * ``annotation`` -- pure description of the artifact. Nothing reads it after
#     load, so any well-formed value is genuinely harmless.
#   * ``annotation-execution`` -- the loader cannot cross-check it against the
#     original source, but the executor *does* send it, so a changed value
#     changes the run. Type and range are still checked where the field has
#     them; fidelity to the source is not checkable from the artifact alone.
#   * ``annotation-provenance`` -- a record of where the plan came from. It
#     feeds the run summary rather than the request, so a changed value changes
#     what the run reports but not what it does.
#
# Every non-constrained class still loads with an arbitrary well-formed value;
# the distinction is documentation plus the grouping test below, which keeps a
# field from being filed as "pure annotation" when something consumes it.

_PLAN_FIELDS = {
    # Rejected when mutated: the loader constrains them.
    "schema_version": "enum",
    "plan_kind": "enum",
    "execution_ready": "paired with execution_unavailable_reason",
    "execution_unavailable_reason": "paired with execution_ready",
    "source_sha256": "non-empty",
    "source_sessions": "non-negative",
    "replayable_source_sessions": "<= source_sessions",
    "plan_namespace": "non-empty",
    "workload_fingerprint": "non-empty",
    "prompt_calibration": "known status",
    "interval_model": "per-field typed",
    "tasks": "non-empty",
    # Accepted as written: annotation only, no consumer keys off them.
    "planner_version": "annotation",
    "source": "annotation",
    "source_bundle_sha256": "annotation",
    "sample_seed": "annotation (negative seeds are valid)",
}

_TASK_FIELDS = {
    "task_id": "non-empty, unique across the plan",
    "task_index": "non-negative, unique across the plan",
    "runtime_session_id": "non-empty, unique across the plan",
    "source_session_id": "paired with sample_ordinal, unique across the plan",
    "sample_ordinal": "paired with source_session_id, unique across the plan",
    "requests": "non-empty",
    "source_task_id": "annotation",
}

_REQUEST_NODE_FIELDS = {
    "runtime_request_id": "non-empty, unique across the plan",
    "source_key": "non-empty, unique within the task",
    "node_type": "enum",
    "send_after": "reference must resolve",
    "context_after": "reference must resolve to a same-actor request",
    "effective_interval_seconds": "finite, non-negative",
    "prompt_recipe_key": "non-empty",
    "prompt_kind": "enum, not 'none'",
    "actor_id": "non-empty, must match context_after target",
    "actor_role": "enum",
    "context_mode": "enum, paired with context_after",
    "planned_input_tokens": "positive",
    "planned_output_tokens": "non-negative",
    "source_prefix_tokens": "<= planned_input_tokens",
    "prompt_ref": "parsed; one safe path component",
    "source_output_tokens": "annotation-provenance (counted in the run summary)",
    "backend_sampling_seed": "annotation-execution (sent as the backend sampling seed)",
    "response_validation": "enum",
    "parent_actor_id": "annotation-execution (sent as the parent-agent header)",
    "effective_output_tokens": "annotation; ignored on load",
}

_TIMING_NODE_FIELDS = {
    "runtime_request_id": "non-empty, unique across the plan",
    "source_key": "non-empty, unique within the task",
    "node_type": "enum",
    "send_after": "reference must resolve",
    "effective_interval_seconds": "finite, non-negative",
    "effective_duration_seconds": "finite, non-negative",
    "prompt_kind": "must be 'none'",
    "actor_id": "annotation-provenance (whose wait this is)",
    "actor_role": "annotation-provenance",
    "parent_actor_id": "annotation-provenance",
}


# Fields `to_dict` writes only when they carry a value. They legitimately
# disappear from an artifact, so the coverage test cannot demand them in a
# builder-produced sample. Only the names live here; their constraints stay in
# the table above so a field never carries two classifications -- enforced by
# ``test_conditional_fields_are_classified`` below, which is what stops a field
# from being excused here and never classified at all.
_CONDITIONAL_REQUEST_NODE_FIELDS = frozenset(
    {"prompt_ref", "response_validation", "source_prefix_tokens", "effective_output_tokens"}
)


def _serialized_field_levels(tmp_path: Path) -> dict[str, set[str]]:
    """Per-level key sets from a builder-produced plan."""

    plan = _build_plan(tmp_path)
    data = plan.to_dict()
    requests = [node for task in plan.tasks for node in task.requests]
    request_node = next(node for node in requests if node.node_type == "request")
    timing_nodes = [node for node in requests if node.node_type == "timing_dependency"]
    assert timing_nodes, "the shared fixture must contain a timing node or this level is untested"
    return {
        "plan": set(data),
        "task": set(data["tasks"][0]),
        "request_node": set(request_node.to_dict()),
        "timing_node": set(timing_nodes[0].to_dict()),
    }


@pytest.mark.parametrize(
    ("table", "conditional", "level"),
    [
        (_PLAN_FIELDS, set(), "plan"),
        (_TASK_FIELDS, set(), "task"),
        (_REQUEST_NODE_FIELDS, set(_CONDITIONAL_REQUEST_NODE_FIELDS), "request_node"),
        (_TIMING_NODE_FIELDS, set(), "timing_node"),
    ],
)
def test_every_serialized_field_is_classified(tmp_path: Path, table: dict, conditional: set[str], level: str) -> None:
    """``to_dict`` output and the contract table must name the same fields.

    Adding a field to ``to_dict`` without deciding what loading guarantees
    about it fails here. That is the point: the previous rounds of review each
    found one such field, and the cost was a review round apiece.
    """

    serialized = _serialized_field_levels(tmp_path)[level]
    classified = set(table) | conditional
    assert serialized - classified == set(), (
        f"{level}: to_dict writes {sorted(serialized - classified)} but the contract table does not classify them"
    )
    assert classified - conditional - serialized == set(), (
        f"{level}: the contract table classifies {sorted(classified - conditional - serialized)}"
        " which to_dict never writes"
    )


# --- Field contract, second half: the constraint must actually fire ----------
#
# The coverage test above says every serialized field has a stated constraint.
# This one says the constraint is real: a well-formed but wrong value must be
# rejected, and a field classified as pure annotation must still load with any
# well-formed value, so the documented boundary is pinned in both directions
# rather than asserted in a comment.

_FIELD_TABLES = {
    "plan": _PLAN_FIELDS,
    "task": _TASK_FIELDS,
    "request_node": _REQUEST_NODE_FIELDS,
    "timing_node": _TIMING_NODE_FIELDS,
}

# (level, field, mutated value). Values are chosen to be *well-formed* wherever
# the classification is "annotation" -- the question there is whether load
# accepts an arbitrary legal value, not whether it type-checks. Every table field
# must appear here exactly once; the coverage test below enforces that, so a new
# classification cannot be added without a mutation that proves it fires.
# ``effective_output_tokens`` is the one field with no constraint to fire: load
# deliberately ignores it (``planned_output_tokens`` is authoritative), which is
# what "annotation" means here.
_FIELD_MUTATIONS = [
    ("plan", "schema_version", "3"),
    ("plan", "plan_kind", "adaptive"),
    ("plan", "execution_ready", True),
    ("plan", "execution_unavailable_reason", None),
    ("plan", "source_sha256", ""),
    ("plan", "source_sessions", -1),
    ("plan", "replayable_source_sessions", 2),
    ("plan", "plan_namespace", ""),
    ("plan", "workload_fingerprint", ""),
    ("plan", "prompt_calibration", {"status": "half-done"}),
    ("plan", "interval_model", {"mode": "lognormal", "fit_version": "agentinfer-replay-lognormal"}),
    ("plan", "tasks", []),
    ("plan", "planner_version", "agentinfer-replay-other"),
    ("plan", "source", "/somewhere/else.jsonl"),
    ("plan", "source_bundle_sha256", "b" * 64),
    ("plan", "sample_seed", -7),
    ("task", "task_id", ""),
    ("task", "task_index", -1),
    ("task", "runtime_session_id", ""),
    ("task", "source_session_id", ""),
    ("task", "sample_ordinal", -1),
    ("task", "requests", []),
    ("task", "source_task_id", "another-task"),
    ("request_node", "runtime_request_id", ""),
    ("request_node", "source_key", ""),
    ("request_node", "node_type", "sleep"),
    ("request_node", "send_after", "missing"),
    ("request_node", "context_after", "missing"),
    ("request_node", "effective_interval_seconds", -1.0),
    ("request_node", "prompt_recipe_key", ""),
    ("request_node", "prompt_kind", "none"),
    ("request_node", "actor_id", ""),
    ("request_node", "actor_role", "worker"),
    ("request_node", "context_mode", "none"),
    ("request_node", "planned_input_tokens", 0),
    ("request_node", "planned_output_tokens", -1),
    ("request_node", "source_prefix_tokens", 11),
    ("request_node", "prompt_ref", {"session_id": "../escape", "turn_index": 0}),
    ("request_node", "source_output_tokens", 999),
    ("request_node", "backend_sampling_seed", 999),
    ("request_node", "response_validation", "fuzzy"),
    ("request_node", "parent_actor_id", "someone-else"),
    ("request_node", "effective_output_tokens", 5),
    ("timing_node", "runtime_request_id", ""),
    ("timing_node", "source_key", ""),
    ("timing_node", "node_type", "sleep"),
    ("timing_node", "send_after", "missing"),
    ("timing_node", "effective_interval_seconds", -1.0),
    ("timing_node", "effective_duration_seconds", -1.0),
    ("timing_node", "prompt_kind", "lead_main"),
    ("timing_node", "actor_id", ""),
    ("timing_node", "actor_role", "lead"),
    ("timing_node", "parent_actor_id", "someone-else"),
]


def _mutated_plan(level: str, field: str, value: object) -> dict:
    """A minimal plan with one field of one level replaced."""

    data = _minimal_plan_dict()
    if level == "plan":
        target = data
    elif level == "task":
        target = data["tasks"][0]
    elif level == "request_node":
        target = _first_request_node(data)
    elif level == "timing_node":
        target = _timing_node_dict()
        data["tasks"][0]["requests"].append(target)
    else:  # pragma: no cover - guards the table above
        raise AssertionError(f"unknown level {level!r}")
    target[field] = value
    return data


@pytest.mark.parametrize(("level", "field", "value"), _FIELD_MUTATIONS)
def test_serialized_field_constraint_matches_its_classification(level: str, field: str, value: object) -> None:
    """A constrained field rejects a wrong value; an annotation field accepts one."""

    data = _mutated_plan(level, field, value)
    classification = _FIELD_TABLES[level][field]
    if classification.startswith("annotation"):
        validate_loaded_plan(ReplayPlan.from_dict(data))
        return
    with pytest.raises(ValueError):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_every_classified_field_has_a_mutation() -> None:
    """Each classification above is pinned by a mutation that exercises it.

    Without this, a constraint added to the tables would be a claim no test
    checks -- exactly the drift that made earlier review rounds necessary.
    """

    mutated = {(level, field) for level, field, _ in _FIELD_MUTATIONS}
    classified = {(level, field) for level, table in _FIELD_TABLES.items() for field in table}
    assert classified - mutated == set(), f"classified but never mutated: {sorted(classified - mutated)}"
    assert mutated - classified == set(), f"mutated but never classified: {sorted(mutated - classified)}"


def test_conditional_fields_are_classified() -> None:
    """A field excused from the coverage test must still be classified.

    ``_CONDITIONAL_REQUEST_NODE_FIELDS`` removes a name from the "must appear in
    a builder sample" requirement. Without this check, adding a field there and
    nowhere else leaves it wholly unclassified while every other test still
    passes -- the exact hole that lets a serialized field escape the contract.
    """

    unclassified = _CONDITIONAL_REQUEST_NODE_FIELDS - set(_REQUEST_NODE_FIELDS)
    assert unclassified == set(), f"conditional but never classified: {sorted(unclassified)}"


# --- Built artifacts, not just the hand-built minimal dict -------------------
#
# Everything above mutates a hand-written dict. A builder-produced artifact is
# the only thing that proves the *real* shapes load: a schema-2 plan written by
# the TraceLab or AgentX paths carries conditional fields the minimal dict never
# sets, so a constraint that fires on them would be invisible above. These
# build through the real converter, save, reload, and compare field by field.


def _trace_ir_roundtrip_plan(tmp_path: Path, kind: str) -> ReplayPlan:
    from agentinfer.agentbench.replay.converters import TraceLabConverter
    from agentinfer.agentbench.replay.converters.agentx import AgentXConverter, analyze_agentx_trace_ir
    from agentinfer.agentbench.replay.unified_trace_ir import (
        analyze_explicit_trace_ir,
        validate_trace_ir,
    )

    source = tmp_path / f"{kind}-source.jsonl"
    output = tmp_path / f"{kind}-converted"
    if kind == "tracelab":
        rows = [
            {
                "provider": "codex",
                "project": "fixture",
                "session_id": "session",
                "round_index": index,
                "round_id": f"round-{index}",
                "model": "source-model",
                "input_tokens_total": input_tokens,
                "prefix_tokens": prefix_tokens,
                "newly_append_tokens": input_tokens - prefix_tokens,
                "output_tokens": 7 + index,
                "reasoning_output_tokens": 1,
                "timing_events": [
                    {"event_type": "user_message", "timestamp": f"2026-01-01T00:00:0{index}Z"},
                    {"event_type": "text", "timestamp": f"2026-01-01T00:00:0{index + 1}Z"},
                ],
                "tools": [{"is_error": True}],
            }
            for index, (input_tokens, prefix_tokens) in enumerate([(30, 20), (35, 29)])
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        TraceLabConverter().convert(source, output)
        trace_ir = validate_trace_ir(output / "requests.jsonl")
        analysis = analyze_explicit_trace_ir(trace_ir)
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_num": 1}, "replay": {"trace_type": "tracelab", "trace_path": source}}
        )
    else:
        source.write_text(
            json.dumps(
                {
                    "id": "source-session",
                    "models": ["model-a"],
                    "block_size": 64,
                    "hash_id_scope": "local",
                    "requests": [
                        {"t": 0, "api_time": 4, "in": 64, "out": 3, "hash_ids": [0], "model": "model-a", "type": "n"},
                        {
                            "t": 1,
                            "api_time": 1,
                            "in": 128,
                            "out": 3,
                            "hash_ids": [0, 1],
                            "model": "model-a",
                            "type": "n",
                        },
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        AgentXConverter().convert(source, output)
        trace_ir = validate_trace_ir(output / "requests.jsonl")
        analysis = analyze_agentx_trace_ir(trace_ir)
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend", "model": "served-model", "endpoint": "/v1/completions"},
                "replay": {"trace_type": "agentX", "trace_path": source},
            }
        )
    return build_replay_plan(config, analysis, trace_ir)


@pytest.mark.parametrize("kind", ["tracelab", "agentx"])
def test_builder_produced_schema2_plan_loads_unchanged(tmp_path: Path, kind: str) -> None:
    """A real schema-2 artifact survives save/load with every field intact.

    The hand-built fixtures above are schema 1. The schema-2 shapes set fields
    the minimal dict leaves at their defaults -- TraceLab's zero output target
    for a warmup request in particular -- so this is the case that would catch a
    load-time rule that rejects a plan the builder itself wrote.
    """

    plan = _trace_ir_roundtrip_plan(tmp_path, kind)
    assert plan.schema_version == "2"

    path = tmp_path / f"{kind}-plan.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    loaded = ReplayPlan.load(path)
    validate_loaded_plan(loaded)

    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(plan.to_dict(), sort_keys=True)


# --- Review round 2026-11-12 -------------------------------------------------


def test_validate_loaded_plan_rejects_timing_node_sharing_a_request_id() -> None:
    """A timing node and a request may not share an id, even across tasks.

    The executor removes per-node preparation state by id at task end and the
    tokenizer cancels in-flight work by id, so a shared id deletes the other
    node's records. The check used to sit inside the ``node_type == "request"``
    branch, so a timing node slipped past it.
    """

    data = _minimal_plan_dict()
    second = json.loads(json.dumps(data["tasks"][0]))
    second.update(
        {
            "task_id": "replay-000001",
            "task_index": 1,
            "sample_ordinal": 1,
            "runtime_session_id": "sess-2",
            "source_session_id": "s2",
        }
    )
    second["requests"][0].update({"source_key": "k2", "runtime_request_id": "rid-2", "prompt_recipe_key": "k2"})
    data["tasks"].append(second)

    timing = _timing_node_dict()
    timing["runtime_request_id"] = "rid-1"
    timing["source_key"] = "wait-1"
    second["requests"].append(timing)

    with pytest.raises(ValueError, match="timing_dependency runtime_request_id rid-1 must be unique"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


@pytest.mark.parametrize("status", [[], {}])
def test_validate_loaded_plan_rejects_non_string_calibration_status(status: object) -> None:
    """A non-string status raises ``ValueError``, as the docstring promises.

    A membership test on an unhashable value raises ``TypeError`` instead, which
    a caller catching the documented ``ValueError`` would not see.
    """

    data = _minimal_plan_dict()
    data["prompt_calibration"]["status"] = status
    with pytest.raises(ValueError, match="prompt_calibration.status must be one of"):
        validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_rejects_zero_output_in_a_schema1_plan() -> None:
    """Schema 1 never carries a zero output target; schema 2 legitimately does."""

    data = _minimal_plan_dict()
    _first_request_node(data)["planned_output_tokens"] = 0
    with pytest.raises(ValueError, match="positive output target in a schema-1 plan"):
        validate_loaded_plan(ReplayPlan.from_dict(data))

    data["schema_version"] = "2"
    validate_loaded_plan(ReplayPlan.from_dict(data))


def _long_chain(count: int, *, reverse: bool) -> dict:
    data = _minimal_plan_dict()
    base = _first_request_node(data)
    nodes = []
    for index in range(count):
        node = json.loads(json.dumps(base))
        node.update(
            {
                "source_key": f"k{index}",
                "runtime_request_id": f"rid-{index}",
                "prompt_recipe_key": f"k{index}",
                "send_after": f"k{index - 1}" if index else None,
            }
        )
        nodes.append(node)
    data["tasks"][0]["requests"] = list(reversed(nodes)) if reverse else nodes
    return data


@pytest.mark.parametrize("reverse", [False, True])
def test_validate_loaded_plan_handles_a_long_dependency_chain(reverse: bool) -> None:
    """Node order in the artifact must not decide whether a plan loads.

    The acyclic check recursed once per edge, so a valid 1000-request chain
    loaded in build order and raised ``RecursionError`` when the same chain was
    serialized in reverse. The traversal is iterative now.
    """

    data = _long_chain(1000, reverse=reverse)
    validate_loaded_plan(ReplayPlan.from_dict(data))


def test_validate_loaded_plan_still_rejects_a_cycle_in_a_long_chain() -> None:
    """The iterative traversal keeps detecting cycles, not just avoiding the crash."""

    data = _long_chain(1000, reverse=True)
    data["tasks"][0]["requests"][-1]["send_after"] = "k999"  # k999 -> ... -> k0 -> k999
    with pytest.raises(ValueError, match="dependency cycle detected"):
        validate_loaded_plan(ReplayPlan.from_dict(data))
