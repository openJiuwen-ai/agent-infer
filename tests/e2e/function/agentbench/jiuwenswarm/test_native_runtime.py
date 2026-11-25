# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Real 0.2.8 dependency checks; opt in with AGENTBENCH_JIUWEN_NATIVE=1."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.marks import test_marks

pytestmark = [
    *test_marks(hardware="cpu"),
    pytest.mark.skipif(
        os.environ.get("AGENTBENCH_JIUWEN_NATIVE") != "1", reason="requires the WorkSwarm 0.2.8 environment"
    ),
]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp("jiuwen-native")
    patch = pytest.MonkeyPatch()
    patch.chdir(root)
    for name, value in {
        "HOME": str(root),
        "JIUWENSWARM_DATA_DIR": str(root / "data"),
        "AGENTBENCH_ROOT_SESSION_ID": "root",
        "AGENTBENCH_MAX_SUBAGENTS": "8",
        "AGENTBENCH_MAX_CONCURRENT_SUBAGENTS": "1",
        "AGENTBENCH_LIMITS_LOG": str(root / "limits.jsonl"),
    }.items():
        patch.setenv(name, value)
    assert version("workswarm") == "0.2.8"
    assert version("openjiuwen") == "0.1.20"
    subprocess.run(
        [str(Path(sys.executable).with_name("jiuwenswarm-init"))],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=120,
    )
    from agentinfer.agentbench.agents.jiuwenswarm import extension, hooks

    yield SimpleNamespace(extension=extension, hooks=hooks, root=root)
    patch.undo()


@pytest.fixture
def backend():
    class Requests(list):
        team = False
        team_members = ("worker",)
        step = 0
        block_children = False
        child_started = threading.Event()
        release = threading.Event()
        file_checks = False
        member_step = 0
        canary = ""
        nested = False
        nested_step = 0
        list_result = ""

    requests = Requests()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            message = {"role": "assistant", "content": "done"}
            finish_reason = "stop"
            identity = json.loads(body["vllm_xargs"]["agentic_context"])
            if requests.block_children and identity["agent_role"] == "subagent":
                requests.child_started.set()
                requests.release.wait(60)
            if requests.team and identity["agent_role"] == "lead":
                actions = [
                    (
                        "build_team",
                        {
                            "display_name": "Test",
                            "team_desc": "Reply done",
                            "leader_display_name": "Leader",
                            "leader_desc": "Coordinate",
                            "enable_hitt": False,
                        },
                    ),
                ]
                actions.extend(
                    (
                        "spawn_teammate",
                        {
                            "member_name": member,
                            "display_name": member.title(),
                            "desc": "Reply done",
                            "prompt": "Reply done and stop.",
                        },
                    )
                    for member in requests.team_members
                )
                actions.append(("send_message", {"to": "worker", "content": "Reply done and stop."}))
                if requests.step < len(actions):
                    name, arguments = actions[requests.step]
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call-{requests.step}",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                requests.step += 1
            elif requests.file_checks and (not requests.team or identity["agent_role"] == "subagent"):
                paths = ["member-result.txt", requests.canary, "escape-link"]
                if requests.member_step < len(paths):
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"write-{requests.member_step}",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {"file_path": paths[requests.member_step], "content": "NATIVE_TEAM_OK\n"}
                                    ),
                                },
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                requests.member_step += 1
            elif requests.nested and identity.get("parent_program_id") == "root":
                if requests.nested_step == 0:
                    name = "subagent_spawn"
                    arguments = {
                        "subagent_type": "worker",
                        "task_description": "Reply done.",
                        "display_name": "Grandchild",
                        "role": "analysis",
                    }
                elif requests.nested_step == 1:
                    content = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "tool")
                    child_id = next(
                        line.partition(": ")[2] for line in content.splitlines() if line.startswith("subagent_id:")
                    )
                    name, arguments = "subagent_wait", {"subagent_ids": [child_id], "timeout_ms": 10000}
                elif requests.nested_step == 2:
                    name, arguments = "subagent_list", {}
                else:
                    requests.list_result = body["messages"][-1]["content"]
                if requests.nested_step < 3:
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"nested-{requests.nested_step}",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                requests.nested_step += 1
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frames = [
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test",
                        "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                    },
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                ]
                for frame in frames:
                    self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                payload = {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()
    thread.join()


def _model(url):
    from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

    return Model(
        ModelClientConfig(client_provider="AgentBenchAffinity", api_base=url, api_key="test"),
        ModelRequestConfig(model="test"),
    )


def test_sdk_sends_identity_and_streaming_usage(native, backend):
    from openjiuwen.core.foundation.llm import UserMessage

    hooks = native.hooks
    url, requests = backend
    model = _model(url)

    async def scenario():
        token = hooks._actor.set(hooks._budget.root)
        try:
            await model.invoke([UserMessage(content="hello")])
            chunks = [chunk async for chunk in model.stream([UserMessage(content="hello")])]
            assert chunks
        finally:
            hooks._actor.reset(token)

    asyncio.run(scenario())
    assert len(requests) == 2
    assert requests[1]["stream_options"]["include_usage"] is True
    for body in requests:
        identity = json.loads(body["vllm_xargs"]["agentic_context"])
        assert identity["task_id"] == identity["program_id"] == "root"
        assert body["agent_hint"]["session_id"] == "root"


def test_real_subagent_creation_inherits_nested_delegation(native, backend):
    from openjiuwen.core.single_agent import AgentCard
    from openjiuwen.harness.factory import create_deep_agent
    from openjiuwen.harness.rails.subagent.subagent_rail import SubagentRail
    from openjiuwen.harness.schema.config import SubAgentConfig

    from agentinfer.agentbench.agents.jiuwenswarm.limits import QuotaExceeded, TaskBudget

    hooks = native.hooks
    hooks._budget = TaskBudget("root", 2, 1)
    spec = SubAgentConfig(agent_card=AgentCard(id="worker", name="worker"), system_prompt="Do the task.")
    parent = create_deep_agent(
        model=_model(backend[0]),
        card=AgentCard(id="parent", name="parent"),
        subagents=[spec],
        add_general_purpose_agent=False,
        enable_subagent_runtime=True,
        enable_read_image_multimodal=False,
    )

    async def scenario():
        token = hooks._actor.set(hooks._budget.root)
        try:
            child = parent.create_subagent("worker", "child")
            assert child.deep_config.subagents == parent.deep_config.subagents
            assert child.find_rails_by_type((SubagentRail,))
            hooks._actor.set(hooks._budget.actors["child"])
            grandchild = child.create_subagent("worker", "grandchild")
            assert grandchild.deep_config.enable_subagent_runtime
            assert hooks._budget.actors["grandchild"].parent.program_id == "child"
            with pytest.raises(QuotaExceeded):
                child.create_subagent("worker", "over-quota")
        finally:
            hooks._actor.reset(token)

    asyncio.run(scenario())


def test_team_sessions_have_distinct_identities_with_one_product_session(native):
    from openjiuwen.agent_teams.harness.team_harness import TeamHarness
    from openjiuwen.agent_teams.schema.team import TeamRole
    from openjiuwen.core.session.agent_team import create_agent_team_session
    from openjiuwen.core.single_agent import AgentCard

    from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget

    hooks = native.hooks
    hooks._budget = TaskBudget("root", 2, 1)
    team = create_agent_team_session(session_id="root", team_id="team")
    sessions = []
    for name, role in (("leader", TeamRole.LEADER), ("one", TeamRole.TEAMMATE), ("two", TeamRole.TEAMMATE)):
        harness = object.__new__(TeamHarness)
        harness._native = SimpleNamespace(card=AgentCard(id=name, name=name))
        harness._role, harness._member_name = role, name
        sessions.append(harness._make_child_session(team))
    actors = [hooks._budget.actors[s.get_cache_identity().cache_id] for s in sessions]
    assert actors[0] is hooks._budget.root
    assert len({a.program_id for a in actors}) == 3
    assert all(s.get_session_id() == "root" for s in sessions)
    assert hooks._budget.launched == 2


def test_real_command_tool_is_sealed(native):
    from jiuwenswarm.agents.harness.common.tools import command_tools

    canary = native.root / "unconfined"
    result = asyncio.run(command_tools.mcp_exec_command.invoke({"command": f"touch {canary}"}))
    assert "mcp_exec_command is disabled" in result
    assert not canary.exists()


def test_native_deep_agent_invokes_instrumented_model(native, backend):
    from openjiuwen.core.single_agent import AgentCard
    from openjiuwen.harness import create_deep_agent

    from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget

    hooks = native.hooks
    hooks._budget = TaskBudget("root", 2, 1, native.root / "invoke-limits.jsonl")
    agent = create_deep_agent(
        model=_model(backend[0]),
        card=AgentCard(id="root", name="root"),
        add_general_purpose_agent=False,
        enable_task_loop=False,
        enable_task_planning=False,
        enable_read_image_multimodal=False,
        enable_skill_discovery=False,
    )
    result = asyncio.run(agent.invoke({"query": "Reply done", "conversation_id": "root"}))
    assert "done" in str(result)
    assert backend[1]
    context = json.loads(backend[1][-1]["vllm_xargs"]["agentic_context"])
    assert context["program_id"] == context["task_id"] == "root"
    rows = [json.loads(line) for line in (native.root / "invoke-limits.jsonl").read_text().splitlines()]
    assert any(row["event"] == "work_start" for row in rows)
    assert hooks._budget.running == 0


def test_nested_wait_and_parallel_tool_work_share_one_slot(native):
    from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget

    hooks = native.hooks

    async def scenario():
        budget = TaskBudget("root", 3, 1)
        hooks._budget = budget
        member = budget.admit("member", budget.root)
        child = budget.admit("child", member)
        grandchild = budget.admit("grandchild", child)

        def ctx(actor, name=""):
            return SimpleNamespace(
                session=SimpleNamespace(get_cache_identity=lambda: SimpleNamespace(cache_id=actor.program_id)),
                inputs=SimpleNamespace(tool_name=name),
            )

        probes = []

        async def leaf(self, context):
            probes.append(hooks.current_actor().program_id)
            return "done"

        leaf_work = hooks._wrap_work(leaf)

        async def child_wait(self, context):
            return await leaf_work(None, ctx(grandchild))

        child_wait = hooks._wrap_work(child_wait, tool=True)

        async def member_wait(self, context):
            return await child_wait(None, ctx(child, "subagent_wait"))

        member_wait = hooks._wrap_work(member_wait, tool=True)
        assert await asyncio.wait_for(member_wait(None, ctx(member, "subagent_wait")), 2) == "done"
        assert probes == ["grandchild"]
        assert budget.running == 0 and budget.peak == 1
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_tool(self, context):
            started.set()
            await finish.wait()

        slow = hooks._wrap_work(slow_tool, tool=True)
        worker = asyncio.create_task(slow(None, ctx(member, "read_file")))
        await started.wait()
        waiter = asyncio.create_task(member_wait(None, ctx(member, "subagent_wait")))
        await asyncio.sleep(0)
        assert member.work == 1 and budget.running == 1
        assert len(probes) == 1
        finish.set()
        await asyncio.wait_for(asyncio.gather(worker, waiter), 2)
        assert probes == ["grandchild", "grandchild"]
        assert budget.running == 0 and budget.peak == 1

    asyncio.run(scenario())


def test_native_subagent_runtime_executes_through_shared_budget(native, backend):
    from openjiuwen.core.session.agent import create_agent_session
    from openjiuwen.core.single_agent import AgentCard
    from openjiuwen.harness.factory import create_deep_agent
    from openjiuwen.harness.schema.config import SubAgentConfig
    from openjiuwen.harness.subagent_runtime.control import SubagentControl

    from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget

    hooks = native.hooks
    backend[1].nested = True
    hooks._budget = TaskBudget("root", 3, 1, native.root / "subagent-limits.jsonl")
    spec = SubAgentConfig(
        agent_card=AgentCard(id="worker", name="worker"), system_prompt="Reply done.", enable_task_loop=False
    )
    parent = create_deep_agent(
        model=_model(backend[0]),
        card=AgentCard(id="parent", name="parent"),
        subagents=[spec],
        add_general_purpose_agent=False,
        enable_task_loop=False,
        enable_subagent_runtime=True,
        enable_read_image_multimodal=False,
    )

    async def scenario():
        session = create_agent_session(session_id="root", card=parent.card)
        control = SubagentControl(parent, "root", parent_session=session)
        token = hooks._actor.set(hooks._budget.root)
        try:
            spawned = await control.spawn("worker", "Reply done.", display_name="worker", role="analysis")
            result = await control.wait([spawned.subagent_id], timeout_ms=10000)
            assert result is not None
            assert backend[1]
            identities = [json.loads(body["vllm_xargs"]["agentic_context"]) for body in backend[1]]
            assert any(identity["parent_program_id"] == spawned.subagent_id for identity in identities)
            waits = [
                message["content"]
                for body in backend[1]
                for message in body["messages"]
                if message["role"] == "tool" and "\nresult:\n" in str(message["content"])
            ]
            assert waits and all("not_found" not in text and "completed" in text for text in waits)
            grandchild_id = next(
                identity["program_id"]
                for identity in identities
                if identity["parent_program_id"] == spawned.subagent_id
            )
            assert grandchild_id in backend[1].list_result
            assert hooks._budget.launched == 2
            assert hooks._budget.peak == 1 and hooks._budget.running == 0
        finally:
            await control.cancel_all("test complete")
            hooks._actor.reset(token)

    asyncio.run(scenario())


@pytest.mark.parametrize("profile", ["agent.code.normal", "team.code.normal"])
def test_task_local_services_start_and_stop_with_real_cli(native, backend, profile, tmp_path):
    from agentinfer.agentbench.agents.jiuwenswarm.instance import JiuwenInstance

    artifacts = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    artifacts.mkdir()
    workspace.mkdir()
    instance = JiuwenInstance(
        executable=Path(sys.executable).with_name("jiuwenswarm"),
        artifact_dir=artifacts,
        workspace=workspace,
        session_id="service-root",
        api_base_url=backend[0].removesuffix("/v1"),
        model="test",
        completion_timeout_seconds=180,
        profile_name=profile,
        max_subagents=3,
        max_concurrent_subagents=1,
    )

    async def scenario():
        try:
            await asyncio.wait_for(instance.start(), 180)
            assert instance._process.returncode is None
            assert instance._jiuwenbox_process.returncode is None
        finally:
            await instance.stop()
        assert instance._process is None and instance._jiuwenbox_process is None
        assert instance._runtime_dir is None
        assert not Path(instance.environment["JIUWENSWARM_RUNTIME_SOCKET"]).exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("profile", ["agent.code.normal", "team.code.normal"])
def test_real_cli_jsonl_round(native, backend, profile, tmp_path):
    from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest
    from agentinfer.agentbench.agents.jiuwenswarm.runner import run_jiuwenswarm
    from agentinfer.agentbench.benchkit.dataset import Task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("test workspace\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=workspace,
        check=True,
    )
    backend[1].team = profile == "team.code.normal"
    backend[1].file_checks = True
    backend[1].canary = str(tmp_path / "outside-workspace.txt")
    (workspace / "escape-link").symlink_to(tmp_path / "symlink-escape.txt")
    request = AgentRunRequest(
        agent_type="jiuwenswarm",
        profile_name=profile,
        task=Task("fixture", "fixture/repo", "HEAD", "Reply done and stop."),
        executable=Path(sys.executable).with_name("jiuwenswarm"),
        model="test",
        api_base_url=backend[0].removesuffix("/v1"),
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="cli-root",
        timeout_seconds=180,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=10,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
        max_subagents=3,
        max_concurrent_subagents=1,
    )
    result = asyncio.run(run_jiuwenswarm(request))
    assert result.outcome is AgentRunOutcome.COMPLETED, (result, str(request.artifact_dir))
    assert backend[1]
    identities = [json.loads(body["vllm_xargs"]["agentic_context"]) for body in backend[1]]
    assert all(identity["task_id"] == "cli-root" for identity in identities)
    if profile == "team.code.normal":
        assert any(identity["agent_role"] == "subagent" for identity in identities)
    assert (workspace / "member-result.txt").read_text() == "NATIVE_TEAM_OK\n"
    assert "member-result.txt" in (request.artifact_dir / "model.patch").read_text()
    assert not (tmp_path / "outside-workspace.txt").exists()
    assert not (tmp_path / "symlink-escape.txt").exists()


def test_child_session_overrides_inherited_parent_context(native):
    from openjiuwen.core.session import _current_session
    from openjiuwen.core.session.agent import create_agent_session

    from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget

    hooks = native.hooks
    hooks._budget = TaskBudget("root", 1, 1)
    child = hooks._budget.admit("child", hooks._budget.root)
    actor_token = hooks._actor.set(hooks._budget.root)
    session_token = _current_session.set(create_agent_session(session_id="child"))
    try:
        assert hooks.current_actor() is child
    finally:
        _current_session.reset(session_token)
        hooks._actor.reset(actor_token)


def test_team_quota_denial_reaches_calling_tool(native, backend, tmp_path):
    from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunRequest
    from agentinfer.agentbench.agents.jiuwenswarm.runner import run_jiuwenswarm
    from agentinfer.agentbench.benchkit.dataset import Task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=workspace,
        check=True,
    )
    url, requests = backend
    requests.team = True
    requests.team_members = ("worker", "worker2")
    request = AgentRunRequest(
        agent_type="jiuwenswarm",
        profile_name="team.code.normal",
        task=Task("quota", "fixture/repo", "HEAD", "Ask a team member to reply done."),
        executable=Path(sys.executable).with_name("jiuwenswarm"),
        model="test",
        api_base_url=url.removesuffix("/v1"),
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="quota-root",
        timeout_seconds=180,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=10,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
        max_subagents=1,
        max_concurrent_subagents=1,
    )

    result = asyncio.run(run_jiuwenswarm(request))
    assert result.outcome is AgentRunOutcome.COMPLETED, result
    tool_replies = [
        message["content"] for body in requests for message in body["messages"] if message["role"] == "tool"
    ]
    assert any("Member spawned: member_name=worker " in reply for reply in tool_replies)
    assert any("AgentBench task launch quota exhausted (1/1)" in reply for reply in tool_replies)
    assert not any("Member spawned: member_name=worker2 " in reply for reply in tool_replies)
    events = [
        json.loads(line) for line in (request.artifact_dir / "jiuwenswarm-agent-limits.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events].count("launch") == 1
    assert any(event["event"] == "quota_denied" and event["program_id"].endswith("_worker2") for event in events)


def test_team_cancel_stops_active_member(native, backend, tmp_path):
    from agentinfer.agentbench.agents.contracts import AgentRunRequest
    from agentinfer.agentbench.agents.jiuwenswarm.runner import run_jiuwenswarm
    from agentinfer.agentbench.benchkit.dataset import Task

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=workspace,
        check=True,
    )
    requests = backend[1]
    requests.team = requests.block_children = True
    request = AgentRunRequest(
        agent_type="jiuwenswarm",
        profile_name="team.code.normal",
        task=Task("cancel", "fixture/repo", "HEAD", "Build a team and wait for its work."),
        executable=Path(sys.executable).with_name("jiuwenswarm"),
        model="test",
        api_base_url=backend[0].removesuffix("/v1"),
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        session_id="cancel-root",
        timeout_seconds=180,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=10,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
        max_subagents=3,
        max_concurrent_subagents=1,
    )

    async def scenario():
        task = asyncio.create_task(run_jiuwenswarm(request))
        try:
            assert await asyncio.to_thread(requests.child_started.wait, 90), str(request.artifact_dir)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 40)
        finally:
            requests.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert (request.artifact_dir / "jiuwenswarm-worker-stop.log").exists()
    assert not (request.artifact_dir / "jiuwenswarm-worker/worker.sock").exists()
