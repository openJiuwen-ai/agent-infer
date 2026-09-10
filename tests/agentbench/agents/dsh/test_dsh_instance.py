# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify task-local DSH home and settings preparation."""

import json
import os
import subprocess
from pathlib import Path

import yaml

from agentinfer.agentbench.agents.dsh.instance import DshInstance


def _instance(tmp_path: Path) -> DshInstance:
    return DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180/",
        model="spike-model",
        session_id="root-session",
    )


def test_bootstrap_writes_isolated_home_and_settings(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    instance.bootstrap()

    assert instance.home_dir == tmp_path / "artifacts" / "dsh-home"
    patch = instance.policy_patch_path.read_text(encoding="utf-8")
    assert patch.startswith("- id: session-title-llm\n  disabled: true\n")
    assert "forcePlanMode: false" in patch
    assert "autoApprove: false" in patch
    assert instance.settings_path.is_file()
    settings = yaml.safe_load(instance.settings_path.read_text(encoding="utf-8"))
    assert settings["agent-default-model"] == {"provider": "deepseek-official", "model": "spike-model"}
    route = settings["llm-deepseek"]
    assert route["apiKeyEnv"] == "DEEPSEEK_API_KEY"
    assert route["baseURL"] == "http://127.0.0.1:18180/v1"
    assert route["thinking"] == "disabled"
    assert route["retryPolicy"] == {"mode": "normal", "maxRetries": 0}
    assert [model["id"] for model in route["models"]] == ["spike-model"]


def test_bootstrap_writes_settings_snapshot(tmp_path: Path) -> None:
    instance = DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180",
        model="spike-model",
        session_id="root-session-abc",
    )
    instance.bootstrap()

    assert (tmp_path / "artifacts" / "settings.yaml").read_text(encoding="utf-8") == instance.settings_path.read_text(
        encoding="utf-8"
    )
    snapshot = json.loads(instance.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["settings"] == instance.settings
    assert snapshot["environment"]["AGENTBENCH_ROOT_SESSION_ID"] == "root-session-abc"


def test_bootstrap_writes_profile_policy_and_records_it_in_snapshot(tmp_path: Path) -> None:
    policy_patch = "- id: tool-subagent\n  disabled: true\n"
    instance = DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180",
        model="spike-model",
        session_id="root-session",
        policy_patch=policy_patch,
    )
    instance.bootstrap()

    patch = instance.policy_patch_path.read_text(encoding="utf-8")
    assert patch.startswith("- id: session-title-llm\n  disabled: true\n" + policy_patch)
    assert "forcePlanMode: false" in patch
    snapshot = json.loads(instance.snapshot_path.read_text(encoding="utf-8"))
    assert "agentbench-bridge" in snapshot["policy_patch"]


def test_bridge_mounts_lineage_for_every_profile(tmp_path: Path) -> None:
    for enforce_plan_mode in (False, True):
        instance = DshInstance(
            artifact_dir=tmp_path / str(enforce_plan_mode),
            api_base_url="http://127.0.0.1:18180",
            model="spike-model",
            session_id="root-session",
            enforce_plan_mode=enforce_plan_mode,
        )
        instance.bootstrap()

        assert instance.bridge_path.is_file()
        source = instance.bridge_path.read_text(encoding="utf-8")
        assert "AsyncLocalStorage" in source
        assert "llm/stream" in source
        assert "agentic_context" in source
        assert "q.options" in source
        patch = instance.policy_patch_path.read_text(encoding="utf-8")
        bridge_name = str(instance.bridge_path).replace("\\", "/")
        assert f"name: {bridge_name}" in patch
        assert f"forcePlanMode: {str(enforce_plan_mode).lower()}" in patch
        assert f"autoApprove: {str(enforce_plan_mode).lower()}" in patch


def test_bridge_injects_concurrent_session_lineage(tmp_path: Path) -> None:
    instance = DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180",
        model="spike-model",
        session_id="root-session",
    )
    instance.bootstrap()
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        """\
const { apply } = await import(process.argv[2])
async function collect(stream) {
  const items = []
  for await (const item of stream) items.push(item)
  return items
}

const handlers = new Map()
const ctx = {
  get: () => undefined,
  logger: { warn: () => undefined },
  on(event, handler) {
    handlers.set(event, handler)
    return () => handlers.delete(event)
  },
}
const sent = []
globalThis.fetch = async (request) => {
  sent.push(JSON.parse(await request.text()))
  return new Response('{}', { status: 200 })
}
const dispose = apply(ctx)
const created = handlers.get('agent/created')
created({ agent: { session: { header: { id: 'root-dsh', delegationDepth: 0 } } } })
created({ agent: { session: { header: { id: 'child-dsh', parentSession: 'root-dsh', delegationDepth: 1 } } } })
const stream = handlers.get('llm/stream')
async function* request() {
  await fetch('http://proxy/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [] }),
  })
  yield { type: 'done' }
}
await Promise.all([
  collect(stream({ sessionId: 'root-dsh' }, request)),
  collect(stream({ sessionId: 'child-dsh' }, request)),
])
dispose()
console.log(JSON.stringify(sent.map((body) => JSON.parse(body.vllm_xargs.agentic_context))))
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(harness), instance.bridge_path.as_uri()],
        capture_output=True,
        check=True,
        text=True,
        env=os.environ | {"AGENTBENCH_ROOT_SESSION_ID": "root-session"},
    )
    contexts = {context["session_id"]: context for context in json.loads(result.stdout)}

    assert contexts["root-dsh"] == {
        "program_id": "root-dsh",
        "task_id": "root-session",
        "session_id": "root-dsh",
        "agent_id": "root-dsh",
        "blocks_parent": False,
        "expected_resume": True,
        "agent_role": "lead",
    }
    assert contexts["child-dsh"] == {
        "program_id": "child-dsh",
        "task_id": "root-session",
        "session_id": "child-dsh",
        "agent_id": "child-dsh",
        "parent_program_id": "root-dsh",
        "blocks_parent": True,
        "expected_resume": False,
        "agent_role": "subagent",
    }


def test_bridge_preserves_request_body_and_fetch_wrapper_stack(tmp_path: Path) -> None:
    instance = DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180",
        model="spike-model",
        session_id="root-session",
    )
    instance.bootstrap()
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        """\
const { apply } = await import(process.argv[2])
async function collect(stream) {
  const items = []
  for await (const item of stream) items.push(item)
  return items
}

const handlers = []
const ctx = {
  get: () => undefined,
  logger: { warn: () => undefined },
  on(event, handler) {
    handlers.push([event, handler])
    return () => undefined
  },
}
const originalFetch = async (request) => {
  const payload = await request.json()
  return new Response(JSON.stringify(payload), { status: 200 })
}
globalThis.fetch = originalFetch
const firstDispose = apply(ctx)
const firstFetch = globalThis.fetch
const secondDispose = apply(ctx)
const secondFetch = globalThis.fetch
const created = handlers.find(([event]) => event === 'agent/created')[1]
created({ agent: { session: { header: { id: 'root-dsh', delegationDepth: 0 } } } })
const stream = handlers.find(([event]) => event === 'llm/stream')[1]
async function* request() {
  const response = await fetch('http://proxy/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] }),
  })
  console.log(await response.text())
  yield { type: 'done' }
}
await collect(stream({ sessionId: 'root-dsh' }, request))
firstDispose()
const afterFirstDispose = globalThis.fetch
secondDispose()
const afterSecondDispose = globalThis.fetch
console.log(JSON.stringify({
  firstFetchInstalled: firstFetch !== originalFetch,
  secondFetchInstalled: secondFetch !== firstFetch,
  firstDisposePreservedSecond: afterFirstDispose === secondFetch,
  secondDisposeRestoredOriginal: afterSecondDispose === originalFetch,
}))
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(harness), instance.bridge_path.as_uri()],
        capture_output=True,
        check=True,
        text=True,
        env=os.environ | {"AGENTBENCH_ROOT_SESSION_ID": "root-session"},
    )
    payload, wrappers = result.stdout.splitlines()

    rewritten = json.loads(payload)
    assert rewritten["messages"] == [{"role": "user", "content": "hello"}]
    assert json.loads(rewritten["vllm_xargs"]["agentic_context"])["session_id"] == "root-dsh"
    assert json.loads(wrappers) == {
        "firstFetchInstalled": True,
        "secondFetchInstalled": True,
        "firstDisposePreservedSecond": True,
        "secondDisposeRestoredOriginal": True,
    }


def test_environment_isolates_home_and_forces_autonomy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DSH_HOME", raising=False)
    instance = _instance(tmp_path)

    env = instance.environment()

    assert env["DSH_HOME"] == str(tmp_path / "artifacts" / "dsh-home")
    assert env["DSH_PERMISSION_MODE"] == "workspace-write"
    assert env["DEEPSEEK_API_KEY"] == "agentbench"
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:18180/v1"
    assert env["AGENTBENCH_ROOT_SESSION_ID"] == "root-session"


def test_bridge_fails_closed_without_plan_enforcement_services(tmp_path: Path) -> None:
    instance = DshInstance(
        artifact_dir=tmp_path / "artifacts",
        api_base_url="http://127.0.0.1:18180",
        model="spike-model",
        session_id="root-session",
        enforce_plan_mode=True,
    )
    instance.bootstrap()
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        """\
const { apply } = await import(process.argv[2])
async function collect(stream) {
  const items = []
  for await (const item of stream) items.push(item)
  return items
}

const outcomes = []
const services = new Map()
const handlers = new Map()
function runScenario(name, ctx, created) {
  try {
    const dispose = apply(ctx, { forcePlanMode: true, autoApprove: true })
    created()
    dispose()
    outcomes.push([name, 'ok'])
  } catch (error) {
    outcomes.push([name, error.message])
  }
}
const base = {
  logger: { warn: () => undefined },
  get: (name) => services.get(name),
  on: (_event, handler) => { handlers.set(_event, handler); return () => handlers.delete(_event) },
}
// userQuestions is missing: autoApprove fails at plugin apply time.
runScenario('no-user-questions', base, () => undefined)
services.set('userQuestions', { registerProvider: () => undefined })
// planMode is missing: forcePlanMode vetoes agent publication at created time.
const created = () => handlers.get('agent/created')({ agent: { session: { header: { id: 'lead', delegationDepth: 0 } } } })
runScenario('no-plan-mode', base, created)
services.set('planMode', { set: () => { throw new Error('set boom') } })
runScenario('set-throws', base, created)
services.set('planMode', { set: () => 'queued' })
runScenario('both-present', base, created)
console.log(JSON.stringify(outcomes))
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(harness), instance.bridge_path.as_uri()],
        capture_output=True,
        check=True,
        text=True,
        env=os.environ | {"AGENTBENCH_ROOT_SESSION_ID": "root-session"},
    )
    outcomes = json.loads(result.stdout)

    assert outcomes == [
        [
            "no-user-questions",
            "agentbench-bridge: userQuestions service unavailable; cannot auto-approve the plan review",
        ],
        ["no-plan-mode", "agentbench-bridge: planMode service unavailable; cannot enforce plan mode"],
        ["set-throws", "set boom"],
        ["both-present", "ok"],
    ]


def test_bridge_skips_plan_enforcement_without_the_flag(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    instance.bootstrap()
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        """\
const { apply } = await import(process.argv[2])
async function collect(stream) {
  const items = []
  for await (const item of stream) items.push(item)
  return items
}

const ctx = {
  logger: { warn: () => undefined },
  get: () => undefined,
  on: () => () => undefined,
}
apply(ctx)
console.log('ok')
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(harness), instance.bridge_path.as_uri()],
        capture_output=True,
        check=True,
        text=True,
        env=os.environ | {"AGENTBENCH_ROOT_SESSION_ID": "root-session"},
    )

    assert result.stdout.strip() == "ok"
