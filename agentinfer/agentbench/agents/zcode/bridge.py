# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Own task identity and synchronous ZCode hook policy before tools execute."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager, contextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from ..contracts import AgentRunRequest, TerminationReason
from .profiles import EXPLORE_TOOL_ROUNDS, explore_name, get_profile

READ_TOOLS = {"Read", "Glob", "Grep"}
AGENT_TOOLS = {"Agent", "Task"}


def wire_session(session_id: str) -> str:
    return session_id.removeprefix("sess_").removeprefix("subagent_agent_")


class ZCodeBridge:
    def __init__(self, request: AgentRunRequest):
        self.request = request
        self.planning = get_profile(request.profile_name).planning
        self.sessions: dict[str, dict] = {}
        self.pending: set[str] = set()
        self.reports = 0
        self.child_tool_rounds: dict[str, int] = {}
        self.approved = False
        self.failure: TerminationReason | None = None

    def register(self, session_id: str) -> dict:
        key = wire_session(session_id)
        if key not in self.sessions:
            child = session_id.startswith("sess_subagent_")
            row = {
                "task_id": self.request.session_id,
                "session_id": session_id,
                "agent_id": key if child else "lead",
                "program_id": f"{self.request.session_id}:{key if child else 'lead'}",
                "agent_role": "subagent" if child else "lead",
                "blocks_parent": child,
                "expected_resume": not child,
            }
            if child:
                row["parent_program_id"] = f"{self.request.session_id}:lead"
            self.sessions[key] = row
        return self.sessions[key]

    def observe_hook(self, event: dict) -> dict:
        row = self.register(event["sessionId"])
        child = row["agent_role"] == "subagent"
        kind = event["hookEventName"]
        name = event.get("toolName")
        args = event.get("toolInput", {})
        response = {}
        if kind == "SessionStart" and not child and self.planning:
            # Ignore only this session's generated plan, not the project's .zcode files.
            plan = f".zcode/plans/plan-{event['sessionId']}.md"
            with (self.request.workspace / ".git/info/exclude").open("a") as handle:
                handle.write(f"\n/{plan}\n")
        if kind == "PreToolUse":
            reason = self.denial(child, name, args)
            if reason:
                response = {"decision": "block", "reason": reason}
            else:
                if name in AGENT_TOOLS:
                    self.pending.add(event["toolCallId"])
                # Headless's immediate deny broker races PermissionRequest hooks.
                # PreToolUse applies this checked approval before that race starts.
                response = {"hookSpecificOutput": {"hookEventName": kind, "permissionDecision": "allow"}}
        elif kind == "PostToolUse":
            if name in AGENT_TOOLS:
                self.pending.discard(event["toolCallId"])
                if event["toolResponse"]["status"] == "completed":
                    self.reports += 1
            elif name == "ExitPlanMode":
                self.approved = True
        elif kind == "PostToolUseFailure" and name in AGENT_TOOLS:
            if event["toolCallId"] in self.pending:
                self.pending.remove(event["toolCallId"])
                self.failure = TerminationReason.SUBAGENT_FAILED
        elif kind == "Stop" and not child and self.planning and not self.approved:
            if event["stopHookActive"]:
                self.failure = TerminationReason.PLAN_NOT_APPROVED
            else:
                response = {
                    "decision": "block",
                    "reason": "Complete Explore delegation and ExitPlanMode before implementing and finishing.",
                }
        with (self.request.artifact_dir / "zcode-hooks.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": event, "response": response}) + "\n")
        self.save()
        return response

    def denial(self, child: bool, name: str, args: dict) -> str | None:
        if child:
            return None if name in READ_TOOLS else "Explore is read-only; use only Read, Glob and Grep."
        if name == "AskUserQuestion" or args.get("run_in_background"):
            return "Interactive questions and background work are disabled in AgentBench."
        if name in AGENT_TOOLS:
            if not self.planning or self.approved or self.reports or self.pending:
                return "Delegate once, then use the report to submit your plan through ExitPlanMode."
            if args.get("subagent_type") != explore_name(self.request.session_id):
                return f"Use subagent_type={explore_name(self.request.session_id)} for read-only exploration."
        if self.planning and not self.approved:
            if not self.reports and not self.pending and name not in AGENT_TOOLS:
                return "Delegate read-only exploration using Agent before other tools."
            if name == "ExitPlanMode":
                if not self.reports or self.pending or not args.get("plan", "").strip():
                    return "Wait for all Explore reports and submit a nonempty plan."
            elif name not in AGENT_TOOLS:
                return "Use the Explore report to submit ExitPlanMode before implementation."
        return None

    def model_tools(self, tools: list[dict], *, child: bool) -> list[dict]:
        """Advertise the same phase policy that the synchronous hook enforces."""
        if child:
            return [tool for tool in tools if tool["function"]["name"] in READ_TOOLS]
        if not self.planning:
            return tools
        if self.approved:
            return [tool for tool in tools if tool["function"]["name"] not in AGENT_TOOLS | {"ExitPlanMode"}]
        if self.reports:
            return [tool for tool in tools if tool["function"]["name"] == "ExitPlanMode"]
        selected = [tool for tool in tools if tool["function"]["name"] in AGENT_TOOLS]
        for tool in selected:
            function = tool["function"]
            name = explore_name(self.request.session_id)
            function["description"] = f"Delegate one read-only exploration to {name}, then submit ExitPlanMode."
            parameters = function["parameters"]
            parameters["properties"]["subagent_type"]["enum"] = [name]
            parameters["required"] = sorted(set(parameters.get("required", [])) | {"subagent_type"})
        return selected

    def save(self) -> None:
        (self.request.artifact_dir / "zcode-lineage.json").write_text(
            json.dumps(
                {
                    "task_id": self.request.session_id,
                    "sessions": list(self.sessions.values()),
                    "auto_plan_approvals": int(self.approved),
                    "exploration_reports": self.reports,
                    "pending": sorted(self.pending),
                    "failure": self.failure,
                    "child_tool_rounds": self.child_tool_rounds,
                },
                indent=2,
            )
            + "\n"
        )

    @asynccontextmanager
    async def serve(self):
        app = FastAPI()
        async with httpx.AsyncClient(timeout=self.request.timeout_seconds, trust_env=False) as client:

            @app.post("/hook")
            async def hook(request: Request):
                return self.observe_hook(await request.json())

            @app.post("/v1/chat/completions")
            async def completion(request: Request):
                payload = await request.json()
                key = request.headers["x-session-id"]
                if key not in self.sessions:
                    # Only synchronous, depth-one Explore delegation is admitted by the hook.
                    if request.headers["x-zcode-session-type"] != "subagent" or not self.pending:
                        raise ValueError("ZCode model request has no registered session")
                    self.register(f"sess_subagent_agent_{key}")
                payload.setdefault("vllm_xargs", {})["agentic_context"] = json.dumps(self.sessions[key])
                if "tools" in payload:
                    child = self.sessions[key]["agent_role"] == "subagent"
                    payload["tools"] = self.model_tools(payload["tools"], child=child)
                    if child and payload["tools"]:
                        rounds = self.child_tool_rounds.get(key, 0)
                        if rounds >= EXPLORE_TOOL_ROUNDS:
                            payload.pop("tools")
                            payload["tool_choice"] = "none"
                        else:
                            self.child_tool_rounds[key] = rounds + 1
                    elif self.planning and not self.approved and payload["tools"]:
                        payload["tool_choice"] = {
                            "type": "function",
                            "function": {"name": payload["tools"][0]["function"]["name"]},
                        }
                self.save()
                upstream = await client.send(
                    client.build_request(
                        "POST",
                        f"{self.request.api_base_url.rstrip('/')}/v1/chat/completions",
                        json=payload,
                        headers={"authorization": "Bearer agentbench"},
                    ),
                    stream=True,
                )

                async def stream():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream.aclose()

                return StreamingResponse(
                    stream(),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type"),
                    background=BackgroundTask(upstream.aclose),
                )

            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                server = uvicorn.Server(
                    uvicorn.Config(app, log_level="error", lifespan="off", timeout_graceful_shutdown=1)
                )
                server.capture_signals = _no_signals
                task = asyncio.create_task(server.serve(sockets=[listener]))
                try:
                    while not server.started:
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
                    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
                finally:
                    server.should_exit = True
                    await task
                    self.save()


@contextmanager
def _no_signals():
    """Leave signal ownership with the benchmark process."""
    yield
