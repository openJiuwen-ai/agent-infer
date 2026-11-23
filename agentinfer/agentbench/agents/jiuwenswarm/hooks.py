# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""WorkSwarm 0.2.8 execution seams; loaded only inside task-local services."""

import json
import os
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from openjiuwen.agent_teams.harness.team_harness import TeamHarness
from openjiuwen.core.session import get_current_session
from openjiuwen.core.single_agent.ability_manager import AbilityManager
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.rails.subagent.subagent_rail import SubagentRail
from openjiuwen.harness.subagent_runtime.config import SubagentRuntimeConfig
from openjiuwen.harness.subagent_runtime.control import SubagentControl
from openjiuwen.harness.subagent_runtime.session_manager import SubagentSessionManager

from .limits import Actor, TaskBudget

_actor: ContextVar[Actor | None] = ContextVar("agentbench_jiuwen_actor", default=None)
_restoring = ContextVar("agentbench_jiuwen_restoring", default=False)
_synchronous = ContextVar("agentbench_jiuwen_synchronous", default=False)
_budget: TaskBudget | None = None
_team_actors: WeakKeyDictionary = WeakKeyDictionary()


def current_actor() -> Actor:
    budget = cast(TaskBudget, _budget)
    session = get_current_session()
    if session is not None:
        key = session.get_cache_identity().cache_id
        if key in budget.actors:
            return budget.actors[key]
    # Model helpers can have their own session; attribute them to the active work.
    return cast(Actor, _actor.get())


def _wrap_start(original):
    @wraps(original)
    async def start(self, *args, **kwargs):
        with _budget.transaction():
            return await original(self, *args, **kwargs)

    return start


def _wrap_work(original, *, tool=False):
    @wraps(original)
    async def work(self, ctx, *args, **kwargs):
        actor = _budget.actors[ctx.session.get_cache_identity().cache_id]
        token = _actor.set(actor)
        name = ctx.inputs.tool_name if tool else ""
        # Waiting for a child performs no model or host IO in the parent.
        waiting = name in {"subagent_wait", "task_tool"}
        sync_token = _synchronous.set(name == "task_tool")
        try:
            if waiting:
                actor.waiting += 1
                try:
                    return await original(self, ctx, *args, **kwargs)
                finally:
                    actor.waiting -= 1
            async with _budget.executing(actor):
                return await original(self, ctx, *args, **kwargs)
        finally:
            _synchronous.reset(sync_token)
            _actor.reset(token)

    return work


def install() -> None:
    global _budget
    if _budget is not None:
        return
    _budget = TaskBudget(
        os.environ["AGENTBENCH_ROOT_SESSION_ID"],
        int(os.environ["AGENTBENCH_MAX_SUBAGENTS"]),
        int(os.environ["AGENTBENCH_MAX_CONCURRENT_SUBAGENTS"]),
        Path(os.environ["AGENTBENCH_LIMITS_LOG"]),
    )
    cast(Path, _budget.log_path).with_name("jiuwenswarm-runtime.json").write_text(
        json.dumps(
            {
                "workswarm": version("workswarm"),
                "openjiuwen": version("openjiuwen"),
                "profile": os.environ.get("AGENTBENCH_PROFILE", "agent.code.normal"),
                "max_subagents": _budget.total,
                "max_concurrent_subagents": _budget.concurrent,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ReActAgent._railed_model_call = _wrap_work(ReActAgent._railed_model_call)
    AbilityManager._railed_execute_single_tool_call = _wrap_work(
        AbilityManager._railed_execute_single_tool_call, tool=True
    )

    original_create = DeepAgent.create_subagent

    @wraps(original_create)
    def create_subagent(self, subagent_type, subsession_id, browser_capabilities=None):
        with _budget.transaction():
            _budget.admit(subsession_id, current_actor(), persistent=not _synchronous.get(), restore=_restoring.get())
            child = original_create(self, subagent_type, subsession_id, browser_capabilities)
            child.deep_config.subagents = self.deep_config.subagents
            child.deep_config.enable_subagent_runtime = True
            child.add_rail(SubagentRail(enable_subagent_runtime=True))
            return child

    DeepAgent.create_subagent = create_subagent
    SubagentSessionManager.create = _wrap_start(SubagentSessionManager.create)

    original_control_init = SubagentControl.__init__

    @wraps(original_control_init)
    def control_init(self, parent_agent, parent_session_id, config=None, parent_session=None):
        config = replace(
            config or SubagentRuntimeConfig(),
            max_subagents=_budget.total,
            max_concurrent_running=_budget.concurrent,
        )
        original_control_init(self, parent_agent, parent_session_id, config, parent_session)

    SubagentControl.__init__ = control_init
    original_resume = SubagentControl.resume

    @wraps(original_resume)
    async def resume(self, subagent_id):
        token = _restoring.set(True)
        try:
            result = await original_resume(self, subagent_id)
            _budget.actors[subagent_id].closed = False
            return result
        finally:
            _restoring.reset(token)

    SubagentControl.resume = resume
    original_close = SubagentControl.close

    @wraps(original_close)
    async def close(self, subagent_id, reason="manual"):
        result = await original_close(self, subagent_id, reason)
        actor = _budget.actors.get(subagent_id)
        if actor is not None:
            actor.closed = True
        return result

    SubagentControl.close = close
    original_session = TeamHarness._make_child_session

    @wraps(original_session)
    def team_session(self, team_session):
        session = original_session(self, team_session)
        key = session.get_cache_identity().cache_id
        if self._role.value == "leader":
            _budget.actors[key] = _budget.root
            _team_actors[self] = _budget.root
        else:
            _team_actors[self] = _budget.admit(key, _budget.root)
        return session

    TeamHarness._make_child_session = team_session
    TeamHarness.start = _wrap_start(TeamHarness.start)
    TeamHarness.run_once = _wrap_start(TeamHarness.run_once)
    original_dispose = TeamHarness.dispose

    @wraps(original_dispose)
    async def dispose(self):
        actor = _team_actors.pop(self, None)
        try:
            return await original_dispose(self)
        finally:
            if actor is not None:
                actor.closed = True

    TeamHarness.dispose = dispose
