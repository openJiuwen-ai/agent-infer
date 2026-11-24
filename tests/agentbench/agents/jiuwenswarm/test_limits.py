# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio

import pytest

from agentinfer.agentbench.agents.jiuwenswarm.limits import QuotaExceeded, TaskBudget
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


def test_quota_cumulative_resume_rebuild_and_failed_creation():
    async def scenario():
        budget = TaskBudget("root", 2, 1)
        with pytest.raises(RuntimeError), budget.transaction():
            budget.admit("failed", budget.root)
            raise RuntimeError("startup failed")
        assert budget.launched == 0
        child = budget.admit("child", budget.root)
        child.closed = True
        assert budget.admit("child", budget.root, restore=True) is child
        assert budget.launched == 1
        child.closed = True
        assert budget.admit("child", budget.root) is not child
        with pytest.raises(QuotaExceeded):
            budget.admit("third", budget.root)
        assert budget.launched == 2

    asyncio.run(scenario())


def test_parallel_creation_cannot_oversubscribe():
    async def scenario():
        budget = TaskBudget("root", 2, 2)

        async def spawn(i):
            try:
                return budget.admit(str(i), budget.root)
            except QuotaExceeded:
                return None

        actors = await asyncio.gather(*(spawn(i) for i in range(8)))
        assert sum(a is not None for a in actors) == 2
        assert budget.launched == 2

    asyncio.run(scenario())


def test_parallel_work_reaches_limit_and_cancellation_releases_slots():
    async def scenario():
        budget = TaskBudget("root", 4, 2)
        actors = [budget.admit(str(i), budget.root) for i in range(4)]
        ready = asyncio.Event()
        release = asyncio.Event()
        entered = set()

        async def run(actor):
            async with budget.executing(actor):
                entered.add(actor.program_id)
                if len(entered) == 2:
                    ready.set()
                await release.wait()

        tasks = [asyncio.create_task(run(actor)) for actor in actors]
        await asyncio.wait_for(ready.wait(), 2)
        assert len(entered) == budget.running == budget.peak == 2
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert budget.running == 0
        async with budget.executing(actors[0]):
            assert budget.running == 1

    asyncio.run(scenario())


def test_one_actor_parallel_tools_count_once_and_other_tasks_are_independent():
    async def scenario():
        budget = TaskBudget("root", 1, 1)
        actor = budget.admit("child", budget.root)
        other = TaskBudget("other", 1, 1)
        async with budget.executing(actor), budget.executing(actor), other.executing(other.root):
            assert budget.running == 1
            assert actor.work == 2
            assert other.running == 0
        assert budget.running == actor.work == 0

    asyncio.run(scenario())


def test_transactions_do_not_follow_a_spawned_task_context():
    async def scenario():
        budget = TaskBudget("root", 3, 1)
        ready = asyncio.Event()

        async def child():
            await ready.wait()
            with pytest.raises(RuntimeError), budget.transaction():
                budget.admit("failed-in-child", budget.root)
                raise RuntimeError("child startup failed")

        with budget.transaction():
            budget.admit("one", budget.root)
            task = asyncio.create_task(child())
        ready.set()
        await task
        assert budget.launched == 1
        assert "one" in budget.actors and "failed-in-child" not in budget.actors

    asyncio.run(scenario())
