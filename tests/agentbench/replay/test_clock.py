# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exact scheduler assertions use one shared clock, advanced only by the test."""

import asyncio
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.executor import ReplayExecutor
from agentinfer.agentbench.replay.prompt import SyntheticPrompt
from agentinfer.agentbench.replay.transport import TransportResult
from tests.agentbench.replay.test_executor import _node

pytestmark = pytest.mark.cpu_test


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.waiters = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + seconds, future))
        await future

    async def settle(self):
        # All work here is in-memory; allow nested gather/future callbacks to drain.
        for _ in range(30):
            await asyncio.sleep(0)

    async def advance(self, now):
        assert now >= self.now
        self.now = now
        for due, future in self.waiters:
            if due <= now + 1e-10 and not future.done():
                future.set_result(None)
        self.waiters = [(due, f) for due, f in self.waiters if not f.done()]
        await self.settle()


class Harness:
    def __init__(self, requests, *, durations=None, preparation=None, cleanup=0, tasks=1):
        self.clock = ManualClock()
        self.events, self.sends = [], []
        self.durations = durations or {}
        self.preparation = preparation or {}
        self.cleanup = cleanup
        self.tasks = [
            SimpleNamespace(task_id=f"t{i}", runtime_session_id=f"s{i}", requests=requests) for i in range(tasks)
        ]
        config = ReplayBenchConfig.model_validate(
            {"experiment": {"task_num": tasks, "max_concurrency": 1}, "replay": {}}
        )
        self.executor = ReplayExecutor(
            config, SimpleNamespace(tasks=self.tasks), self, self, clock=self.clock, observer=self
        )

    def emit(self, event):
        self.events.append(event)

    async def build(self, task, node, context):
        await self.clock.sleep(self.preparation.get(node.source_key, 0))
        return SyntheticPrompt("", (), ())

    async def release_task(self, task):
        await self.clock.sleep(self.cleanup)

    async def send(self, task, node, prompt):
        self.sends.append((task.runtime_session_id, node.source_key, self.clock.monotonic()))
        await self.clock.sleep(
            self.durations.get((task.runtime_session_id, node.source_key), self.durations.get(node.source_key, 0))
        )
        return TransportResult(True, "answer", self.clock.monotonic(), 200, None)

    async def start(self):
        self.running = asyncio.create_task(self.executor.execute())
        await self.clock.settle()

    async def finish(self):
        assert self.running.done(), (self.sends, self.clock.waiters)
        return await self.running


def run(exercise):
    asyncio.run(asyncio.wait_for(exercise(), 5))  # Real time is only a deadlock guard.


@pytest.mark.parametrize("prepare,send_time,lag", [(0, 0.35, 0), (0.12, 0.42, 0.07)])
def test_completion_plus_gap_is_preserved_when_preparation_is_slow(prepare, send_time, lag):
    async def exercise():
        h = Harness(
            (
                _node("a", "lead_main"),
                _node("b", "continuation", send_after="a", context_after="a", effective_interval=0.05),
            ),
            durations={"a": 0.30},
            preparation={"b": prepare},
        )
        await h.start()
        await h.clock.advance(0.299)
        assert [s[1] for s in h.sends] == ["a"]
        await h.clock.advance(0.30)
        await h.clock.advance(0.349)
        assert [s[1] for s in h.sends] == ["a"]
        await h.clock.advance(send_time)
        result = (await h.finish())[0]
        b = next(n for n in result.nodes if n.source_key == "b")
        assert b.planned_send_offset_seconds == pytest.approx(0.35)
        assert b.actual_send_offset_seconds == pytest.approx(send_time)
        assert b.scheduler_lag_seconds == pytest.approx(lag)
        assert b.prompt_prepare_seconds == pytest.approx(prepare)

    run(exercise)


@pytest.mark.parametrize("main_duration", [0.05, 1.0])
def test_title_release_is_relative_to_graph_start(main_duration):
    async def exercise():
        h = Harness(
            (_node("main", "lead_main"), _node("title", "lead_title", effective_interval=0.2)),
            durations={"main": main_duration},
        )
        await h.start()
        if main_duration < 0.2:
            await h.clock.advance(main_duration)
        await h.clock.advance(0.199)
        assert [s[1] for s in h.sends] == ["main"]
        await h.clock.advance(0.2)
        assert h.sends[-1] == ("s0", "title", 0.2)
        if main_duration > 0.2:
            assert not h.running.done()
            await h.clock.advance(main_duration)
        assert (await h.finish())[0].status == "completed"

    run(exercise)


def test_timing_node_consumes_time_but_does_not_call_transport():
    async def exercise():
        timing = _node("timer", "continuation", effective_interval=0.1)
        timing.node_type, timing.effective_duration_seconds = "timing_dependency", 0.2
        h = Harness((timing, _node("next", "lead_main", send_after="timer", effective_interval=0.05)))
        await h.start()
        await h.clock.advance(0.1)
        await h.clock.advance(0.3)
        assert h.sends == []
        await h.clock.advance(0.349)
        assert h.sends == []
        await h.clock.advance(0.35)
        assert h.sends == [("s0", "next", 0.35)]
        result = (await h.finish())[0]
        timer = next(n for n in result.nodes if n.source_key == "timer")
        assert timer.actual_send_offset_seconds is None

    run(exercise)


def test_task_duration_excludes_admission_queue():
    async def exercise():
        h = Harness((_node("root", "lead_main"),), tasks=2, durations={("s0", "root"): 10, ("s1", "root"): 8})
        await h.start()
        await h.clock.advance(10)
        assert h.sends == [("s0", "root", 0), ("s1", "root", 10)]
        await h.clock.advance(18)
        assert [t.duration_seconds for t in await h.finish()] == [10, 8]
        queued = [e for e in h.events if e.get("runtime_session_id") == "s1"]
        times = {e["event"]: e["t_monotonic"] for e in queued}
        assert times["task_admitted"] - times["task_submitted"] == 10
        assert times["task_finished"] - times["task_submitted"] == 18

    run(exercise)


def test_task_duration_includes_root_preparation_and_cleanup():
    async def exercise():
        h = Harness((_node("root", "lead_main"),), preparation={"root": 2}, durations={"root": 3}, cleanup=4)
        await h.start()
        await h.clock.advance(2)
        await h.clock.advance(5)
        assert not h.running.done()
        await h.clock.advance(9)
        result = (await h.finish())[0]
        assert result.duration_seconds == 9
        assert result.nodes[0].prompt_prepare_start_offset_seconds == -2
        assert result.nodes[0].actual_send_offset_seconds == 0

    run(exercise)


@pytest.mark.parametrize("main_time,title_time,send_time,release", [(0.3, 0.5, 0.55, 0.55), (0.5, 0.3, 0.52, 0.35)])
def test_distinct_context_and_timing_predecessors(main_time, title_time, send_time, release):
    async def exercise():
        h = Harness(
            (
                _node("main", "lead_main"),
                _node("title", "lead_title"),
                _node("next", "continuation", send_after="title", context_after="main", effective_interval=0.05),
            ),
            durations={"main": main_time, "title": title_time},
            preparation={"next": 0.02},
        )
        await h.start()
        for now in sorted({main_time, title_time, main_time + 0.02, send_time}):
            if now < send_time:
                assert all(s[1] != "next" for s in h.sends)
            await h.clock.advance(now)
        result = (await h.finish())[0]
        next_node = next(n for n in result.nodes if n.source_key == "next")
        assert next_node.planned_send_offset_seconds == pytest.approx(release)
        assert next_node.actual_send_offset_seconds == pytest.approx(send_time)
        assert next_node.prompt_prepare_start_offset_seconds == pytest.approx(main_time)

    run(exercise)
