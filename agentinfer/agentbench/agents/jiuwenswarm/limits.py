# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""One task's launch quota and concurrent model/tool execution slots."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Actor:
    budget: "TaskBudget"
    program_id: str
    parent: "Actor | None" = None
    persistent: bool = True
    closed: bool = False
    work: int = 0
    waiting: int = 0


class QuotaExceeded(ValueError):
    """The task has used its cumulative child launch allowance."""


class TaskBudget:
    def __init__(self, root_id: str, total: int, concurrent: int, log_path: Path | None = None) -> None:
        self.total = total
        self.concurrent = concurrent
        self.launched = 0
        self.running = 0
        self.peak = 0
        self.root = Actor(self, root_id)
        self.actors = {root_id: self.root}
        self.log_path = log_path
        self._log_failed = False
        self._teammate_reservations: dict[str, object] = {}
        self._condition = asyncio.Condition()
        self._transaction: ContextVar[tuple[asyncio.Task | None, list[tuple[Actor, Actor | None]]] | None] = ContextVar(
            "jiuwen_launch_transaction", default=None
        )

    def emit(self, event: str, actor: Actor) -> None:
        if self.log_path is None or self._log_failed:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": event,
                            "program_id": actor.program_id,
                            "parent_program_id": actor.parent.program_id if actor.parent else None,
                            "launched": self.launched,
                            "running": self.running,
                            "peak": self.peak,
                            "time": time.monotonic(),
                        }
                    )
                    + "\n"
                )
        except OSError:
            self._log_failed = True
            logger.warning("Cannot write AgentBench task limits log %s", self.log_path, exc_info=True)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Undo admitted children when their enclosing startup fails."""
        owner = asyncio.current_task()
        current = self._transaction.get()
        if current is not None and current[0] is owner:
            yield
            return
        admitted: list[tuple[Actor, Actor | None]] = []
        token = self._transaction.set((owner, admitted))
        try:
            yield
        except BaseException:
            for actor, previous in reversed(admitted):
                if previous is None:
                    self.actors.pop(actor.program_id)
                else:
                    self.actors[actor.program_id] = previous
                self.launched -= 1
                self.emit("launch_failed", actor)
            raise
        finally:
            self._transaction.reset(token)

    def _check_capacity(self, actor: Actor) -> None:
        allocated = self.launched + len(self._teammate_reservations)
        if allocated >= self.total:
            self.emit("quota_denied", actor)
            raise QuotaExceeded(f"AgentBench task launch quota exhausted ({allocated}/{self.total})")

    def reserve_teammate(self, member_id: str) -> object:
        """Hold a launch for a roster entry until WorkSwarm starts the member."""
        if member_id in self._teammate_reservations:
            raise QuotaExceeded(f"Team member {member_id!r} already has a pending launch")
        self._check_capacity(Actor(self, member_id, self.root))
        token = object()
        self._teammate_reservations[member_id] = token
        return token

    def teammate_reservation(self, member_id: str) -> object | None:
        return self._teammate_reservations.get(member_id)

    def release_teammate(self, member_id: str, token: object | None) -> None:
        if token is not None and self._teammate_reservations.get(member_id) is token:
            del self._teammate_reservations[member_id]

    def admit(
        self,
        program_id: str,
        parent: Actor,
        *,
        persistent: bool = True,
        restore: bool = False,
        teammate_id: str | None = None,
    ) -> Actor:
        reserved = teammate_id is not None and self._teammate_reservations.pop(teammate_id, None) is not None
        previous = self.actors.get(program_id)
        if previous is not None and (restore or not previous.closed):
            return previous
        actor = Actor(self, program_id, parent, persistent)
        if not reserved:
            self._check_capacity(actor)
        self.actors[program_id] = actor
        self.launched += 1
        transaction = self._transaction.get()
        if transaction is not None and transaction[0] is asyncio.current_task():
            transaction[1].append((actor, previous))
        self.emit("launch", actor)
        return actor

    @asynccontextmanager
    async def executing(self, actor: Actor) -> AsyncIterator[None]:
        """Count each child once even when its tools run in parallel."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: actor is self.root or actor.work > 0 or self.running < self.concurrent
            )
            if actor is not self.root and actor.work == 0:
                self.running += 1
                self.peak = max(self.peak, self.running)
            actor.work += 1
            self.emit("work_start", actor)
        try:
            yield
        finally:
            async with self._condition:
                actor.work -= 1
                if actor is not self.root and actor.work == 0:
                    self.running -= 1
                self.emit("work_end", actor)
                self._condition.notify_all()
