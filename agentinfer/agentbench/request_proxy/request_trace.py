# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Immutable request facts and an asynchronous JSONL writer."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from pydantic import TypeAdapter


@dataclass(frozen=True)
class TraceHealth:
    """Summarize asynchronous request-fact persistence health."""

    submitted: int
    written: int
    pending: int
    writer_error: str | None


@dataclass(frozen=True)
class RequestFact:
    """Store one immutable request observation from the transparent proxy."""

    schema_version: str
    run_id: str
    request_id: str
    session_id: str | None
    actor_id: str
    actor_role: str
    started_at: str
    finished_at: str
    status: str
    status_code: int | None
    latency_seconds: float
    ttft_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_tokens: int | None
    cached_tokens: int | None
    upstream: str
    error: str | None
    request_purpose: str | None = None


_REQUEST_FACT_ADAPTER = TypeAdapter(RequestFact)


class RequestTraceWriter:
    """Persist request facts asynchronously to one append-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: asyncio.Queue[RequestFact | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="request-trace")
        self._handle: TextIO | None = None
        self._submitted = 0
        self._written = 0
        self._error: str | None = None
        self._closed = False

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run())

    def submit(self, fact: RequestFact) -> None:
        if self._closed:
            raise RuntimeError("request trace writer is closed")
        if self._error is not None:
            raise RuntimeError(f"request trace writer failed: {self._error}")
        self._submitted += 1
        self._queue.put_nowait(fact)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._open)
            while True:
                fact = await self._queue.get()
                try:
                    if fact is None:
                        return
                    await loop.run_in_executor(self._executor, self._append, fact)
                    self._written += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._error = str(exc)
            self._discard_pending()
        finally:
            try:
                await loop.run_in_executor(self._executor, self._close_handle)
            except Exception as exc:
                cleanup_error = f"handle close failed: {exc}"
                self._error = f"{self._error}; {cleanup_error}" if self._error else cleanup_error
            self._executor.shutdown(wait=False)

    def _open(self) -> None:
        self._handle = self.path.open("a", encoding="utf-8")

    def _append(self, fact: RequestFact) -> None:
        assert self._handle is not None
        self._handle.write(json.dumps(asdict(fact), ensure_ascii=False) + "\n")
        self._handle.flush()

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _discard_pending(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    def health(self) -> TraceHealth:
        return TraceHealth(self._submitted, self._written, self._submitted - self._written, self._error)

    async def drain(self) -> TraceHealth:
        await self._queue.join()
        return self.health()

    async def close(self) -> TraceHealth:
        if not self._closed:
            self._closed = True
            if self._task and (self._task.done() or self._error is not None):
                await self._task
            else:
                self._queue.put_nowait(None)
                await self._queue.join()
                if self._task:
                    await self._task
        return self.health()


def load_request_facts(path: Path, *, session_id: str | None = None) -> list[RequestFact]:
    """Load persisted request facts, optionally retaining one session."""

    if not path.exists():
        return []
    facts = []
    with path.open(encoding="utf-8") as handle:
        previous = None
        for line in handle:
            if previous is not None:
                _append_request_fact(facts, previous, session_id)
            previous = line
        if previous is not None:
            try:
                _append_request_fact(facts, previous, session_id)
            except json.JSONDecodeError:
                if previous.endswith(("\n", "\r")):
                    raise
    return facts


def _append_request_fact(facts: list[RequestFact], line: str, session_id: str | None) -> None:
    if not line.strip():
        return
    fact = _REQUEST_FACT_ADAPTER.validate_python(json.loads(line))
    if session_id is None or fact.session_id == session_id:
        facts.append(fact)
