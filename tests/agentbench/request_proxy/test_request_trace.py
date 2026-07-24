import asyncio
from dataclasses import asdict
from pathlib import Path

import pytest

from agentinfer.agentbench.request_proxy.request_trace import (
    RequestFact,
    RequestTraceWriter,
    load_request_facts,
)


def fact(request_id: str = "r1") -> RequestFact:
    return RequestFact(
        schema_version="1",
        run_id="run",
        request_id=request_id,
        session_id="session",
        actor_id="lead",
        actor_role="lead",
        started_at="start",
        finished_at="finish",
        status="success",
        status_code=200,
        latency_seconds=1.0,
        ttft_seconds=0.1,
        input_tokens=4,
        output_tokens=2,
        cache_creation_tokens=1,
        cached_tokens=3,
        upstream="upstream",
        error=None,
    )


def test_writer_drains_and_close_is_idempotent(tmp_path: Path):
    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        writer.submit(fact())
        health = await writer.drain()
        assert asdict(health) == {"submitted": 1, "written": 1, "pending": 0, "writer_error": None}
        assert await writer.close() == await writer.close()

    asyncio.run(run())


def test_writer_keeps_one_handle_open_for_all_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "requests.jsonl"
    original_open = Path.open
    open_calls = 0

    def counted_open(self, *args, **kwargs):
        nonlocal open_calls
        if self == path:
            open_calls += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    async def run():
        writer = RequestTraceWriter(path)
        await writer.start()
        writer.submit(fact())
        writer.submit(fact("r2"))
        health = await writer.close()
        assert health.written == 2

    asyncio.run(run())
    assert open_calls == 1


def test_writer_reports_file_error(tmp_path: Path):
    async def run():
        writer = RequestTraceWriter(tmp_path)
        await writer.start()
        writer.submit(fact())
        health = await asyncio.wait_for(writer.close(), 1)
        assert health.writer_error is not None
        assert health.pending == 1

    asyncio.run(run())


def test_writer_post_dequeue_flush_failure_does_not_deadlock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class BrokenHandle:
        def write(self, _value):
            return None

        def flush(self):
            raise OSError("flush failed")

        def close(self):
            return None

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: BrokenHandle())

    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        writer.submit(fact())
        health = await asyncio.wait_for(writer.close(), 1)
        assert health.writer_error == "flush failed"
        assert health.pending == 1

    asyncio.run(run())


def test_writer_preserves_write_and_handle_close_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class BrokenHandle:
        def write(self, _value):
            raise OSError("write failed")

        def flush(self):
            return None

        def close(self):
            raise OSError("close failed")

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: BrokenHandle())

    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        writer.submit(fact())
        health = await writer.close()
        assert health.writer_error == "write failed; handle close failed: close failed"

    asyncio.run(run())


def test_writer_rejects_submissions_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_open(*_args, **_kwargs):
        raise OSError("open failed")

    monkeypatch.setattr(Path, "open", fail_open)

    async def run():
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        writer.submit(fact())
        health = await asyncio.wait_for(writer.drain(), 1)
        assert health.writer_error == "open failed"
        with pytest.raises(RuntimeError, match="request trace writer failed"):
            writer.submit(fact("r2"))
        assert await writer.close() == health

    asyncio.run(run())


def test_load_request_facts_filters_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    path = tmp_path / "requests.jsonl"
    other = RequestFact(**(asdict(fact("r2")) | {"session_id": "other"}))
    path.write_text(
        json.dumps(asdict(fact())) + "\n" + json.dumps(asdict(other)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("read_text called"))

    assert load_request_facts(path) == [fact(), other]
    assert load_request_facts(path, session_id="session") == [fact()]


def test_load_request_facts_preserves_valid_prefix_before_incomplete_tail(tmp_path: Path) -> None:
    import json

    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(asdict(fact())) + '\n{"request_id":', encoding="utf-8")

    assert load_request_facts(path) == [fact()]


def test_load_request_facts_rejects_complete_malformed_record(tmp_path: Path) -> None:
    import json

    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(asdict(fact())) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_request_facts(path)
