# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.replay import inferact_source
from agentinfer.agentbench.replay.agentinfer_source import builtin_agentinfer_source
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.inferact_source import materialize_inferact_source
from agentinfer.agentbench.replay.runner import _resolve_replay_source


def test_builtin_agentinfer_source_is_packaged() -> None:
    source = builtin_agentinfer_source()

    assert source.name == "agentinfer_trace_requests.jsonl"
    assert len(source.read_text(encoding="utf-8").splitlines()) == 597


def test_materialize_inferact_source_keeps_up_to_first_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [{"id": index} for index in range(3)]
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=records)

    _mock_inferact_http(monkeypatch, respond)

    result = materialize_inferact_source(2, cache_dir=tmp_path / "cache")
    assert json.loads(result.read_text(encoding="utf-8")) == records[:2]
    assert calls[0].url.path == (
        "/datasets/Inferact/codex_swebenchpro_traces/resolve/"
        "0d52ae8c75738117be9e58c7071bd9a5b43ff78f/codex_swebenchpro.json"
    )
    assert materialize_inferact_source(2, cache_dir=tmp_path / "cache") == result
    assert len(calls) == 1

    all_result = materialize_inferact_source(8, cache_dir=tmp_path / "cache-all")
    assert json.loads(all_result.read_text(encoding="utf-8")) == records


def _mock_inferact_http(monkeypatch: pytest.MonkeyPatch, respond: Callable[[httpx.Request], httpx.Response]) -> None:
    client = httpx.Client
    monkeypatch.setattr(inferact_source, "try_to_load_from_cache", lambda **kwargs: None)
    monkeypatch.setattr(inferact_source.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(inferact_source, "build_hf_headers", lambda: {"user-agent": "agentinfer-test"})
    monkeypatch.setattr(
        inferact_source.httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), trust_env=False, **kwargs),
    )


class _TrackedStream(httpx.SyncByteStream):
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


def test_inferact_stops_network_before_next_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'[{"id":0}]'.ljust(inferact_source.INFERACT_STREAM_CHUNK_SIZE)
        raise AssertionError("must not read the unrequested tail")

    stream = _TrackedStream(chunks())
    _mock_inferact_http(monkeypatch, lambda request: httpx.Response(200, stream=stream))
    result = materialize_inferact_source(1, cache_dir=tmp_path)
    assert json.loads(result.read_text()) == [{"id": 0}]
    assert stream.closed


def test_inferact_does_not_parse_next_buffered_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The malformed next object must not be parsed, even if it was prefetched.
    _mock_inferact_http(monkeypatch, lambda request: httpx.Response(200, content=b'[{"id":0},invalid'))
    result = materialize_inferact_source(1, cache_dir=tmp_path)
    assert json.loads(result.read_text()) == [{"id": 0}]


def test_inferact_decodes_split_utf8_and_nested_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"text": '中文🙂 \\" [,]', "nested": {"items": [1, 2]}}]
    data = json.dumps(records, ensure_ascii=False).encode("utf-8")
    stream = _TrackedStream(iter(bytes([byte]) for byte in data))
    monkeypatch.setattr(inferact_source, "INFERACT_STREAM_CHUNK_SIZE", 3)
    _mock_inferact_http(monkeypatch, lambda request: httpx.Response(200, stream=stream))
    result = materialize_inferact_source(2, cache_dir=tmp_path)
    assert json.loads(result.read_text()) == records
    assert stream.closed


@pytest.mark.parametrize(
    "body",
    [b"[]", b'[{"id":', b'[{"id":0},]', b"[1]", b"[{}]garbage"],
    ids=["empty", "truncated", "trailing-comma", "non-object", "trailing-data"],
)
def test_inferact_invalid_source_leaves_no_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    stream = _TrackedStream(iter([body]))
    _mock_inferact_http(monkeypatch, lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(ValueError):
        materialize_inferact_source(8, cache_dir=tmp_path)
    assert stream.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["status", "disconnect"])
def test_inferact_download_failure_leaves_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'[{"id":0},'.ljust(inferact_source.INFERACT_STREAM_CHUNK_SIZE)
        raise httpx.ReadError("connection lost")

    stream = _TrackedStream(chunks())
    _mock_inferact_http(monkeypatch, lambda request: httpx.Response(503 if failure == "status" else 200, stream=stream))
    with pytest.raises(httpx.HTTPError):
        materialize_inferact_source(8, cache_dir=tmp_path)
    assert stream.closed
    assert list(tmp_path.iterdir()) == []


def test_inferact_uses_existing_hub_cache_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "cached.json"
    source.write_text('[{"id":0},{"id":1}]')
    monkeypatch.setattr(inferact_source, "try_to_load_from_cache", lambda **kwargs: str(source))
    monkeypatch.setattr(inferact_source.constants, "HF_HUB_OFFLINE", True)
    result = materialize_inferact_source(1, cache_dir=tmp_path / "subset")
    assert json.loads(result.read_text()) == [{"id": 0}]


def test_inferact_offline_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inferact_source, "try_to_load_from_cache", lambda **kwargs: None)
    monkeypatch.setattr(inferact_source.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(inferact_source.OfflineModeIsEnabled):
        materialize_inferact_source(1, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("trace_type", ["agentinfer", "inferact_codex_swebenchpro", "tracelab"])
def test_resolve_default_source_updates_only_a_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trace_type: str,
) -> None:
    source = tmp_path / "source.jsonl"
    config_data: dict[str, object] = {
        "experiment": {"task_num": 8, "max_concurrency": 4},
        "replay": {"trace_type": trace_type, "trace_path": None},
    }
    if trace_type == "inferact_codex_swebenchpro":
        config_data["replay"] = {
            "trace_type": trace_type,
            "trace_path": None,
            "interval_mode": "lognormal",
            "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
        }
    config = ReplayBenchConfig.model_validate(config_data)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.builtin_agentinfer_source", lambda: source)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.materialize_inferact_source", lambda count: source)
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.materialize_tracelab_source", lambda count: source)

    resolved = _resolve_replay_source(config)

    assert resolved.replay.trace_path == source
    assert config.replay.trace_path is None


def test_agentx_without_trace_path_remains_unimplemented() -> None:
    config = ReplayBenchConfig.model_validate(
        {"experiment": {"task_num": 8}, "replay": {"trace_type": "agentX", "trace_path": None}}
    )

    with pytest.raises(NotImplementedError, match="agentX"):
        _resolve_replay_source(config)
