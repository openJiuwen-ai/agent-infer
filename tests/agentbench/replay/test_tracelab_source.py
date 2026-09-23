# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import gzip
import json
from pathlib import Path

import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.runner import _resolve_replay_source
from agentinfer.agentbench.replay.tracelab_source import materialize_tracelab_source


def _write_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def test_materialize_tracelab_source_keeps_first_complete_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl.gz"
    rows = [
        {"provider": "claude", "session_id": "first", "round_index": 0},
        {"provider": "codex", "session_id": "second", "round_index": 0},
        {"provider": "claude", "session_id": "first", "round_index": 1},
        {"provider": "claude", "session_id": "third", "round_index": 0},
        {"provider": "codex", "session_id": "second", "round_index": 1},
    ]
    _write_gzip(source, rows)
    calls = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr("agentinfer.agentbench.replay.tracelab_source.hf_hub_download", download)

    result = materialize_tracelab_source(2, cache_dir=tmp_path / "cache")
    actual = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]

    assert actual == [rows[0], rows[1], rows[2], rows[4]]
    assert calls == [
        {
            "repo_id": "UW-SyFI/TraceLab",
            "repo_type": "dataset",
            "filename": "data/v0.0.2/syfi_coding_trace.jsonl.gz",
            "revision": "v0.0.2",
        }
    ]

    monkeypatch.setattr(
        "agentinfer.agentbench.replay.tracelab_source.hf_hub_download",
        lambda **kwargs: pytest.fail(f"unexpected repeated download: {kwargs}"),
    )
    assert materialize_tracelab_source(2, cache_dir=tmp_path / "cache") == result


def test_materialize_tracelab_source_keeps_all_when_fewer_than_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl.gz"
    _write_gzip(source, [{"provider": "claude", "session_id": "only", "round_index": 0}])
    monkeypatch.setattr(
        "agentinfer.agentbench.replay.tracelab_source.hf_hub_download",
        lambda **kwargs: str(source),
    )

    result = materialize_tracelab_source(2, cache_dir=tmp_path / "cache")

    assert [json.loads(line)["session_id"] for line in result.read_text().splitlines()] == ["only"]


def test_resolve_replay_source_updates_copy_with_downloaded_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = tmp_path / "first-2-sessions.jsonl"
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 2},
            "replay": {"trace_type": "tracelab", "trace_path": None},
        }
    )
    monkeypatch.setattr("agentinfer.agentbench.replay.runner.materialize_tracelab_source", lambda count: downloaded)

    resolved = _resolve_replay_source(config)

    assert resolved.replay.trace_path == downloaded
    assert config.replay.trace_path is None
