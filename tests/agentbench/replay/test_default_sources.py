# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from pathlib import Path

import pytest

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
    source = tmp_path / "codex_swebenchpro.json"
    records = [{"id": index} for index in range(3)]
    source.write_text(json.dumps(records), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr("agentinfer.agentbench.replay.inferact_source.hf_hub_download", download)

    result = materialize_inferact_source(2, cache_dir=tmp_path / "cache")
    assert json.loads(result.read_text(encoding="utf-8")) == records[:2]
    assert calls[0]["revision"] == "0d52ae8c75738117be9e58c7071bd9a5b43ff78f"

    all_result = materialize_inferact_source(8, cache_dir=tmp_path / "cache-all")
    assert json.loads(all_result.read_text(encoding="utf-8")) == records


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
