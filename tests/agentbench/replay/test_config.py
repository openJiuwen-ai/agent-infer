# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentinfer.agentbench.replay.config import (
    ReplayBenchConfig,
    load_replay_config,
)


def test_replay_config_resolves_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "replay.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
experiment:
  result_dir: ../results
replay:
  trace_path: ../source.jsonl
""",
        encoding="utf-8",
    )

    config = load_replay_config(config_path)

    assert config.experiment.result_dir == tmp_path / "results"
    assert config.replay.trace_path == tmp_path / "source.jsonl"


def test_tokenizer_url_follows_backend_until_explicitly_overridden() -> None:
    defaulted = ReplayBenchConfig.model_validate(
        {
            "backend": {"base_url": "http://backend"},
            "replay": {"trace_path": "source.jsonl"},
        }
    )
    explicit = ReplayBenchConfig.model_validate(
        {
            "backend": {
                "base_url": "http://backend",
                "tokenizer_base_url": "http://tokenizer",
            },
            "replay": {"trace_path": "source.jsonl"},
        }
    )

    assert defaulted.backend.resolved_tokenizer_base_url == "http://backend"
    assert explicit.backend.resolved_tokenizer_base_url == "http://tokenizer"


def test_replay_sample_yaml_loads() -> None:
    config = load_replay_config(Path("agentinfer/agentbench/configs/replay_benchmark.yaml"))

    assert config.backend.endpoint == "/v1/messages"
    assert config.replay.trace_path.name == "requests.jsonl"
    assert config.replay.trace_type == "agentinfer"
    assert config.replay.trace_same_agent_gap_scale == 1.0
    assert config.replay.trace_same_agent_gap_offset_seconds == 0.0
    assert config.backend.metrics_url is None
    assert config.backend.effective_metrics_url == "http://127.0.0.1:8000/metrics"
    assert config.replay.interval_mode == "trace"
    assert config.replay.prompt_shape == "claude_code_minimal_v1"
    assert config.replay.lead_title_sys_shared_prefix == 296
    assert config.replay.lead_name_sys_shared_prefix == 105
    assert config.replay.lead_1st_sys_shared_prefix == 18
    assert config.replay.lead_1st_sys_session_prefix == 1466
    assert config.replay.lead_1st_tool_shared_prefix == 17550
    assert config.replay.lead_1st_msg_shared_prefix == 74
    assert config.replay.lead_1st_trailing_system_prefix == 459
    assert config.replay.subagent_tool_1st_shared_prefix == 6900
    assert config.replay.subagent_1st_sys_shared_prefix == 18
    assert config.replay.subagent_1st_sys_session_prefix == 732
    assert config.replay.subagent_1st_msg_shared_prefix == 74
    assert config.replay.lead_continuation_extra_system_ratio == 0.2146
    assert config.replay.subagent_continuation_extra_system_ratio == 0.2344
    assert config.replay.context_adjustment_mode == "adaptive"
    assert config.replay.context_micro_trim_max_tokens == 64
    assert config.replay.context_micro_trim_max_ratio == 0.005
    assert config.replay.prompt_calibration_tolerance_tokens == 1
    assert config.replay.context_micro_trim_limit(100) == 64
    assert config.replay.context_micro_trim_limit(20_000) == 100


def test_replay_rejects_obsolete_router_config() -> None:
    with pytest.raises(ValidationError, match="router"):
        ReplayBenchConfig.model_validate(
            {
                "replay": {"trace_path": "source.jsonl"},
                "router": {"enabled": False},
            }
        )


def test_replay_defaults_to_minimal_and_rejects_removed_legacy_shape() -> None:
    config = ReplayBenchConfig.model_validate({"replay": {"trace_path": "source.jsonl"}})

    assert config.replay.prompt_shape == "claude_code_minimal_v1"
    with pytest.raises(ValidationError, match="prompt_shape"):
        ReplayBenchConfig.model_validate({"replay": {"trace_path": "source.jsonl", "prompt_shape": "legacy"}})


def test_lognormal_requires_three_quantile_anchors() -> None:
    with pytest.raises(ValidationError, match="requires interval_lognormal anchors"):
        ReplayBenchConfig.model_validate(
            {
                "replay": {
                    "trace_path": "source.jsonl",
                    "interval_mode": "lognormal",
                }
            }
        )


def test_removed_timing_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timing_mode"):
        ReplayBenchConfig.model_validate({"replay": {"trace_path": "source.jsonl", "timing_mode": "closed_loop"}})


def test_trace_mode_rejects_unused_lognormal_anchors() -> None:
    with pytest.raises(ValidationError, match="anchors require interval_mode=lognormal"):
        ReplayBenchConfig.model_validate(
            {
                "replay": {
                    "trace_path": "source.jsonl",
                    "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
                }
            }
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_removed_cache_control_option_is_rejected(enabled: bool) -> None:
    with pytest.raises(ValidationError, match="enable_cache_control"):
        ReplayBenchConfig.model_validate({"replay": {"trace_path": "source.jsonl", "enable_cache_control": enabled}})


def test_inferact_requires_chat_completions_endpoint() -> None:
    payload = {
        "backend": {"endpoint": "/v1/messages"},
        "replay": {
            "trace_type": "inferact_codex_swebenchpro",
            "trace_path": "source.json",
            "prompt_shape": "trace_record",
            "interval_mode": "lognormal",
            "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
        },
    }
    with pytest.raises(ValidationError, match="backend.endpoint=/v1/chat/completions"):
        ReplayBenchConfig.model_validate(payload)


def test_trace_path_is_resolved_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "replay.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
replay:
  trace_path: ../traces/source.jsonl
""",
        encoding="utf-8",
    )

    config = load_replay_config(config_path)

    assert config.replay.trace_path == tmp_path / "traces" / "source.jsonl"
