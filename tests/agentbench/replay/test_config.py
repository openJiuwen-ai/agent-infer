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


def test_backend_accepts_chat_template_kwargs() -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "backend": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "clear_thinking": True,
                }
            },
            "replay": {"trace_path": "source.jsonl"},
        }
    )

    assert config.backend.chat_template_kwargs == {
        "enable_thinking": False,
        "clear_thinking": True,
    }


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
    assert config.replay.prompt_shape == "agentinfer_synthetic"
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
    assert config.replay.prompt_calibration_tolerance_tokens == 0
    assert "inferact_synthetic_calibration_mode" not in type(config.replay).model_fields
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


def test_replay_defaults_to_agentinfer_shape() -> None:
    config = ReplayBenchConfig.model_validate({"replay": {"trace_path": "source.jsonl"}})

    assert config.replay.prompt_shape == "agentinfer_synthetic"


@pytest.mark.parametrize(
    ("trace_type", "shape"),
    [
        ("inferact_codex_swebenchpro", "inferact_synthetic"),
        ("agentinfer", "agentinfer_synthetic"),
        ("tracelab", "tracelab_synthetic"),
        ("agentX", "agentX_synthetic"),
    ],
)
def test_replay_yaml_derives_prompt_shape_and_round_trips(tmp_path: Path, trace_type: str, shape: str) -> None:
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(
        f"""
replay:
  trace_type: {trace_type}
  trace_path: source.json
  interval_mode: lognormal
  interval_lognormal:
    p50_seconds: 2
    p95_seconds: 30
    p99_seconds: 90
""",
        encoding="utf-8",
    )

    config = load_replay_config(config_path)

    assert config.replay.prompt_shape == shape
    payload = config.model_dump(mode="json")
    assert "prompt_shape" not in payload["replay"]
    assert ReplayBenchConfig.model_validate(payload).replay.prompt_shape == shape
    with pytest.raises(AttributeError):
        config.replay.prompt_shape = "agentinfer_synthetic"


@pytest.mark.parametrize("shape", ["agentinfer_synthetic", "inferact_synthetic", "trace_record", "legacy"])
def test_replay_rejects_explicit_prompt_shape(tmp_path: Path, shape: str) -> None:
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(f"replay:\n  trace_path: source.jsonl\n  prompt_shape: {shape}\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_replay_config(config_path)

    assert exc_info.value.errors()[0]["loc"] == ("replay", "prompt_shape")
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_inferact_still_requires_lognormal_intervals() -> None:
    with pytest.raises(ValidationError, match="requires interval_mode=lognormal"):
        ReplayBenchConfig.model_validate(
            {"replay": {"trace_type": "inferact_codex_swebenchpro", "trace_path": "source.json"}}
        )


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


@pytest.mark.parametrize("field", ["max_input_tokens", "max_output_tokens"])
@pytest.mark.parametrize("value", ["null", "1024"])
def test_removed_token_caps_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(f"replay:\n  trace_path: source.jsonl\n  {field}: {value}\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_replay_config(config_path)

    assert exc_info.value.errors()[0]["loc"] == ("replay", field)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


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
            "interval_mode": "lognormal",
            "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
        },
    }
    with pytest.raises(ValidationError, match="backend.endpoint=/v1/chat/completions"):
        ReplayBenchConfig.model_validate(payload)


@pytest.mark.parametrize("tolerance", [None, 0, 1, 8])
def test_inferact_rejects_nonzero_tolerance_instead_of_normalizing(tolerance) -> None:
    replay = {
        "trace_type": "inferact_codex_swebenchpro",
        "trace_path": "source.json",
        "interval_mode": "lognormal",
        "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
    }
    if tolerance is not None:
        replay["prompt_calibration_tolerance_tokens"] = tolerance
    if tolerance:
        with pytest.raises(
            ValidationError, match="inferact_codex_swebenchpro requires prompt_calibration_tolerance_tokens=0"
        ):
            ReplayBenchConfig.model_validate({"replay": replay})
        assert replay["prompt_calibration_tolerance_tokens"] == tolerance
        synthetic = ReplayBenchConfig.model_validate(
            {"replay": {"trace_path": "source.jsonl", "prompt_calibration_tolerance_tokens": tolerance}}
        )
        assert synthetic.replay.prompt_calibration_tolerance_tokens == tolerance
    else:
        assert ReplayBenchConfig.model_validate({"replay": replay}).replay.prompt_calibration_tolerance_tokens == 0


def test_tracelab_requires_exact_calibration_and_chat_endpoint() -> None:
    valid = {
        "backend": {"endpoint": "/v1/chat/completions"},
        "replay": {
            "trace_type": "tracelab",
            "trace_path": "rounds.jsonl",
            "prompt_calibration_tolerance_tokens": 0,
        },
    }
    assert ReplayBenchConfig.model_validate(valid).replay.trace_type == "tracelab"

    for section, field, value, message in (
        ("replay", "prompt_calibration_tolerance_tokens", 1, "tolerance_tokens=0"),
        ("backend", "endpoint", "/v1/messages", "endpoint=/v1/chat/completions"),
    ):
        payload = {name: dict(settings) for name, settings in valid.items()}
        payload[section][field] = value
        with pytest.raises(ValidationError, match=message):
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
