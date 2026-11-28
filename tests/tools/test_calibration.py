# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate calibration metadata, DP scope, fitting, and final profile assembly."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import httpx
import pytest

from tests.helpers.marks import test_marks
from tools.calibration import calibrate_decode, calibrate_prefill
from tools.calibration.calibrate_decode import fit_decode, parser as decode_parser
from tools.calibration.calibrate_prefill import fit_prefill, parser as prefill_parser
from tools.calibration.metadata import (
    KV_CAPACITY_OVERRIDE_ENV,
    PREEMPTIONS_COUNTER,
    assemble_profile,
    collect_metadata,
    collect_throughput_counters,
    counter_delta,
    infer_kv_capacity_tokens,
    metadata_hash,
    persist_metadata,
    resolve_calibration_engine,
)

pytestmark = test_marks(hardware="cpu", ci="ready")


@pytest.fixture(autouse=True)
def clear_capacity_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep calibration tests independent of the operator's shell environment."""
    monkeypatch.delenv(KV_CAPACITY_OVERRIDE_ENV, raising=False)


METRICS = """
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{engine="0",block_size="16",num_gpu_blocks="100"} 1
vllm:cache_config_info{engine="1",block_size="16",num_gpu_blocks="100"} 1
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name="model-a",engine="0"} 100
vllm:prompt_tokens_total{model_name="model-a",engine="1"} 200
# TYPE vllm:prompt_tokens_cached counter
vllm:prompt_tokens_cached_total{model_name="model-a",engine="0"} 40
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{model_name="model-a",engine="0"} 30
# TYPE vllm:request_success counter
vllm:request_success_total{model_name="model-a",engine="0",finished_reason="stop"} 2
"""


def _transport(request: httpx.Request, *, metrics: str = METRICS) -> httpx.Response:
    if request.url.path == "/version":
        return httpx.Response(200, json={"version": "0.23.0"})
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "model-a",
                        "root": "/models/a",
                        "max_model_len": 131072,
                    }
                ]
            },
        )
    if request.url.path == "/metrics":
        return httpx.Response(200, text=metrics)
    return httpx.Response(404)


def _collect_metadata(metrics: str = METRICS) -> dict[str, object]:
    async def collect() -> dict[str, object]:
        transport = httpx.MockTransport(partial(_transport, metrics=metrics))
        async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
            metadata, _ = await collect_metadata(client, "model-a")
            return metadata

    return asyncio.run(collect())


def test_http_metadata_discovers_homogeneous_internal_dp() -> None:
    """Endpoint metadata should discover ranks without hashing dynamic values."""

    async def run() -> tuple[dict[str, object], tuple[int, ...], dict[str, float]]:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(_transport)) as client:
            metadata, engines = await collect_metadata(client, "model-a")
            counters = await collect_throughput_counters(client, 0)
            return metadata, engines, counters

    metadata, engines, counters = asyncio.run(run())
    assert engines == (0, 1)
    assert metadata["deployment"] == {
        "engine_count": 2,
        "cache_config": {"block_size": "16", "num_gpu_blocks": "100"},
        "kv_capacity": {"tokens": 800, "source": "block_arithmetic"},
        "throughput_metric_names": [
            "vllm:generation_tokens_total",
            "vllm:prompt_tokens_cached_total",
            "vllm:prompt_tokens_total",
            "vllm:request_success_total",
        ],
    }
    assert counters["vllm:prompt_tokens_total"] == 100
    assert PREEMPTIONS_COUNTER not in counters
    assert infer_kv_capacity_tokens(metadata) == 800


def test_capacity_prefers_reported_logical_tokens() -> None:
    """A reported logical KV size wins over block arithmetic.

    Compressed or sparse attention backends can report a block_size that is not the
    number of logical tokens per block. On such a deployment the legacy formula can be
    an order of magnitude too small and silently removes every long-context decode
    case from the calibration matrix.
    """
    metadata = {
        "deployment": {
            "engine_count": 4,
            "cache_config": {
                "block_size": "2",
                "num_gpu_blocks": "125403",
                "kv_cache_size_tokens": "1457130",
            },
        }
    }
    assert infer_kv_capacity_tokens(metadata) == 1457130


@pytest.mark.parametrize(("label", "tokens"), [("1457130", 1457130), ("800.0", 800), ("  '800.0'  ", 800)])
def test_metadata_records_logical_capacity_source(label: str, tokens: int) -> None:
    """Even when both paths yield 800 tokens, metadata must identify the label as the source."""
    metrics = METRICS.replace('block_size="16"', f'block_size="16",kv_cache_size_tokens="{label}"')
    metadata = _collect_metadata(metrics)
    assert metadata["deployment"]["kv_capacity"] == {"tokens": tokens, "source": "kv_cache_size_tokens"}
    assert infer_kv_capacity_tokens(metadata) == tokens


@pytest.mark.parametrize("value", ["None", "none", "", "null", "0", "-1", "inf", "Infinity", "-inf", "1e400", "nan"])
def test_capacity_ignores_invalid_logical_tokens(value: str) -> None:
    """Placeholder, nonpositive and non-finite labels fall back to block arithmetic."""
    metadata = {
        "deployment": {
            "engine_count": 2,
            "cache_config": {
                "block_size": "16",
                "num_gpu_blocks": "100",
                "kv_cache_size_tokens": value,
            },
        }
    }
    assert infer_kv_capacity_tokens(metadata) == 800


def test_capacity_override_environment_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator override wins over both the label and the block arithmetic."""
    monkeypatch.setenv("AGENTINFER_KV_CAPACITY_TOKENS", "999999")
    metadata = {
        "deployment": {
            "engine_count": 2,
            "cache_config": {
                "block_size": "16",
                "num_gpu_blocks": "100",
                "kv_cache_size_tokens": "1457130",
            },
        }
    }
    assert infer_kv_capacity_tokens(metadata) == 999999


@pytest.mark.parametrize(
    "value", ["", " ", "0", "-1", "1457130 tokens", "1,457,130", "inf", "Infinity", "1e400", "nan", "1.5"]
)
def test_capacity_rejects_invalid_override(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly configured invalid capacity must never silently fall back."""
    monkeypatch.setenv(KV_CAPACITY_OVERRIDE_ENV, value)
    metadata = {
        "deployment": {
            "engine_count": 2,
            "cache_config": {"block_size": "16", "num_gpu_blocks": "100", "kv_cache_size_tokens": "1457130"},
        }
    }
    with pytest.raises(ValueError, match=KV_CAPACITY_OVERRIDE_ENV) as exc:
        infer_kv_capacity_tokens(metadata)
    assert repr(value) in str(exc.value)


def test_capacity_override_is_captured_and_hashed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Changing the override must reject mixing stages in the same results directory."""
    baseline = _collect_metadata()
    monkeypatch.setenv(KV_CAPACITY_OVERRIDE_ENV, "999999")
    overridden = _collect_metadata()
    assert overridden["deployment"]["kv_capacity"] == {"tokens": 999999, "source": KV_CAPACITY_OVERRIDE_ENV}
    assert metadata_hash(baseline) != metadata_hash(overridden)
    persist_metadata(tmp_path, overridden)
    monkeypatch.setenv(KV_CAPACITY_OVERRIDE_ENV, "1000000")
    assert infer_kv_capacity_tokens(overridden) == 999999
    with pytest.raises(ValueError, match="metadata do not match"):
        persist_metadata(tmp_path, _collect_metadata())
    # Even an override equal to the derived capacity records a different source.
    monkeypatch.setenv(KV_CAPACITY_OVERRIDE_ENV, "800")
    assert metadata_hash(baseline) != metadata_hash(_collect_metadata())


@pytest.mark.parametrize(("stage", "module"), [("decode", calibrate_decode), ("prefill", calibrate_prefill)])
@pytest.mark.parametrize(("override", "error"), [("1,457,130", KV_CAPACITY_OVERRIDE_ENV), ("1457130", "do not match")])
def test_overwrite_preserves_completed_stage_on_capacity_error(
    stage, module, override, error, tmp_path, monkeypatch
) -> None:
    """Capacity validation must leave every previous artifact intact."""
    directory = tmp_path / stage
    directory.mkdir()
    persist_metadata(tmp_path, _collect_metadata())
    for path in (directory / "requests.jsonl", directory / "result.json", tmp_path / "calibration.json"):
        path.write_text('{"previous": "result"}\n', encoding="utf-8")

    def contents():
        return {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = contents()
    monkeypatch.setenv(KV_CAPACITY_OVERRIDE_ENV, override)
    monkeypatch.setattr(httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(_transport)))
    args = module.parser().parse_args(["--model", "model-a", "--output-dir", str(tmp_path), "--overwrite"])
    with pytest.raises(ValueError, match=error):
        asyncio.run(module.calibrate(args))
    assert contents() == before


def test_decode_records_preemptions_on_calibrated_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Read the target rank before and after the matrix and persist the delta with the fit."""
    counts = iter([(10, 100), (10, 100), (12, 150)])

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/metrics":
            return _transport(request)
        counters = "".join(f'{PREEMPTIONS_COUNTER}{{engine="{rank}"}} {n}\n' for rank, n in enumerate(next(counts)))
        return _transport(request, metrics=METRICS + counters)

    async def run_case(client, args, context, batch, repeat, case_id, recorder, request_rank):
        return {"batch_size": batch, "context_tokens_per_request": context, "estimated_iteration_seconds": 0.1}

    monkeypatch.setattr(httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(calibrate_decode, "run_case", run_case)
    args = decode_parser().parse_args(
        ["--model", "model-a", "--output-dir", str(tmp_path), "--batch-sizes", "1,2", "--context-lengths", "16,32"]
    )
    asyncio.run(calibrate_decode.calibrate(args))
    result = json.loads((tmp_path / "decode/result.json").read_text(encoding="utf-8"))["calibration"]
    assert result["throughput_counter_delta"][PREEMPTIONS_COUNTER] == 2
    assert result["throughput_counters_after"][PREEMPTIONS_COUNTER] == 12
    assert "2 preemptions on engine 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ({}, {PREEMPTIONS_COUNTER: 2}, {}),
        ({PREEMPTIONS_COUNTER: 2}, {}, {}),
        ({PREEMPTIONS_COUNTER: 3}, {PREEMPTIONS_COUNTER: 1}, {}),
        ({PREEMPTIONS_COUNTER: 2}, {PREEMPTIONS_COUNTER: 2}, {PREEMPTIONS_COUNTER: 0}),
    ],
)
def test_preemptions_require_two_comparable_snapshots(before, after, expected) -> None:
    assert counter_delta(before, after) == expected


def test_cli_uses_repository_results_and_observed_capacity() -> None:
    """Output should have a safe default and decode capacity should not be manual."""
    prefill = prefill_parser().parse_args(["--model", "model-a"])
    decode = decode_parser().parse_args(["--model", "model-a"])
    assert prefill.output_dir.as_posix() == "tools/calibration/results"
    assert decode.output_dir.as_posix() == "tools/calibration/results"
    assert prefill.repeats == 2
    assert decode.repeats == 1
    assert decode.output_tokens == 128
    assert decode.warmup_settle_seconds == 1.0
    assert prefill.data_parallel_rank is None
    assert decode.data_parallel_rank is None
    assert not hasattr(decode, "kv_capacity_tokens")


@pytest.mark.parametrize(
    ("build_parser", "invalid_option"),
    [
        (prefill_parser, ("--repeats", "0")),
        (prefill_parser, ("--timeout", "0")),
        (prefill_parser, ("--data-parallel-rank", "-1")),
        (decode_parser, ("--output-tokens", "0")),
        (decode_parser, ("--warmup-settle-seconds", "-1")),
        (decode_parser, ("--repeats", "0")),
        (decode_parser, ("--max-kv-fraction", "0")),
        (decode_parser, ("--max-kv-fraction", "1.1")),
        (decode_parser, ("--timeout", "nan")),
        (decode_parser, ("--data-parallel-rank", "-1")),
    ],
)
def test_cli_rejects_invalid_numeric_options(
    build_parser: Callable[[], argparse.ArgumentParser],
    invalid_option: tuple[str, str],
) -> None:
    """Scalar options should reject invalid ranges before calibration starts."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "model-a", *invalid_option])


def test_engine_selection_omits_unnecessary_dp_header() -> None:
    """DP=1 should use native routing while multi-DP pins one discovered rank."""
    assert resolve_calibration_engine(None, (0,)) == (0, None)
    assert resolve_calibration_engine(None, (0, 1)) == (0, 0)
    assert resolve_calibration_engine(1, (0, 1)) == (1, 1)


def test_metadata_hash_is_key_order_independent() -> None:
    """Canonical JSON order should be the only Hash input serialization."""
    assert metadata_hash({"a": 1, "b": {"x": 2}}) == metadata_hash({"b": {"x": 2}, "a": 1})


def test_profile_has_exactly_three_top_level_fields(tmp_path) -> None:
    """Two matching stage results should assemble one reusable DP profile."""
    metadata = {"schema_version": 1, "backend": {"type": "vllm"}}
    digest = persist_metadata(tmp_path, metadata)
    scope = {
        "kind": "homogeneous_internal_dp",
        "calibrated_engine": 0,
        "reusable_engine_ids": [0, 1],
    }
    for stage in ("prefill", "decode"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "metadata_hash": digest,
                    "scope": scope,
                    "calibration": {"stage": stage},
                }
            ),
            encoding="utf-8",
        )
    output = assemble_profile(tmp_path)
    assert output is not None
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert set(profile) == {"metadata", "metadata_hash", "calibration"}
    assert profile["calibration"]["scope"] == scope


def test_prefill_and_decode_models_fit_exact_points() -> None:
    """Reference points should reproduce the deployable formulas."""
    prefill = fit_prefill(
        [
            {"prompt_tokens": 1000, "median_ttft_seconds": 6, "sample_count": 1},
            {"prompt_tokens": 2000, "median_ttft_seconds": 17, "sample_count": 1},
            {"prompt_tokens": 3000, "median_ttft_seconds": 34, "sample_count": 1},
        ]
    )
    assert prefill["fit"]["rmse_seconds"] < 1e-10
    decode = fit_decode(
        [
            {"batch_size": 1, "total_context_tokens": 1000, "median_iteration_seconds": 0.021},
            {"batch_size": 2, "total_context_tokens": 2000, "median_iteration_seconds": 0.032},
            {"batch_size": 1, "total_context_tokens": 2000, "median_iteration_seconds": 0.022},
        ]
    )
    assert decode["fit"]["rmse_seconds"] < 1e-10


def test_prefill_model_rejects_rank_deficient_prompt_lengths() -> None:
    """A quadratic Prefill model requires at least three distinct lengths."""
    with pytest.raises(ValueError, match="three distinct prompt lengths"):
        fit_prefill(
            [
                {"prompt_tokens": 1000, "median_ttft_seconds": 1, "sample_count": 1},
                {"prompt_tokens": 1000, "median_ttft_seconds": 2, "sample_count": 1},
                {"prompt_tokens": 2000, "median_ttft_seconds": 3, "sample_count": 1},
            ]
        )
