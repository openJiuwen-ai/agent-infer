#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Collect stable vLLM HTTP metadata and per-DP throughput observations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

from tools.calibration.common import write_json

THROUGHPUT_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)
PREEMPTIONS_COUNTER = "vllm:num_preemptions_total"

# Optional operator override for the per-EngineCore logical KV capacity. It exists for
# deployments whose cache_config_info does not expose a trustworthy logical size.
KV_CAPACITY_OVERRIDE_ENV = "AGENTINFER_KV_CAPACITY_TOKENS"


async def collect_metadata(client: httpx.AsyncClient, model: str) -> tuple[dict[str, object], tuple[int, ...]]:
    """Return hashable deployment metadata and the engine IDs exposed by /metrics.

    Both stages record the per-engine token count and its source in
    ``deployment.kv_capacity``. An invalid capacity override raises ValueError.
    Without an override, unusable logical and block capacity fields also raise ValueError.
    """
    version_response = await client.get("/version")
    version_response.raise_for_status()
    version_body = version_response.json()
    if not isinstance(version_body, dict) or not isinstance(version_body.get("version"), str):
        raise ValueError("/version did not return a string version")

    models_response = await client.get("/v1/models")
    models_response.raise_for_status()
    models_body = models_response.json()
    entries = models_body.get("data") if isinstance(models_body, dict) else None
    if not isinstance(entries, list):
        raise ValueError("/v1/models did not return a model list")
    selected = next(
        (entry for entry in entries if isinstance(entry, dict) and entry.get("id") == model),
        None,
    )
    if selected is None:
        raise ValueError(f"model {model!r} is not served by /v1/models")

    metrics_response = await client.get("/metrics")
    metrics_response.raise_for_status()
    samples = _parse_samples(metrics_response.text)
    engine_ids = _engine_ids(samples)
    cache_configs = _cache_configs(samples, engine_ids)
    metric_names = sorted(name for name in THROUGHPUT_COUNTERS if any(sample[0] == name for sample in samples))
    metadata: dict[str, object] = {
        "schema_version": 1,
        "backend": {"type": "vllm", "version": version_body["version"]},
        "model": {key: selected[key] for key in ("id", "root", "max_model_len") if key in selected},
        "deployment": {
            "engine_count": len(engine_ids),
            "cache_config": cache_configs,
            "throughput_metric_names": metric_names,
        },
    }
    capacity, source = _resolve_kv_capacity_tokens(metadata)
    metadata["deployment"]["kv_capacity"] = {"tokens": capacity, "source": source}
    return metadata, engine_ids


async def collect_throughput_counters(client: httpx.AsyncClient, engine: int) -> dict[str, float]:
    """Read throughput counters and, when exposed, preemptions for one EngineCore."""
    response = await client.get("/metrics")
    response.raise_for_status()
    values = dict.fromkeys(THROUGHPUT_COUNTERS, 0.0)
    found_engine = False
    for name, labels, value in _parse_samples(response.text):
        if labels.get("engine") != str(engine):
            continue
        found_engine = True
        if name in THROUGHPUT_COUNTERS or name == PREEMPTIONS_COUNTER:
            values[name] = values.get(name, 0.0) + value
    if not found_engine:
        raise ValueError(f"/metrics does not expose engine={engine}")
    return values


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Calculate stage deltas, omitting preemptions if missing or reset during the stage."""
    deltas = {}
    for name, start in before.items():
        if name == PREEMPTIONS_COUNTER and (name not in after or after[name] < start):
            continue
        deltas[name] = max(after.get(name, 0) - start, 0)
    return deltas


def metadata_hash(metadata: dict[str, object]) -> str:
    """Return SHA-256 over canonical UTF-8 metadata JSON."""
    canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def persist_metadata(root: Path, metadata: dict[str, object]) -> str:
    """Persist metadata once and reject mixing stages from different deployments."""
    path = root / "metadata.json"
    digest = metadata_hash(metadata)
    record = {"metadata": metadata, "metadata_hash": digest}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != record:
            raise ValueError("prefill and decode metadata do not match")
    else:
        write_json(path, record)
    return digest


def prepare_stage(root: Path, stage: str, overwrite: bool, metadata: dict[str, object]) -> tuple[Path, str]:
    """Check the shared metadata before replacing any completed stage artifacts."""
    path = root / stage
    occupied = path.exists() and any(path.iterdir())
    if occupied and not overwrite:
        raise FileExistsError(f"stage directory is not empty: {path}")
    digest = persist_metadata(root, metadata)
    if occupied:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path, digest


def assemble_profile(root: Path) -> Path | None:
    """Create the three-field profile after both measurement stages complete."""
    metadata_path = root / "metadata.json"
    prefill_path = root / "prefill" / "result.json"
    decode_path = root / "decode" / "result.json"
    if not all(path.exists() for path in (metadata_path, prefill_path, decode_path)):
        return None
    metadata_record = json.loads(metadata_path.read_text(encoding="utf-8"))
    prefill = json.loads(prefill_path.read_text(encoding="utf-8"))
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    if prefill["metadata_hash"] != decode["metadata_hash"]:
        raise ValueError("prefill and decode metadata hashes do not match")
    if prefill["scope"] != decode["scope"]:
        raise ValueError("prefill and decode calibration scopes do not match")
    profile = {
        "metadata": metadata_record["metadata"],
        "metadata_hash": metadata_record["metadata_hash"],
        "calibration": {
            "scope": prefill["scope"],
            "prefill": prefill["calibration"],
            "decode": decode["calibration"],
        },
    }
    output = root / "calibration.json"
    write_json(output, profile)
    return output


def validate_engine(engine: int, engine_ids: tuple[int, ...]) -> None:
    """Require calibration traffic to target an exposed internal-DP rank."""
    if engine not in engine_ids:
        raise ValueError(f"engine {engine} is not present in /metrics: {engine_ids}")


def resolve_calibration_engine(requested_engine: int | None, engine_ids: tuple[int, ...]) -> tuple[int, int | None]:
    """Select the observed engine and the optional request-routing header value."""
    engine = engine_ids[0] if requested_engine is None else requested_engine
    validate_engine(engine, engine_ids)
    request_rank = engine if len(engine_ids) > 1 else None
    return engine, request_rank


def _positive_int(value: object) -> int | None:
    """Parse a positive integer from a metrics label value, or return None."""
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    if text in ("", "None", "none", "null"):
        return None
    try:
        parsed = int(text)
    except ValueError:
        try:
            parsed = int(float(text))
        except (ValueError, OverflowError):
            return None
    return parsed if parsed > 0 else None


def infer_kv_capacity_tokens(metadata: dict[str, object]) -> int:
    """Return the capacity captured with metadata, or resolve uncaptured metadata."""
    deployment = metadata.get("deployment")
    if isinstance(deployment, dict) and "kv_capacity" in deployment:
        return deployment["kv_capacity"]["tokens"]
    return _resolve_kv_capacity_tokens(metadata)[0]


def _resolve_kv_capacity_tokens(metadata: dict[str, object]) -> tuple[int, str]:
    """Return one EngineCore rank's logical HBM KV capacity in tokens.

    Resolution order:

    1. ``AGENTINFER_KV_CAPACITY_TOKENS`` (explicit operator override);
    2. ``cache_config_info.kv_cache_size_tokens`` -- the logical per-engine capacity
       reported by the backend itself;
    3. ``block_size * num_gpu_blocks / engine_count`` -- legacy fallback.

    Step 3 is only a fallback because some backends report a ``block_size`` that is not
    the number of logical tokens per block. Compressed or sparse attention layouts can
    make that arithmetic off by more than an order of magnitude, and the resulting
    capacity silently truncates the decode calibration matrix: every case whose
    ``batch * context`` exceeds ``max_kv_fraction * capacity`` is skipped, so the fitted
    context coefficient collapses to zero and the profile underestimates decode cost
    on production-length contexts.
    """
    deployment = metadata.get("deployment")
    cache_config = deployment.get("cache_config") if isinstance(deployment, dict) else None
    if not isinstance(cache_config, dict):
        raise ValueError("/metrics does not expose vllm:cache_config_info")

    raw_override = os.environ.get(KV_CAPACITY_OVERRIDE_ENV)
    if raw_override is not None:
        try:
            override = int(raw_override)
        except ValueError:
            override = 0
        if override <= 0:
            raise ValueError(f"{KV_CAPACITY_OVERRIDE_ENV} must be a positive integer, got {raw_override!r}")
        return override, KV_CAPACITY_OVERRIDE_ENV

    logical = _positive_int(cache_config.get("kv_cache_size_tokens"))
    if logical is not None:
        # vLLM reports this per EngineCore, so it is already the pinned rank's capacity.
        return logical, "kv_cache_size_tokens"

    try:
        block_size = int(str(cache_config["block_size"]))
        num_gpu_blocks = int(str(cache_config["num_gpu_blocks"]))
        engine_count = int(deployment["engine_count"])
    except (KeyError, ValueError) as exc:
        raise ValueError("cache_config_info and deployment must expose integer capacity fields") from exc
    if engine_count <= 0:
        raise ValueError("engine_count must be positive")
    # In an internally managed DP deployment, the aggregated /metrics endpoint
    # repeats the deployment-wide num_gpu_blocks value for every engine label.
    # Calibration pins one EngineCore, so its usable share is the aggregate
    # logical token capacity divided by the homogeneous engine count.
    capacity = block_size * num_gpu_blocks // engine_count
    if capacity <= 0:
        raise ValueError("derived KV capacity must be positive")
    return capacity, "block_arithmetic"


def _parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus exposition into stable sample tuples."""
    return [
        (sample.name, dict(sample.labels), float(sample.value))
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    ]


def _engine_ids(samples: list[tuple[str, dict[str, str], float]]) -> tuple[int, ...]:
    """Discover numeric EngineCore ranks from metric labels."""
    values = {int(labels["engine"]) for _, labels, _ in samples if labels.get("engine", "").isdigit()}
    if not values:
        raise ValueError("/metrics does not expose numeric engine labels")
    return tuple(sorted(values))


def _cache_configs(samples: list[tuple[str, dict[str, str], float]], engine_ids: tuple[int, ...]) -> dict[str, str]:
    """Return one common cache config and reject heterogeneous internal DP."""
    configs: dict[int, dict[str, str]] = {}
    for name, labels, _ in samples:
        if name != "vllm:cache_config_info" or "engine" not in labels:
            continue
        engine = int(labels["engine"])
        configs[engine] = {key: value for key, value in labels.items() if key != "engine"}
    if not configs:
        raise ValueError("/metrics does not expose vllm:cache_config_info")
    missing = set(engine_ids).difference(configs)
    if missing:
        raise ValueError(f"cache_config_info is missing engines: {sorted(missing)}")
    first = configs[engine_ids[0]]
    if any(configs[engine] != first for engine in engine_ids[1:]):
        raise ValueError("initial calibration supports only homogeneous internal DP")
    return dict(sorted(first.items()))
