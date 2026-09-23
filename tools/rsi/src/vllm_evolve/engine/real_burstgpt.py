"""Official BurstGPT preparation for the pure real-vLLM evolution path.

This module never imports the Frontier scenario builder. It keeps chronological
train/validation/calibration fragments separate, hides the fourth band until a
winner SHA is frozen, and scales only inter-arrival time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from vllm_evolve.bench.datasets import burstgpt
from vllm_evolve.bench.datasets.burstgpt import expand_scaled_payload

__all__ = [
    "expand_scaled_payload",
    "materialize_heldout_fragment",
    "materialize_search_fragments",
    "scale_fragment",
]

OFFICIAL_RELEASE = "v2.0"
OFFICIAL_FILENAME = "BurstGPT_1.csv"
OFFICIAL_URL = (
    "https://github.com/HPMLL/BurstGPT/releases/download/"
    f"{OFFICIAL_RELEASE}/{OFFICIAL_FILENAME}"
)
SPLITS = ("train", "validation", "calibration")
HELDOUT_SPLIT = "test"
TOTAL_BANDS = 4
TOKEN_GENERATOR_VERSION = "burstgpt-unique-prompt-v1"


def _canonical_sha(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_payload(path: Path, core: dict) -> dict:
    payload = {**core, "payload_sha256": _canonical_sha(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _fragment_core(
    source: Path,
    split: str,
    dataset,
    *,
    selection: dict | None = None,
) -> dict:
    rows = list(dataset.requests)
    if len(rows) < 2:
        raise ValueError("BurstGPT fragment must contain at least two requests")
    return {
        "schema_version": 2,
        "source_kind": "official_burstgpt",
        "official_release": OFFICIAL_RELEASE,
        "official_filename": OFFICIAL_FILENAME,
        "official_url": OFFICIAL_URL,
        "source_sha256": burstgpt.source_sha256(source),
        "split": split,
        "chronological_band": {
            "index": (*SPLITS, HELDOUT_SPLIT).index(split),
            "total_bands": TOTAL_BANDS,
        },
        "dataset_name": dataset.name,
        "selection": dict(selection or {"strategy": "densest"}),
        "token_generator_version": TOKEN_GENERATOR_VERSION,
        "requests": [
            {
                "request_id": request.request_id,
                "source_arrival_s": float(request.arrival_s),
                "num_prompt_tokens": int(request.prompt_tokens),
                "num_output_tokens": int(request.output_tokens),
            }
            for request in rows
        ],
        "source_window": {
            "first_request_id": rows[0].request_id,
            "last_request_id": rows[-1].request_id,
            "start_s": float(rows[0].arrival_s),
            "end_s": float(rows[-1].arrival_s),
            "duration_s": float(rows[-1].arrival_s - rows[0].arrival_s),
            "requests": len(rows),
        },
        "integrity": {
            "arrival_scaling_only": True,
            "token_lengths_unchanged": True,
            "request_order_unchanged": True,
            "tenant_or_prefix_labels_inferred": False,
        },
    }


def materialize_search_fragments(
    source_path: str | Path,
    out_dir: str | Path,
    *,
    fragment_size: int = 512,
    minimum_fragment_size: int = 256,
    selection_strategy: str = "densest",
    target_mean_total_tokens: float | None = None,
    max_request_total_tokens: int | None = None,
    max_request_output_tokens: int | None = None,
) -> dict:
    """Write bands 0..2 and only a locator for the still-hidden test band."""
    if fragment_size < minimum_fragment_size:
        raise ValueError(
            f"real BurstGPT fragments require at least {minimum_fragment_size} requests"
        )
    source = Path(source_path).resolve()
    out = Path(out_dir).resolve()
    fragments = burstgpt.extract_bursty_fragments(
        source,
        fragment_size=fragment_size,
        split_names=SPLITS,
        band_indices=(0, 1, 2),
        total_bands=TOTAL_BANDS,
        selection_strategy=selection_strategy,
        target_mean_total_tokens=target_mean_total_tokens,
        max_request_total_tokens=max_request_total_tokens,
        max_request_output_tokens=max_request_output_tokens,
    )
    selection = {
        "strategy": selection_strategy,
        "target_mean_total_tokens": target_mean_total_tokens,
        "max_request_total_tokens": max_request_total_tokens,
        "max_request_output_tokens": max_request_output_tokens,
    }
    files = {}
    for split in SPLITS:
        path = out / "base" / f"{split}.json"
        payload = _write_payload(
            path,
            _fragment_core(
                source,
                split,
                fragments[split],
                selection=selection,
            ),
        )
        files[split] = {
            "path": str(path),
            "payload_sha256": payload["payload_sha256"],
            "requests": len(payload["requests"]),
            "source_window": payload["source_window"],
        }
    manifest_core = {
        "schema_version": 2,
        "source_kind": "official_burstgpt",
        "official_release": OFFICIAL_RELEASE,
        "official_url": OFFICIAL_URL,
        "source_path": str(source),
        "source_sha256": burstgpt.source_sha256(source),
        "fragment_size": int(fragment_size),
        "selection": selection,
        "files": files,
        "heldout": {
            "materialized": False,
            "split": HELDOUT_SPLIT,
            "chronological_band": {"index": 3, "total_bands": TOTAL_BANDS},
            "requires_frozen_winner_sha256": True,
        },
    }
    manifest = _write_payload(out / "scaled_workload_manifest.json", manifest_core)
    return manifest


def materialize_heldout_fragment(
    source_path: str | Path,
    out_path: str | Path,
    *,
    frozen_winner_sha256: str,
    fragment_size: int = 512,
    selection_strategy: str = "densest",
    target_mean_total_tokens: float | None = None,
    max_request_total_tokens: int | None = None,
    max_request_output_tokens: int | None = None,
    chronological_band_index: int = 3,
    total_bands: int = TOTAL_BANDS,
) -> dict:
    if len(frozen_winner_sha256) != 64:
        raise ValueError("heldout materialization requires a frozen 64-char winner SHA")
    source = Path(source_path).resolve()
    fragment = burstgpt.extract_bursty_fragments(
        source,
        fragment_size=fragment_size,
        split_names=(HELDOUT_SPLIT,),
        band_indices=(chronological_band_index,),
        total_bands=total_bands,
        selection_strategy=selection_strategy,
        target_mean_total_tokens=target_mean_total_tokens,
        max_request_total_tokens=max_request_total_tokens,
        max_request_output_tokens=max_request_output_tokens,
    )[HELDOUT_SPLIT]
    core = _fragment_core(
        source,
        HELDOUT_SPLIT,
        fragment,
        selection={
            "strategy": selection_strategy,
            "target_mean_total_tokens": target_mean_total_tokens,
            "max_request_total_tokens": max_request_total_tokens,
            "max_request_output_tokens": max_request_output_tokens,
        },
    )
    # A failed post-freeze held-out run consumes that fragment.  A later
    # winner must be testable on a disjoint chronological sub-band without
    # weakening the same freeze-before-materialization boundary.  The default
    # remains the original fourth-of-four split; callers must opt in to a
    # narrower later band explicitly.
    core["chronological_band"] = {
        "index": int(chronological_band_index),
        "total_bands": int(total_bands),
    }
    core["heldout_lock"] = {
        "winner_source_sha256": frozen_winner_sha256,
        "materialized_after_winner_freeze": True,
    }
    return _write_payload(Path(out_path).resolve(), core)


def scale_fragment(
    base_payload: dict,
    *,
    target_qps: float,
    min_duration_s: float = 0.0,
    min_requests: int = 512,
    scenario: str,
) -> dict:
    """Return a compact deterministic replay plan; token lengths never change.

    ``min_duration_s`` controls the nominal *arrival* span only.  It does not
    assert the real measured-window duration, which is enforced from runtime
    telemetry after the queue has drained.  Saturation calibration should
    normally leave this at zero and size the finite replay with
    ``min_requests``.
    """
    if target_qps <= 0:
        raise ValueError("target_qps must be positive")
    if min_duration_s < 0:
        raise ValueError("min_duration_s must be non-negative")
    requests = list(base_payload.get("requests") or [])
    if len(requests) < 2:
        raise ValueError("base fragment must contain at least two requests")
    source_start = float(requests[0]["source_arrival_s"])
    source_end = float(requests[-1]["source_arrival_s"])
    source_duration = source_end - source_start
    if source_duration <= 0:
        raise ValueError("source fragment must have a positive duration")
    scaled_span = (len(requests) - 1) / float(target_qps)
    time_scale = scaled_span / source_duration
    cycle_period_s = len(requests) / float(target_qps)
    min_cycles_for_requests = math.ceil(min_requests / len(requests))
    remaining_duration = max(0.0, min_duration_s - scaled_span)
    min_cycles_for_duration = (
        math.ceil(remaining_duration / cycle_period_s) + 1
    )
    replay_cycles = max(min_cycles_for_requests, min_cycles_for_duration)
    scaled_requests = [
        {
            **request,
            "arrival_s": (
                float(request["source_arrival_s"]) - source_start
            ) * time_scale,
        }
        for request in requests
    ]
    core = {
        "schema_version": 2,
        "source_kind": "official_burstgpt",
        "source_payload_sha256": base_payload.get("payload_sha256"),
        "source_sha256": base_payload.get("source_sha256"),
        "official_release": base_payload.get("official_release"),
        "split": base_payload.get("split"),
        "scenario": scenario,
        "requests": scaled_requests,
        "replay": {
            "target_qps": float(target_qps),
            "time_scale": time_scale,
            "cycle_period_s": cycle_period_s,
            "cycles": replay_cycles,
            "base_requests": len(requests),
            "measured_requests": replay_cycles * len(requests),
            "nominal_arrival_span_s": (
                (replay_cycles - 1) * cycle_period_s + scaled_span
            ),
            "min_duration_s": float(min_duration_s),
            "min_requests": int(min_requests),
            "arrival_scaling_only": True,
        },
        "token_generator_version": TOKEN_GENERATOR_VERSION,
        "integrity": dict(base_payload.get("integrity") or {}),
    }
    if base_payload.get("heldout_lock") is not None:
        # The scaled replay remains visibly bound to the winner that was frozen
        # before the chronological test fragment was materialized.
        core["heldout_lock"] = dict(base_payload["heldout_lock"])
    return {**core, "payload_sha256": _canonical_sha(core)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_evolve.engine.real_burstgpt"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    search = sub.add_parser("extract-search")
    search.add_argument("--source", required=True)
    search.add_argument("--out", required=True)
    search.add_argument("--fragment-size", type=int, default=512)
    search.add_argument(
        "--selection",
        choices=("densest", "token_pressure"),
        default="densest",
    )
    search.add_argument("--target-mean-total-tokens", type=float)
    search.add_argument("--max-request-total-tokens", type=int)
    search.add_argument("--max-request-output-tokens", type=int)
    heldout = sub.add_parser("extract-heldout")
    heldout.add_argument("--source", required=True)
    heldout.add_argument("--out", required=True)
    heldout.add_argument("--winner-sha256", required=True)
    heldout.add_argument("--fragment-size", type=int, default=512)
    heldout.add_argument(
        "--selection",
        choices=("densest", "token_pressure"),
        default="densest",
    )
    heldout.add_argument("--target-mean-total-tokens", type=float)
    heldout.add_argument("--max-request-total-tokens", type=int)
    heldout.add_argument("--max-request-output-tokens", type=int)
    heldout.add_argument("--band-index", type=int, default=3)
    heldout.add_argument("--total-bands", type=int, default=TOTAL_BANDS)
    scale = sub.add_parser("scale")
    scale.add_argument("--base", required=True)
    scale.add_argument("--out", required=True)
    scale.add_argument("--target-qps", type=float, required=True)
    scale.add_argument("--scenario", required=True)
    scale.add_argument(
        "--min-duration-s",
        type=float,
        default=0.0,
        help=(
            "minimum nominal arrival span, not the measured-window duration; "
            "defaults to 0 because runtime validity enforces the 120s gate"
        ),
    )
    scale.add_argument("--min-requests", type=int, default=512)
    args = parser.parse_args(argv)
    if args.action == "extract-search":
        result = materialize_search_fragments(
            args.source,
            args.out,
            fragment_size=args.fragment_size,
            selection_strategy=args.selection,
            target_mean_total_tokens=args.target_mean_total_tokens,
            max_request_total_tokens=args.max_request_total_tokens,
            max_request_output_tokens=args.max_request_output_tokens,
        )
    elif args.action == "extract-heldout":
        result = materialize_heldout_fragment(
            args.source,
            args.out,
            frozen_winner_sha256=args.winner_sha256,
            fragment_size=args.fragment_size,
            selection_strategy=args.selection,
            target_mean_total_tokens=args.target_mean_total_tokens,
            max_request_total_tokens=args.max_request_total_tokens,
            max_request_output_tokens=args.max_request_output_tokens,
            chronological_band_index=args.band_index,
            total_bands=args.total_bands,
        )
    else:
        base = json.loads(Path(args.base).read_text(encoding="utf-8"))
        result = scale_fragment(
            base,
            target_qps=args.target_qps,
            min_duration_s=args.min_duration_s,
            min_requests=args.min_requests,
            scenario=args.scenario,
        )
        _write_payload(
            Path(args.out),
            {key: value for key, value in result.items() if key != "payload_sha256"},
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
