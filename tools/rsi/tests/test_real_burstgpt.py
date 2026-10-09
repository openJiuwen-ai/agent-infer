from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_evolve.engine.real_burstgpt import (
    expand_scaled_payload,
    materialize_heldout_fragment,
    materialize_search_fragments,
    scale_fragment,
)

FIX = Path(__file__).parent / "bench" / "fixtures" / "burstgpt_fragments.csv"


def test_search_fragments_are_disjoint_and_heldout_stays_hidden(tmp_path):
    manifest = materialize_search_fragments(
        FIX, tmp_path, fragment_size=2, minimum_fragment_size=2
    )
    assert set(manifest["files"]) == {"train", "validation", "calibration"}
    assert manifest["heldout"]["materialized"] is False
    ids = []
    for item in manifest["files"].values():
        payload = json.loads(Path(item["path"]).read_text())
        ids.append({request["request_id"] for request in payload["requests"]})
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_real_fragment_minimum_is_256_except_small_test_fixture(tmp_path):
    with pytest.raises(ValueError, match="at least 256"):
        materialize_search_fragments(FIX, tmp_path, fragment_size=2)


def test_search_manifest_records_token_pressure_selection(tmp_path):
    manifest = materialize_search_fragments(
        FIX,
        tmp_path,
        fragment_size=2,
        minimum_fragment_size=2,
        selection_strategy="token_pressure",
        target_mean_total_tokens=200,
        max_request_total_tokens=500,
        max_request_output_tokens=40,
    )
    assert manifest["selection"] == {
        "strategy": "token_pressure",
        "target_mean_total_tokens": 200,
        "max_request_total_tokens": 500,
        "max_request_output_tokens": 40,
    }
    validation = json.loads(
        Path(manifest["files"]["validation"]["path"]).read_text()
    )
    assert validation["selection"] == manifest["selection"]
    assert validation["integrity"]["token_lengths_unchanged"] is True


def test_scale_changes_only_arrivals_and_expands_both_hard_minima():
    base = {
        "payload_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "official_release": "v2.0",
        "split": "train",
        "integrity": {"token_lengths_unchanged": True},
        "requests": [
            {
                "request_id": "a",
                "source_arrival_s": 10,
                "num_prompt_tokens": 8,
                "num_output_tokens": 4,
            },
            {
                "request_id": "b",
                "source_arrival_s": 12,
                "num_prompt_tokens": 16,
                "num_output_tokens": 6,
            },
        ],
    }
    payload = scale_fragment(
        base,
        target_qps=5,
        min_duration_s=2,
        min_requests=8,
        scenario="burstgpt_saturated",
    )
    expanded = expand_scaled_payload(payload)
    assert len(expanded) >= 8
    assert expanded[-1]["arrival_s"] >= 2
    assert {row["num_prompt_tokens"] for row in expanded} == {8, 16}
    assert {row["num_output_tokens"] for row in expanded} == {4, 6}
    # Replayed cycles use different prompt IDs, so repetition does not invent
    # prefix-cache reuse that is absent from BurstGPT v1.
    assert expanded[0]["prompt_token_ids"] != expanded[2]["prompt_token_ids"]


def test_one_cycle_is_enough_when_its_arrival_span_exceeds_duration():
    requests = [
        {
            "request_id": str(index),
            "source_arrival_s": float(index),
            "num_prompt_tokens": 8,
            "num_output_tokens": 4,
        }
        for index in range(512)
    ]
    payload = scale_fragment(
        {"split": "calibration", "requests": requests},
        target_qps=4,
        min_duration_s=120,
        min_requests=512,
        scenario="calibration",
    )
    assert payload["replay"]["cycles"] == 1
    assert payload["replay"]["nominal_arrival_span_s"] >= 120


def test_default_scale_does_not_confuse_arrival_span_with_measured_window():
    requests = [
        {
            "request_id": str(index),
            "source_arrival_s": float(index),
            "num_prompt_tokens": 800,
            "num_output_tokens": 800,
        }
        for index in range(512)
    ]

    payload = scale_fragment(
        {"split": "calibration", "requests": requests},
        target_qps=500,
        scenario="calibration",
    )

    assert payload["replay"]["cycles"] == 1
    assert payload["replay"]["measured_requests"] == 512
    assert payload["replay"]["nominal_arrival_span_s"] < 2
    assert payload["replay"]["min_duration_s"] == 0


def test_heldout_requires_a_frozen_winner_sha(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        materialize_heldout_fragment(
            FIX,
            tmp_path / "test.json",
            frozen_winner_sha256="not-frozen",
            fragment_size=2,
        )


def test_heldout_can_use_a_later_disjoint_chronological_subband(tmp_path):
    payload = materialize_heldout_fragment(
        FIX,
        tmp_path / "test.json",
        frozen_winner_sha256="a" * 64,
        fragment_size=2,
        chronological_band_index=5,
        total_bands=6,
    )

    assert payload["chronological_band"] == {
        "index": 5,
        "total_bands": 6,
    }
    assert payload["heldout_lock"]["winner_source_sha256"] == "a" * 64


def test_scaled_heldout_preserves_frozen_winner_lock():
    base = {
        "split": "test",
        "heldout_lock": {
            "winner_source_sha256": "a" * 64,
            "materialized_after_winner_freeze": True,
        },
        "requests": [
            {
                "request_id": "a",
                "source_arrival_s": 1.0,
                "num_prompt_tokens": 8,
                "num_output_tokens": 4,
            },
            {
                "request_id": "b",
                "source_arrival_s": 2.0,
                "num_prompt_tokens": 16,
                "num_output_tokens": 6,
            },
        ],
    }

    payload = scale_fragment(
        base,
        target_qps=5,
        min_requests=2,
        scenario="burstgpt_saturated",
    )

    assert payload["heldout_lock"] == base["heldout_lock"]
