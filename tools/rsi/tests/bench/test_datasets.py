"""Unit tests for the four real-trace adapters, against verified-format fixtures."""
from __future__ import annotations

import math
from pathlib import Path

from vllm_evolve.bench.datasets import available, azure, burstgpt, load_dataset, mooncake, sharegpt

FIX = Path(__file__).parent / "fixtures"


# ---- BurstGPT -------------------------------------------------------------

def test_burstgpt_6col_mapping_and_failure_filter():
    # drop_failures=True drops the Response tokens == 0 row, leaving 3, sorted.
    ds = burstgpt.load(FIX / "burstgpt_6col.csv", drop_failures=True)
    assert ds.total_requests == 3
    arrivals = [r.arrival_s for r in ds.requests]
    assert arrivals == sorted(arrivals)
    assert arrivals == [5.0, 45.0, 118.0]
    first = ds.requests[0]
    assert first.prompt_tokens == 472   # Request tokens
    assert first.output_tokens == 18    # Response tokens


def test_burstgpt_keeps_failures_when_asked():
    ds = burstgpt.load(FIX / "burstgpt_6col.csv", drop_failures=False)
    assert ds.total_requests == 4
    # sorting still applied -> the arrival=60 zero-output row lands in order
    assert [r.arrival_s for r in ds.requests] == [5.0, 45.0, 60.0, 118.0]


def test_burstgpt_8col_detected_by_name_not_position():
    # extra Session ID / Elapsed time columns must not shift token parsing.
    ds = burstgpt.load(FIX / "burstgpt_8col.csv")
    assert ds.total_requests == 2
    assert ds.requests[0].prompt_tokens == 500
    assert ds.requests[0].output_tokens == 100
    assert ds.requests[0].arrival_s == 10.0


def test_burstgpt_dense_splits_are_contiguous_disjoint_and_deterministic():
    path = FIX / "burstgpt_fragments.csv"
    a = burstgpt.extract_bursty_fragments(path, fragment_size=2)
    b = burstgpt.extract_bursty_fragments(path, fragment_size=2)
    assert list(a) == ["train", "validation", "test"]
    assert {
        name: [r.request_id for r in ds.requests] for name, ds in a.items()
    } == {
        name: [r.request_id for r in ds.requests] for name, ds in b.items()
    }
    ids = [[r.request_id for r in ds.requests] for ds in a.values()]
    assert not (set(ids[0]) & set(ids[1]) or set(ids[1]) & set(ids[2]))
    assert all(ds.duration_s == 1.0 for ds in a.values())  # densest pair in every band


def test_burstgpt_can_materialize_only_the_heldout_chronological_third():
    path = FIX / "burstgpt_fragments.csv"
    all_fragments = burstgpt.extract_bursty_fragments(path, fragment_size=2)
    heldout = burstgpt.extract_bursty_fragments(
        path,
        fragment_size=2,
        split_names=("test",),
        band_indices=(2,),
        total_bands=3,
    )
    assert [r.request_id for r in heldout["test"].requests] == [
        r.request_id for r in all_fragments["test"].requests
    ]


def test_burstgpt_token_pressure_keeps_contiguous_unmodified_rows():
    path = FIX / "burstgpt_fragments.csv"
    fragments = burstgpt.extract_bursty_fragments(
        path,
        fragment_size=2,
        selection_strategy="token_pressure",
        target_mean_total_tokens=200,
        max_request_total_tokens=500,
        max_request_output_tokens=40,
    )
    for dataset in fragments.values():
        ids = [int(request.request_id.rsplit("_", 1)[-1]) for request in dataset.requests]
        assert ids[1] == ids[0] + 1
        assert all(
            request.prompt_tokens + request.output_tokens <= 500
            for request in dataset.requests
        )
        assert all(request.output_tokens <= 40 for request in dataset.requests)


def test_burstgpt_frontier_conversion_preserves_shape_without_fake_prefixes():
    ds = burstgpt.load(FIX / "burstgpt_6col.csv")
    rows, meta = burstgpt.to_frontier_rows(ds, target_qps=20.0)
    assert rows[0]["arrived_at"] == 0.0
    assert rows[-1]["arrived_at"] == (len(rows) - 1) / 20.0
    assert [r["num_prefill_tokens"] for r in rows] == [472, 1087, 417]
    assert {r["session_id"] for r in rows} == {0}
    assert {r["block_hash_ids"] for r in rows} == {""}
    assert meta["tenant_policy"].startswith("single neutral")


# ---- Azure ----------------------------------------------------------------

def test_azure_datetime_relative_arrival_and_token_mapping():
    ds = azure.load(FIX / "azure.csv")
    assert ds.total_requests == 3
    assert ds.requests[0].arrival_s == 0.0                 # earliest -> 0
    # 18:15:50.9951690 - 18:15:46.6805900 = 4.314579 s (7th frac digit truncated)
    assert math.isclose(ds.requests[1].arrival_s, 4.314579, abs_tol=1e-6)
    assert ds.requests[0].prompt_tokens == 374             # ContextTokens
    assert ds.requests[0].output_tokens == 44              # GeneratedTokens


# ---- ShareGPT -------------------------------------------------------------

def test_sharegpt_requires_two_turns_and_derives_tokens():
    # b_0 has < 2 turns -> dropped, leaving a_0 and c_0.
    ds = sharegpt.load(FIX / "sharegpt.json")
    assert ds.total_requests == 2
    assert all(r.prompt_tokens > 0 and r.output_tokens > 0 for r in ds.requests)


def test_sharegpt_uses_injected_token_counter():
    ds = sharegpt.load(FIX / "sharegpt.json", count_tokens=lambda t: len(t.split()))
    # a_0 prompt "hello world this is a prompt" -> 6 words
    assert ds.requests[0].prompt_tokens == 6


def test_sharegpt_output_len_override():
    ds = sharegpt.load(FIX / "sharegpt.json", output_len=128)
    assert all(r.output_tokens == 128 for r in ds.requests)


# ---- Mooncake -------------------------------------------------------------

def test_mooncake_ms_to_seconds_and_prefix_and_bad_line_skip():
    ds = mooncake.load(FIX / "mooncake.jsonl")
    assert ds.total_requests == 2        # the malformed middle line is skipped
    assert ds.parse_errors == 1
    # sorted: timestamp 0 -> 0.0 s first, then 27000 ms -> 27.0 s
    assert ds.requests[0].arrival_s == 0.0
    assert ds.requests[1].arrival_s == 27.0
    assert ds.requests[0].prompt_tokens == 100
    assert ds.requests[1].prefix_hashes == (46, 47, 48)


# ---- Registry -------------------------------------------------------------

def test_registry_dispatch_and_listing():
    assert set(available()) == {"burstgpt", "azure", "sharegpt", "mooncake"}
    ds = load_dataset("mooncake", FIX / "mooncake.jsonl")
    assert ds.name == "mooncake"
    assert ds.total_requests == 2


def test_registry_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        load_dataset("nope", FIX / "mooncake.jsonl")
