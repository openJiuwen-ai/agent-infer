from __future__ import annotations

from vllm_evolve.engine.local_frontier_evolve import Scenario
from vllm_evolve.engine.real_workloads import prompt_token_ids, scenario_payload


def test_prompt_lengths_are_exact_and_unlabeled_requests_do_not_share_prefix():
    a = {"num_prefill_tokens": 32, "block_hash_ids": ""}
    b = {"num_prefill_tokens": 32, "block_hash_ids": ""}
    ta = prompt_token_ids(a, 1)
    tb = prompt_token_ids(b, 2)
    assert len(ta) == len(tb) == 32
    assert ta[:16] != tb[:16]


def test_only_explicit_prefix_blocks_are_shared():
    a = {"num_prefill_tokens": 48, "block_hash_ids": "tenant-a|shared"}
    b = {"num_prefill_tokens": 80, "block_hash_ids": "tenant-a|shared"}
    c = {"num_prefill_tokens": 48, "block_hash_ids": "tenant-b|other"}
    ta = prompt_token_ids(a, 1)
    tb = prompt_token_ids(b, 2)
    tc = prompt_token_ids(c, 3)
    assert ta[:32] == tb[:32]
    assert ta[:16] != tc[:16]
    assert ta[32:] != tb[32:48]


def test_scenario_payload_binds_rows_and_exact_requests():
    scenario = Scenario(
        name="stress",
        split="test",
        source_kind="synthetic",
        rows=[{
            "arrived_at": 0.25,
            "num_prefill_tokens": 24,
            "num_decode_tokens": 12,
            "session_id": 7,
            "block_hash_ids": "x",
        }],
        slo_ttft_ms=200,
        max_num_seqs=4,
        enable_prefix_caching=True,
    )
    payload = scenario_payload(scenario)
    request = payload["requests"][0]
    assert request["arrival_s"] == 0.25
    assert request["num_prompt_tokens"] == len(request["prompt_token_ids"]) == 24
    assert request["num_output_tokens"] == 12
    assert len(payload["payload_sha256"]) == 64
