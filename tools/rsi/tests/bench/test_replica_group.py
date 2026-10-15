from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vllm_evolve.bench.config import CANDIDATE, STRONG_BASELINE, BenchConfig
from vllm_evolve.bench.datasets.burstgpt import (
    expand_scaled_payload,
    paired_mass_balance_owners,
)
from vllm_evolve.bench.dispatch import RemoteBench
from vllm_evolve.bench.native import _wait_replica_seed_barrier
from vllm_evolve.bench.replica_group import (
    DUAL_REPLICA,
    run_remote_replica_group,
    split_scaled_replay_payload,
)


def _sha(core: dict) -> str:
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _payload() -> dict:
    core = {
        "schema_version": 2,
        "source_kind": "official_burstgpt",
        "official_release": "v2.0",
        "split": "validation",
        "scenario": "burstgpt_saturated",
        "requests": [
            {
                "request_id": f"official-{index}",
                "source_arrival_s": float(index),
                "arrival_s": index * 0.25,
                "num_prompt_tokens": 100 + index,
                "num_output_tokens": 200 + index,
            }
            for index in range(4)
        ],
        "replay": {
            "target_qps": 4.0,
            "time_scale": 0.25,
            "cycle_period_s": 1.0,
            "cycles": 128,
            "base_requests": 4,
            "measured_requests": 512,
            "nominal_arrival_span_s": 130.0,
            "min_duration_s": 120.0,
            "min_requests": 512,
            "arrival_scaling_only": True,
        },
        "integrity": {},
    }
    return {**core, "payload_sha256": _sha(core)}


def _config(tmp_path: Path, payload_path: Path) -> BenchConfig:
    config = BenchConfig()
    config.engine.model = "/models/qwen32b"
    config.engine.tensor_parallel_size = 1
    config.engine.max_num_seqs = 16
    config.engine.max_num_batched_tokens = 256
    config.environment.gpus = "0,1"
    config.runner.runner_kind = STRONG_BASELINE
    config.runner.local_artifact_root = str(tmp_path / "groups")
    config.workload.regime = "evolve_real"
    config.workload.trace_path = str(payload_path)
    config.workload.n_requests = 512
    config.workload.concurrency = 512
    config.workload.arrival_rate_qps = 4.0
    config.workload.workload_spec = {
        "parallel_mode": DUAL_REPLICA,
        "validity": {
            "required": True,
            "offered_qps": 4.0,
            "plateau_reference": {
                "base_offered_qps": 4.0,
                "higher_offered_qps": 4.8,
                "base_achieved_throughput": 100.0,
                "higher_achieved_throughput": 103.0,
            },
        },
    }
    config.statistical.primary_metric = "output_throughput_tok_s"
    config.statistical.seed_tiers = [0, 1, 2]
    config.statistical.max_seeds = 3
    return config


def test_split_scaled_replay_is_lossless_and_deterministic():
    payload = _payload()

    first = split_scaled_replay_payload(payload)
    second = split_scaled_replay_payload(payload)

    assert first == second
    expected_ids = [f"official-{index}" for index in range(4)]
    assert [row["request_id"] for row in first[0]["requests"]] == expected_ids
    assert [row["request_id"] for row in first[1]["requests"]] == expected_ids
    assert sum(row["replay"]["measured_requests"] for row in first) == 512
    assert sum(row["replay"]["target_qps"] for row in first) == 4.0
    assert [row["replay"]["measured_requests"] for row in first] == [256, 256]
    for member in first:
        core = {key: value for key, value in member.items() if key != "payload_sha256"}
        assert member["payload_sha256"] == _sha(core)
        assert member["replica_partition"]["strategy"] == (
            "cycle_rotated_paired_mass_balance"
        )
        assert member["replica_partition"]["tokens_unchanged"] is True
        assert member["replica_partition"]["arrival_timestamps_unchanged"] is True
        assert member["replica_partition"]["member_source_order_unchanged"] is True
        assert member["replica_partition"]["source_cycle_indices"] == list(
            range(payload["replay"]["cycles"])
        )
    assert [
        member["replica_partition"]["base_prompt_tokens"]
        for member in first
    ] == [406, 406]
    assert [
        member["replica_partition"]["base_output_tokens"]
        for member in first
    ] == [806, 806]
    assert [
        member["replica_partition"]["member_prompt_tokens"]
        for member in first
    ] == [25984, 25984]
    assert [
        member["replica_partition"]["member_output_tokens"]
        for member in first
    ] == [51584, 51584]


def test_cycle_rotated_stripes_preserve_every_expanded_occurrence():
    payload = _payload()
    full = expand_scaled_payload(payload)
    members = [
        row
        for member in split_scaled_replay_payload(payload)
        for row in expand_scaled_payload(member)
    ]
    source_order = {
        request["request_id"]: index
        for index, request in enumerate(payload["requests"])
    }

    def ordered(rows):
        return sorted(
            rows,
            key=lambda row: (
                row["source_cycle"],
                source_order[row["source_request_id"]],
            ),
        )

    assert len(members) == len(full)
    for expected, actual in zip(ordered(full), ordered(members)):
        for key in (
            "arrival_s",
            "prompt_token_ids",
            "num_prompt_tokens",
            "num_output_tokens",
            "source_request_id",
            "source_cycle",
        ):
            assert actual[key] == expected[key]


def test_cycle_rotated_pairs_swap_each_request_between_replicas():
    payload = _payload()
    first, second = [
        expand_scaled_payload(member)
        for member in split_scaled_replay_payload(payload)
    ]

    for cycle in range(2):
        first_indices = {
            int(row["request_id"].rsplit("-r", 1)[1])
            for row in first
            if row["source_cycle"] == cycle
        }
        second_indices = {
            int(row["request_id"].rsplit("-r", 1)[1])
            for row in second
            if row["source_cycle"] == cycle
        }
        assert first_indices == ({0, 3} if cycle == 0 else {1, 2})
        assert second_indices == ({1, 2} if cycle == 0 else {0, 3})


def test_paired_owner_map_reduces_cumulative_token_mass_skew():
    requests = [
        {"num_prompt_tokens": 1000, "num_output_tokens": 100},
        {"num_prompt_tokens": 10, "num_output_tokens": 900},
        {"num_prompt_tokens": 900, "num_output_tokens": 20},
        {"num_prompt_tokens": 20, "num_output_tokens": 800},
        {"num_prompt_tokens": 800, "num_output_tokens": 30},
        {"num_prompt_tokens": 30, "num_output_tokens": 700},
        {"num_prompt_tokens": 700, "num_output_tokens": 40},
        {"num_prompt_tokens": 40, "num_output_tokens": 600},
    ]
    owners = paired_mass_balance_owners(requests)

    assert owners.count(0) == owners.count(1) == 4
    for pair_start in range(0, len(owners), 2):
        assert owners[pair_start] != owners[pair_start + 1]
    prompt_mass = [
        sum(
            request["num_prompt_tokens"]
            for request, owner in zip(requests, owners)
            if owner == replica
        )
        for replica in (0, 1)
    ]
    output_mass = [
        sum(
            request["num_output_tokens"]
            for request, owner in zip(requests, owners)
            if owner == replica
        )
        for replica in (0, 1)
    ]
    assert abs(prompt_mass[0] - prompt_mass[1]) < 100
    assert abs(output_mass[0] - output_mass[1]) < 100


def test_replica_seed_barrier_returns_one_common_start(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _wait_replica_seed_barrier,
                "unit_test",
                replica_index=index,
                replica_count=2,
                seed=0,
                timeout_s=2.0,
                lead_s=0.05,
                root=tmp_path,
            )
            for index in range(2)
        ]
        starts = [future.result() for future in futures]

    assert starts[0] == starts[1]


def _write_member_evidence(
    config: BenchConfig,
    *,
    utilization: int = 96,
) -> RemoteBench:
    spec = config.workload.workload_spec["replica_member"]
    index = int(spec["index"])
    gpu = config.environment.gpus
    run = Path(config.runner.local_artifact_root) / "run"
    run.mkdir(parents=True)
    windows = []
    gpu_rows = []
    vllm_rows = []
    raw_rows = []
    eval_rows = []
    for seed in (0, 1, 2):
        start = 1000.0 + seed * 200.0 + index * 0.05
        end = start + 130.0
        windows.append({
            "seed": seed,
            "started_at_unix_s": start,
            "ended_at_unix_s": end,
            "duration_s": 130.0,
            "measured_requests": 256,
            "replica_barrier_id": spec["barrier_id"],
            "barrier_start_at_unix_s": 999.95 + seed * 200.0,
            "replica_index": index,
            "replica_count": 2,
        })
        for tick in range(20):
            timestamp = start + tick * 5.0
            gpu_rows.append({
                "captured_at_unix_s": timestamp,
                "gpus": [{
                    "index": gpu,
                    "uuid": f"GPU-{gpu}",
                    "name": "L20X",
                    "memory_total_mib": 1000,
                    "memory_used_mib": 900,
                    "utilization_gpu_pct": utilization,
                    "utilization_memory_pct": 80,
                    "power_draw_w": 300,
                    "clocks_sm_mhz": 1500,
                    "clocks_memory_mhz": 3000,
                    "temperature_gpu_c": 65,
                    "thermal_throttle_active": False,
                    "power_throttle_active": False,
                }],
            })
            vllm_rows.append({
                "captured_at_unix_s": timestamp,
                "seed": seed,
                "metrics": {
                    "running_requests": 16,
                    "waiting_requests": 8,
                    "kv_cache_occupancy": 0.92,
                    "preemptions_total": tick,
                    "scheduler_invocations_total": tick * 10,
                },
            })
        for request_index in range(256):
            submit = start + request_index / 2.0
            raw_rows.append({
                "seed": seed,
                "request_id": f"p{index}-s{seed}-r{request_index}",
                "submit_time_s": submit,
                "first_token_time_s": submit + 0.1,
                "end_time_s": submit + 0.5,
                "num_prompt_tokens": 100,
                "num_output_tokens": 200,
                "success": True,
                "error": None,
            })
        eval_rows.append({
            "seed": seed,
            "primary_value": 400.0,
            "metrics": {
                "num_requests": 256,
                "num_completed": 256,
                "num_failed": 0,
            },
        })
    (run / "measurement_windows.json").write_text(
        json.dumps({"windows": windows}),
        encoding="utf-8",
    )
    for name, rows in (
        ("gpu_samples.jsonl", gpu_rows),
        ("vllm_metrics.jsonl", vllm_rows),
        ("raw_requests.jsonl", raw_rows),
    ):
        (run / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    candidate = config.runner.runner_kind == CANDIDATE
    eval_result = {
        "schema_version": "1.0",
        "source": "real_vllm",
        "policy_sha256": "",
        "config_sha256": "config",
        "git_sha": "abc123",
        "vllm_version": "0.21.0",
        "hardware_profile": "L20X",
        "profile": "evolve_real",
        "regime": "replay",
        "outcome_class": "eval_result",
        "primary_metric": "output_throughput_tok_s",
        "seeds": [0, 1, 2],
        "raw_per_seed_metrics": eval_rows,
        "aggregate_metrics": {
            "median": 400.0,
            "mean": 400.0,
            "std": 0.0,
            "cv": 0.0,
            "n": 3,
        },
        "slo": {},
        "score_aggregation_formula_version": "median-v1",
        "command_line": "fake-member",
        "wall_time_s": 130.0,
        "marker_verified": candidate,
        "effective": candidate,
        "mechanism_applicable": True if candidate else None,
    }
    if candidate:
        eval_result["plugin_provenance"] = {
            "invoked": True,
            "fallback": False,
            "effective": True,
            "execution_valid": True,
            "mechanism_applicable": True,
            "deferred_request_actions": index + 1,
        }
    status = {
        "ok": True,
        "duration_s": 500,
        "gpu_summary": {
            "contaminated": False,
            "per_gpu": {
                gpu: {
                    "name": "L20X",
                    "max_memory_used_mib": 900,
                    "max_utilization_gpu_pct": 96,
                },
            },
        },
    }
    return RemoteBench(
        eval_result,
        "server log",
        str(run / "eval_result.json"),
        str(run / "serve.log"),
        remote_cmd=f"replica-{index}",
        remote_run_dir=f"/remote/replica-{index}",
        local_run_dir=str(run),
        status=status,
    )


def test_dual_replica_group_combines_metrics_and_requires_both_gpus(tmp_path):
    payload = _payload()
    payload_path = tmp_path / "workload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config(tmp_path, payload_path)
    seen = []

    def fake_runner(member, *, git_sha=None):
        seen.append(member)
        return _write_member_evidence(member)

    result = run_remote_replica_group(
        config,
        git_sha="abc123",
        single_runner=fake_runner,
        cleanup=lambda remote, port: None,
    )

    assert sorted(member.environment.gpus for member in seen) == ["0", "1"]
    assert all(member.engine.tensor_parallel_size == 1 for member in seen)
    assert result.eval_result["parallel_mode"] == DUAL_REPLICA
    assert result.eval_result["workload_validity"]["valid"] is True
    per_seed = result.eval_result["workload_validity"]["per_seed"]
    assert all(set(row["gpu"]) == {"0", "1"} for row in per_seed)
    assert all(row["measurement"]["requests"] == 512 for row in per_seed)
    assert all(
        row["metrics"]["num_completed"] == 512
        for row in result.eval_result["raw_per_seed_metrics"]
    )
    assert result.status["gpu_summary"]["per_gpu"]["0"]["name"] == "L20X"


def test_dual_replica_group_rejects_one_underloaded_gpu(tmp_path):
    payload = _payload()
    payload_path = tmp_path / "workload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config(tmp_path, payload_path)

    def fake_runner(member, *, git_sha=None):
        return _write_member_evidence(
            member,
            utilization=40 if member.environment.gpus == "1" else 96,
        )

    result = run_remote_replica_group(
        config,
        git_sha="abc123",
        single_runner=fake_runner,
        cleanup=lambda remote, port: None,
    )

    validity = result.eval_result["workload_validity"]
    assert validity["valid"] is False
    assert validity["verdict"] == "invalid_gpu_imbalance"
    assert "invalid_underloaded_gpu" in validity["per_seed"][0]["failed_categories"]


def test_dual_replica_candidate_combines_action_counters(tmp_path):
    payload = _payload()
    payload_path = tmp_path / "workload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _config(tmp_path, payload_path)
    config.runner.runner_kind = CANDIDATE
    config.runner.policy_path = str(tmp_path / "candidate.py")
    config.engine.scheduler_cls = "generated_scheduler.EvolvedScheduler"

    result = run_remote_replica_group(
        config,
        git_sha="abc123",
        single_runner=lambda member, git_sha=None: _write_member_evidence(member),
        cleanup=lambda remote, port: None,
    )

    assert result.eval_result["marker_verified"] is True
    assert result.eval_result["effective"] is True
    provenance = result.eval_result["plugin_provenance"]
    assert provenance["deferred_request_actions"] == 3
    assert len(provenance["replicas"]) == 2


def test_split_rejects_non_official_payload():
    with pytest.raises(ValueError, match="official BurstGPT"):
        split_scaled_replay_payload({"schema_version": 2, "requests": []})
