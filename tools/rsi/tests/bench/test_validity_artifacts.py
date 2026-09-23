from __future__ import annotations

import json

from vllm_evolve.bench.validity_artifacts import (
    attach_workload_validity,
    summarize_client_inflight,
)


def test_client_inflight_summary_is_time_weighted():
    summary = summarize_client_inflight(
        [
            {"submit_time_s": 0.0, "end_time_s": 4.0},
            {"submit_time_s": 1.0, "end_time_s": 3.0},
        ]
    )
    assert summary["mean"] == 1.5
    assert summary["p50"] == 1.0
    assert summary["p95"] == 2.0
    assert summary["max"] == 2


def test_missing_formal_protocol_never_defaults_to_valid(tmp_path):
    result = attach_workload_validity(
        {"source": "real_vllm"},
        status={},
        run_dir=tmp_path,
        bench_config={},
    )
    assert result["workload_validity"]["valid"] is False
    assert result["workload_validity"]["required"] is False


def test_missing_measurement_window_is_invalid_and_persisted(tmp_path):
    result = attach_workload_validity(
        {
            "source": "real_vllm",
            "seeds": [0],
            "raw_per_seed_metrics": [{"seed": 0, "metrics": {}}],
        },
        status={},
        run_dir=tmp_path,
        bench_config={
            "engine": {"max_num_seqs": 16},
            "environment": {"gpus": "0,1"},
            "workload": {
                "workload_spec": {
                    "validity": {"required": True, "offered_qps": 10}
                }
            },
        },
    )
    assert result["workload_validity"]["valid"] is False
    assert result["workload_validity"]["per_seed"][0]["reasons"] == [
        "measurement window is missing"
    ]
    assert json.loads((tmp_path / "validity.json").read_text())["valid"] is False
