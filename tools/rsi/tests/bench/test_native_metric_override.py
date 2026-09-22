from __future__ import annotations

import json

import pytest

import vllm_evolve.bench.native as native


def test_native_cli_overrides_profile_metric_and_slo(monkeypatch, tmp_path):
    captured = {}

    def fake_native_bench(_source, profile, **_kwargs):
        captured["metric"] = profile.primary_metric
        captured["slo"] = profile.slo
        return (
            {
                "outcome_class": "eval_result",
                "primary_metric": profile.primary_metric,
                "aggregate_metrics": {"median": 1.0},
                "seeds": [0],
                "sample_count": 1,
                "wall_time_s": 1.0,
            },
            "",
        )

    monkeypatch.setattr(native, "native_bench", fake_native_bench)
    monkeypatch.setattr(native, "_hardware_name", lambda: "test-gpu")
    monkeypatch.setattr(
        "vllm_evolve.bench.eval_result.write_eval_result",
        lambda _result, _path: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "vllm", type("V", (), {"__version__": "x"}))
    out = tmp_path / "eval.json"
    rc = native.main(
        [
            "--runner-kind",
            "strong_baseline",
            "--profile",
            "config/bench/profiles/evolve_real.yaml",
            "--primary-metric",
            "output_throughput_tok_s",
            "--slo-json",
            "{}",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert captured["metric"] == "output_throughput_tok_s"
    assert captured["slo"].ttft_ms is None


def test_native_cli_rejects_unknown_slo_field(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        native.main(
            [
                "--runner-kind",
                "strong_baseline",
                "--profile",
                "config/bench/profiles/evolve_real.yaml",
                "--slo-json",
                json.dumps({"invented": 1}),
                "--out",
                str(tmp_path / "eval.json"),
            ]
        )
    assert exc.value.code == 2
    assert "unknown fields" in capsys.readouterr().err
