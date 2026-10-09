"""M1: L1 profile assembly (parse log + summarize GPU + assemble) — zero GPU."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core import schemas as S  # noqa: E402
from vllm_evolve.engine.diagnose import diagnose  # noqa: E402
from vllm_evolve.engine.profile import (  # noqa: E402
    assemble_profile,
    parse_vllm_log,
    summarize_gpu,
)

_LOG = """\
vllm-evolve: scheduler plugin invoked
INFO Running: 12 reqs, Waiting: 30 reqs, GPU KV cache usage: 88.5%
INFO Avg prompt throughput: 800.0 tokens/s, Avg generation throughput: 200.0 tokens/s
WARNING Preemption is happening: 3 reqs
INFO Running: 8 reqs, Waiting: 45 reqs, GPU KV cache usage: 96.1%
"""


def test_parse_vllm_log():
    v = parse_vllm_log(_LOG)
    assert v["kv_util"] == 0.961          # peak KV
    assert v["running"] == 12 and v["waiting"] == 45
    assert v["preempt"] >= 1
    assert v["prefill_token_frac"] == 0.8  # 800/(800+200), a token-rate proxy


def test_summarize_gpu_reports_true_mean_and_duty_cycle():
    # honest: idle samples are NOT silently dropped from the mean; duty_cycle
    # carries the "how often busy" signal so bursty/idle can't read as compute.
    samples = [{"sm_util": 2, "mem_used_mb": 1000, "mem_total_mb": 49140},  # load idle
               {"sm_util": 95, "mem_bw_util": 70, "mem_used_mb": 44000, "mem_total_mb": 49140},
               {"sm_util": 99, "mem_bw_util": 80, "mem_used_mb": 44200, "mem_total_mb": 49140}]
    g = summarize_gpu(samples)
    assert g["sm_util_max"] == 99
    assert g["sm_util"] == round((2 + 95 + 99) / 3, 1)   # mean over ALL (includes idle)
    assert g["duty_cycle"] == round(2 / 3, 2)            # 2 of 3 samples busy
    assert g["sm_util_busy"] == 97.0                     # mean of the busy window
    assert g["mem_used_mb"] == 44200 and g["mem_total_mb"] == 49140


def test_assemble_and_diagnose_end_to_end():
    er = {"outcome_class": "eval_result", "primary_metric": "goodput_req_s",
          "aggregate_metrics": {"median": 6.7},
          "raw_per_seed_metrics": [{"metrics": {"ttft_ms": 5000, "tpot_ms": 40,
                                                "total_token_throughput_tok_s": 1800}}]}
    samples = [{"sm_util": 96, "mem_bw_util": 60, "mem_used_mb": 44000, "mem_total_mb": 49140}]
    p = assemble_profile({"max_num_seqs": 128}, er, samples, _LOG)
    assert p.marker_verified is True
    assert p.metrics["goodput_req_s"] == 6.7 and p.metrics["tok_s"] == 1800
    assert p.vllm["kv_util"] == 0.961 and p.gpu["sm_util_max"] == 96
    assert p.saturated is True
    # KV peak 96% + preemption -> diagnose should call it kv_capacity
    d = diagnose(p)
    assert d.bottleneck == S.KV_CAPACITY


def test_assemble_extracts_nested_ttft_percentile():
    # a real eval_result may keep TTFT as metrics.ttft_ms.{p99,...}; ttft_p99_ms
    # must still be extracted (flat name absent) so L0 latency goals can score.
    er = {"raw_per_seed_metrics": [{"metrics": {"ttft_ms": {"p99": 410.0, "p50": 200.0}}},
                                   {"metrics": {"ttft_ms": {"p99": 430.0, "p50": 210.0}}}]}
    p = assemble_profile({}, er, [], _LOG)
    assert p.metrics["ttft_p99_ms"] == 420.0            # median of [410, 430]
    assert p.per_seed["ttft_p99_ms"] == [410.0, 430.0]   # per-seed for L4 bootstrap


def test_assemble_without_marker_flags_unverified():
    p = assemble_profile({}, {"aggregate_metrics": {"median": 1.0}}, [], "no marker here")
    assert p.marker_verified is False
    assert any("marker" in lim for lim in diagnose(p).limitations)
