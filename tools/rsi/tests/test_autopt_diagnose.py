"""M1: L1 diagnose rule engine — synthetic profiles -> bottleneck (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core import schemas as S  # noqa: E402
from vllm_evolve.engine.diagnose import diagnose  # noqa: E402


def _prof(gpu=None, vllm=None, metrics=None, config=None, marker=True):
    return S.Profile(
        config=config or {"max_num_seqs": 128}, metrics=metrics or {},
        vllm=vllm or {}, gpu=gpu or {}, marker_verified=marker,
    )


def test_compute_bound_confirmed_needs_full_signature():
    # sustained-high SM (duty>=0.7) + KV measured (ruled out) -> confirmed
    d = diagnose(_prof(gpu={"sm_util": 96, "sm_util_max": 96, "duty_cycle": 0.95},
                       vllm={"kv_util": 0.4, "waiting": 1}))
    assert d.bottleneck == S.COMPUTE and d.status == "confirmed"


def test_compute_only_suspected_when_evidence_partial():
    # high SM but NO duty and NO KV -> can't confirm it's not KV/bursty -> suspected
    d = diagnose(_prof(gpu={"sm_util": 95}, vllm={"waiting": 1}))
    assert d.bottleneck == S.COMPUTE and d.status == "suspected"


def test_kv_capacity_bound():
    d = diagnose(_prof(gpu={"sm_util": 88},
                       vllm={"kv_util": 0.95, "preempt": 5, "waiting": 20}))
    assert d.bottleneck == S.KV_CAPACITY and d.status == "confirmed"


def test_under_saturated():
    # sustained idle (duty<0.4) + known short queue -> confirmed
    d = diagnose(_prof(gpu={"sm_util": 18, "sm_util_max": 35, "duty_cycle": 0.1},
                       vllm={"kv_util": 0.2, "waiting": 0}))
    assert d.bottleneck == S.UNDER_SATURATED and d.status == "confirmed"


def test_prefill_proxy_is_only_suspected():
    # prefill comes from a token-rate proxy -> never above suspected, with a limitation
    d = diagnose(_prof(gpu={"sm_util": 70},
                       vllm={"prefill_token_frac": 0.8, "waiting": 2},
                       metrics={"ttft_p99_ms": 5000}))
    assert d.bottleneck == S.PREFILL and d.status == "suspected"
    assert any("proxy" in lim for lim in d.limitations)


def test_mem_bandwidth_not_under_saturated():
    # low SM but high bandwidth -> bandwidth-bound, NOT under-saturated
    d = diagnose(_prof(gpu={"sm_util": 50, "mem_bw_util": 90}, vllm={"kv_util": 0.3,
                                                                     "waiting": 1}))
    assert d.bottleneck == S.MEM_BANDWIDTH


def test_scheduling_queue_bound():
    # free capacity (SM mid) + standing queue + KV not full + no preempt
    d = diagnose(_prof(gpu={"sm_util": 70},
                       vllm={"kv_util": 0.4, "waiting": 20, "preempt": 0}))
    assert d.bottleneck == S.SCHEDULING_QUEUE


def test_scheduling_queue_via_duty_when_no_sm_source():
    # no SM source (e.g. simulator backend) but a measured engine duty < 0.9 + standing
    # queue -> scheduling_queue on weaker evidence, capped at suspected.
    d = diagnose(_prof(gpu={"duty_cycle": 0.7},
                       vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0}))
    assert d.bottleneck == S.SCHEDULING_QUEUE and d.status == "suspected"


def test_no_scheduling_queue_when_engine_fully_busy_and_no_sm():
    # duty ~1.0 means NO free capacity -> the duty-evidence branch must not fire
    d = diagnose(_prof(gpu={"duty_cycle": 0.98},
                       vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0}))
    assert d.bottleneck != S.SCHEDULING_QUEUE


def test_comm_bound():
    d = diagnose(_prof(gpu={"sm_util": 88, "nccl_frac": 0.3},
                       vllm={"kv_util": 0.4, "waiting": 1}))
    assert d.bottleneck == S.COMM and d.status == "confirmed"


def test_unknown_when_signals_missing():
    d = diagnose(_prof(gpu={}, vllm={}, metrics={}))
    assert d.bottleneck == S.UNKNOWN and d.status == "unknown"
    assert d.limitations


def test_unverified_marker_is_a_limitation():
    d = diagnose(_prof(gpu={"sm_util": 95}, vllm={"kv_util": 0.4}, marker=False))
    assert any("marker" in lim for lim in d.limitations)


def test_diagnosis_roundtrips():
    d = diagnose(_prof(gpu={"sm_util": 95}, vllm={"kv_util": 0.4}))
    assert S.Diagnosis.from_dict(d.to_dict()).bottleneck == d.bottleneck


def test_cmd_diagnose_cli(tmp_path, capsys):
    import json

    from vllm_evolve.cli import main as cli_main
    f = tmp_path / "prof.json"
    f.write_text(json.dumps(_prof(gpu={"sm_util": 95}, vllm={"kv_util": 0.4}).to_dict()),
                 encoding="utf-8")
    rc = cli_main.main(["diagnose", str(f)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True and out["bottleneck"] == S.COMPUTE


def test_cmd_diagnose_missing_file(tmp_path, capsys):
    import json

    from vllm_evolve.cli import main as cli_main
    rc = cli_main.main(["diagnose", str(tmp_path / "nope.json")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "profile_not_found"
