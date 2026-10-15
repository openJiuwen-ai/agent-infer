"""R3/AC4: synthesize Frontier trace_replay CSVs. The workload primitive is a TRACE (5-col
contract); named shapes are template generators with a RANGE guardrail (not an enum). Deterministic.
Pure/offline."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.workload_synth import (  # noqa: E402
    TRACE_COLUMNS,
    TraceRanges,
    WorkloadSpec,
    bursty_multi_tenant,
    heterogeneous_bursty_multi_tenant,
    validate_trace,
    write_trace_csv,
)


def test_bursty_multi_tenant_native_5col_and_shared_prefix():
    rows = bursty_multi_tenant(n_tenants=3, bursts_per_tenant=2, burst_size=4)
    assert len(rows) == 3 * 2 * 4
    # native 5-column contract present on every row
    for r in rows:
        assert set(r) == set(TRACE_COLUMNS)
    # multiple tenants (session_id groups)
    assert {r["session_id"] for r in rows} == {0, 1, 2}
    # every request from a tenant shares THAT tenant's block_hash_ids (prefix-cache reuse), and
    # different tenants have different prefixes
    blocks_by_tenant = {t: {r["block_hash_ids"] for r in rows if r["session_id"] == t}
                        for t in (0, 1, 2)}
    assert all(len(v) == 1 for v in blocks_by_tenant.values())          # one prefix per tenant
    assert len({next(iter(v)) for v in blocks_by_tenant.values()}) == 3  # tenants differ
    # sorted by arrival
    assert rows == sorted(rows, key=lambda r: (r["arrived_at"], r["session_id"]))


def test_synthesis_is_deterministic():
    a = bursty_multi_tenant(n_tenants=2, bursts_per_tenant=3, burst_size=3)
    b = bursty_multi_tenant(n_tenants=2, bursts_per_tenant=3, burst_size=3)
    assert a == b                                                       # no RNG -> reproducible


def test_heterogeneous_burst_has_deceptive_lengths_and_is_deterministic():
    a = heterogeneous_bursty_multi_tenant(
        n_tenants=2, bursts_per_tenant=1, burst_size=4
    )
    b = heterogeneous_bursty_multi_tenant(
        n_tenants=2, bursts_per_tenant=1, burst_size=4
    )
    assert a == b
    profiles = {
        (row["num_prefill_tokens"], row["num_decode_tokens"]) for row in a
    }
    assert (256, 2048) in profiles
    assert (768, 64) in profiles
    assert (1536, 2048) in profiles
    assert all(row["block_hash_ids"] for row in a)


def test_validate_trace_is_range_not_enum(tmp_path):
    ok = bursty_multi_tenant(n_tenants=2, bursts_per_tenant=2, burst_size=2,
                             prefill_tokens=256, decode_tokens=32)
    assert validate_trace(ok) == []                                    # in-range -> admissible
    # an ARBITRARY trace (not from a named shape) is admitted purely on ranges
    custom = [{"arrived_at": 0.0, "num_prefill_tokens": 10, "num_decode_tokens": 5,
               "session_id": 7, "block_hash_ids": "1|2"}]
    assert validate_trace(custom) == []
    # out-of-range tokens / too many rows / too long -> reported (range guardrail)
    assert validate_trace([{**custom[0], "num_prefill_tokens": 999999}])
    assert validate_trace(ok, TraceRanges(max_rows=1))
    assert validate_trace([{**custom[0], "arrived_at": 10_000.0}], TraceRanges(max_duration_s=1.0))
    assert validate_trace([]) == ["empty trace"]


def test_validate_trace_never_raises_on_malformed_open_world_input():
    # the open-world admission path must REPORT type/range errors, never crash on untrusted values
    bad = [
        {"arrived_at": "bad", "num_prefill_tokens": "x", "num_decode_tokens": 4,
         "session_id": 0, "block_hash_ids": "0"},
        {"arrived_at": 0.1, "num_prefill_tokens": 8, "num_decode_tokens": "y",
         "session_id": 1, "block_hash_ids": "1"},
    ]
    errs = validate_trace(bad)                                         # must not raise
    assert any("num_prefill_tokens not an integer" in e for e in errs)
    assert any("num_decode_tokens not an integer" in e for e in errs)
    assert any("arrived_at not a number" in e for e in errs)
    # scanning continued past the first bad row (row 1's decode error is present)
    assert any(e.startswith("row 1") for e in errs)
    # a trace whose every arrived_at is non-numeric -> explicit arrival error, no crash
    no_arrivals = [{"arrived_at": "nope", "num_prefill_tokens": 8, "num_decode_tokens": 4,
                    "session_id": 0, "block_hash_ids": "0"}]
    assert "no valid numeric arrived_at in trace" in validate_trace(no_arrivals)


def test_write_trace_csv_roundtrip(tmp_path):
    rows = bursty_multi_tenant(n_tenants=2, bursts_per_tenant=1, burst_size=2)
    p = write_trace_csv(rows, tmp_path / "trace.csv")
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(TRACE_COLUMNS)
        back = list(reader)
    assert len(back) == len(rows)
    assert back[0]["block_hash_ids"] == rows[0]["block_hash_ids"]       # prefix blocks survive CSV


def test_workload_spec_template_and_inline():
    rows = WorkloadSpec(template="bursty_multi_tenant", params={"n_tenants": 2}).build()
    assert {r["session_id"] for r in rows} == {0, 1}
    inline = [{"arrived_at": 0.0, "num_prefill_tokens": 8, "num_decode_tokens": 4,
               "session_id": 0, "block_hash_ids": "0"}]
    assert WorkloadSpec(inline_rows=inline).build() == inline           # custom trace passthrough
    try:
        WorkloadSpec(template="nonsense").build()
        raise AssertionError("expected ValueError for unknown template")
    except ValueError:
        pass
