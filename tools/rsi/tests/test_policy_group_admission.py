"""Offline unit test for the H* B-arm fixture (tests/fixtures/policy_group_admission.py).

Proves the hand-written grouped-admission policy is a legitimate, deterministic tenant-grouping
policy that uses BOTH admission channels: it admits exactly the current tenant's requests
(prefill_batch) and explicitly DEFERS every other tenant (defer_ids). No Frontier needed — the
policy is driven with SimpleNamespace requests carrying the same fields the bridge synthesizes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "policy_group_admission.py"


def _load_policy():
    spec = importlib.util.spec_from_file_location("policy_group_admission", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _req(rid, session, arrival, is_prefill=True):
    return SimpleNamespace(request_id=rid, session_id=session, arrival_time_s=arrival,
                           is_prefill=is_prefill)


def test_admits_current_tenant_defers_the_rest():
    pol = _load_policy()
    waiting = [
        _req("t0-a", 0, 0.00), _req("t1-a", 1, 0.05),
        _req("t0-b", 0, 0.10), _req("t1-b", 1, 0.15),
    ]
    d = pol.schedule_batch(waiting, [], 8192, 256, 1 << 30, 0.0)
    # tenant 0 arrived first -> it is the current group; both of its requests are admitted in order
    assert d.prefill_batch == ["t0-a", "t0-b"]
    # tenant 1 is explicitly DEFERRED (the second channel), not silently forgotten
    assert sorted(d.defer_ids) == ["t1-a", "t1-b"]
    # the two channels are disjoint and cover every waiting id
    assert set(d.prefill_batch).isdisjoint(d.defer_ids)
    assert set(d.prefill_batch) | set(d.defer_ids) == {"t0-a", "t0-b", "t1-a", "t1-b"}


def test_decode_passthrough_and_empty_waiting():
    pol = _load_policy()
    running = [_req("r0", 0, 0.0, is_prefill=False)]
    d = pol.schedule_batch([], running, 8192, 256, 1 << 30, 0.0)
    assert d.prefill_batch == [] and d.defer_ids == []
    assert d.decode_batch == ["r0"]


def test_single_tenant_defers_nothing():
    pol = _load_policy()
    waiting = [_req("a", 7, 0.0), _req("b", 7, 0.1)]
    d = pol.schedule_batch(waiting, [], 8192, 256, 1 << 30, 0.0)
    assert d.prefill_batch == ["a", "b"] and d.defer_ids == []


def test_deterministic_across_calls():
    pol = _load_policy()
    waiting = [_req("t1-a", 1, 0.05), _req("t0-a", 0, 0.00)]
    d1 = pol.schedule_batch(waiting, [], 8192, 256, 1 << 30, 0.0)
    d2 = pol.schedule_batch(waiting, [], 8192, 256, 1 << 30, 0.0)
    assert d1.prefill_batch == d2.prefill_batch == ["t0-a"]
    assert d1.defer_ids == d2.defer_ids == ["t1-a"]
