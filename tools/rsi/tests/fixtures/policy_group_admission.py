"""H* B-arm: a hand-written tenant-grouped admission policy (NOT evolved/authored).

This is the deterministic B arm for the AC8 acceptance loop. The hypothesis H* is that under bursty
multi-tenant load, prefix-cache thrash drives the TTFT tail, and admitting one TENANT's requests at a
time (instead of interleaving all tenants FCFS) reduces that thrash. So this policy:

  * groups the waiting requests by tenant (``session_id``, exposed by the ve_policy bridge);
  * admits the CURRENT group — the tenant whose oldest waiting request arrived earliest (a stateless,
    deterministic, FCFS-fair choice of "current");
  * explicitly DEFERS every other tenant's requests via ``defer_ids`` (the bridge moves them to the
    step and force-admits any tenant starved for >= N consecutive events — anti-starvation is the
    bridge's job, not the policy's).

It is a *test asset*, deliberately simple and deterministic — the acceptance test proves the research
LOOP closes (express -> measure -> execute -> adjudicate -> ledger -> reuse), not that this particular
policy wins. The verdict (supported / falsified / inconclusive) is decided by the deterministic
adjudicator from the real measurement, never asserted here.
"""
from __future__ import annotations

import os
import sys

# Import the sanctioned defer-aware decision API. The bridge runs this policy inside the Frontier
# interpreter and exports VE_POLICY_ROOT (the repo root); offline tests run from the repo root too.
_ROOT = os.environ.get("VE_POLICY_ROOT") or os.getcwd()
_API_DIR = os.path.join(_ROOT, "integrations", "frontier")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from ve_policy_api import ScheduleDecision  # noqa: E402


def _tenant(req):
    """The tenant signal: the bridge exposes ``session_id``; absent that, fall back to request_id
    (each request is then its own tenant — a safe, if trivial, grouping)."""
    sid = getattr(req, "session_id", None)
    return sid if sid is not None else getattr(req, "request_id", "")


def _arrival(req) -> float:
    try:
        return float(getattr(req, "arrival_time_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def schedule_batch(
    waiting_requests,
    running_requests,
    max_num_batched_tokens,
    max_num_seqs,
    available_kv_blocks,
    prefix_cache_hit_rate,
) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests
                    if not getattr(r, "is_prefill", False)]
    if not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    groups: dict = {}
    for r in waiting_requests:
        groups.setdefault(_tenant(r), []).append(r)

    # current tenant = the one whose earliest-arriving waiting request is oldest (FCFS-fair across
    # tenants; deterministic tie-break on the tenant key).
    current = min(groups, key=lambda t: (min(_arrival(r) for r in groups[t]), str(t)))

    admit = sorted(groups[current], key=_arrival)
    prefill_batch = [r.request_id for r in admit]
    # everyone NOT in the current tenant is explicitly deferred (the second channel) — not merely
    # left unmentioned, so the bridge throttles their admission rather than FCFS-appending them.
    defer_ids = [r.request_id for r in waiting_requests if _tenant(r) != current]
    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch,
                            defer_ids=defer_ids)
