"""Out-of-process effectiveness probe (Codex HIGH#3) — the TRUSTWORTHY effective signal.

A candidate's ``schedule_batch`` is arbitrary in-process Python, so any stdout marker
it emits is forgeable (it can reflectively reach the wrapper's emit + the per-run
nonce). The harness therefore does NOT decide effectiveness from candidate-emitted
text. Instead it DIRECTLY calls the candidate's ordering function on harness-chosen
scenarios and compares the RETURNED prefill order to FCFS. A candidate cannot make the
harness observe a reordering it did not actually return; and one that reorders only
when probed (not when serving) gains nothing on the unforgeable throughput gate (the
real proof of improvement, measured externally by the client).

Stdout markers (runtime.py) remain only a corroborating signal; this probe is the
effective signal of record.
"""
from __future__ import annotations

from targets.scheduling.seed import schedule_batch as fcfs_schedule
from targets.scheduling.skeleton import RequestInfo

_BIG = 10 ** 9


def _req(rid: str, arrival: float, prompt_tokens: int) -> RequestInfo:
    return RequestInfo(
        request_id=rid, num_prompt_tokens=prompt_tokens, num_computed_tokens=0,
        num_output_tokens=0, arrival_time_s=arrival, is_prefill=True,
        kv_blocks_used=0, prefix_cached_tokens=0,
    )


def _scenarios():
    # Budgets are huge so EVERY waiting request is admittable -> the prefill ORDER is
    # purely the candidate's free choice (isolates "did it change admission order?").
    kw = dict(max_num_batched_tokens=_BIG, max_num_seqs=256,
              available_kv_blocks=_BIG, prefix_cache_hit_rate=0.0)
    s1 = [_req("a", 0.0, 50), _req("b", 1.0, 10), _req("c", 2.0, 30)]
    s2 = [_req("x", 0.0, 100), _req("y", 1.0, 20), _req("z", 2.0, 200), _req("w", 3.0, 5)]
    return [(s1, [], kw), (s2, [], kw)]


def probe_effectiveness(candidate_fn) -> dict:
    """Directly invoke ``candidate_fn`` on harness scenarios; judge by its RETURN value.

    effective iff, on >=1 scenario, the candidate returns a VALID prefill order (no
    fabricated request_ids) that DIFFERS from FCFS. Never reads candidate stdout, so an
    emitted marker cannot make a no-op look effective. A candidate that errors, or that
    fabricates ids, or that matches FCFS everywhere, is NOT effective.
    """
    changed = 0
    errors = 0
    invalid = 0
    scenarios = _scenarios()
    for waiting, running, kw in scenarios:
        waiting_ids = {r.request_id for r in waiting}
        base = list(fcfs_schedule(waiting, running, **kw).prefill_batch)
        try:
            cand = list(candidate_fn(waiting, running, **kw).prefill_batch)
        except Exception:                       # noqa: BLE001 - any policy error => not effective
            errors += 1
            continue
        # valid = a subset of the real waiting ids with NO fabricated and NO duplicate ids
        if any(rid not in waiting_ids for rid in cand) or len(cand) != len(set(cand)):
            invalid += 1
            continue
        if cand != base:
            changed += 1
    return {
        "effective": changed > 0,
        "changed_scenarios": changed,
        "n_scenarios": len(scenarios),
        "errors": errors,
        "invalid": invalid,
        "method": "out_of_process_return_value_probe",
    }
