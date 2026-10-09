"""HIGH#3 close: out-of-process effectiveness probe (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:  # pragma: no cover - import path bootstrap
        sys.path.insert(0, p)

from targets.scheduling.seed import schedule_batch as fcfs  # noqa: E402
from targets.scheduling.skeleton import ScheduleDecision  # noqa: E402
from vllm_evolve.bench.probe import probe_effectiveness  # noqa: E402


def test_fcfs_is_not_effective():
    # the seed equals FCFS -> no reordering -> not effective
    assert probe_effectiveness(fcfs)["effective"] is False


def test_reordering_policy_is_effective():
    # shortest-prompt-first: genuinely different admission order from FCFS
    def sjf(waiting, running, **kw):
        order = [r.request_id for r in sorted(waiting, key=lambda r: r.remaining_prompt_tokens)]
        return ScheduleDecision(prefill_batch=order, decode_batch=[])
    r = probe_effectiveness(sjf)
    assert r["effective"] is True and r["changed_scenarios"] >= 1


def test_stdout_marker_cannot_forge_effectiveness():
    # a no-op policy that PRINTS the marker is still judged by its RETURN value -> not effective
    def forger(waiting, running, **kw):
        print("vllm-evolve: waiting reordered")   # forged stdout — probe ignores it
        return fcfs(waiting, running, **kw)        # but the order is unchanged
    assert probe_effectiveness(forger)["effective"] is False


def test_fabricated_ids_not_effective():
    def faker(waiting, running, **kw):
        return ScheduleDecision(prefill_batch=["FAKE_ID"], decode_batch=[])
    r = probe_effectiveness(faker)
    assert r["effective"] is False and r["invalid"] >= 1


def test_duplicate_ids_not_effective():
    def dup(waiting, running, **kw):
        rid = waiting[0].request_id
        return ScheduleDecision(prefill_batch=[rid, rid], decode_batch=[])
    r = probe_effectiveness(dup)
    assert r["effective"] is False and r["invalid"] >= 1


def test_erroring_policy_not_effective():
    def boom(waiting, running, **kw):
        raise RuntimeError("boom")
    r = probe_effectiveness(boom)
    assert r["effective"] is False and r["errors"] >= 1
