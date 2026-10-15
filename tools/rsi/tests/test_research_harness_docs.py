"""R10: the research-harness docs must stay in sync with the implementation.

A light drift guard — asserts CLAUDE.md / DESIGN.md / MODE_E2E_MATRIX.md actually document the
shipped surface (the `ve experiment` verbs, the prediction-first chain, `hypothesis:<event_id>`
citations, the two-channel `defer_ids` + N=8 decay, missing-data -> inconclusive, the real Frontier
trace flag, and the AC8 marker evidence) so the docs can't silently fall behind the code.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_claude_md_documents_the_research_workflow():
    body = _read("CLAUDE.md")
    for token in ("ve experiment run", "ve experiment list", "hypothesis:<event_id>",
                  "MetricExpr", "inconclusive", "PROPOSAL-layer", "adoption gate"):
        assert token in body, token


def test_design_md_documents_the_invariants():
    body = _read("DESIGN.md")
    for token in ("MetricExpr", "ve_policy_api.ScheduleDecision", "defer_ids",
                  "adjudicate", "inconclusive", "N = 8",
                  "--trace_request_generator_config_trace_file"):
        assert token in body, token
    # missing data -> None -> inconclusive (no fabrication)
    assert any(s in body for s in ("never fabricated", "never a fabricated", "explicitly missing"))
    # append-only ledger
    assert "insert-only" in body or "append-only" in body


def test_mode_matrix_phase_f_documents_the_e2e():
    body = _read("docs/restructure/MODE_E2E_MATRIX.md")
    assert "Phase F" in body
    for token in ("VE_FRONTIER_REPO", "VE_FRONTIER_PYTHON", "ve_policy_marker.json",
                  "invocations", "defers", "hypothesis:<event_id>", "inconclusive",
                  "skip", "DoD-B"):
        assert token in body, token
    # the exact mechanism-proof rule (marker), not just loop closure
    assert "fallbacks==0" in body or "fallbacks == 0" in body
