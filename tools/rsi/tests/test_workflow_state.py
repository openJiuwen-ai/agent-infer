from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_evolve.engine.workflow_state import (
    LEGAL_TRANSITIONS,
    TERMINAL_STAGES,
    WORKFLOW_STAGES,
    ArtifactRef,
    ArtifactValidationError,
    IllegalTransitionError,
    ResumeRequestMismatchError,
    WorkflowLedger,
    WorkflowStage,
    request_fingerprint,
)

EXPECTED_STAGES = [
    "GOAL_NORMALIZED",
    "CONTEXT_FROZEN",
    "DIAGNOSIS_FROZEN",
    "RESEARCH_FROZEN",
    "SEARCHING",
    "SELECTION_FROZEN",
    "WINNER_FROZEN",
    "HELDOUT_MATERIALIZED",
    "BASELINES_EVALUATED",
    "HELDOUT_EVALUATED",
    "ABLATIONS_EVALUATED",
    "SIM_WINNER",
    "NO_WINNER",
    "FAILED_RETRIABLE",
    "FAILED_TERMINAL",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


def test_canonical_order_and_legal_transition_table():
    assert [stage.value for stage in WORKFLOW_STAGES] == EXPECTED_STAGES
    assert all(LEGAL_TRANSITIONS[terminal] == () for terminal in TERMINAL_STAGES)
    assert WorkflowStage.NO_WINNER in LEGAL_TRANSITIONS[WorkflowStage.SEARCHING]
    assert WorkflowStage.NO_WINNER in LEGAL_TRANSITIONS[WorkflowStage.HELDOUT_EVALUATED]
    assert WorkflowStage.SIM_WINNER not in LEGAL_TRANSITIONS[WorkflowStage.SEARCHING]
    assert LEGAL_TRANSITIONS[WorkflowStage.ABLATIONS_EVALUATED][0] == WorkflowStage.SIM_WINNER


def test_create_is_run_scoped_and_records_deterministic_request_fingerprint(tmp_path: Path):
    request_a = {"goal": "reduce p99", "seeds": [0, 1, 2]}
    request_b = {"seeds": [0, 1, 2], "goal": "reduce p99"}
    root = tmp_path / "run-a"

    ledger = WorkflowLedger.create(root, request_a, run_id="run-a-id")
    state = ledger.state

    assert request_fingerprint(request_a) == request_fingerprint(request_b)
    assert state.request_fingerprint == request_fingerprint(request_b)
    assert state.run_id == "run-a-id"
    assert state.stage is WorkflowStage.GOAL_NORMALIZED
    assert state.sequence == 1
    assert json.loads((root / "state.json").read_text())["stage"] == "GOAL_NORMALIZED"
    rows = _event_rows(root)
    assert [(row["sequence"], row["kind"], row["to_stage"]) for row in rows] == [
        (1, "created", "GOAL_NORMALIZED")
    ]


def test_happy_path_is_resumable_and_reaches_sim_winner(tmp_path: Path):
    request = {"target": "scheduler", "budget": 4}
    root = tmp_path / "run"
    ledger = WorkflowLedger.create(root, request)
    path = [
        WorkflowStage.CONTEXT_FROZEN,
        WorkflowStage.DIAGNOSIS_FROZEN,
        WorkflowStage.RESEARCH_FROZEN,
        WorkflowStage.SEARCHING,
        WorkflowStage.SELECTION_FROZEN,
        WorkflowStage.WINNER_FROZEN,
        WorkflowStage.HELDOUT_MATERIALIZED,
        WorkflowStage.BASELINES_EVALUATED,
        WorkflowStage.HELDOUT_EVALUATED,
        WorkflowStage.ABLATIONS_EVALUATED,
        WorkflowStage.SIM_WINNER,
    ]

    for stage in path[:4]:
        ledger.transition(stage)
    ledger = WorkflowLedger.resume(root, {"budget": 4, "target": "scheduler"})
    for stage in path[4:]:
        ledger.transition(stage)

    assert ledger.state.stage is WorkflowStage.SIM_WINNER
    assert ledger.state.is_terminal
    assert len(_event_rows(root)) == len(path) + 1
    with pytest.raises(IllegalTransitionError, match="terminal"):
        ledger.transition(WorkflowStage.NO_WINNER)


def test_illegal_transition_does_not_mutate_state_or_events(tmp_path: Path):
    root = tmp_path / "run"
    ledger = WorkflowLedger.create(root, {"goal": "x"})
    before_state = (root / "state.json").read_bytes()
    before_events = (root / "events.jsonl").read_bytes()

    with pytest.raises(IllegalTransitionError, match="GOAL_NORMALIZED -> SEARCHING"):
        ledger.transition(WorkflowStage.SEARCHING)

    assert (root / "state.json").read_bytes() == before_state
    assert (root / "events.jsonl").read_bytes() == before_events


def test_resume_rejects_a_different_request(tmp_path: Path):
    root = tmp_path / "run"
    WorkflowLedger.create(root, {"goal": "original", "seed": 0})

    with pytest.raises(ResumeRequestMismatchError, match="fingerprint mismatch"):
        WorkflowLedger.resume(root, {"goal": "changed", "seed": 0})

    resumed = WorkflowLedger.resume(
        root,
        expected_request_fingerprint=request_fingerprint({"seed": 0, "goal": "original"}),
    )
    assert resumed.current_stage is WorkflowStage.GOAL_NORMALIZED


def test_artifacts_are_confined_hashed_and_revalidated_before_each_transition(
    tmp_path: Path,
):
    root = tmp_path / "run"
    ledger = WorkflowLedger.create(root, {"goal": "x"})
    artifact = root / "context.json"
    artifact.write_text('{"frozen":true}\n')

    state = ledger.transition(
        WorkflowStage.CONTEXT_FROZEN,
        artifacts=[ArtifactRef(path="context.json", sha256=_sha(artifact))],
    )
    assert state.artifacts[0].path == "context.json"

    artifact.write_text('{"frozen":false}\n')
    before_events = (root / "events.jsonl").read_bytes()
    with pytest.raises(ArtifactValidationError, match="sha256 mismatch"):
        ledger.transition(WorkflowStage.DIAGNOSIS_FROZEN)
    assert (root / "events.jsonl").read_bytes() == before_events
    assert ledger.state.stage is WorkflowStage.CONTEXT_FROZEN


def test_bad_or_outside_artifact_blocks_transition_without_a_write(tmp_path: Path):
    root = tmp_path / "run"
    ledger = WorkflowLedger.create(root, {"goal": "x"})
    outside = tmp_path / "outside.json"
    outside.write_text("external")
    before_events = (root / "events.jsonl").read_bytes()

    with pytest.raises(ArtifactValidationError, match="escapes run root"):
        ledger.transition(
            WorkflowStage.CONTEXT_FROZEN,
            artifacts=[{"path": str(outside), "sha256": _sha(outside)}],
        )
    with pytest.raises(ArtifactValidationError, match="invalid sha256"):
        ledger.transition(
            WorkflowStage.CONTEXT_FROZEN,
            artifacts=[{"path": "missing", "sha256": "not-a-sha"}],
        )
    assert (root / "events.jsonl").read_bytes() == before_events


@pytest.mark.parametrize(
    ("target", "retriable"),
    [
        (WorkflowStage.FAILED_RETRIABLE, True),
        (WorkflowStage.FAILED_TERMINAL, False),
    ],
)
def test_failure_terminals_require_and_persist_explicit_diagnostics(
    tmp_path: Path,
    target: WorkflowStage,
    retriable: bool,
):
    root = tmp_path / target.value
    ledger = WorkflowLedger.create(root, {"goal": "x"})

    with pytest.raises(ValueError, match="requires explicit failure diagnostics"):
        ledger.transition(target)
    failed = ledger.fail(
        "Frontier worker exited",
        retriable=retriable,
        code="frontier_exit",
        details={"exit_code": 75},
    )

    assert failed.stage is target
    assert failed.diagnostics == {
        "code": "frontier_exit",
        "details": {"exit_code": 75},
        "message": "Frontier worker exited",
    }
    assert _event_rows(root)[-1]["diagnostics"] == failed.diagnostics


def test_no_winner_may_terminate_search_without_selection(tmp_path: Path):
    ledger = WorkflowLedger.create(tmp_path / "run", {"goal": "x"})
    for stage in (
        WorkflowStage.CONTEXT_FROZEN,
        WorkflowStage.DIAGNOSIS_FROZEN,
        WorkflowStage.RESEARCH_FROZEN,
        WorkflowStage.SEARCHING,
    ):
        ledger.transition(stage)

    state = ledger.transition(
        WorkflowStage.NO_WINNER,
        diagnostics={"reason": "all candidates failed verification"},
    )
    assert state.stage is WorkflowStage.NO_WINNER
    assert state.is_terminal


def test_run_roots_are_independent_and_missing_snapshot_repairs_from_events(tmp_path: Path):
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first = WorkflowLedger.create(first_root, {"goal": "one"})
    second = WorkflowLedger.create(second_root, {"goal": "two"})
    first.transition(WorkflowStage.CONTEXT_FROZEN)

    assert second.current_stage is WorkflowStage.GOAL_NORMALIZED
    (first_root / "state.json").unlink()
    resumed = WorkflowLedger.resume(first_root, {"goal": "one"})
    assert resumed.current_stage is WorkflowStage.CONTEXT_FROZEN
    assert (first_root / "state.json").is_file()


def test_protected_dot_ve_is_not_a_valid_ledger_root(tmp_path: Path):
    with pytest.raises(ValueError, match="protected .ve"):
        WorkflowLedger.create(tmp_path / ".ve" / "run", {"goal": "x"})
