"""Durable, run-scoped state for the top-level Frontier evolution workflow.

The event log is the write-ahead record and ``state.json`` is its atomically
replaced snapshot.  A ledger is deliberately rooted in a caller-supplied run
directory; it never reads or writes the repository's protected ``.ve`` state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

STATE_FILENAME = "state.json"
EVENTS_FILENAME = "events.jsonl"
SCHEMA_VERSION = 1
_LOCK_FILENAME = ".workflow.lock"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowStage(str, Enum):
    """Canonical top-level workflow stages, in declaration order."""

    GOAL_NORMALIZED = "GOAL_NORMALIZED"
    CONTEXT_FROZEN = "CONTEXT_FROZEN"
    DIAGNOSIS_FROZEN = "DIAGNOSIS_FROZEN"
    RESEARCH_FROZEN = "RESEARCH_FROZEN"
    SEARCHING = "SEARCHING"
    SELECTION_FROZEN = "SELECTION_FROZEN"
    WINNER_FROZEN = "WINNER_FROZEN"
    HELDOUT_MATERIALIZED = "HELDOUT_MATERIALIZED"
    BASELINES_EVALUATED = "BASELINES_EVALUATED"
    HELDOUT_EVALUATED = "HELDOUT_EVALUATED"
    ABLATIONS_EVALUATED = "ABLATIONS_EVALUATED"
    SIM_WINNER = "SIM_WINNER"
    NO_WINNER = "NO_WINNER"
    FAILED_RETRIABLE = "FAILED_RETRIABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"


# Tuples, rather than sets, make introspection/reporting deterministic.
WORKFLOW_STAGES: tuple[WorkflowStage, ...] = tuple(WorkflowStage)
TOP_LEVEL_STAGES = WORKFLOW_STAGES
TERMINAL_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage.SIM_WINNER,
    WorkflowStage.NO_WINNER,
    WorkflowStage.FAILED_RETRIABLE,
    WorkflowStage.FAILED_TERMINAL,
)
ACTIVE_STAGES: tuple[WorkflowStage, ...] = tuple(
    stage for stage in WORKFLOW_STAGES if stage not in TERMINAL_STAGES
)

_HAPPY_PATH: tuple[WorkflowStage, ...] = ACTIVE_STAGES
_transition_lists: dict[WorkflowStage, list[WorkflowStage]] = {
    stage: [] for stage in WORKFLOW_STAGES
}
for current, following in zip(_HAPPY_PATH, _HAPPY_PATH[1:]):
    _transition_lists[current].append(following)

# An exhausted/invalid search can produce no winner immediately.  A selected
# candidate can also fail any subsequent adjudication gate.
for stage in ACTIVE_STAGES[ACTIVE_STAGES.index(WorkflowStage.SEARCHING) :]:
    _transition_lists[stage].append(WorkflowStage.NO_WINNER)

_transition_lists[WorkflowStage.ABLATIONS_EVALUATED].insert(0, WorkflowStage.SIM_WINNER)
for stage in ACTIVE_STAGES:
    _transition_lists[stage].extend(
        (WorkflowStage.FAILED_RETRIABLE, WorkflowStage.FAILED_TERMINAL)
    )

LEGAL_TRANSITIONS: dict[WorkflowStage, tuple[WorkflowStage, ...]] = {
    stage: tuple(targets) for stage, targets in _transition_lists.items()
}
del _transition_lists


class WorkflowLedgerError(RuntimeError):
    """Base class for persisted workflow errors."""


class WorkflowAlreadyExistsError(WorkflowLedgerError):
    """Raised when creation would replace an existing run ledger."""


class WorkflowStateCorruptionError(WorkflowLedgerError):
    """Raised when persisted workflow state cannot be trusted."""


class IllegalTransitionError(WorkflowLedgerError):
    """Raised when a requested stage edge is absent from ``LEGAL_TRANSITIONS``."""


class ResumeRequestMismatchError(WorkflowLedgerError):
    """Raised when a resume request does not identify the original request."""


class ArtifactValidationError(WorkflowLedgerError):
    """Raised when referenced evidence is missing, escapes the run, or changed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("workflow request and diagnostics must be canonical JSON values") from exc


def request_fingerprint(request: Any) -> str:
    """Return the stable SHA-256 of a JSON-serializable workflow request."""

    return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()


compute_request_fingerprint = request_fingerprint


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_stage(stage: WorkflowStage | str) -> WorkflowStage:
    if isinstance(stage, WorkflowStage):
        return stage
    try:
        return WorkflowStage(stage)
    except ValueError as exc:
        known = ", ".join(item.value for item in WORKFLOW_STAGES)
        raise IllegalTransitionError(
            f"unknown workflow stage {stage!r}; expected one of: {known}"
        ) from exc


@dataclass(frozen=True)
class ArtifactRef:
    """A run-relative immutable evidence reference."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ArtifactValidationError("artifact path must be non-empty")
        normalized_sha = self.sha256.lower()
        if not _SHA256_RE.fullmatch(normalized_sha):
            raise ArtifactValidationError(
                f"artifact {self.path!r} has invalid sha256 {self.sha256!r}"
            )
        object.__setattr__(self, "sha256", normalized_sha)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_value(cls, value: ArtifactRef | Mapping[str, Any]) -> ArtifactRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ArtifactValidationError("artifact references must be ArtifactRef or mappings")
        try:
            path = value["path"]
            sha256 = value["sha256"]
        except KeyError as exc:
            raise ArtifactValidationError(
                "artifact reference requires both 'path' and 'sha256'"
            ) from exc
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise ArtifactValidationError("artifact path and sha256 must be strings")
        return cls(path=path, sha256=sha256)


@dataclass(frozen=True)
class WorkflowState:
    """The latest durable snapshot of one workflow run."""

    run_id: str
    request_fingerprint: str
    stage: WorkflowStage
    sequence: int
    created_at: str
    updated_at: str
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_fingerprint": self.request_fingerprint,
            "stage": self.stage.value,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowState:
        try:
            schema_version = payload["schema_version"]
            run_id = payload["run_id"]
            fingerprint = payload["request_fingerprint"]
            stage = WorkflowStage(payload["stage"])
            sequence = payload["sequence"]
            created_at = payload["created_at"]
            updated_at = payload["updated_at"]
            raw_artifacts = payload.get("artifacts", [])
            diagnostics = payload.get("diagnostics", {})
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowStateCorruptionError("state.json has an invalid workflow schema") from exc

        if schema_version != SCHEMA_VERSION:
            raise WorkflowStateCorruptionError(
                f"unsupported workflow state schema version {schema_version!r}"
            )
        if not isinstance(run_id, str) or not run_id:
            raise WorkflowStateCorruptionError("workflow run_id must be a non-empty string")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise WorkflowStateCorruptionError("workflow request_fingerprint is not a SHA-256")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise WorkflowStateCorruptionError("workflow sequence must be a positive integer")
        if not isinstance(created_at, str) or not isinstance(updated_at, str):
            raise WorkflowStateCorruptionError("workflow timestamps must be strings")
        if not isinstance(raw_artifacts, list):
            raise WorkflowStateCorruptionError("workflow artifacts must be a list")
        if not isinstance(diagnostics, Mapping):
            raise WorkflowStateCorruptionError("workflow diagnostics must be an object")
        try:
            artifacts = tuple(ArtifactRef.from_value(value) for value in raw_artifacts)
            safe_diagnostics = json.loads(_canonical_json(diagnostics))
        except (ArtifactValidationError, ValueError) as exc:
            raise WorkflowStateCorruptionError("workflow state contains invalid evidence") from exc
        return cls(
            schema_version=schema_version,
            run_id=run_id,
            request_fingerprint=fingerprint,
            stage=stage,
            sequence=sequence,
            created_at=created_at,
            updated_at=updated_at,
            artifacts=artifacts,
            diagnostics=safe_diagnostics,
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # Directory handles cannot be opened this way on Windows.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError(f"could not append workflow event to {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Serialize state/event pairs across ledger objects and processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class WorkflowLedger:
    """Create, resume, and advance one durable workflow run."""

    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        if ".ve" in self.run_root.parts:
            raise ValueError("workflow ledgers must not use protected .ve state")
        self.state_path = self.run_root / STATE_FILENAME
        self.events_path = self.run_root / EVENTS_FILENAME
        self._lock_path = self.run_root / _LOCK_FILENAME
        self._thread_lock = threading.RLock()

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        request: Any,
        *,
        run_id: str | None = None,
    ) -> WorkflowLedger:
        ledger = cls(run_root)
        ledger.run_root.mkdir(parents=True, exist_ok=True)
        with ledger._locked():
            if ledger.state_path.exists() or ledger.events_path.exists():
                raise WorkflowAlreadyExistsError(
                    f"workflow ledger already exists in {ledger.run_root}"
                )
            fingerprint = request_fingerprint(request)
            now = _utc_now()
            state = WorkflowState(
                run_id=run_id or ledger.run_root.name,
                request_fingerprint=fingerprint,
                stage=WorkflowStage.GOAL_NORMALIZED,
                sequence=1,
                created_at=now,
                updated_at=now,
            )
            ledger._commit_event("created", None, state)
        return ledger

    @classmethod
    def resume(
        cls,
        run_root: str | Path,
        request: Any | None = None,
        *,
        expected_request_fingerprint: str | None = None,
    ) -> WorkflowLedger:
        """Open a run only when the supplied request identifies the same work."""

        if (request is None) == (expected_request_fingerprint is None):
            raise ValueError(
                "resume requires exactly one of request or expected_request_fingerprint"
            )
        expected = (
            request_fingerprint(request)
            if expected_request_fingerprint is None
            else expected_request_fingerprint.lower()
        )
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError("expected_request_fingerprint must be a 64-character SHA-256")
        ledger = cls(run_root)
        with ledger._locked():
            state = ledger._read_consistent_state(repair=True)
            if state.request_fingerprint != expected:
                raise ResumeRequestMismatchError(
                    "resume request fingerprint mismatch: "
                    f"expected {state.request_fingerprint}, got {expected}"
                )
            ledger._validate_artifacts(state.artifacts)
        return ledger

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            with _file_lock(self._lock_path):
                yield

    @property
    def state(self) -> WorkflowState:
        return self.read()

    @property
    def current_stage(self) -> WorkflowStage:
        return self.read().stage

    def read(self) -> WorkflowState:
        with self._locked():
            return self._read_consistent_state(repair=True)

    def events(self) -> tuple[dict[str, Any], ...]:
        """Read the append-only audit events in sequence order."""

        with self._locked():
            return tuple(self._read_events())

    def transition(
        self,
        target: WorkflowStage | str,
        *,
        artifacts: Iterable[ArtifactRef | Mapping[str, Any]] = (),
        diagnostics: Mapping[str, Any] | None = None,
    ) -> WorkflowState:
        target_stage = _coerce_stage(target)
        new_refs = tuple(ArtifactRef.from_value(value) for value in artifacts)
        safe_diagnostics = json.loads(_canonical_json(diagnostics or {}))
        if not isinstance(safe_diagnostics, dict):
            raise ValueError("workflow diagnostics must be a JSON object")
        if target_stage in (WorkflowStage.FAILED_RETRIABLE, WorkflowStage.FAILED_TERMINAL):
            if not safe_diagnostics:
                raise ValueError(f"{target_stage.value} requires explicit failure diagnostics")

        with self._locked():
            current = self._read_consistent_state(repair=True)
            allowed = LEGAL_TRANSITIONS[current.stage]
            if target_stage not in allowed:
                choices = ", ".join(stage.value for stage in allowed) or "(none; terminal)"
                raise IllegalTransitionError(
                    f"cannot transition {current.stage.value} -> {target_stage.value}; "
                    f"allowed next stages: {choices}"
                )

            # Frozen evidence remains immutable throughout the rest of the run.
            normalized_new_refs = self._validate_artifacts(new_refs)
            self._validate_artifacts(current.artifacts)
            now = _utc_now()
            next_state = WorkflowState(
                run_id=current.run_id,
                request_fingerprint=current.request_fingerprint,
                stage=target_stage,
                sequence=current.sequence + 1,
                created_at=current.created_at,
                updated_at=now,
                artifacts=current.artifacts + normalized_new_refs,
                diagnostics=safe_diagnostics,
            )
            self._commit_event("transition", current.stage, next_state)
            return next_state

    def fail(
        self,
        message: str,
        *,
        retriable: bool,
        code: str = "workflow_failed",
        details: Mapping[str, Any] | None = None,
    ) -> WorkflowState:
        """Enter an explicit failure terminal with structured diagnostics."""

        if not message.strip():
            raise ValueError("failure message must be non-empty")
        diagnostics: dict[str, Any] = {"code": code, "message": message}
        if details:
            diagnostics["details"] = dict(details)
        target = (
            WorkflowStage.FAILED_RETRIABLE if retriable else WorkflowStage.FAILED_TERMINAL
        )
        return self.transition(target, diagnostics=diagnostics)

    def _validate_artifacts(self, artifacts: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
        normalized: list[ArtifactRef] = []
        run_root = self.run_root.resolve()
        for artifact in artifacts:
            candidate = Path(artifact.path)
            candidate = candidate if candidate.is_absolute() else self.run_root / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ArtifactValidationError(
                    f"referenced artifact does not exist: {artifact.path}"
                ) from exc
            try:
                relative = resolved.relative_to(run_root)
            except ValueError as exc:
                raise ArtifactValidationError(
                    f"referenced artifact escapes run root: {artifact.path}"
                ) from exc
            if not resolved.is_file():
                raise ArtifactValidationError(
                    f"referenced artifact is not a regular file: {artifact.path}"
                )
            actual = file_sha256(resolved)
            if actual != artifact.sha256:
                raise ArtifactValidationError(
                    f"artifact sha256 mismatch for {artifact.path}: "
                    f"expected {artifact.sha256}, got {actual}"
                )
            normalized.append(ArtifactRef(path=relative.as_posix(), sha256=actual))
        return tuple(normalized)

    def _commit_event(
        self,
        kind: str,
        previous_stage: WorkflowStage | None,
        state: WorkflowState,
    ) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "sequence": state.sequence,
            "timestamp": state.updated_at,
            "run_id": state.run_id,
            "request_fingerprint": state.request_fingerprint,
            "from_stage": previous_stage.value if previous_stage is not None else None,
            "to_stage": state.stage.value,
            "artifacts": [artifact.to_dict() for artifact in state.artifacts],
            "diagnostics": dict(state.diagnostics),
            # The complete snapshot makes a write-ahead event sufficient to
            # repair state.json after interruption between append and replace.
            "state": state.to_dict(),
        }
        _append_json_line(self.events_path, event)
        _atomic_write_json(self.state_path, state.to_dict())

    def _read_consistent_state(self, *, repair: bool) -> WorkflowState:
        events = self._read_events()
        disk_state: WorkflowState | None = None
        if self.state_path.is_file():
            try:
                raw_state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkflowStateCorruptionError(
                    f"could not parse workflow state at {self.state_path}"
                ) from exc
            if not isinstance(raw_state, Mapping):
                raise WorkflowStateCorruptionError("state.json root must be an object")
            disk_state = WorkflowState.from_dict(raw_state)

        if not events:
            if disk_state is None:
                raise WorkflowStateCorruptionError(
                    f"no workflow ledger exists in {self.run_root}"
                )
            raise WorkflowStateCorruptionError("state.json exists without an event ledger")

        last = events[-1]
        raw_event_state = last.get("state")
        if not isinstance(raw_event_state, Mapping):
            raise WorkflowStateCorruptionError("latest workflow event has no state snapshot")
        event_state = WorkflowState.from_dict(raw_event_state)
        if last.get("sequence") != event_state.sequence:
            raise WorkflowStateCorruptionError("latest workflow event sequence is inconsistent")
        if disk_state is None or disk_state.sequence < event_state.sequence:
            if repair:
                _atomic_write_json(self.state_path, event_state.to_dict())
            return event_state
        if disk_state.sequence > event_state.sequence:
            raise WorkflowStateCorruptionError("state.json is ahead of append-only events.jsonl")
        if disk_state.to_dict() != event_state.to_dict():
            raise WorkflowStateCorruptionError(
                "state.json disagrees with the latest append-only workflow event"
            )
        return disk_state

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise WorkflowStateCorruptionError(
                            f"blank workflow event at line {line_number}"
                        )
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise WorkflowStateCorruptionError(
                            f"workflow event {line_number} is not an object"
                        )
                    if event.get("sequence") != line_number:
                        raise WorkflowStateCorruptionError(
                            f"workflow event sequence mismatch at line {line_number}"
                        )
                    events.append(event)
        except json.JSONDecodeError as exc:
            raise WorkflowStateCorruptionError(
                f"events.jsonl contains invalid JSON at line {exc.lineno}"
            ) from exc
        return events


def create_workflow(
    run_root: str | Path,
    request: Any,
    *,
    run_id: str | None = None,
) -> WorkflowLedger:
    """Functional wrapper for ``WorkflowLedger.create``."""

    return WorkflowLedger.create(run_root, request, run_id=run_id)


def resume_workflow(
    run_root: str | Path,
    request: Any | None = None,
    *,
    expected_request_fingerprint: str | None = None,
) -> WorkflowLedger:
    """Functional wrapper for ``WorkflowLedger.resume``."""

    return WorkflowLedger.resume(
        run_root,
        request,
        expected_request_fingerprint=expected_request_fingerprint,
    )


__all__ = [
    "ACTIVE_STAGES",
    "ArtifactRef",
    "ArtifactValidationError",
    "EVENTS_FILENAME",
    "IllegalTransitionError",
    "LEGAL_TRANSITIONS",
    "ResumeRequestMismatchError",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "TERMINAL_STAGES",
    "TOP_LEVEL_STAGES",
    "WORKFLOW_STAGES",
    "WorkflowAlreadyExistsError",
    "WorkflowLedger",
    "WorkflowLedgerError",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowStateCorruptionError",
    "compute_request_fingerprint",
    "create_workflow",
    "file_sha256",
    "request_fingerprint",
    "resume_workflow",
]
