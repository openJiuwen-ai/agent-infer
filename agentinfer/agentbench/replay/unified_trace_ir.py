# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Define and validate the unified Trace Replay intermediate representation.

This module owns the multi-file IR contract shared by heterogeneous source
converters and runtime consumers. It does not parse source datasets or
construct Backend prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import ReplayAnalysis, ReplaySourceEvidence, ReplayTimingEvidence

BUNDLE_SCHEMA_VERSION = "2"
TOKEN_RECIPE_SCHEMA_VERSION = "3"
BUNDLE_MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    """Return the streaming SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    """Count newline-terminated lines without decoding large sidecars."""

    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def canonical_sha256(value: object) -> str:
    """Hash JSON using the canonical encoding shared by writer and validator."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PromptReference:
    """Reference one human turn in the sidecar transcript store."""

    session_id: str
    turn_index: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the sidecar lookup key for plan artifacts."""

        return {"session_id": self.session_id, "turn_index": self.turn_index}


@dataclass(frozen=True)
class UnifiedTraceIR:
    """A validated requests/text/manifest intermediate representation."""

    root: Path
    requests_path: Path
    text_dir: Path | None
    manifest_path: Path
    bundle_sha256: str
    prompt_source_kind: str = "text_turns"

    def text_path(self, reference: PromptReference) -> Path:
        """Resolve a validated prompt reference beneath this IR's text root."""

        if self.text_dir is None:
            raise ValueError("this unified Trace IR has no text sidecars")
        return self.text_dir / reference.session_id / f"turn_{reference.turn_index}.txt"


def parse_prompt_reference(raw: object) -> PromptReference | None:
    """Parse an optional prompt reference without accepting path traversal."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("prompt_ref must be an object")
    session_id = raw.get("session_id")
    turn_index = raw.get("turn_index")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("prompt_ref.session_id must be a non-empty string")
    path = PurePosixPath(session_id)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError("prompt_ref.session_id must be one safe path component")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("prompt_ref.turn_index must be a non-negative integer")
    return PromptReference(session_id, turn_index)


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _safe_relative_path(raw: object, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a non-empty string")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path must stay inside the unified Trace IR")
    return path


def _validate_file(root: Path, entry: object, label: str) -> Path:
    if not isinstance(entry, dict):
        raise ValueError(f"manifest {label} entry must be an object")
    relative = _safe_relative_path(entry.get("path"), label)
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"manifest {label} file is missing: {relative}")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
        raise ValueError(f"manifest {label} SHA256 mismatch: {relative}")
    expected_lines = entry.get("lines")
    if isinstance(expected_lines, bool) or not isinstance(expected_lines, int) or expected_lines < 0:
        raise ValueError(f"manifest {label} lines must be a non-negative integer")
    if line_count(path) != expected_lines:
        raise ValueError(f"manifest {label} line-count mismatch: {relative}")
    return path


def _trace_ir_digest_payload(manifest: dict[str, object]) -> dict[str, object]:
    """Select every manifest field covered by the portable Trace IR digest."""

    payload = {
        "schema_version": manifest.get("schema_version"),
        "converter": manifest.get("converter"),
        "source": manifest.get("source"),
        "requests": manifest.get("requests"),
        "text_dir": manifest.get("text_dir"),
        "texts": manifest.get("texts"),
        "summary": manifest.get("summary"),
    }
    if manifest.get("schema_version") == TOKEN_RECIPE_SCHEMA_VERSION:
        payload["prompt_source"] = manifest.get("prompt_source")
    return payload


def validate_trace_ir(requests_path: Path, text_dir: Path | None = None) -> UnifiedTraceIR:
    """Validate the complete unified Trace IR and all cross-file references.

    The manifest is discovered next to ``requests.jsonl`` and binds the request
    rows to their text files. Dataset semantics belong to the selected converter.
    """

    requests_path = requests_path.resolve()
    text_dir = text_dir.resolve() if text_dir is not None else None
    root = requests_path.parent
    manifest_path = root / BUNDLE_MANIFEST_NAME
    manifest = _load_object(manifest_path, "unified Trace IR manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in {BUNDLE_SCHEMA_VERSION, TOKEN_RECIPE_SCHEMA_VERSION}:
        raise ValueError(f"unsupported unified Trace IR schema_version: {manifest.get('schema_version')!r}")
    prompt_source = manifest.get("prompt_source")
    if schema_version == BUNDLE_SCHEMA_VERSION:
        prompt_source_kind = "text_turns"
        if prompt_source not in (None, {"kind": "text_turns"}):
            raise ValueError("unified Trace IR v2 only supports text_turns")
    else:
        if prompt_source != {"kind": "token_recipe"}:
            raise ValueError("unified Trace IR v3 requires prompt_source.kind=token_recipe")
        prompt_source_kind = "token_recipe"
    calculated_digest = canonical_sha256(_trace_ir_digest_payload(manifest))
    if manifest.get("bundle_sha256") != calculated_digest:
        raise ValueError("bundle_sha256 does not match manifest contents")
    manifest_requests = _validate_file(root, manifest.get("requests"), "requests")
    if manifest_requests != requests_path:
        raise ValueError("converted requests path does not match the Trace IR manifest")
    expected_paths: set[Path] = set()
    if prompt_source_kind == "text_turns":
        if text_dir is None:
            raise ValueError("text_turns unified Trace IR requires a text directory")
        manifest_text_dir = _safe_relative_path(manifest.get("text_dir"), "text_dir")
        if root.joinpath(*manifest_text_dir.parts).resolve() != text_dir:
            raise ValueError("converted texts path does not match the Trace IR manifest")
        if not text_dir.is_dir():
            raise ValueError(f"Trace IR text directory is missing: {text_dir}")
        raw_texts = manifest.get("texts")
        if not isinstance(raw_texts, list):
            raise ValueError("manifest texts must be a list")
        for index, entry in enumerate(raw_texts):
            path = _validate_file(root, entry, f"texts[{index}]")
            if path in expected_paths:
                raise ValueError(f"duplicate text manifest entry: {path.relative_to(root)}")
            if text_dir not in path.parents:
                raise ValueError(f"text manifest entry escapes text_dir: {path.relative_to(root)}")
            expected_paths.add(path)
        actual_paths = {path.resolve() for path in text_dir.rglob("*") if path.is_file()}
        if actual_paths != expected_paths:
            missing = expected_paths - actual_paths
            extra = actual_paths - expected_paths
            raise ValueError(f"text manifest inventory mismatch: missing={len(missing)} extra={len(extra)}")
    elif text_dir is not None:
        raise ValueError("token_recipe unified Trace IR must not supply a text directory")

    request_rows = 0
    referenced_paths: set[Path] = set()
    references: set[PromptReference] = set()
    turns_by_session: dict[str, set[int]] = {}
    recipe_ids: set[str] = set()
    recipe_sequences: dict[str, set[int]] = {}
    recipe_previous: dict[str, str | None] = {}
    with requests_path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid requests JSON at line {source_line}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"requests line {source_line} is not an object")
            if prompt_source_kind == "text_turns":
                reference = parse_prompt_reference(row.get("prompt_ref"))
                if reference is None:
                    raise ValueError(f"requests line {source_line} has no prompt_ref")
                if row.get("session_id") != reference.session_id:
                    raise ValueError(f"requests line {source_line} prompt_ref session does not match session_id")
                if reference in references:
                    raise ValueError(f"requests line {source_line} duplicates a prompt_ref")
                references.add(reference)
                turns_by_session.setdefault(reference.session_id, set()).add(reference.turn_index)
                assert text_dir is not None
                path = text_dir / reference.session_id / f"turn_{reference.turn_index}.txt"
                if path.resolve() not in expected_paths:
                    raise ValueError(f"requests line {source_line} references an unmanifested text file")
                referenced_paths.add(path.resolve())
            else:
                request_id = row.get("request_id")
                session_id = row.get("session_id")
                sequence = row.get("sequence_index")
                input_after = row.get("input_after")
                send_after = row.get("send_after")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError(f"requests line {source_line} has no request_id")
                if request_id in recipe_ids:
                    raise ValueError(f"requests line {source_line} duplicates request_id {request_id}")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError(f"requests line {source_line} has no session_id")
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                    raise ValueError(f"requests line {source_line} has invalid sequence_index")
                if input_after is not None and (not isinstance(input_after, str) or input_after not in recipe_ids):
                    raise ValueError(f"requests line {source_line} input_after must reference an earlier request")
                if sequence == 0 and input_after is not None:
                    raise ValueError(f"requests line {source_line} first recipe request has input_after")
                if sequence > 0 and input_after != recipe_previous.get(session_id):
                    raise ValueError(f"requests line {source_line} input_after does not reference the previous request")
                if send_after != input_after:
                    raise ValueError(f"requests line {source_line} send_after must match input_after")
                if row.get("context_after") is not None or row.get("prompt_ref") is not None:
                    raise ValueError(f"requests line {source_line} token_recipe cannot reference text context")
                for field in ("input_tokens", "output_tokens"):
                    value = row.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError(f"requests line {source_line} has invalid {field}")
                cached = row.get("cached_tokens")
                if (
                    isinstance(cached, bool)
                    or not isinstance(cached, int)
                    or cached < 0
                    or cached > row["input_tokens"]
                ):
                    raise ValueError(f"requests line {source_line} has invalid cached_tokens")
                sequences = recipe_sequences.setdefault(session_id, set())
                if sequence in sequences:
                    raise ValueError(f"requests line {source_line} duplicates sequence_index")
                sequences.add(sequence)
                recipe_ids.add(request_id)
                recipe_previous[session_id] = request_id
            request_rows += 1
    if request_rows != int(manifest["requests"]["lines"]):
        raise ValueError("requests row count does not match manifest")
    if prompt_source_kind == "text_turns" and referenced_paths != expected_paths:
        raise ValueError("one or more manifested text files are not referenced by requests.jsonl")
    for session_id, turns in turns_by_session.items():
        if turns != set(range(len(turns))):
            raise ValueError(f"prompt_ref turns are not contiguous for session {session_id}")
    for session_id, sequences in recipe_sequences.items():
        if sequences != set(range(len(sequences))):
            raise ValueError(f"sequence_index values are not contiguous for session {session_id}")

    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("manifest summary must be an object")
    if summary.get("requests") != request_rows or summary.get("text_files") != len(expected_paths):
        raise ValueError("manifest summary request/text coverage does not match the Trace IR")
    session_count = len(turns_by_session) if prompt_source_kind == "text_turns" else len(recipe_sequences)
    if summary.get("sessions") != session_count:
        raise ValueError("manifest summary session coverage does not match the Trace IR")
    return UnifiedTraceIR(
        root,
        requests_path,
        text_dir,
        manifest_path,
        calculated_digest,
        prompt_source_kind,
    )


def write_trace_ir_manifest(
    output_dir: Path,
    *,
    converter_name: str,
    converter_version: str,
    source_path: Path,
    summary: dict[str, object],
    prompt_source_kind: str = "text_turns",
) -> dict[str, object]:
    """Inventory a converted Trace IR and atomically publish its manifest."""

    output_dir = output_dir.resolve()
    requests_path = output_dir / "requests.jsonl"
    text_dir = output_dir / "texts"
    if not requests_path.is_file():
        raise ValueError("converter output is missing requests.jsonl")
    if prompt_source_kind == "text_turns" and not text_dir.is_dir():
        raise ValueError("converter output is missing texts/")
    if prompt_source_kind not in {"text_turns", "token_recipe"}:
        raise ValueError(f"unsupported prompt source kind: {prompt_source_kind}")

    def entry(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
            "lines": line_count(path),
            "bytes": path.stat().st_size,
        }

    manifest_without_digest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION if prompt_source_kind == "text_turns" else TOKEN_RECIPE_SCHEMA_VERSION,
        "converter": {"name": converter_name, "version": converter_version},
        "source": {
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "requests": entry(requests_path),
        "summary": summary,
    }
    if prompt_source_kind == "text_turns":
        manifest_without_digest["text_dir"] = "texts"
        manifest_without_digest["texts"] = [entry(path) for path in sorted(text_dir.rglob("*")) if path.is_file()]
    else:
        manifest_without_digest["prompt_source"] = {"kind": prompt_source_kind}
        manifest_without_digest["text_dir"] = None
        manifest_without_digest["texts"] = None
    manifest = {
        **manifest_without_digest,
        "bundle_sha256": canonical_sha256(manifest_without_digest),
    }
    temporary = output_dir / f".{BUNDLE_MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / BUNDLE_MANIFEST_NAME)
    return manifest


class TraceTextStore:
    """Read immutable human turns from a validated unified Trace IR."""

    def __init__(self, trace_ir: UnifiedTraceIR) -> None:
        """Create a read-through cache over a fully validated Trace IR."""

        self.trace_ir = trace_ir
        self._cache: dict[PromptReference, str] = {}

    def read(self, reference: PromptReference) -> str:
        """Return one immutable human turn, caching successful reads by reference."""

        text = self._cache.get(reference)
        if text is None:
            text = self.trace_ir.text_path(reference).read_text(encoding="utf-8")
            self._cache[reference] = text
        return text


_LEGACY_ANALYSIS_COUNTERS = (
    "invalid_json",
    "not_an_object",
    "missing_session_id",
    "missing_agent_id",
    "invalid_timestamps",
    "retry_matched_failures",
    "rows_without_replayable_tokens",
    "sessions_without_replayable_requests",
    "inferred_actor_role",
    "invalid_token_counts",
    "duplicate_request_ids",
    "pruned_terminal_timing_dependency_rows",
    "inferred_late_title_requests",
    "inferred_lead_name_requests",
)


def _explicit_required_string(row: dict[str, object], field: str, source_line: int) -> str:
    """Return a required non-empty string from one validated explicit IR row."""

    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"explicit IR line {source_line} has invalid {field}")
    return value


def _explicit_optional_string(value: object) -> str | None:
    """Return an optional non-empty string without coercing source values."""

    return value if isinstance(value, str) and value else None


def _explicit_required_int(row: dict[str, object], field: str, source_line: int, *, positive: bool) -> int:
    """Return a strict integer satisfying the explicit field's lower bound."""

    value = row.get(field)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"explicit IR line {source_line} has invalid {field}")
    return value


def _explicit_optional_nonnegative_int(value: object) -> int | None:
    """Return a strict optional non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _explicit_timestamp(row: dict[str, object], field: str, source_line: int) -> datetime:
    """Parse a required explicit IR timestamp into UTC."""

    raw = _explicit_required_string(row, field, source_line)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"explicit IR line {source_line} has invalid {field}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _explicit_source_gap(row: dict[str, object], source_line: int) -> float | None:
    """Return a finite source gap, retaining unavailable timing as ``None``."""

    value = row.get("source_gap_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"explicit IR line {source_line} has invalid source_gap_seconds")
    gap = float(value)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError(f"explicit IR line {source_line} has invalid source_gap_seconds")
    return gap


def _explicit_timing_evidence(row: dict[str, object], source_line: int) -> ReplayTimingEvidence:
    """Map selected event references into typed source timing evidence."""

    from .schema import ReplayTimingEvidence

    raw = row.get("timing_proxy")
    if not isinstance(raw, dict):
        raise ValueError(f"explicit IR line {source_line} has invalid timing_proxy")
    return ReplayTimingEvidence(
        valid=row.get("timing_valid") is True,
        basis=_explicit_optional_string(raw.get("basis")),
        reason=_explicit_optional_string(raw.get("reason")),
        input_event_type=_explicit_optional_string(raw.get("input_ready_event_type")),
        input_event_index=_explicit_optional_nonnegative_int(raw.get("input_ready_event_index")),
        output_event_type=_explicit_optional_string(raw.get("output_end_event_type")),
        output_event_index=_explicit_optional_nonnegative_int(raw.get("output_end_event_index")),
    )


def _explicit_source_evidence(row: dict[str, object], source_line: int) -> ReplaySourceEvidence:
    """Map dataset provenance without exposing it as scheduling state."""

    from .schema import ReplaySourceEvidence

    return ReplaySourceEvidence(
        provider=_explicit_required_string(row, "source_provider", source_line),
        session_id=_explicit_required_string(row, "source_session_id", source_line),
        round_index=_explicit_required_int(row, "round_index", source_line, positive=False),
        round_id=_explicit_optional_string(row.get("source_round_id")),
        trace_key=_explicit_optional_string(row.get("source_trace_key")),
        model=_explicit_optional_string(row.get("source_model")),
        newly_append_tokens=_explicit_required_int(row, "newly_append_tokens", source_line, positive=False),
        timing=_explicit_timing_evidence(row, source_line),
        tool_count=_explicit_required_int(row, "tool_count", source_line, positive=False),
        tool_error_count=_explicit_required_int(row, "tool_error_count", source_line, positive=False),
    )


def analyze_explicit_trace_ir(trace_ir: UnifiedTraceIR) -> ReplayAnalysis:
    """Build Replay sessions from a validated token-recipe IR without inference."""

    from .schema import ReplayAnalysis, ReplayRequest, ReplaySession

    if trace_ir.prompt_source_kind != "token_recipe":
        raise ValueError("explicit Replay analysis requires token_recipe unified Trace IR")
    rows_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    with trace_ir.requests_path.open(encoding="utf-8") as handle:
        for ir_line, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"explicit IR line {ir_line} is not an object")
            row = {str(key): value for key, value in raw.items()}
            rows_by_session[_explicit_required_string(row, "session_id", ir_line)].append(row)

    sessions: list[ReplaySession] = []
    unavailable_timing = 0
    round_index_gaps = 0
    for session_id, session_rows in sorted(rows_by_session.items()):
        rows = sorted(session_rows, key=lambda row: int(row["sequence_index"]))
        requests: list[ReplayRequest] = []
        previous_round_index: int | None = None
        for row in rows:
            source_line = _explicit_required_int(row, "source_line", 0, positive=True)
            request_id = _explicit_required_string(row, "request_id", source_line)
            sequence_index = _explicit_required_int(row, "sequence_index", source_line, positive=False)
            input_after = _explicit_optional_string(row.get("input_after"))
            gap = _explicit_source_gap(row, source_line)
            evidence = _explicit_source_evidence(row, source_line)
            if not evidence.timing.valid:
                unavailable_timing += 1
            if previous_round_index is not None and evidence.round_index != previous_round_index + 1:
                round_index_gaps += 1
            previous_round_index = evidence.round_index
            input_tokens = _explicit_required_int(row, "input_tokens", source_line, positive=True)
            output_tokens = _explicit_required_int(row, "output_tokens", source_line, positive=True)
            requests.append(
                ReplayRequest(
                    key=request_id,
                    source_line=source_line,
                    source_request_id=request_id,
                    actor_id="lead",
                    actor_role="lead",
                    parent_actor_id=None,
                    started_at=_explicit_timestamp(row, "started_at", source_line),
                    finished_at=_explicit_timestamp(row, "finished_at", source_line),
                    historical_status="unknown",
                    source_input_tokens=input_tokens,
                    source_output_tokens=output_tokens,
                    source_cached_tokens=_explicit_required_int(row, "cached_tokens", source_line, positive=False),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    replay_kind="request",
                    prompt_kind="lead_main" if sequence_index == 0 else "continuation",
                    prompt_kind_source="explicit",
                    prompt_kind_inference_rule=None,
                    retry_match_key=None,
                    prompt_recipe_key=request_id,
                    prompt_ref=None,
                    send_after=input_after,
                    context_after=None,
                    delay_seconds=gap or 0.0,
                    dependency_kind="session_root" if input_after is None else "same_agent",
                    same_agent_gap_seconds=gap if input_after is not None else None,
                    parallel_with=(),
                    input_after=input_after,
                    source_evidence=evidence,
                )
            )
        source_task_id = _explicit_optional_string(rows[0].get("task_id")) if rows else None
        sessions.append(ReplaySession(session_id, source_task_id, tuple(requests)))

    row_analysis = dict.fromkeys(_LEGACY_ANALYSIS_COUNTERS, 0)
    row_analysis["explicit_timing_unavailable"] = unavailable_timing
    row_analysis["source_round_index_gaps"] = round_index_gaps
    total_rows = sum(len(rows) for rows in rows_by_session.values())
    return ReplayAnalysis(
        source_path=str(trace_ir.requests_path),
        source_sha256=sha256_file(trace_ir.requests_path),
        total_rows=total_rows,
        valid_rows=total_rows,
        row_analysis=dict(sorted(row_analysis.items())),
        sessions=tuple(sessions),
    )
