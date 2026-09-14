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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BUNDLE_SCHEMA_VERSION = "2"
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
    text_dir: Path
    manifest_path: Path
    bundle_sha256: str

    def text_path(self, reference: PromptReference) -> Path:
        """Resolve a validated prompt reference beneath this IR's text root."""

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

    return {
        "schema_version": manifest.get("schema_version"),
        "converter": manifest.get("converter"),
        "source": manifest.get("source"),
        "requests": manifest.get("requests"),
        "text_dir": manifest.get("text_dir"),
        "texts": manifest.get("texts"),
        "summary": manifest.get("summary"),
    }


def validate_trace_ir(requests_path: Path, text_dir: Path) -> UnifiedTraceIR:
    """Validate the complete unified Trace IR and all cross-file references.

    The manifest is discovered next to ``requests.jsonl`` and binds the request
    rows to their text files. Dataset semantics belong to the selected converter.
    """

    requests_path = requests_path.resolve()
    text_dir = text_dir.resolve()
    root = requests_path.parent
    manifest_path = root / BUNDLE_MANIFEST_NAME
    manifest = _load_object(manifest_path, "unified Trace IR manifest")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported unified Trace IR schema_version: {manifest.get('schema_version')!r}")
    calculated_digest = canonical_sha256(_trace_ir_digest_payload(manifest))
    if manifest.get("bundle_sha256") != calculated_digest:
        raise ValueError("bundle_sha256 does not match manifest contents")
    manifest_requests = _validate_file(root, manifest.get("requests"), "requests")
    if manifest_requests != requests_path:
        raise ValueError("converted requests path does not match the Trace IR manifest")
    manifest_text_dir = _safe_relative_path(manifest.get("text_dir"), "text_dir")
    if root.joinpath(*manifest_text_dir.parts).resolve() != text_dir:
        raise ValueError("converted texts path does not match the Trace IR manifest")
    if not text_dir.is_dir():
        raise ValueError(f"Trace IR text directory is missing: {text_dir}")

    raw_texts = manifest.get("texts")
    if not isinstance(raw_texts, list):
        raise ValueError("manifest texts must be a list")
    expected_paths: set[Path] = set()
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

    request_rows = 0
    referenced_paths: set[Path] = set()
    references: set[PromptReference] = set()
    turns_by_session: dict[str, set[int]] = {}
    with requests_path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid requests JSON at line {source_line}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"requests line {source_line} is not an object")
            reference = parse_prompt_reference(row.get("prompt_ref"))
            if reference is None:
                raise ValueError(f"requests line {source_line} has no prompt_ref")
            if row.get("session_id") != reference.session_id:
                raise ValueError(f"requests line {source_line} prompt_ref session does not match session_id")
            if reference in references:
                raise ValueError(f"requests line {source_line} duplicates a prompt_ref")
            references.add(reference)
            turns_by_session.setdefault(reference.session_id, set()).add(reference.turn_index)
            path = text_dir / reference.session_id / f"turn_{reference.turn_index}.txt"
            if path.resolve() not in expected_paths:
                raise ValueError(f"requests line {source_line} references an unmanifested text file")
            referenced_paths.add(path.resolve())
            request_rows += 1
    if request_rows != int(manifest["requests"]["lines"]):
        raise ValueError("requests row count does not match manifest")
    if referenced_paths != expected_paths:
        raise ValueError("one or more manifested text files are not referenced by requests.jsonl")
    for session_id, turns in turns_by_session.items():
        if turns != set(range(len(turns))):
            raise ValueError(f"prompt_ref turns are not contiguous for session {session_id}")

    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("manifest summary must be an object")
    if summary.get("requests") != request_rows or summary.get("text_files") != len(expected_paths):
        raise ValueError("manifest summary request/text coverage does not match the Trace IR")
    if summary.get("sessions") != len(turns_by_session):
        raise ValueError("manifest summary session coverage does not match the Trace IR")
    return UnifiedTraceIR(
        root,
        requests_path,
        text_dir,
        manifest_path,
        calculated_digest,
    )


def write_trace_ir_manifest(
    output_dir: Path,
    *,
    converter_name: str,
    converter_version: str,
    source_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    """Inventory a converted Trace IR and atomically publish its manifest."""

    output_dir = output_dir.resolve()
    requests_path = output_dir / "requests.jsonl"
    text_dir = output_dir / "texts"
    if not requests_path.is_file():
        raise ValueError("converter output is missing requests.jsonl")
    if not text_dir.is_dir():
        raise ValueError("converter output is missing texts/")

    def entry(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
            "lines": line_count(path),
            "bytes": path.stat().st_size,
        }

    manifest_without_digest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "converter": {"name": converter_name, "version": converter_version},
        "source": {
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "requests": entry(requests_path),
        "text_dir": "texts",
        "texts": [entry(path) for path in sorted(text_dir.rglob("*")) if path.is_file()],
        "summary": summary,
    }
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
