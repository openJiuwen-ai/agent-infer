# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Download and materialize bounded TraceLab replay sources."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

TRACELAB_REPO_ID = "UW-SyFI/TraceLab"
TRACELAB_REVISION = "v0.0.2"
TRACELAB_DATASET_FILE = "data/v0.0.2/syfi_coding_trace.jsonl.gz"


def _default_cache_dir() -> Path:
    """Return the AgentInfer cache without changing Hugging Face's own cache."""

    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "agentinfer" / "datasets" / "tracelab" / TRACELAB_REVISION


def _session_key(row: object, source_line: int) -> tuple[str, str]:
    """Read the provider-scoped session identity used by TraceLabConverter."""

    if not isinstance(row, dict):
        raise ValueError(f"TraceLab dataset line {source_line} is not an object")
    provider = row.get("provider")
    session_id = row.get("session_id")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"TraceLab dataset line {source_line} has invalid provider")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"TraceLab dataset line {source_line} has invalid session_id")
    return provider.strip(), session_id.strip()


def _write_session_subset(source: Path, destination: Path, task_num: int) -> None:
    """Write every round belonging to the first ``task_num`` encountered sessions."""

    selected: set[tuple[str, str]] = set()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with gzip.open(source, mode="rt", encoding="utf-8") as rows:
                for source_line, line in enumerate(rows, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid TraceLab dataset JSON at line {source_line}: {exc}") from exc
                    key = _session_key(row, source_line)
                    if key not in selected and len(selected) < task_num:
                        selected.add(key)
                    if key in selected:
                        output.write(line if line.endswith("\n") else f"{line}\n")
        if len(selected) < task_num:
            raise ValueError(f"TraceLab dataset contains only {len(selected)} sessions; requested task_num={task_num}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_tracelab_source(task_num: int, *, cache_dir: Path | None = None) -> Path:
    """Return a cached JSONL containing the first N complete TraceLab sessions."""

    if task_num < 1:
        raise ValueError("TraceLab task_num must be at least 1")
    target_dir = (cache_dir or _default_cache_dir()).expanduser().resolve()
    destination = target_dir / f"first-{task_num}-sessions.jsonl"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=TRACELAB_REPO_ID,
            repo_type="dataset",
            filename=TRACELAB_DATASET_FILE,
            revision=TRACELAB_REVISION,
        )
    )
    _write_session_subset(downloaded, destination, task_num)
    return destination
