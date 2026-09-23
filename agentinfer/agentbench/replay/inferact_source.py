# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Download and materialize bounded Inferact Replay sources."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from .converters.codex_swebenchpro import _iter_json_array

INFERACT_REPO_ID = "Inferact/codex_swebenchpro_traces"
INFERACT_REVISION = "0d52ae8c75738117be9e58c7071bd9a5b43ff78f"
INFERACT_DATASET_FILE = "codex_swebenchpro.json"


def _default_cache_dir() -> Path:
    """Return the AgentInfer cache without changing Hugging Face's own cache."""

    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "agentinfer" / "datasets" / "inferact_codex_swebenchpro" / INFERACT_REVISION


def _write_record_subset(source: Path, destination: Path, task_num: int) -> None:
    """Write the first N complete top-level records as a valid JSON array."""

    temporary_path: Path | None = None
    written = 0
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
            output.write("[")
            records = _iter_json_array(source)
            try:
                for record in records:
                    if written >= task_num:
                        break
                    if written:
                        output.write(",")
                    json.dump(record, output, ensure_ascii=False, separators=(",", ":"))
                    written += 1
            finally:
                records.close()
            output.write("]\n")
        if written == 0:
            raise ValueError("Inferact dataset contains no records")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_inferact_source(task_num: int, *, cache_dir: Path | None = None) -> Path:
    """Return a cached JSON array containing up to the first N records."""

    if task_num < 1:
        raise ValueError("Inferact task_num must be at least 1")
    target_dir = (cache_dir or _default_cache_dir()).expanduser().resolve()
    destination = target_dir / f"first-{task_num}-records.json"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=INFERACT_REPO_ID,
            repo_type="dataset",
            filename=INFERACT_DATASET_FILE,
            revision=INFERACT_REVISION,
        )
    )
    _write_record_subset(downloaded, destination, task_num)
    return destination
