# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Download and materialize bounded Inferact Replay sources."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import httpx
from huggingface_hub import constants, hf_hub_url, try_to_load_from_cache
from huggingface_hub.utils import OfflineModeIsEnabled, build_hf_headers

from .converters.codex_swebenchpro import _iter_json_array, _iter_json_array_chunks

INFERACT_REPO_ID = "Inferact/codex_swebenchpro_traces"
INFERACT_REVISION = "0d52ae8c75738117be9e58c7071bd9a5b43ff78f"
INFERACT_DATASET_FILE = "codex_swebenchpro.json"
INFERACT_STREAM_CHUNK_SIZE = 64 * 1024


def _default_cache_dir() -> Path:
    """Return the AgentInfer cache without changing Hugging Face's own cache."""

    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "agentinfer" / "datasets" / "inferact_codex_swebenchpro" / INFERACT_REVISION


def _iter_source_records() -> Iterator[dict[str, object]]:
    """Reuse a cached blob, otherwise stream the pinned remote JSON array."""

    source = {"repo_id": INFERACT_REPO_ID, "repo_type": "dataset", "revision": INFERACT_REVISION}
    cached = try_to_load_from_cache(filename=INFERACT_DATASET_FILE, **source)
    if isinstance(cached, str):
        yield from _iter_json_array(Path(cached))
        return
    if constants.HF_HUB_OFFLINE:
        raise OfflineModeIsEnabled("Inferact source is not cached and HF_HUB_OFFLINE is enabled")

    url = hf_hub_url(filename=INFERACT_DATASET_FILE, **source)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        with client.stream("GET", url, headers=build_hf_headers()) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            yield from _iter_json_array_chunks(response.iter_text(chunk_size=INFERACT_STREAM_CHUNK_SIZE))


def _write_record_subset(destination: Path, task_num: int) -> None:
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
            with closing(_iter_source_records()) as records:
                for record in records:
                    if written:
                        output.write(",")
                    json.dump(record, output, ensure_ascii=False, separators=(",", ":"))
                    written += 1
                    if written == task_num:
                        break
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
    _write_record_subset(destination, task_num)
    return destination
