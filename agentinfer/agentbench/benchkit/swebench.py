# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare SWE-bench Verified metadata in the shared local dataset format."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from .dataset import PreparedDataset, _as_tuple


def prepare_swebench(output_dir: Path) -> PreparedDataset:
    """Download SWE-bench Verified metadata and write benchkit local inputs.

    The command writes `instances.jsonl`, `task-lists/default.txt`, and
    `manifest.json`. `default.txt` includes every prepared task ID in upstream
    dataset iteration order; `experiment.task_num` can later choose the first N
    tasks from that generated list.
    """

    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path already exists and is not an empty directory: {output_dir}")

    dataset = _load_swebench_verified()
    rows = [_normalize_swebench_row(row) for row in dataset]

    task_list_dir = output_dir / "task-lists"
    task_list_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "instances.jsonl"
    selection_path = task_list_dir / "default.txt"
    manifest_path = output_dir / "manifest.json"

    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selection_path.write_text("\n".join(cast(str, row["instance_id"]) for row in rows) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "swebench",
                "variant": "verified",
                "source": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return PreparedDataset(
        index_path=index_path,
        selection_path=selection_path,
        manifest_path=manifest_path,
        rows=len(rows),
    )


def _load_swebench_verified() -> Iterable[Mapping[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return cast(Iterable[Mapping[str, object]], dataset)


def _normalize_swebench_row(row: Mapping[str, object]) -> dict[str, object]:
    fail_to_pass = list(_as_tuple(row.get("fail_to_pass", row.get("FAIL_TO_PASS", []))))
    pass_to_pass = list(_as_tuple(row.get("pass_to_pass", row.get("PASS_TO_PASS", []))))
    return {
        "repo": row.get("repo", ""),
        "instance_id": row.get("instance_id", ""),
        "base_commit": row.get("base_commit", ""),
        "patch": row.get("patch", ""),
        "test_patch": row.get("test_patch", ""),
        "problem_statement": row.get("problem_statement", ""),
        "created_at": row.get("created_at"),
        "version": row.get("version"),
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "environment_setup_commit": row.get("environment_setup_commit"),
        "difficulty": row.get("difficulty"),
    }
