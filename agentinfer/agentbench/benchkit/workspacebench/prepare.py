# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Materialize two independent Full workspaces with the pinned official scripts."""

import json
import os
import runpy
import subprocess
import tempfile
from pathlib import Path

from ..common import atomic_write_json
from ..dataset import PreparedDataset
from . import FILES_REVISION, LANGUAGES, REVISION, TASK_REVISION, UPSTREAM


def prepare(output: Path, checkout: Path | None = None) -> PreparedDataset:
    from filelock import FileLock

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".prepare.lock"), timeout=0):
        return _prepare(output, checkout)


def _prepare(output: Path, checkout: Path | None) -> PreparedDataset:
    manifest = output / "manifest.json"
    marker = output / ".workspacebench-preparing"
    if manifest.exists() or (not marker.exists() and any(p.name != ".prepare.lock" for p in output.iterdir())):
        raise FileExistsError(f"Choose an empty Workspace-Bench destination: {output}")
    marker.touch()
    if checkout is None:
        checkout = output / "upstream"
        if not (checkout / ".git").exists():
            subprocess.run(["git", "init", str(checkout)], check=True)
        if subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", REVISION], capture_output=True).returncode:
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", UPSTREAM, REVISION], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", REVISION], check=True)
    evaluation = evaluation_path(checkout)
    roles = runpy.run_path(str(evaluation / "scripts/build_run_config.py"))["ROLE_DIRS"]
    make_filesys = runpy.run_path(str(evaluation / "src/filesys_utils.py"))["make_filesys"]
    rows = []
    for language in LANGUAGES:
        root = output / language
        root.mkdir(exist_ok=True)
        _assets(evaluation, root, language)
        language_rows = _index(root, language, roles)
        for role, prefix in roles.items():
            standard = root / "filesys" / f"{prefix}_standard"
            if not standard.exists():
                staging = standard.with_suffix(".preparing")
                make_filesys(str(root / "filesys" / f"{prefix}_raw"), str(staging), role, str(root / "tasks"))
                staging.rename(standard)
        rows.extend(language_rows)
    rows.sort(key=lambda row: (row["absolute_id"], LANGUAGES.index(row["language"])))
    index = output / "instances.jsonl"
    temp = index.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, index)
    lists = output / "task-lists"
    lists.mkdir(exist_ok=True)
    all_ids = [row["instance_id"] for row in rows]
    (lists / "all.txt").write_text("\n".join(all_ids) + "\n")
    selected = lists / "selected.txt"
    selected.write_text("\n".join(f"wsb-{lang}-{n:04d}" for n in (2, 1) for lang in LANGUAGES) + "\n")
    atomic_write_json(
        manifest,
        {
            "dataset": "workspacebench",
            "split": "full",
            "rows": len(rows),
            "languages": LANGUAGES,
            "evaluation_revision": REVISION,
            "task_revision": TASK_REVISION,
            "workspace_revision": FILES_REVISION,
        },
    )
    marker.unlink()
    return PreparedDataset(index, selected, manifest, len(rows))


def evaluation_path(checkout: Path) -> Path:
    """Resolve the pinned, unmodified upstream evaluation code."""
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if (
        revision != REVISION
        or subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--", "evaluation"]).returncode
    ):
        raise ValueError(f"Workspace-Bench requires the unmodified evaluation sources at {REVISION}")
    return checkout.resolve() / "evaluation"


def _assets(evaluation: Path, root: Path, language: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    official = runpy.run_path(str(evaluation / "scripts/download_hf_assets.py"))
    if not (root / "tasks").is_dir():
        directory = f"task_clean_{language}"
        staging = root / ".download"
        snapshot_download(
            "Workspace-Bench/Workspace-Bench",
            repo_type="dataset",
            revision=TASK_REVISION,
            allow_patterns=[f"{directory}/**"],
            local_dir=staging,
        )
        official["_normalize_task_metadata_files"](staging / directory, expected_language=language)
        (staging / directory).rename(root / "tasks")
    if not (root / "filesys").is_dir():
        archive = hf_hub_download(
            "Workspace-Bench/Workspace-Bench-Workspaces",
            f"filesys_{language}.zip",
            repo_type="dataset",
            revision=FILES_REVISION,
        )
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staging = Path(tmp) / "filesys"
            official["_extract_workspace_archive"](Path(archive), staging, False, language)
            staging.rename(root / "filesys")


def _index(root: Path, language: str, roles: dict[str, str]) -> list[dict]:
    rows = {}
    for path in sorted((root / "tasks").glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        number = int(meta["absolute_id"])
        if number in rows or meta.get("language") != language:
            raise ValueError(f"Duplicate task or wrong language: {path}")
        for item in meta.get("data_manifest", []):
            source = (path.parent / item["stored_relpath"]).resolve()
            if not source.is_relative_to(path.parent.resolve()) or not source.is_file():
                raise ValueError(f"Missing task input at {source}")
        prefix = roles[meta["file_system"]]
        rows[number] = {
            "instance_id": f"wsb-{language}-{number:04d}",
            "absolute_id": number,
            "language": language,
            "problem_statement": meta["task"],
            "difficulty": meta.get("task_diff"),
            "metadata_path": str(path.relative_to(root.parent)),
            "standard_path": f"{language}/filesys/{prefix}_standard",
        }
    if set(rows) != set(range(1, 389)):
        raise ValueError(f"Incomplete {language} Full dataset ({len(rows)}/388)")
    return list(rows.values())
