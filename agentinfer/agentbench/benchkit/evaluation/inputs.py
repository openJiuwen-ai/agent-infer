# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Validate completed tasks and freeze the exact inputs supplied to the harness."""

import hashlib
import json
import re
from pathlib import Path

from ..artifacts.task_index import load_task_results
from ..common import atomic_write_json
from ..compare import load_summary
from ..dataset import Task


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(value: object) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_object(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def prepare_inputs(run: Path) -> dict:
    """Read without changing the original run or substituting reference patches."""
    summary = load_summary(run)
    manifest = read_object(run / "manifest.json")
    config = manifest["config"]
    if config["dataset"]["name"] != "swebench_verified":
        raise ValueError("evaluate supports swebench_verified only")
    index = read_object(run / "task_index.json")
    if index.get("run_id") != manifest["run_id"] or index.get("schema_version") != "1":
        raise ValueError("task_index identifies a different run or unsupported schema")
    rows = index["tasks"]
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ID in task_index")
    loaded_results = load_task_results(run)
    results = {result.instance_id: result for result in loaded_results}
    if len(results) != len(loaded_results):
        raise ValueError("duplicate task ID in result files")
    if set(results) != set(ids):
        raise ValueError("task_index and task results differ")
    for row in rows:
        result = results[row["instance_id"]]
        if (row["outcome"], row["session_id"], row["task_position"]) != (
            result.outcome,
            result.session_id,
            result.task_position,
        ):
            raise ValueError("task_index and task result identity/status differ")
    selected = [row["instance_id"] for row in rows if row["outcome"] == "completed"]
    if len(selected) != summary["tasks"]["completed"]:
        raise ValueError("completed task count differs from summary")
    if not selected:
        raise ValueError("no_completed_tasks")
    dataset_path = Path(config["dataset"]["index_path"])
    if not dataset_path.is_absolute():
        raise ValueError("source dataset index_path must be absolute for reproducible evaluation")
    dataset_manifest = read_object(dataset_path.with_name("manifest.json"))
    if (dataset_manifest.get("variant"), dataset_manifest.get("split")) != ("verified", "test"):
        raise ValueError("dataset manifest must identify Verified/test")
    if dataset_manifest.get("source") not in {"princeton-nlp/SWE-bench_Verified", "SWE-bench/SWE-bench_Verified"}:
        raise ValueError("unsupported dataset source")
    dataset = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        iid = item["instance_id"]
        if iid in selected:
            if iid in dataset:
                raise ValueError("duplicate selected ID in dataset")
            # Reuse the existing task validation, including filesystem-safe IDs.
            Task(iid, item["repo"], item["base_commit"], item["problem_statement"])
            for key in ("version", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
                if key not in item:
                    raise ValueError(f"dataset missing {key}: {iid}")
            dataset[iid] = item
    if set(dataset) != set(selected):
        raise ValueError("selected tasks are absent from dataset")

    tasks, predictions, patches, prompts = [], [], {}, {}
    source_hashes = {
        name: digest((run / name).read_bytes()) for name in ("manifest.json", "summary.json", "task_index.json")
    }
    for iid in selected:
        result = results[iid]
        task_dir = run / "tasks" / iid
        if read_object(task_dir / "result.json").get("instance_id") != iid:
            raise ValueError("task directory and result ID differ")
        prompt = (task_dir / "prompt.txt").read_bytes()
        for name in ("result.json", "prompt.txt"):
            source_hashes[f"tasks/{iid}/{name}"] = digest((task_dir / name).read_bytes())
        text = prompt.decode("utf-8")
        for label, expected in (
            ("Instance", iid),
            ("Repository", dataset[iid]["repo"]),
            ("Base commit", dataset[iid]["base_commit"]),
        ):
            if not re.search(rf"^{label}: {re.escape(expected)}\s*$", text, flags=re.MULTILINE):
                raise ValueError(f"source prompt {label} differs from dataset: {iid}")
        prompts[iid] = digest(prompt)
        record = {"instance_id": iid, "status": "error", "patch_sha256": None, "reason": None}
        try:
            path = task_dir / "model.patch"
            if path.exists():
                patch = path.read_bytes()
                if len(patch) != result.patch_bytes or bool(patch) != result.has_patch:
                    raise ValueError("artifact_mismatch")
            else:
                status_path = task_dir / "dsh-run-status.json"
                if (
                    result.has_patch
                    or result.patch_bytes != 0
                    or not status_path.is_file()
                    or read_object(status_path).get("patch") != "no_changes"
                ):
                    raise ValueError("missing_artifact")
                patch = b""
            prediction = patch.decode("utf-8")
            patches[iid] = patch
            predictions.append(
                {"instance_id": iid, "model_name_or_path": config["backend"]["model"], "model_patch": prediction}
            )
            record.update(
                patch_sha256=digest(patch),
                status="empty_patch" if not patch else "error",
                reason=None if not patch else "not_evaluated",
            )
        except (OSError, UnicodeError, ValueError) as exc:
            record["reason"] = f"input_error: {exc}"
        tasks.append(record)
    generation = {
        "agent": config["agent"],
        "backend": {k: v for k, v in config["backend"].items() if k not in {"base_url", "metrics_url", "api_key_env"}},
        "experiment": {k: v for k, v in config["experiment"].items() if k != "result_dir"},
        "prompts": prompts,
    }
    return {
        "source_run_id": manifest["run_id"],
        "instance_ids": selected,
        "source_hashes": source_hashes,
        "dataset": [dataset[iid] for iid in selected],
        "dataset_source": dataset_manifest,
        "generation": generation,
        "tasks": tasks,
        "predictions": predictions,
        "patches": patches,
    }


def freeze_inputs(output: Path, prepared: dict) -> dict[str, str]:
    directory = output / "inputs"
    (directory / "patches").mkdir(parents=True)
    atomic_write_json(directory / "task_plan.json", prepared["instance_ids"])
    atomic_write_json(directory / "dataset.json", prepared["dataset"])
    predictions = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in prepared["predictions"])
    (directory / "predictions.jsonl").write_text(predictions, encoding="utf-8")
    for iid, patch in prepared["patches"].items():
        (directory / "patches" / f"{iid}.patch").write_bytes(patch)
    return {
        str(path.relative_to(output)): digest(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
