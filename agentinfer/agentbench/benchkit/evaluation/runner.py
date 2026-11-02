# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run the pinned official harness and finalize an independent evaluation."""

import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from ..artifacts.correctness import EvaluationSummary
from ..collectors.correctness import collect_swebench_reports
from ..common import atomic_write_json, utc_now
from ..metrics.correctness import correctness_metrics
from .config import EvaluationConfig
from .inputs import digest, freeze_inputs, json_digest, prepare_inputs

_PROBE = """
import hashlib, importlib.metadata as m, json, pathlib, platform, sys
# Check version before importing any version-specific harness module.
version = m.version('swebench')
if version != sys.argv[2]:
    sys.exit(f'expected SWE-bench {sys.argv[2]}, got {version}')
import swebench
from swebench.harness.test_spec.test_spec import make_test_spec
root = pathlib.Path(swebench.__file__).parent
h = hashlib.sha256()
for p in sorted(root.rglob('*.py')):
    h.update(str(p.relative_to(root)).encode())
    h.update(p.read_bytes())
rows = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
print(json.dumps({
    'version': version, 'source_sha256': h.hexdigest(),
    'python': platform.python_version(),
    'packages': {d.metadata['Name']: d.version for d in m.distributions()},
    'images': {r['instance_id']: make_test_spec(r, namespace='swebench').instance_image_key for r in rows}
}))
"""


def harness_identity(config: EvaluationConfig, dataset: Path, timeout: float) -> dict:
    result = subprocess.run(
        [str(config.harness_python), "-c", _PROBE, str(dataset), config.expected_harness_version],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise ValueError(f"harness preflight failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def image_identities(images: dict[str, str], timeout: float) -> dict:
    """Inspect unique image names in one bounded call and map them to task IDs.

    Docker may return JSON for available images with a nonzero exit code when
    other images are missing. Preserve that evidence and mark only missing names
    unavailable; never assign results by position after a partial response.
    """
    if not images:
        return {}
    probe = subprocess.run(
        ["docker", "image", "inspect", *dict.fromkeys(images.values())],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    items = json.loads(probe.stdout) if probe.stdout.strip() else []
    if not isinstance(items, list) or (not items and probe.returncode == 0):
        raise ValueError("docker image inspect returned no image identity list")
    by_name = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("Id"), str):
            raise ValueError("docker image inspect returned an invalid image identity")
        for name in [item["Id"], *(item.get("RepoTags") or []), *(item.get("RepoDigests") or [])]:
            by_name[name] = item
    result = {}
    for iid, name in images.items():
        item = by_name.get(name)
        result[iid] = (
            {"name": name, "id": None}
            if item is None
            else {
                "name": name,
                "id": item["Id"],
                "digests": item.get("RepoDigests") or [],
            }
        )
    return result


def run_harness(command: list[str], output: Path, timeout: float, cleanup_timeout: int) -> int:
    """Bound the process tree, retaining logs even if the parent is interrupted."""
    with (output / "harness.stdout.log").open("wb") as stdout, (output / "harness.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=output / "raw", stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=cleanup_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def cleanup_containers(run_id: str, timeout: int) -> list[str]:
    """Only remove containers ending in this unique harness run ID."""
    failures = []
    deadline = time.monotonic() + timeout
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=timeout, check=True
    )
    for name in result.stdout.splitlines():
        if name.startswith("sweb.eval.") and name.endswith("." + run_id):
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    check=True,
                    timeout=max(0.1, deadline - time.monotonic()),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{name}: {exc}")
    return failures


def evaluate(run: Path, config: EvaluationConfig, output_dir: Path | None = None) -> tuple[Path, int]:
    prepared = prepare_inputs(run.resolve())
    evaluation_id = "eval-" + uuid.uuid4().hex
    output = (output_dir or run / "evaluations" / evaluation_id).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    hashes = freeze_inputs(output, prepared)
    manifest = {
        "schema_version": "1",
        "adapter_version": "1",
        "evaluation_id": evaluation_id,
        "source_run_id": prepared["source_run_id"],
        "status": "preparing",
        "created_at": utc_now().isoformat(),
        "finished_at": None,
        "source_hashes": prepared["source_hashes"],
        "input_hashes": hashes,
        "instance_ids": prepared["instance_ids"],
        "dataset_source": prepared["dataset_source"],
        "dataset_sha256": json_digest(sorted(prepared["dataset"], key=lambda row: row["instance_id"])),
        "generation": prepared["generation"],
        "config": config.model_dump(mode="json"),
        "harness": None,
        "images": {},
        "command": [],
        "returncode": None,
        "errors": [],
    }
    manifest_path = output / "evaluation_manifest.json"
    atomic_write_json(manifest_path, manifest)
    started = time.monotonic()
    launched = False
    interrupted = False
    active = [p["instance_id"] for p in prepared["predictions"] if p["model_patch"]]
    try:
        manifest["harness"] = harness_identity(config, output / "inputs/dataset.json", config.overall_timeout_seconds)
        if active:
            command = [
                str(config.harness_python),
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(output / "inputs/dataset.json"),
                "--split",
                "test",
                "--predictions_path",
                str(output / "inputs/predictions.jsonl"),
                "--instance_ids",
                *[p["instance_id"] for p in prepared["predictions"]],
                "--max_workers",
                str(config.max_workers),
                "--cache_level",
                "instance",
                "--timeout",
                str(config.timeout_seconds),
                "--run_id",
                evaluation_id,
            ]
            manifest.update(command=command, status="running")
            atomic_write_json(manifest_path, manifest)
            launched = True
            remaining = max(0.1, config.overall_timeout_seconds - (time.monotonic() - started))
            manifest["returncode"] = run_harness(command, output, remaining, config.cleanup_timeout_seconds)
            if manifest["returncode"] != 0:
                manifest["errors"].append(f"harness exited with {manifest['returncode']}")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        interrupted = True
        manifest["errors"].append("evaluation interrupted")
    finally:
        if launched:
            try:
                manifest["errors"].extend(cleanup_containers(evaluation_id, config.cleanup_timeout_seconds))
            except (OSError, subprocess.SubprocessError) as exc:
                manifest["errors"].append(f"container cleanup failed: {exc}")
        if manifest["harness"] and active:
            try:
                manifest["images"] = image_identities(manifest["harness"]["images"], config.cleanup_timeout_seconds)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                manifest["errors"].append(f"image capture failed: {exc}")

    model = prepared["generation"]["backend"]["model"]
    results = collect_swebench_reports(output, evaluation_id, model, prepared["tasks"])
    summary = EvaluationSummary(
        source_run_id=prepared["source_run_id"],
        evaluation_id=evaluation_id,
        **correctness_metrics(results, prepared["instance_ids"]),
    )
    # Raw evidence and normalized results are bound by the final manifest.
    temporary = output / "task_results.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    temporary.replace(output / "task_results.jsonl")
    atomic_write_json(output / "correctness.json", summary.model_dump())
    manifest["output_hashes"] = {
        str(path.relative_to(output)): digest(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and (
            path.is_relative_to(output / "raw")
            or path.name in {"correctness.json", "task_results.jsonl", "harness.stdout.log", "harness.stderr.log"}
        )
    }
    manifest["finished_at"] = utc_now().isoformat()
    manifest["status"] = "failed" if summary.counts.error or manifest["errors"] else "completed"
    atomic_write_json(manifest_path, manifest)
    return output, 130 if interrupted else int(manifest["status"] != "completed")
