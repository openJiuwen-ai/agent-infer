# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Grade frozen task outputs in separate official Claude judge containers."""

import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

import yaml
from pydantic import ConfigDict, Field, SecretStr

from ..common import atomic_write_json
from ..config import StrictModel
from . import LANGUAGES
from .runner import WORKER, claude_model_environment, container_args, image_id, writable_directory


class JudgeConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    baseUrl: str = Field(min_length=1)
    model: str = Field(min_length=1)
    apiKey: SecretStr
    model_name: str = "official"
    timeout_seconds: int = Field(3600, ge=1)


def load_judge_config(path: Path | None) -> JudgeConfig:
    data = (
        yaml.safe_load(path.read_text())
        if path
        else {"baseUrl": "${JUDGE_BASE_URL}", "model": "${JUDGE_MODEL}", "apiKey": "${JUDGE_API_KEY}"}
    )
    if not isinstance(data, dict):
        raise ValueError("Expected Workspace-Bench judge YAML")
    for key, value in data.items():
        if isinstance(value, str) and (variable := re.fullmatch(r"\$\{(\w+)\}", value)):
            data[key] = os.environ.get(variable[1])
            if not data[key]:
                raise ValueError(f"Missing judge environment variable {variable[1]}")
    if "model_name" not in data and isinstance(data.get("model"), str):
        data["model_name"] = re.sub(r"[^\w.-]+", "_", data["model"]).strip("._") or "judge"
    config = JudgeConfig.model_validate(data)
    if not config.apiKey.get_secret_value() or not re.fullmatch(r"[\w.-]+", config.model_name):
        raise ValueError("A judge API key and a filename-safe model_name are required")
    return config


def score(raw: dict, rubrics: list[str]) -> int:
    judge = raw.get("judge") or {}
    if not rubrics or not judge or judge.get("error"):
        raise ValueError("Official judge failed or returned no judge evidence")
    items = raw["rubrics"]
    by_index = {row["index"]: row for row in items if type(row.get("index")) is int}
    if len(items) != len(rubrics) or set(by_index) != set(range(len(rubrics))):
        raise ValueError("Missing, duplicate or extra rubric indices")
    for index, text in enumerate(rubrics):
        row = by_index[index]
        if row.get("rubric") != text or type(row.get("passed")) is not bool:
            raise ValueError(f"Invalid official evidence for rubric {index}")
    return sum(row["passed"] for row in items)


def _judge_one(run: Path, row: dict, directory: Path, config: JudgeConfig, image: str) -> tuple[int, int]:
    root = run / "tasks" / row["instance_id"]
    case = root / "official" / str(row["absolute_id"])
    metadata = json.loads((root / "metadata.json").read_text())
    source = Path(row["metadata_path"]).parent / "data"
    if not (case / "agent.json").is_file() or not (case / "output").is_dir():
        raise ValueError("No official execution evidence")
    if not source.is_dir() and metadata.get("data_manifest"):
        raise FileNotFoundError(f"Original task inputs unavailable: {source}")
    writable_directory(directory)
    name = f"agentbench-wsb-judge-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory() as tmp:
        # Only the judge receives a key. Its temporary raw SDK configs disappear with the container.
        settings = Path(tmp) / "evaluation.yaml"
        settings.write_text(
            yaml.safe_dump(
                {
                    "model": config.model,
                    "model_name": config.model_name,
                    "baseUrl": config.baseUrl,
                    "apiKey": config.apiKey.get_secret_value(),
                }
            )
        )
        settings.chmod(0o400)
        if os.getuid() == 0:
            os.chown(settings, 65534, 65534)
        command = [
            *container_args(name),
            *claude_model_environment(config.model),
            "-v",
            f"{settings}:/evaluation.yaml:ro",
            "-v",
            f"{root / 'metadata.json'}:/evidence/metadata.json:ro",
            "-v",
            f"{case / 'agent.json'}:/evidence/agent.json:ro",
            "-v",
            f"{case / 'output'}:/evidence/output:ro",
            "-v",
            f"{directory}:/scores:rw",
        ]
        if source.exists():
            command += ["-v", f"{source}:/inputs/data:ro"]
        with (directory / "judge.log").open("wb") as log:
            try:
                subprocess.run(
                    [*command, image, "python", "-m", WORKER, "judge"],
                    check=True,
                    timeout=config.timeout_seconds,
                    stdout=log,
                    stderr=log,
                )
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False, timeout=30)
    raw = json.loads((directory / f"rubrics_judge--{config.model_name}.json").read_text())
    rubrics = metadata["rubrics"]
    return score(raw, rubrics), len(rubrics)


def evaluate(run: Path, config: JudgeConfig, output_dir: Path | None = None) -> tuple[Path, int]:
    run = run.resolve()
    manifest = json.loads((run / "workspacebench.json").read_text())
    judge_image = manifest.get("image_id") or image_id()
    directory = (output_dir or run / "evaluations" / f"eval-{uuid.uuid4().hex}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tasks = []
    summary = {lang: {"planned": 0, "scored": 0, "passed_rubrics": 0, "evaluated_rubrics": 0} for lang in LANGUAGES}
    for row in manifest["tasks"]:
        language = row["language"]
        counts = summary[language]
        counts["planned"] += 1
        record = {"instance_id": row["instance_id"], "language": language}
        try:
            passed, total = _judge_one(run, row, directory / row["instance_id"], config, judge_image)
            record.update(status="scored", passed=passed, total=total)
            counts["passed_rubrics"] += passed
            counts["evaluated_rubrics"] += total
            counts["scored"] += 1
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            record.update(status="error", error=str(exc))
        tasks.append(record)
    languages = {
        lang: {
            **counts,
            "rubric_pass_rate": counts["passed_rubrics"] / counts["evaluated_rubrics"]
            if counts["evaluated_rubrics"]
            else None,
            "coverage": counts["scored"] / counts["planned"] if counts["planned"] else None,
        }
        for lang, counts in summary.items()
    }
    complete = bool(tasks) and all(task["status"] == "scored" for task in tasks)
    atomic_write_json(
        directory / "correctness.json",
        {
            "dataset": "workspacebench",
            "source_run_id": run.name,
            "source": manifest["source"],
            "image_id": judge_image,
            "judge_model": config.model,
            "complete": complete,
            "languages": languages,
            "tasks": tasks,
        },
    )
    return directory, 0 if complete else 1
