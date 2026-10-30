#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expand mirror_hardwares presets and upload the GPU pipeline to Buildkite.

Buildkite does not read hardware.yml. This script is the step that turns
`mirror_hardwares: l20_1` into agents, plugins, and env.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pyyaml"],
        check=True,
    )
    import yaml


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return data


def _apply_preset(step: dict, presets: dict) -> None:
    name = step.pop("mirror_hardwares")
    try:
        preset = copy.deepcopy(presets[name])
    except KeyError as exc:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"unknown mirror_hardwares {name!r}; known: {known}") from exc

    host_only = bool(step.pop("host_only", False))
    image_build = step.pop("docker_compose_build", None)

    env = dict(preset.get("env") or {})
    env.update(step.get("env") or {})
    if env:
        step["env"] = env
    elif "env" in step and not step["env"]:
        step.pop("env")

    if "agents" not in step:
        step["agents"] = preset["agents"]

    if host_only:
        return

    plugins = copy.deepcopy(preset.get("plugins") or [])
    if image_build:
        for plugin in plugins:
            for config in plugin.values():
                if isinstance(config, dict):
                    config.pop("run", None)
                    config["build"] = image_build
    if plugins:
        step["plugins"] = plugins


def expand(node, presets: dict):
    if isinstance(node, list):
        return [expand(item, presets) for item in node]
    if not isinstance(node, dict):
        return node
    if "mirror_hardwares" in node:
        _apply_preset(node, presets)
    if "steps" in node:
        node["steps"] = [expand(step, presets) for step in node["steps"]]
    return node


def render(pipeline_path: Path, hardware_path: Path) -> str:
    hardware = _load(hardware_path)
    presets = hardware.get("mirror_hardwares")
    if not isinstance(presets, dict) or not presets:
        raise SystemExit(f"{hardware_path} has no mirror_hardwares presets")
    pipeline = expand(_load(pipeline_path), presets)
    return yaml.safe_dump(pipeline, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipeline",
        nargs="?",
        default=str(ROOT / "test-ready.yml"),
    )
    parser.add_argument(
        "--hardware",
        default=str(ROOT / "hardware.yml"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the expanded pipeline instead of uploading",
    )
    args = parser.parse_args()

    rendered = render(Path(args.pipeline), Path(args.hardware))
    if args.dry_run or not os.environ.get("BUILDKITE_JOB_ID"):
        sys.stdout.write(rendered)
        return

    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=rendered,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
