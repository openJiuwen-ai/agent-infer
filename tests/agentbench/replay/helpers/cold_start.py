# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Two independent children per round, with bounded control and unconditional cleanup."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tests.agentbench.helpers.mock_server import MockServerProcess
from tests.agentbench.helpers.mock_server.process import REPO

from .fixtures import config
from .validation import validate_plan


def run_round(
    root: Path,
    source_path: Path,
    expected: dict,
    concurrency: int,
    round_index: int,
    *,
    control=None,
    expect_failure=False,
    replay_options=None,
    experiment_options=None,
):
    """Run the real CLI once, optionally arm validated node IDs, and reap both children."""
    root.mkdir(parents=True)
    (root / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (root / "input-manifest.json").write_text(
        json.dumps(
            {"source_path": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )
    scenario = {"profile": "replay", "round_index": round_index, "wait_for_arm": control is not None}
    with MockServerProcess(root / "server", scenario) as server:
        payload = config(source_path, expected, concurrency, server.base_url, root / "result")
        payload["replay"].update(replay_options or {})
        payload["experiment"].update(experiment_options or {})
        (root / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
        with (root / "stdout.log").open("w") as stdout, (root / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.agentbench.replay.helpers.entrypoint",
                    "--config",
                    str(root / "config.yaml"),
                    "--artifact-dir",
                    str(root),
                ],
                cwd=REPO,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                if control is not None:
                    deadline = time.monotonic() + 10
                    plan_path = root / "result" / "replay-plan.json"
                    while not plan_path.exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise AssertionError(f"plan unavailable: {(root / 'stderr.log').read_text()}")
                        time.sleep(0.01)
                    plan = json.loads(plan_path.read_text())
                    validate_plan(expected, plan)
                    control(server, plan, root)
                process.wait(timeout=40)
                assert (process.returncode != 0) == expect_failure, (
                    process.returncode,
                    (root / "stderr.log").read_text(),
                )
            finally:
                forced = process.poll() is None
                if forced:
                    process.kill()
                    process.wait(timeout=5)
                (root / "replay-exit.json").write_text(
                    json.dumps({"returncode": process.returncode, "forced": forced}), encoding="utf-8"
                )
    return root
