# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Own exactly one model child, preserving diagnostics on every failure path."""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .protocol import Scenario

REPO = Path(__file__).resolve().parents[4]


def read_rows(path: Path) -> list[dict]:
    """Read complete JSONL evidence; a not-yet-created file is empty while polling."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class MockServerProcess:
    """Start and reap one fresh loopback server, retaining all artifacts in root."""

    def __init__(self, root: Path, scenario: dict):
        self.root = root
        self.scenario = Scenario.model_validate(scenario)
        self.process = None
        self.base_url = ""
        self.ready = {}
        self._handles = []
        self.client = httpx.Client(timeout=2, trust_env=False)

    def start(self):
        """Wait for atomic readiness and verify this child's initial profile/counters."""
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "scenario.json").write_text(self.scenario.model_dump_json(), encoding="utf-8")
        self._handles = [(self.root / name).open("w") for name in ("stdout.log", "stderr.log")]
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "tests.agentbench.helpers.mock_server", "--artifact-dir", str(self.root)],
                cwd=REPO,
                stdout=self._handles[0],
                stderr=self._handles[1],
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"mock startup failed: {(self.root / 'stderr.log').read_text()}")
                if (self.root / "ready.json").exists():
                    self.ready = json.loads((self.root / "ready.json").read_text())
                    self.base_url = self.ready["base_url"]
                    health = self.get("/__mock__/health")
                    assert health["nonce"] == self.ready["nonce"]
                    assert health["profile"] == self.scenario.profile and health["successes"] == 0
                    return self
                time.sleep(0.01)
            raise TimeoutError(f"mock startup timeout: {self.root}")
        except BaseException:
            self.close(check_complete=False)
            raise

    def get(self, path: str):
        response = self.client.get(self.base_url + path)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, body=None):
        response = self.client.post(self.base_url + path, json=body or {})
        response.raise_for_status()
        return response.json()

    def wait_event(self, name: str, *, timeout=5, **fields):
        deadline = time.monotonic() + timeout
        cursor = 0
        while time.monotonic() < deadline:
            page = self.get(f"/__mock__/events?after_sequence={cursor}")
            for event in page["events"]:
                if event["event"] == name and all(event.get(key) == value for key, value in fields.items()):
                    return event
            cursor = page["cursor"]
            time.sleep(0.01)
        raise AssertionError(f"missing event {name} {fields}: {self.get('/__mock__/events')}")

    @property
    def requests(self):
        return [
            row["body"]
            for row in read_rows(self.root / "backend-requests.jsonl")
            if row["kind"] in {"generation", "unidentified_generation"}
        ]

    def close(self, *, check_complete=True):
        """Reap the owned child; require drained evidence except after startup failure."""
        forced = False
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.post("/__mock__/shutdown")
                except httpx.HTTPError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    forced = True
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process is not None:
                rows = read_rows(self.root / "server-events.jsonl")
                complete = (
                    not forced
                    and self.process.returncode == 0
                    and bool(rows)
                    and rows[-1]["event"] == "recorder_closed"
                )
                (self.root / "exit.json").write_text(
                    json.dumps({"returncode": self.process.returncode, "forced": forced, "complete": complete}),
                    encoding="utf-8",
                )
                if check_complete and not complete:
                    raise RuntimeError(f"incomplete mock evidence: {self.root}")
        finally:
            self.client.close()
            for handle in self._handles:
                handle.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()
