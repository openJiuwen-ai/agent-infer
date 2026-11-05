# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Install observation only, then execute the real Replay CLI in a fresh process."""

import argparse
import functools
import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from agentinfer.agentbench.benchkit.cli import main
from agentinfer.agentbench.replay import runner
from tests.agentbench.helpers.mock_server.recorder import Recorder


class Observer:
    def __init__(self, root):
        self.nonce = uuid.uuid4().hex
        self.sequence = 0
        self.recorder = Recorder(root, self.nonce)
        self.root = root
        (root / "replay-process.json").write_text(
            json.dumps({"pid": os.getpid(), "nonce": self.nonce}), encoding="utf-8"
        )

    def emit(self, event):
        self.sequence += 1
        self.recorder.submit("client-events.jsonl", {**event, "sequence": self.sequence, "process_nonce": self.nonce})

    def close(self):
        self.emit({"event": "observer_closed", "t_monotonic": time.monotonic()})
        self.recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    observer = Observer(args.artifact_dir)
    try:
        with (
            patch.object(runner, "ReplayExecutor", functools.partial(runner.ReplayExecutor, observer=observer)),
            patch.object(runner, "ReplayTransport", functools.partial(runner.ReplayTransport, observer=observer)),
        ):
            exit_code = main(["replay", "--config", args.config])
    finally:
        observer.close()
    raise SystemExit(exit_code)
