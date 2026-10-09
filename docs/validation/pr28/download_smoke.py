# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Measure real Inferact prefix downloads without populating the user's cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from agentinfer.agentbench.replay import inferact_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-num", type=int, default=1)
    args = parser.parse_args()
    transfers = []

    class MeasuredClient(httpx.Client):
        @contextmanager
        def stream(self, *args, **kwargs):
            # This wraps the real network response; no response data is mocked.
            with super().stream(*args, **kwargs) as response:
                try:
                    yield response
                finally:
                    transfers.append(
                        {
                            "status_code": response.status_code,
                            "content_length": response.headers.get("content-length"),
                            "received_body_bytes": response.num_bytes_downloaded,
                        }
                    )

    started = time.monotonic()
    with TemporaryDirectory(prefix="inferact-download-smoke-") as temporary:
        # Bypass any pre-existing Hub blob to exercise the cold network path.
        with (
            patch.object(inferact_source, "try_to_load_from_cache", return_value=None),
            patch.object(inferact_source.httpx, "Client", MeasuredClient),
        ):
            output = inferact_source.materialize_inferact_source(args.task_num, cache_dir=Path(temporary))
            data = output.read_bytes()
            rows = json.loads(data)
            assert len(rows) == args.task_num
            assert len(transfers) == 1
            assert inferact_source.materialize_inferact_source(args.task_num, cache_dir=Path(temporary)) == output
            assert len(transfers) == 1, "subset cache hit must not access the network"
            assert transfers[0]["received_body_bytes"] < 218648292, "must not consume the complete pinned blob"
    print(
        json.dumps(
            {
                "status": "passed",
                "revision": inferact_source.INFERACT_REVISION,
                "task_num": args.task_num,
                "records": len(rows),
                "subset_bytes": len(data),
                "subset_sha256": hashlib.sha256(data).hexdigest(),
                "transfers": transfers,
                "cache_hit_network_requests": 0,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
