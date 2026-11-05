# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Independent loopback service; bind once and announce readiness after serving."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import uuid
from pathlib import Path

import uvicorn

from .app import ModelService
from .protocol import Scenario


async def serve(root: Path):
    raw = (root / "scenario.json").read_bytes()
    scenario = Scenario.model_validate_json(raw)
    nonce = uuid.uuid4().hex
    service = ModelService(scenario, root, nonce, lambda: setattr(server, "should_exit", True))
    server = uvicorn.Server(
        uvicorn.Config(service.app, log_level="error", access_log=False, timeout_graceful_shutdown=2)
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("server exited before ready")
                await asyncio.sleep(0.01)
            ready = {
                "base_url": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "pid": os.getpid(),
                "nonce": nonce,
                "profile": scenario.profile,
                "scenario_sha256": hashlib.sha256(raw).hexdigest(),
            }
            temporary = root / "ready.tmp"
            temporary.write_text(json.dumps(ready), encoding="utf-8")
            temporary.replace(root / "ready.json")
            await task
        finally:
            server.should_exit = True
            if not task.done():
                await task
            service.recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    asyncio.run(serve(parser.parse_args().artifact_dir))
