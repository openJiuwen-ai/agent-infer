# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Request Proxy process ownership and termination."""

import asyncio
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

from .server import RequestProxyLaunchConfig, run_request_proxy


@dataclass(frozen=True)
class ProcessExit:
    """Record a completed proxy process and its diagnostic output paths."""

    return_code: int
    stdout_path: Path
    stderr_path: Path


class RequestProxyProcess:
    """Own the spawned proxy process and bounded abnormal-termination fallback."""

    def __init__(self, stdout_path: Path, stderr_path: Path):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._process = None
        self._exit: ProcessExit | None = None

    async def start(self, config: RequestProxyLaunchConfig) -> None:
        if self._process is not None:
            raise RuntimeError("process has already started")
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(target=run_request_proxy, args=(config,))
        self._process.start()

    async def poll(self) -> int | None:
        if self._process is None:
            raise RuntimeError("process has not started")
        return self._process.exitcode

    async def wait(self, timeout_seconds: float) -> ProcessExit:
        if self._exit is not None:
            return self._exit
        if self._process is None:
            raise RuntimeError("process has not started")
        await asyncio.to_thread(self._process.join, timeout_seconds)
        if self._process.is_alive():
            raise TimeoutError(f"request proxy did not exit within {timeout_seconds}s")
        return self._record_exit()

    async def terminate(self, timeout_seconds: float) -> ProcessExit:
        if self._exit is not None:
            return self._exit
        if self._process is None:
            raise RuntimeError("process has not started")
        if self._process.is_alive():
            self._process.terminate()
            await asyncio.to_thread(self._process.join, timeout_seconds)
        if self._process.is_alive():
            self._process.kill()
            await asyncio.to_thread(self._process.join, timeout_seconds)
        if self._process.is_alive():
            raise TimeoutError(f"request proxy did not stop within {timeout_seconds}s")
        return self._record_exit()

    def _record_exit(self) -> ProcessExit:
        assert self._process is not None
        return_code = self._process.exitcode
        if return_code is None:
            raise RuntimeError("request proxy is still running")
        self._exit = ProcessExit(return_code, self.stdout_path, self.stderr_path)
        self._process.close()
        return self._exit
