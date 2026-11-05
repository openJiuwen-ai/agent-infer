# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Nonblocking event submission with explicit write/close failure reporting."""

import json
import queue
import threading
import time
from pathlib import Path


class Recorder:
    def __init__(self, root: Path, nonce: str):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.nonce = root, nonce
        self.error: str | None = None
        self.events: list[dict] = []
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._write, name="mock-recorder", daemon=True)
        self._thread.start()

    def submit(self, file: str, row: dict) -> None:
        self._queue.put((file, row))

    def event(self, event: str, **fields) -> dict:
        row = {
            "sequence": len(self.events) + 1,
            "process_nonce": self.nonce,
            "t_monotonic": time.monotonic(),
            "event": event,
            **fields,
        }
        self.events.append(row)
        self.submit("server-events.jsonl", row)
        return row

    def _write(self) -> None:
        handles = {}
        try:
            while (item := self._queue.get()) is not None:
                file, row = item
                if file not in handles:
                    handles[file] = (self.root / file).open("a", encoding="utf-8")
                handles[file].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[file].flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            for handle in handles.values():
                handle.close()

    def close(self) -> None:
        self.event("recorder_closed")
        self._queue.put(None)
        self._thread.join(5)
        if self._thread.is_alive() or self.error:
            raise RuntimeError(f"recorder failed: {self.error or 'drain timeout'}")
