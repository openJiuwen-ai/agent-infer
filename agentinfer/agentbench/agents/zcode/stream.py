# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Keep the CLI HTTP connection alive without producing model events."""

import asyncio
import re

SSE_HEARTBEAT_SECONDS = 15.0
_FRAME_END = re.compile(rb"\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r")


async def with_heartbeats(chunks):
    iterator = aiter(chunks)
    pending = asyncio.create_task(anext(iterator))
    buffer = b""
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=SSE_HEARTBEAT_SECONDS)
            if not done:
                yield b": agentbench keep-alive\n\n"
                continue
            try:
                buffer += pending.result()
            except StopAsyncIteration:
                if buffer:
                    yield buffer
                break
            # Hold incomplete frames so a heartbeat cannot split tool arguments or UTF-8.
            while match := _FRAME_END.search(buffer):
                yield buffer[: match.end()]
                buffer = buffer[match.end() :]
            pending = asyncio.create_task(anext(iterator))
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await iterator.aclose()
