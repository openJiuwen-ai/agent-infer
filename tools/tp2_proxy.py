# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Small sticky proxy for a two-instance TP2 benchmark."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response

BACKENDS = ("http://127.0.0.1:8001", "http://127.0.0.1:8002")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=600, trust_env=False)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def _backend(path: str, body: bytes) -> str:
    if path != "/v1/chat/completions":
        return BACKENDS[0]
    try:
        payload = json.loads(body)
        messages = payload.get("messages", [])
        sticky_key = messages[0] if messages else payload
        canonical = json.dumps(sticky_key, ensure_ascii=False, sort_keys=True).encode()
    except (ValueError, TypeError):
        canonical = body
    index = hashlib.sha256(canonical).digest()[0] % len(BACKENDS)
    return BACKENDS[index]


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request) -> Response:
    body = await request.body()
    target = f"{_backend('/' + path, body)}/{path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length"}}
    upstream = await request.app.state.client.request(
        request.method,
        target,
        content=body,
        headers=headers,
    )
    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
