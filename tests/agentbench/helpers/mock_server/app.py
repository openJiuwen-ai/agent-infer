# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Shared HTTP service. It never reads a Replay plan or enforces dependencies."""

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .protocol import Rule, Scenario, answer, decode, tokens
from .recorder import Recorder


class ModelService:
    def __init__(self, scenario: Scenario, root: Path, nonce: str, shutdown):
        self.scenario, self.root, self.nonce = scenario, root, nonce
        self.recorder = Recorder(root, nonce)
        self.shutdown = shutdown
        self.started_at = time.time()
        self.received = self.successes = self.input_tokens = self.active = 0
        self.stopping = False
        self.armed = asyncio.Event()
        if not scenario.wait_for_arm:
            self.armed.set()
        self.install_rules(scenario.rules)
        self.app = FastAPI()
        for path, handler, methods in [
            ("/v1/models", self.models, ["GET"]),
            ("/tokenizer_info", self.tokenizer_info, ["GET"]),
            ("/tokenize", self.tokenize, ["POST"]),
            ("/detokenize", self.detokenize, ["POST"]),
            ("/v1/completions", self.generate, ["POST"]),
            ("/v1/chat/completions", self.generate, ["POST"]),
            ("/metrics", self.metrics, ["GET"]),
            ("/__mock__/health", self.health, ["GET"]),
            ("/__mock__/events", self.events, ["GET"]),
            ("/__mock__/arm", self.arm, ["POST"]),
            ("/__mock__/gates/{gate_id}/release", self.release, ["POST"]),
            ("/__mock__/shutdown", self.stop, ["POST"]),
        ]:
            self.app.add_api_route(path, handler, methods=methods)

    def install_rules(self, rules: list[Rule]) -> None:
        if len({rule.request_id for rule in rules}) != len(rules):
            raise ValueError("duplicate request rule")
        self.rules = {rule.request_id: rule for rule in rules}
        self.gates = {rule.gate.id: asyncio.Event() for rule in rules if rule.gate}

    def capture(self, request: Request, body=None) -> dict:
        self.received += 1
        identity = {}
        if isinstance(body, dict):
            try:
                identity = json.loads(body.get("vllm_xargs", {}).get("agentic_context", "{}"))
                if not isinstance(identity, dict):
                    identity = {}
            except (ValueError, TypeError, AttributeError):
                identity = {}
        generation = request.url.path in {"/v1/completions", "/v1/chat/completions"}
        kind = "generation" if generation else "auxiliary"
        if generation and not identity.get("request_id"):
            kind = "unidentified_generation"
        row = {
            "capture_id": self.received,
            "method": request.method,
            "path": request.url.path,
            "body": body,
            "identity": identity,
            "kind": kind,
        }
        row.update(
            self.recorder.event("received", capture_id=self.received, request_id=identity.get("request_id"), kind=kind)
        )
        self.recorder.submit("backend-requests.jsonl", row)
        return row

    async def body(self, request: Request) -> tuple[dict, dict]:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError as exc:
            self.capture(request, {"invalid_json": raw.decode(errors="replace")})
            raise HTTPException(400, "invalid JSON") from exc
        row = self.capture(request, body)
        if not isinstance(body, dict):
            raise HTTPException(400, "expected object")
        return body, row

    async def models(self, request: Request):
        self.capture(request)
        return {"object": "list", "data": [{"id": self.scenario.model, "max_model_len": self.scenario.max_model_len}]}

    async def tokenizer_info(self, request: Request):
        self.capture(request)
        return {}

    async def tokenize(self, request: Request):
        body, _ = await self.body(request)
        try:
            values = tokens(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"tokens": values, "count": len(values)}

    async def detokenize(self, request: Request):
        body, _ = await self.body(request)
        try:
            return {"prompt": decode(body.get("tokens"))}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    async def health(self):
        return {
            "pid": os.getpid(),
            "nonce": self.nonce,
            "profile": self.scenario.profile,
            "armed": self.armed.is_set(),
            "active": self.active,
            "successes": self.successes,
            "recorder_error": self.recorder.error,
        }

    async def events(self, after_sequence: int = 0):
        return {"events": self.recorder.events[after_sequence:], "cursor": len(self.recorder.events)}

    async def arm(self, request: Request):
        if self.armed.is_set():
            raise HTTPException(409, "already armed")
        try:
            payload = await request.json()
            rules = [Rule.model_validate(rule) for rule in payload["rules"]]
            self.install_rules(rules)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        (self.root / "resolved-rules.json").write_text(json.dumps(payload), encoding="utf-8")
        self.recorder.event("armed")
        self.armed.set()
        return {"armed": True}

    async def release(self, gate_id: str):
        if gate_id not in self.gates:
            raise HTTPException(404, "unknown gate")
        self.gates[gate_id].set()
        self.recorder.event("gate_released", gate_id=gate_id)
        return {"released": gate_id}

    async def stop(self):
        self.stopping = True
        self.armed.set()
        for gate in self.gates.values():
            gate.set()
        self.shutdown()
        return {"stopping": True}

    async def gate(self, rule, phase, capture_id):
        if rule and rule.gate and rule.gate.phase == phase:
            self.recorder.event("gate_waiting", gate_id=rule.gate.id, capture_id=capture_id)
            await self.gates[rule.gate.id].wait()
        if self.stopping:
            raise asyncio.CancelledError

    def record_response(self, capture_id, payload):
        self.recorder.submit("backend-responses.jsonl", {"capture_id": capture_id, "payload": payload})
        self.recorder.event("chunk_sent", capture_id=capture_id)

    async def generate(self, request: Request):
        body, record = await self.body(request)
        capture_id = record["capture_id"]
        if self.stopping:
            raise HTTPException(503, "stopping")
        if body.get("model") != self.scenario.model:
            raise HTTPException(400, "wrong model")
        if self.scenario.profile == "run_legacy":
            self.input_tokens += 7
            payload = {
                "id": "fake-completion",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }
            self.record_response(capture_id, payload)
            self.successes += 1
            self.recorder.event("response_finished", capture_id=capture_id)
            return payload
        identity = record["identity"]
        request_id = identity.get("request_id")
        if body.get("stream") and (not isinstance(request_id, str) or not request_id or not identity.get("session_id")):
            raise HTTPException(400, "missing Replay identity")
        try:
            count = body.get("max_tokens")
            if type(count) is not int or count < 1 or body.get("min_tokens") != count:
                raise ValueError("expected positive equal min_tokens/max_tokens")
            input_count = len(tokens(body))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        self.input_tokens += input_count
        # Receive and record BEFORE arming; this must not conceal early requests.
        if body.get("stream"):
            await self.armed.wait()
        rule = self.rules.get(request_id)
        if rule and rule.endpoint and rule.endpoint != request.url.path:
            raise HTTPException(400, "rule endpoint mismatch")
        if rule:
            self.recorder.event("rule_matched", capture_id=capture_id, request_id=request_id, fault=rule.fault)
        if rule and rule.fault == "http_error":
            self.recorder.event("response_failed", capture_id=capture_id)
            raise HTTPException(500, "injected HTTP error")
        text = answer(self.scenario, request_id or "warmup", count)
        usage = {"prompt_tokens": input_count, "completion_tokens": count, "total_tokens": input_count + count}
        if rule and rule.fault == "usage_mismatch":
            usage["prompt_tokens"] += 1
            usage["total_tokens"] += 1
        if not body.get("stream"):
            payload = {"choices": [{"text": text, "finish_reason": "length"}], "usage": usage}
            self.record_response(capture_id, payload)
            self.successes += 1
            self.recorder.event("response_finished", capture_id=capture_id)
            return payload

        async def stream():
            self.active += 1
            terminal = False
            try:
                await self.gate(rule, "before_first_chunk", capture_id)
                await asyncio.sleep((self.scenario.first_chunk_delay_ms + (rule.delay_ms if rule else 0)) / 1000)
                for start in range(0, count, self.scenario.chunk_tokens):
                    part = text[start : start + self.scenario.chunk_tokens]
                    choice = {"text": part} if request.url.path == "/v1/completions" else {"delta": {"content": part}}
                    if body.get("return_token_ids") and not (rule and rule.fault == "token_ids_missing"):
                        choice["token_ids"] = list(map(ord, part))
                        if rule and rule.fault == "token_ids_short" and start == 0:
                            choice["token_ids"] = choice["token_ids"][1:]
                    payload = {"choices": [choice]}
                    self.record_response(capture_id, payload)
                    yield f"data: {json.dumps(payload)}\n\n"
                    if rule and rule.fault == "truncate_stream":
                        return
                    await asyncio.sleep(self.scenario.chunk_interval_ms / 1000)
                await self.gate(rule, "before_terminal", capture_id)
                payload = {"choices": [{"finish_reason": "length"}], "usage": usage}
                self.record_response(capture_id, payload)
                yield f"data: {json.dumps(payload)}\n\n"
                if rule and rule.fault == "omit_done":
                    return
                self.record_response(capture_id, "[DONE]")
                yield "data: [DONE]\n\n"
                terminal = True
                if not rule or not rule.fault:
                    self.successes += 1
            finally:
                self.active -= 1
                self.recorder.event("response_finished" if terminal else "response_aborted", capture_id=capture_id)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def metrics(self, request: Request):
        self.capture(request)
        legacy = self.scenario.profile == "run_legacy"
        labels = f'model_name="{self.scenario.model}"'
        lines = [
            f"process_start_time_seconds {self.started_at}",
            f"vllm:prompt_tokens_total{{{labels}}} {self.input_tokens}",
            f'vllm:request_success_total{{finished_reason="{"stop" if legacy else "length"}",{labels}}} {self.successes}',
        ]
        if legacy:
            lines.append(f"vllm:prompt_tokens_cached_total{{{labels}}} {self.successes * 3}")
        return Response("\n".join(lines) + "\n", media_type="text/plain")
