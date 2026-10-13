# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Streaming Backend transport for Replay requests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity

from ..request_proxy.observers import _normalize_usage
from ..request_proxy.request_trace import RequestFact, RequestTraceWriter
from .config import ReplayBenchConfig
from .planner import ReplayPlan, ReplayPlanNode, ReplayTaskPlan
from .prompt import SyntheticPrompt


@dataclass(frozen=True)
class TransportResult:
    success: bool
    assistant_content: str
    finished_clock: float
    status_code: int | None
    error: str | None
    assistant_reasoning_content: str | None = None
    input_tokens: int | None = None
    assistant_token_ids: tuple[int, ...] | None = None


class ReplayTransport:
    def __init__(self, config: ReplayBenchConfig, run_id: str, writer: RequestTraceWriter | None) -> None:
        self.config = config
        self.run_id = run_id
        self.writer = writer
        self.upstream = config.backend.base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=config.replay.request_timeout_seconds,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=0),
            trust_env=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _effective_output_tokens(node: ReplayPlanNode) -> int:
        """vLLM requires at least one generated token for an exact-token request."""

        assert node.planned_output_tokens is not None
        if getattr(node, "response_validation", None) == "exact_tokens":
            return max(1, node.planned_output_tokens)
        return node.planned_output_tokens

    async def validate_tracelab_context(self, plan: ReplayPlan, warmup_tokens: int) -> None:
        """Reject an impossible token-exact plan before filling the Backend cache."""

        response = await self.client.get(f"{self.upstream}/v1/models", headers=self._auth_headers())
        response.raise_for_status()
        models = response.json().get("data", [])
        model = next(
            (item for item in models if isinstance(item, dict) and item.get("id") == self.config.backend.model),
            None,
        )
        if model is None:
            raise ValueError(f"TraceLab model {self.config.backend.model!r} is not served by the Backend")
        limit = model.get("max_model_len")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Backend /v1/models did not report a positive max_model_len")
        if warmup_tokens + 1 > limit:
            raise ValueError(f"TraceLab warmup needs {warmup_tokens + 1} context tokens, Backend limit is {limit}")
        for task in plan.tasks:
            for node in task.requests:
                if node.node_type != "request":
                    continue
                assert node.planned_input_tokens is not None and node.planned_output_tokens is not None
                required = node.planned_input_tokens + self._effective_output_tokens(node)
                if required > limit:
                    raise ValueError(
                        f"TraceLab request {node.source_key} needs {required} context tokens, Backend limit is {limit}"
                    )

    async def warmup(self, token_ids: tuple[int, ...]) -> dict[str, object]:
        """Prefill the shared TraceLab root without writing a measured request fact."""

        started = time.monotonic()
        response = await self.client.post(
            f"{self.upstream}/v1/completions",
            json={
                "model": self.config.backend.model,
                "prompt": list(token_ids),
                "add_special_tokens": False,
                "max_tokens": 1,
                "min_tokens": 1,
                "ignore_eos": True,
                "seed": self.config.replay.sample_seed,
            },
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        usage = _normalize_usage(response.json().get("usage", {}))
        if usage.get("input_tokens") != len(token_ids):
            raise ValueError(
                f"TraceLab warmup input mismatch: expected={len(token_ids)} observed={usage.get('input_tokens')}"
            )
        return {
            "input_tokens": len(token_ids),
            "output_tokens": usage.get("output_tokens"),
            "duration_seconds": time.monotonic() - started,
        }

    async def send(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        prompt: SyntheticPrompt,
    ) -> TransportResult:
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        status_code: int | None = None
        error: str | None = None
        ttft: float | None = None
        usage: dict[str, int] = {}
        content: list[str] = []
        reasoning: list[str] | None = (
            [] if self.config.replay.prompt_shape in {"inferact_synthetic", "tracelab_synthetic"} else None
        )
        generated_ids: list[int] = []
        saw_token_ids = False
        saw_done = False
        try:
            async with self.client.stream(
                "POST",
                f"{self.upstream}{'/v1/completions' if prompt.token_ids is not None else self.config.backend.endpoint}",
                headers=self._headers(task, node),
                json=self._body(task, node, prompt),
            ) as response:
                status_code = response.status_code
                if response.is_error:
                    error = (await response.aread()).decode(errors="replace")
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            saw_done = True
                            continue
                        if not raw:
                            continue
                        payload = json.loads(raw)
                        if prompt.token_ids is not None:
                            choices = payload.get("choices")
                            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                                chunk_ids = choices[0].get("token_ids")
                                if chunk_ids is not None:
                                    if not isinstance(chunk_ids, list) or any(
                                        type(item) is not int for item in chunk_ids
                                    ):
                                        error = "TraceLab Backend returned invalid token_ids"
                                    else:
                                        generated_ids.extend(chunk_ids)
                                        saw_token_ids = True
                        reasoning_size = len(reasoning) if reasoning is not None else 0
                        delta = self._observe(payload, usage, reasoning, completion=prompt.token_ids is not None)
                        if delta or (reasoning is not None and len(reasoning) > reasoning_size) or generated_ids:
                            if ttft is None:
                                ttft = time.monotonic() - started_clock
                        if delta:
                            content.append(delta)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        if (
            error is None
            and status_code is not None
            and status_code < 400
            and self.config.replay.prompt_shape == "inferact_synthetic"
        ):
            actual = usage.get("input_tokens")
            target = node.planned_input_tokens
            if actual is None:
                error = "Current-turn input verification failed: Backend usage.prompt_tokens is missing"
            elif actual != target:
                error = f"Current-turn input verification failed: actual={actual} target={target}; exact count required"
        finished_clock = time.monotonic()
        if error is None and getattr(node, "response_validation", None) == "exact_tokens":
            expected_input = node.planned_input_tokens
            expected_output = self._effective_output_tokens(node)
            if not saw_done:
                error = "exact token validation failed: stream ended without [DONE]"
            elif usage.get("input_tokens") != expected_input or usage.get("output_tokens") != expected_output:
                error = (
                    "exact token validation failed: "
                    f"input expected={expected_input} observed={usage.get('input_tokens')}; "
                    f"output expected={expected_output} observed={usage.get('output_tokens')}"
                )
            elif self.config.replay.prompt_shape == "tracelab_synthetic" and (
                not saw_token_ids or len(generated_ids) != expected_output
            ):
                error = (
                    "exact token validation failed: Backend output token_ids missing or incomplete; "
                    f"expected={expected_output} observed={len(generated_ids)}"
                )
        success = status_code is not None and status_code < 400 and error is None
        if self.writer is None:
            raise RuntimeError("Replay transport cannot record a measured request without a trace writer")
        self.writer.submit(
            RequestFact(
                schema_version="1",
                run_id=self.run_id,
                request_id=node.runtime_request_id,
                session_id=task.runtime_session_id,
                actor_id=node.actor_id,
                actor_role=node.actor_role,
                started_at=started.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                status="success" if success else "error",
                status_code=status_code,
                latency_seconds=finished_clock - started_clock,
                ttft_seconds=ttft,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                cached_tokens=usage.get("cache_read_input_tokens"),
                upstream=self.upstream,
                error=error,
                request_purpose=node.prompt_kind,
            )
        )
        return TransportResult(
            success,
            "".join(content),
            finished_clock,
            status_code,
            error,
            "".join(reasoning) if reasoning else None,
            usage.get("input_tokens"),
            tuple(generated_ids) if prompt.token_ids is not None and saw_token_ids else None,
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self.config.backend.api_key_env:
            return {}
        return {"authorization": f"Bearer {os.environ[self.config.backend.api_key_env]}"}

    def _headers(self, task: ReplayTaskPlan, node: ReplayPlanNode) -> dict[str, str]:
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "x-claude-code-session-id": task.runtime_session_id,
        }
        if node.actor_role == "subagent":
            headers["x-claude-code-agent-id"] = node.actor_id
        if node.parent_actor_id:
            headers["x-claude-code-parent-agent-id"] = node.parent_actor_id
        if self.config.backend.endpoint == "/v1/messages":
            headers["anthropic-version"] = "2023-06-01"
        if self.config.backend.api_key_env:
            key = os.environ[self.config.backend.api_key_env]
            headers["authorization"] = f"Bearer {key}"
            headers["x-api-key"] = key
        return headers

    def _body(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        prompt: SyntheticPrompt,
    ) -> dict[str, object]:
        output_tokens = self._effective_output_tokens(node)
        identity = AgentIdentity(
            program_id=f"{task.runtime_session_id}:{node.actor_id}",
            task_id=task.runtime_session_id,
            session_id=task.runtime_session_id,
            agent_id=node.actor_id,
            parent_program_id=(f"{task.runtime_session_id}:{node.parent_actor_id}" if node.parent_actor_id else None),
            blocks_parent=node.actor_role == "subagent" and self.config.replay.trace_type != "agentX",
            agent_role=node.actor_role,
            request_id=node.runtime_request_id,
        )
        if prompt.token_ids is not None:
            if len(prompt.token_ids) != node.planned_input_tokens:
                raise ValueError("completion requires exact planned token IDs")
            body: dict[str, object] = {
                "model": self.config.backend.model,
                "prompt": list(prompt.token_ids),
                "add_special_tokens": False,
                "max_tokens": output_tokens,
                "min_tokens": output_tokens,
                "ignore_eos": True,
                "seed": node.backend_sampling_seed,
                "stream": True,
                "stream_options": {"include_usage": True},
                "vllm_xargs": {"agentic_context": encode_agent_identity(identity)},
            }
            if self.config.replay.prompt_shape == "tracelab_synthetic":
                body["return_token_ids"] = True
            return body
        if self.config.backend.endpoint == "/v1/chat/completions":
            messages = ([{"role": "system", "content": prompt.system}] if prompt.system else []) + list(prompt.messages)
            body: dict[str, object] = {
                "model": self.config.backend.model,
                "messages": messages,
                "max_tokens": output_tokens,
                "min_tokens": output_tokens,
                "ignore_eos": True,
                "seed": node.backend_sampling_seed,
                "stream": True,
                "stream_options": {"include_usage": True},
                "vllm_xargs": {"agentic_context": encode_agent_identity(identity)},
            }
            if prompt.tools:
                body["tools"] = list(prompt.tools)
            if self.config.backend.chat_template_kwargs:
                body["chat_template_kwargs"] = dict(self.config.backend.chat_template_kwargs)
            return body
        body = {
            **prompt.anthropic_payload(self.config.backend.model),
            "max_tokens": output_tokens,
            "stream": True,
            "metadata": {
                "_agentinfer_agentic_context": encode_agent_identity(identity),
                "_agentinfer_replay_sampling": {
                    "seed": node.backend_sampling_seed,
                    "min_tokens": output_tokens,
                    "ignore_eos": True,
                },
            },
        }
        return body

    def _observe(
        self,
        payload: dict[str, object],
        usage: dict[str, int],
        reasoning: list[str] | None = None,
        *,
        completion: bool = False,
    ) -> str:
        """Collect usage and keep requested reasoning separate from visible text."""

        raw_usage = payload.get("usage")
        if not isinstance(raw_usage, dict):
            message = payload.get("message")
            raw_usage = message.get("usage") if isinstance(message, dict) else None
        if isinstance(raw_usage, dict):
            usage.update(_normalize_usage(raw_usage))
        if completion:
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                return str(choices[0].get("text") or "")
            return ""
        if self.config.backend.endpoint == "/v1/chat/completions":
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not isinstance(delta, dict):
                return ""
            if reasoning is not None:
                if delta.get("reasoning_content"):
                    reasoning.append(str(delta["reasoning_content"]))
                return str(delta.get("content") or "")
            return str(delta.get("content") or delta.get("reasoning_content") or "")
        delta = payload.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("text") or delta.get("thinking") or delta.get("partial_json") or "")
        block = payload.get("content_block")
        return str(block.get("text") or "") if isinstance(block, dict) else ""
