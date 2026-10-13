# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build and calibrate deterministic synthetic Replay Prompts.

This module owns Prompt shape construction and request-private token calibration;
structural planning and Backend transport remain in their respective modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import Literal, TypeVar, cast

import httpx

from .config import ReplayBenchConfig
from .hash_snapshot import HashSnapshotStore, TokenBlockRenderer
from .local_tokenizer import LocalTokenizerCounter, discover_local_tokenizer
from .planner import ReplayPlan, ReplayPlanNode, ReplayTaskPlan
from .tokenizer_retry import TOKENIZER_REQUEST_MAX_ATTEMPTS, TOKENIZER_RETRY_BACKOFF_SECONDS
from .unified_trace_ir import PromptReference, TraceTextStore, UnifiedTraceIR

logger = logging.getLogger(__name__)

SystemContent = str | tuple[dict[str, object], ...]
T = TypeVar("T")
TokenizerCacheKey = tuple[str, str] | tuple[str, str, int]

# The renderer requires at least two distinct valid token IDs from the Backend tokenizer.
AGENTX_TOKEN_PALETTE_SEED = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !@#$%^&*()-_=+[]{};:,.?/ " * 8
)


@dataclass(frozen=True)
class PromptCalibration:
    """Describe how one Prompt reached, or approached, its Trace token target.

    ``added_filler_tokens`` is retained as the serialized compatibility name for
    requested filler; ``actual_prompt_token_gain`` records the observed gain.
    ``count_history`` contains Backend count results only, never Prompt content.
    """

    target_tokens: int
    initial_tokens: int
    final_tokens: int
    added_filler_tokens: int
    requested_filler_tokens: int
    actual_prompt_token_gain: int
    trimmed_filler_tokens: int
    residual_tokens: int
    target_met: bool
    accepted_with_tolerance: bool
    repair_attempts: int
    count_history: tuple[int, ...]
    adjustment: Literal["none", "pad", "trim", "trim_and_pad", "reset", "reset_and_pad"]
    trimmed_current_user_tokens: int = 0
    trimmed_current_user_characters: int = 0
    preserved_prefix_tokens: int | None = None
    source_prefix_tokens: int | None = None
    effective_prefix_tokens: int | None = None
    prefix_shortfall_tokens: int | None = None


@dataclass(frozen=True)
class SyntheticPrompt:
    """Represent one calibrated synthetic Prompt in Backend-neutral form."""

    system: SystemContent
    tools: tuple[dict[str, object], ...]
    messages: tuple[dict[str, object], ...]
    calibration: PromptCalibration | None = None
    extra_body: dict[str, object] | None = None
    token_ids: tuple[int, ...] | None = None
    # Half-open ranges of synthetic request-private filler; output and shared segments are immutable.
    mutable_token_spans: tuple[tuple[int, int], ...] = ()

    def anthropic_tools(self) -> list[dict[str, object]]:
        """Convert internal function tools to Anthropic Messages tool objects."""

        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in self.tools
        ]

    def anthropic_payload(self, model: str) -> dict[str, object]:
        """Serialize the Prompt fields shared by token counting and inference."""

        payload: dict[str, object] = {
            "model": model,
            "system": list(self.system) if isinstance(self.system, tuple) else self.system,
            "messages": list(self.messages),
            "tools": self.anthropic_tools(),
        }
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def tokenizer_payload(
        self,
        model: str,
        chat_template_kwargs: dict[str, object] | None = None,
    ) -> dict[str, object]:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + list(self.messages)
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if self.tools:
            payload["tools"] = list(self.tools)
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        return payload


@dataclass(frozen=True)
class PromptExchange:
    prompt: SyntheticPrompt
    assistant_content: str
    assistant_reasoning_content: str | None = None
    assistant_token_ids: tuple[int, ...] | None = None


class TokenizerClient:
    """Build deterministic filler text and count its serialized Prompt tokens."""

    def __init__(self, config: ReplayBenchConfig, local_tokenizer: LocalTokenizerCounter | None = None) -> None:
        """Create a tokenizer client, optionally reusing one validated during conversion."""

        self.model = config.backend.model
        self.config = config
        self.endpoint = config.backend.endpoint
        self.base_url = config.backend.resolved_tokenizer_base_url.rstrip("/")
        self.chat_template_kwargs = dict(config.backend.chat_template_kwargs)
        self.local_tokenizer = local_tokenizer if self.endpoint == "/v1/chat/completions" else None
        headers = {}
        if config.backend.api_key_env:
            headers["authorization"] = f"Bearer {os.environ[config.backend.api_key_env]}"
        self.client = httpx.AsyncClient(
            timeout=config.replay.request_timeout_seconds,
            headers=headers,
            limits=httpx.Limits(max_connections=None),
            trust_env=False,
        )
        self._token_ids: dict[str, tuple[int, ...]] = {}
        self._text_cache: dict[tuple[str, int], str] = {}
        self._inflight: dict[TokenizerCacheKey, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(config.replay.tokenizer_max_concurrency)

    async def close(self) -> None:
        pending = tuple(self._inflight.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.client.aclose()

    async def release_task(self, task: ReplayTaskPlan) -> None:
        """Cancel private work and release completed task caches, retaining shared prefixes."""

        prefixes = (f"{task.runtime_session_id}:", f"session:{task.runtime_session_id}:")
        request_ids = {node.runtime_request_id for node in task.requests}

        def private(namespace: str) -> bool:
            return namespace.startswith(prefixes) or (
                namespace.startswith("trace-record:") and namespace.split(":", 2)[1] in request_ids
            )

        pending = tuple(operation for key, operation in self._inflight.items() if private(key[1]))
        for operation in pending:
            operation.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for namespace in tuple(self._token_ids):
            if private(namespace):
                del self._token_ids[namespace]
        for key in tuple(self._text_cache):
            if private(key[0]):
                del self._text_cache[key]

    async def _shared(self, key: TokenizerCacheKey, operation: Callable[[], Awaitable[T]]) -> T:
        """Share cache misses without propagating one waiter's cancellation."""

        task = self._inflight.get(key)
        if task is None:

            async def run() -> T:
                try:
                    return await operation()
                finally:
                    self._inflight.pop(key, None)

            task = asyncio.create_task(run())
            # Retrieve failures even when all waiters have been cancelled.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            self._inflight[key] = task
        return cast(T, await asyncio.shield(task))

    async def prepare_local_tokenizer(self) -> None:
        """Use only backend-validated local tokenization; keep full-template calibration."""

        if self.local_tokenizer is not None or self.endpoint != "/v1/chat/completions":
            return
        loop = asyncio.get_running_loop()

        async def get_json(url: str) -> dict[str, object] | None:
            try:
                async with self._semaphore:
                    response = await self.client.get(url)
                response.raise_for_status()
                value = response.json()
                return value if isinstance(value, dict) else None
            except (httpx.HTTPError, ValueError):
                return None

        async def backend_tokens(payload: dict[str, object]) -> tuple[int, ...] | None:
            response = await self._post("/tokenize", json={"model": self.model, **payload})
            response.raise_for_status()
            value = response.json().get("tokens")
            return tuple(value) if isinstance(value, list) and all(type(t) is int for t in value) else None

        stopped = threading.Event()

        def on_loop(operation):
            if stopped.is_set():
                operation.close()
                raise asyncio.CancelledError
            future = asyncio.run_coroutine_threadsafe(operation, loop)
            try:
                while not stopped.is_set():
                    try:
                        return future.result(timeout=0.1)
                    except FutureTimeoutError:
                        if future.done():
                            raise
                raise asyncio.CancelledError
            finally:
                future.cancel()

        try:
            self.local_tokenizer = await asyncio.to_thread(
                discover_local_tokenizer,
                self.config,
                get_json=lambda url: on_loop(get_json(url)),
                backend_tokens=lambda payload: on_loop(backend_tokens(payload)),
            )
        finally:
            stopped.set()

    async def count(self, prompt: SyntheticPrompt) -> int:
        """Return the Backend tokenizer count for one complete synthetic Prompt."""

        if self.endpoint == "/v1/messages":
            response = await self._post(
                "/v1/messages/count_tokens",
                headers={"anthropic-version": "2023-06-01"},
                json=prompt.anthropic_payload(self.model),
            )
            response.raise_for_status()
            return int(response.json()["input_tokens"])
        if self.local_tokenizer is not None:
            payload = prompt.tokenizer_payload(self.model, self.chat_template_kwargs)
            messages = payload["messages"]
            tools = payload.get("tools")
            assert isinstance(messages, list)
            assert tools is None or isinstance(tools, list)
            tokens = await asyncio.to_thread(self.local_tokenizer.chat_token_ids, messages, tools=tools)
            return len(tokens)
        response = await self._post(
            "/tokenize",
            json=prompt.tokenizer_payload(self.model, self.chat_template_kwargs),
        )
        response.raise_for_status()
        return int(response.json()["count"])

    async def prompt_token_ids(self, prompt: SyntheticPrompt) -> tuple[int, ...]:
        """Encode the complete chat template for calibration prefix verification."""

        if self.endpoint != "/v1/chat/completions":
            raise ValueError("Prompt token IDs require /v1/chat/completions")
        payload = prompt.tokenizer_payload(self.model, self.chat_template_kwargs)
        if self.local_tokenizer is not None:
            return await asyncio.to_thread(
                self.local_tokenizer.chat_token_ids, payload["messages"], tools=payload.get("tools")
            )
        response = await self._post("/tokenize", json=payload)
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("Backend /tokenize response has no integer tokens list")
        return tuple(tokens)

    async def text_token_ids(self, text: str) -> tuple[int, ...]:
        if self.local_tokenizer is not None:
            return await asyncio.to_thread(self.local_tokenizer.text_token_ids, text, cache=False)
        response = await self._post(
            "/tokenize",
            json={"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Backend /tokenize response has no valid integer tokens list")
        return tuple(tokens)

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        if not tokens:
            return ""
        if self.local_tokenizer is not None:
            return await asyncio.to_thread(self.local_tokenizer.detokenize_tokens, tokens)
        response = await self._post(
            "/detokenize",
            json={"model": self.model, "tokens": list(tokens)},
        )
        response.raise_for_status()
        return str(response.json()["prompt"])

    async def _post(
        self,
        path: str,
        *,
        json: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST one stateless tokenizer operation, retrying transport failures."""

        for attempt in range(1, TOKENIZER_REQUEST_MAX_ATTEMPTS + 1):
            try:
                async with self._semaphore:
                    return await self.client.post(f"{self.base_url}{path}", headers=headers, json=json)
            except httpx.TransportError as exc:
                if attempt == TOKENIZER_REQUEST_MAX_ATTEMPTS:
                    raise
                delay = TOKENIZER_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Tokenizer request transport failure; retrying: path=%s attempt=%d/%d error=%s delay_seconds=%s",
                    path,
                    attempt,
                    TOKENIZER_REQUEST_MAX_ATTEMPTS,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def token_text(self, namespace: str, count: int, *, cache: bool = True) -> str:
        if count == 0:
            return ""
        cache_key = (namespace, count)
        if cache and cache_key in self._text_cache:
            return self._text_cache[cache_key]

        async def prepare() -> str:
            tokens = await self.token_sequence(namespace, count)
            text = await self.detokenize_tokens(tokens)
            if cache:
                self._text_cache[cache_key] = text
            return text

        text = await self._shared(("text", namespace, count), prepare)
        if cache:
            self._text_cache[cache_key] = text
        return text

    async def token_sequence(self, namespace: str, count: int) -> tuple[int, ...]:
        """Repeat a cached seed without detokenizing or encoding historical context."""

        if count <= 0:
            return ()
        if namespace not in self._token_ids:

            async def prepare() -> tuple[int, ...]:
                material = hashlib.sha256(namespace.encode()).hexdigest()
                ids = await self.text_token_ids(f" {material}")
                if not ids:
                    raise ValueError(f"Tokenizer returned an empty seed for {namespace}")
                self._token_ids[namespace] = ids
                return ids

            await self._shared(("seed", namespace), prepare)
        ids = self._token_ids[namespace]
        return (ids * ((count + len(ids) - 1) // len(ids)))[:count]

    async def filler_token_ids(
        self, namespace: str, count: int, *, forbidden_first: int | None = None
    ) -> tuple[int, ...]:
        """Generate stable token IDs with an explicit branch at the first new token."""

        if count <= 0:
            return ()
        for variant in range(8):
            key = f"{namespace}:{variant}"
            await self.token_sequence(key, 1)
            seed = self._token_ids[key]
            first = next((index for index, token in enumerate(seed) if token != forbidden_first), None)
            if first is not None:
                return tuple(seed[(first + index) % len(seed)] for index in range(count))
        raise ValueError("cannot construct a divergent TraceLab token suffix")


class PromptBuilder:
    """Construct role-aware Prompt shapes and calibrate them to Trace targets."""

    def __init__(
        self,
        config: ReplayBenchConfig,
        tokenizer: TokenizerClient,
        trace_ir: UnifiedTraceIR | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        if config.replay.prompt_shape == "inferact_synthetic" and trace_ir is None:
            raise ValueError("inferact_synthetic Prompt construction requires a validated unified Trace IR")
        if config.replay.prompt_shape == "tracelab_synthetic" and (
            trace_ir is None or trace_ir.prompt_source_kind != "token_recipe"
        ):
            raise ValueError("token_recipe Prompt construction requires token_recipe unified Trace IR")
        self.trace_texts = TraceTextStore(trace_ir) if trace_ir is not None else None
        self.trace_ir = trace_ir
        self._tracelab_warmup_ids: tuple[int, ...] | None = None
        self._tracelab_root_branch_tokens: dict[str, int] = {}
        if config.replay.prompt_shape == "agentX_snapshot" and (
            trace_ir is None or trace_ir.prompt_source_kind != "hash_snapshot"
        ):
            raise ValueError("AgentX Prompt construction requires hash_snapshot Trace IR")
        self.hash_snapshots = (
            HashSnapshotStore(trace_ir.requests_path)
            if config.replay.prompt_shape == "agentX_snapshot" and trace_ir is not None
            else None
        )
        self._token_blocks: TokenBlockRenderer | None = None
        self._token_blocks_lock = asyncio.Lock()

    async def release_task(self, task: ReplayTaskPlan) -> None:
        """Drop task-private preparation data after its graph has stopped."""

        await self.tokenizer.release_task(task)
        if self.trace_texts is not None:
            for node in task.requests:
                if node.prompt_ref is not None:
                    self.trace_texts.discard(node.prompt_ref)

    @staticmethod
    def _session_scope(task: ReplayTaskPlan, *, lead: bool) -> str:
        return f"session:{task.runtime_session_id}:{'lead' if lead else 'subagents'}"

    def _root_segments(self, task: ReplayTaskPlan, node: ReplayPlanNode) -> list[tuple[str, int]]:
        """Shared namespace recipe used by both preparation and construction."""

        replay = self.config.replay
        if node.prompt_kind in {"lead_title", "lead_name"}:
            name = node.prompt_kind.removeprefix("lead_")
            return [(f"shared:lead-{name}-system", getattr(replay, f"lead_{name}_sys_shared_prefix"))]
        lead = node.actor_role == "lead"
        family = "lead" if lead else "subagent"
        prefix = "lead" if lead else "subagent"
        return [
            (f"shared:{family}:system-zero", 1),
            (f"shared:{family}:system", getattr(replay, f"{prefix}_1st_sys_shared_prefix")),
            (
                f"{self._session_scope(task, lead=lead)}:system-private",
                getattr(replay, f"{prefix}_1st_sys_session_prefix"),
            ),
            (
                f"shared:{family}:tool",
                replay.lead_1st_tool_shared_prefix if lead else replay.subagent_tool_1st_shared_prefix,
            ),
            (f"shared:{family}:message", getattr(replay, f"{prefix}_1st_msg_shared_prefix")),
            ("shared:lead:trailing-system", replay.lead_1st_trailing_system_prefix if lead else 0),
        ]

    def _extra_system_segment(self, node: ReplayPlanNode) -> tuple[str, int]:
        family = "lead" if node.actor_role == "lead" else "subagent"
        return f"shared:{family}:continuation-system", self.config.replay.extra_system_tokens(
            node.source_key, node.actor_role
        )

    async def prepare_tasks(self, plan: ReplayPlan) -> None:
        """Validate the plan and prepare only consumed global shared segments."""

        if self.config.replay.prompt_shape == "agentX_snapshot":
            return

        namespaces: set[str] = set()
        text_segments: set[tuple[str, int]] = set()
        for task in plan.tasks:
            by_key = {node.source_key: node for node in task.requests}
            for node in task.requests:
                if node.node_type != "request":
                    continue
                if node.planned_output_tokens is None or node.planned_output_tokens < 1:
                    raise ValueError(f"request {node.source_key}: backend cannot replay zero output tokens exactly")
                if self.config.replay.prompt_shape in {"tracelab_synthetic", "inferact_synthetic"}:
                    continue
                if node.context_after is None:
                    segments = self._root_segments(task, node)
                else:
                    segments = [self._extra_system_segment(node)]
                    previous = by_key[node.context_after]
                    budget = (
                        node.planned_input_tokens
                        - previous.planned_input_tokens
                        - previous.planned_output_tokens
                        - sum(count for _, count in segments)
                    )
                    if self.config.replay.uses_token_ids and budget < 0 and node.context_mode != "trim":
                        raise ValueError(
                            f"request {node.source_key}: append target cannot contain predecessor and output"
                        )
                for key, count in segments:
                    if count > 0 and key.startswith("shared:"):
                        namespaces.add(key)
                        text_segments.add((key, count))

        async def bounded(items, operation) -> None:
            iterator = iter(items)

            async def worker() -> None:
                for item in iterator:
                    await operation(item)

            workers = [asyncio.create_task(worker()) for _ in range(self.config.replay.tokenizer_max_concurrency)]
            try:
                await asyncio.gather(*workers)
            finally:
                for worker_task in workers:
                    worker_task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        await bounded(sorted(namespaces), lambda key: self.tokenizer.token_sequence(key, 1))
        if not self.config.replay.uses_token_ids:
            await bounded(sorted(text_segments), lambda item: self.tokenizer.token_text(*item))

    async def _token_prompt(
        self, task: ReplayTaskPlan, node: ReplayPlanNode, context: PromptExchange | None
    ) -> SyntheticPrompt:
        """Append actual output IDs, trimming only explicitly recorded private filler."""

        target = node.planned_input_tokens
        assert target is not None
        namespace = f"{task.runtime_session_id}:{node.prompt_recipe_key}"
        mode = node.context_mode
        spans: list[tuple[int, int]] = []
        trimmed = 0
        if context is None:
            if node.context_after is not None:
                raise ValueError("token continuation lacks predecessor context")
            segments = self._root_segments(task, node)
            trailing = segments.pop() if segments[-1][0] == "shared:lead:trailing-system" else ("", 0)
            parts = [await self.tokenizer.token_sequence(key, count) for key, count in segments]
            tokens = tuple(token for part in parts for token in part)
        else:
            if context.prompt.token_ids is None or context.assistant_token_ids is None:
                raise ValueError("token continuation lacks actual predecessor input/output IDs")
            if not context.assistant_token_ids or any(
                type(token) is not int or token < 0 for token in context.assistant_token_ids
            ):
                raise ValueError("token continuation has invalid actual output IDs")
            tokens = context.prompt.token_ids + context.assistant_token_ids
            spans = list(context.prompt.mutable_token_spans)
            trailing = self._extra_system_segment(node)
        initial = len(tokens) + trailing[1]
        preserved = len(tokens)
        deficit = initial - target
        if deficit > 0:
            if context is None or mode != "trim" or self.config.replay.context_adjustment_mode != "adaptive":
                raise ValueError(f"request {node.source_key}: {mode} needs {initial} tokens, target is {target}")
            if deficit > self.config.replay.context_micro_trim_limit(target):
                raise ValueError("planned trim exceeds context micro-trim limit")
            if sum(end - start for start, end in spans) < deficit:
                raise ValueError("planned trim cannot fit without changing shared prefixes or actual outputs")
            # Nearest historical private remainder first. Deleting its tail preserves more prefix.
            for index in range(len(spans) - 1, -1, -1):
                start, end = spans[index]
                removed = min(end - start, deficit)
                cut = end - removed
                tokens = tokens[:cut] + tokens[end:]
                preserved = min(preserved, cut)
                spans[index] = (start, cut)
                for later in range(index + 1, len(spans)):
                    left, right = spans[later]
                    spans[later] = (left - removed, right - removed)
                trimmed += removed
                deficit -= removed
                if deficit == 0:
                    break
            spans = [(start, end) for start, end in spans if start < end]
        filler_count = target - len(tokens) - trailing[1]
        if filler_count:
            spans.append((len(tokens), len(tokens) + filler_count))
            tokens += await self.tokenizer.token_sequence(namespace, filler_count)
        tokens += await self.tokenizer.token_sequence(*trailing)
        adjustment = (
            ("reset_and_pad" if filler_count else "reset")
            if mode == "reset"
            else (
                "trim_and_pad" if trimmed and filler_count else "trim" if trimmed else "pad" if filler_count else "none"
            )
        )
        calibration = PromptCalibration(
            target,
            initial,
            len(tokens),
            filler_count,
            filler_count,
            filler_count,
            trimmed,
            target - len(tokens),
            len(tokens) == target,
            False,
            0,
            (len(tokens),),
            adjustment,
            preserved_prefix_tokens=preserved if context is not None else 0,
        )
        return SyntheticPrompt("", (), (), calibration=calibration, token_ids=tokens, mutable_token_spans=tuple(spans))

    async def prepare_tracelab_warmup(self, plan: ReplayPlan) -> tuple[int, ...]:
        """Build the shared cache root before measured TraceLab execution."""

        if self.config.replay.prompt_shape != "tracelab_synthetic":
            raise ValueError("TraceLab warmup requires a TraceLab Replay plan")
        if self.config.backend.endpoint != "/v1/chat/completions":
            raise ValueError("TraceLab token-exact Replay requires a vLLM OpenAI endpoint")
        assert self.trace_ir is not None
        summary = self.trace_ir.summary
        warmup_input_tokens = summary.get("warmup_input_tokens") if summary is not None else None
        if isinstance(warmup_input_tokens, bool) or not isinstance(warmup_input_tokens, int) or warmup_input_tokens < 1:
            raise ValueError("TraceLab conversion did not provide a valid warmup_input_tokens")
        self._tracelab_warmup_ids = await self.tokenizer.filler_token_ids(
            f"tracelab:warmup:{plan.workload_fingerprint}", warmup_input_tokens
        )
        roots = [
            node
            for task in plan.tasks
            for node in task.requests
            if node.node_type == "request" and node.context_after is None
        ]
        if roots:
            candidates = dict.fromkeys(
                await self.tokenizer.text_token_ids(" ".join(chr(code) for code in range(33, 127)))
            )
            used_by_prefix: dict[int, set[int]] = {}
            for node in roots:
                assert node.planned_input_tokens is not None
                reused = node.source_prefix_tokens or 0
                if reused == node.planned_input_tokens:
                    continue
                used = used_by_prefix.setdefault(reused, set())
                forbidden = self._tracelab_warmup_ids[reused]
                branch = next(
                    (candidate for candidate in candidates if candidate not in used and candidate != forbidden), None
                )
                if branch is None:
                    raise ValueError("not enough distinct token IDs for TraceLab root branches")
                used.add(branch)
                self._tracelab_root_branch_tokens[node.runtime_request_id] = branch
        return self._tracelab_warmup_ids

    async def build(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        context: PromptExchange | None,
    ) -> SyntheticPrompt:
        """Build one node Prompt from its recipe, context mode, and prior exchange."""

        if self.config.replay.prompt_shape == "inferact_synthetic":
            return await self._trace_record_prompt(node, context)
        if self.config.replay.prompt_shape == "tracelab_synthetic":
            return await self._tracelab_prompt(task, node, context)
        if self.config.replay.prompt_shape == "agentX_snapshot":
            return await self._agentx_prompt(task, node, context)
        if self.config.replay.uses_token_ids:
            return await self._token_prompt(task, node, context)
        namespace = f"{task.runtime_session_id}:{node.prompt_recipe_key}"
        context_mode = getattr(node, "context_mode", "independent" if context is None else "append")
        if context_mode == "reset":
            prompt = await self._first_prompt(task, node, namespace)
        elif node.prompt_kind == "lead_name":
            prompt = await self._auxiliary_prompt(node, namespace, name=True)
        elif node.prompt_kind == "lead_title":
            prompt = await self._auxiliary_prompt(node, namespace, name=False)
        elif node.prompt_kind == "lead_main":
            prompt = await self._first_prompt(task, node, namespace)
        elif node.prompt_kind == "subagent_first" or context is None:
            prompt = await self._first_prompt(task, node, namespace)
        else:
            assert context is not None
            prompt = await self._continuation_prompt(node, namespace, context)
        assert node.planned_input_tokens is not None
        return await self._calibrate(
            prompt,
            namespace,
            node.planned_input_tokens,
            context_mode=context_mode,
        )

    async def _agentx_prompt(
        self, task: ReplayTaskPlan, node: ReplayPlanNode, context: PromptExchange | None
    ) -> SyntheticPrompt:
        """Build one complete token-ID snapshot without appending live output."""

        if context is not None or node.context_after is not None:
            raise ValueError("AgentX hash snapshots cannot consume a previous Prompt exchange")
        assert self.hash_snapshots is not None
        if self._token_blocks is None:
            async with self._token_blocks_lock:
                if self._token_blocks is None:
                    palette = await self.tokenizer.text_token_ids(AGENTX_TOKEN_PALETTE_SEED)
                    self._token_blocks = TokenBlockRenderer(palette)
        recipe = self.hash_snapshots.read(node.prompt_recipe_key, node.source_key)
        tokens = self._token_blocks.render(recipe, task.runtime_session_id)
        if len(tokens) != node.planned_input_tokens:
            raise ValueError(f"AgentX request {node.source_key} rendered the wrong input length")
        calibration = PromptCalibration(
            target_tokens=len(tokens),
            initial_tokens=len(tokens),
            final_tokens=len(tokens),
            added_filler_tokens=0,
            requested_filler_tokens=0,
            actual_prompt_token_gain=0,
            trimmed_filler_tokens=0,
            residual_tokens=0,
            target_met=True,
            accepted_with_tolerance=False,
            repair_attempts=0,
            count_history=(),
            adjustment="none",
        )
        return SyntheticPrompt("", (), (), calibration=calibration, token_ids=tokens)

    async def _tracelab_prompt(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        context: PromptExchange | None,
    ) -> SyntheticPrompt:
        """Reuse the requested token prefix of the live predecessor or shared root."""

        if (node.context_after is not None) != (context is not None):
            raise ValueError(f"tracelab request {node.source_key} has inconsistent context")
        assert node.planned_input_tokens is not None
        source_prefix = node.source_prefix_tokens
        if source_prefix is None or source_prefix < 0 or source_prefix > node.planned_input_tokens:
            raise ValueError(f"tracelab request {node.source_key} has invalid source prefix")
        if context is None:
            if self._tracelab_warmup_ids is None:
                raise ValueError("TraceLab warmup has not completed")
            previous = self._tracelab_warmup_ids
        else:
            if context.prompt.token_ids is None or context.assistant_token_ids is None:
                raise ValueError(f"tracelab request {node.source_key} lacks predecessor token IDs")
            previous = context.prompt.token_ids + context.assistant_token_ids
        reused = min(source_prefix, len(previous))
        forbidden = previous[reused] if reused < len(previous) else None
        suffix_count = node.planned_input_tokens - reused
        root_branch = (
            self._tracelab_root_branch_tokens.get(getattr(node, "runtime_request_id", "")) if context is None else None
        )
        if root_branch is None:
            suffix = await self.tokenizer.filler_token_ids(
                f"{task.runtime_session_id}:{node.prompt_recipe_key}",
                suffix_count,
                forbidden_first=forbidden,
            )
        else:
            suffix = (root_branch,) + await self.tokenizer.filler_token_ids(
                f"{task.runtime_session_id}:{node.prompt_recipe_key}", suffix_count - 1
            )
        tokens = previous[:reused] + suffix
        target = node.planned_input_tokens
        calibration = PromptCalibration(
            target_tokens=target,
            initial_tokens=reused,
            final_tokens=len(tokens),
            added_filler_tokens=len(suffix),
            requested_filler_tokens=len(suffix),
            actual_prompt_token_gain=len(suffix),
            trimmed_filler_tokens=0,
            residual_tokens=target - len(tokens),
            target_met=len(tokens) == target,
            accepted_with_tolerance=False,
            repair_attempts=0,
            count_history=(len(tokens),),
            adjustment="pad" if suffix else "none",
            preserved_prefix_tokens=reused,
            source_prefix_tokens=source_prefix,
            effective_prefix_tokens=reused,
            prefix_shortfall_tokens=source_prefix - reused,
        )
        return SyntheticPrompt("", (), (), calibration=calibration, token_ids=tokens)

    async def _auxiliary_prompt(
        self,
        node: ReplayPlanNode,
        namespace: str,
        *,
        name: bool,
    ) -> SyntheticPrompt:
        prefix = (
            self.config.replay.lead_name_sys_shared_prefix if name else self.config.replay.lead_title_sys_shared_prefix
        )
        kind = "name" if name else "title"
        system_text = await self.tokenizer.token_text(f"shared:lead-{kind}-system", prefix)
        request_private_remainder = await self.tokenizer.token_text(namespace, 1)
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {kind: {"type": "string"}},
                "required": [kind],
                "additionalProperties": False,
            },
        }
        return SyntheticPrompt(
            ({"type": "text", "text": system_text},),
            (),
            ({"role": "user", "content": [{"type": "text", "text": request_private_remainder}]},),
            extra_body={"output_config": {"effort": "high", "format": schema}},
        )

    async def _first_prompt(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        namespace: str,
    ) -> SyntheticPrompt:
        lead = getattr(node, "actor_role", "lead") == "lead"
        if lead:
            system_prefix = self.config.replay.lead_1st_sys_shared_prefix
            session_system_prefix = self.config.replay.lead_1st_sys_session_prefix
            tool_prefix = self.config.replay.lead_1st_tool_shared_prefix
            message_prefix = self.config.replay.lead_1st_msg_shared_prefix
            family = "lead"
            session_scope = self._session_scope(task, lead=True)
        else:
            system_prefix = self.config.replay.subagent_1st_sys_shared_prefix
            session_system_prefix = self.config.replay.subagent_1st_sys_session_prefix
            tool_prefix = self.config.replay.subagent_tool_1st_shared_prefix
            message_prefix = self.config.replay.subagent_1st_msg_shared_prefix
            family = "subagent"
            session_scope = self._session_scope(task, lead=False)
        system_namespace = f"shared:{family}:system"
        tool_namespace = f"shared:{family}:tool"
        message_namespace = f"shared:{family}:message"
        system = await self.tokenizer.token_text(system_namespace, system_prefix)
        session_system = await self.tokenizer.token_text(
            f"{session_scope}:system-private",
            session_system_prefix,
        )
        tool_text = await self.tokenizer.token_text(tool_namespace, tool_prefix)
        message = await self.tokenizer.token_text(message_namespace, message_prefix)
        request_private_remainder = await self.tokenizer.token_text(namespace, 1)
        tools = (
            {
                "type": "function",
                "function": {
                    "name": f"replay_{family}_tool",
                    "description": tool_text,
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
        system_blocks = (
            {"type": "text", "text": await self.tokenizer.token_text(f"shared:{family}:system-zero", 1)},
            {"type": "text", "text": system},
            {"type": "text", "text": session_system},
        )
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "text", "text": request_private_remainder},
                ],
            }
        ]
        extra_body: dict[str, object] = {"output_config": {"effort": "high"}}
        if lead:
            trailing = await self.tokenizer.token_text(
                "shared:lead:trailing-system",
                self.config.replay.lead_1st_trailing_system_prefix,
            )
            if trailing:
                messages.append({"role": "system", "content": trailing})
            extra_body.update(
                {
                    "thinking": {"type": "adaptive"},
                    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
                }
            )
        return SyntheticPrompt(system_blocks, tools, tuple(messages), extra_body=extra_body)

    def _add_extra_system_message(self, node: ReplayPlanNode) -> bool:
        ratio = (
            self.config.replay.lead_continuation_extra_system_ratio
            if node.actor_role == "lead"
            else self.config.replay.subagent_continuation_extra_system_ratio
        )
        if ratio <= 0:
            return False
        bucket = int(hashlib.sha256(node.source_key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < ratio

    async def _trace_record_prompt(
        self,
        node: ReplayPlanNode,
        context: PromptExchange | None,
    ) -> SyntheticPrompt:
        """Append a live exchange and exactly calibrate only the newest user text."""

        assert self.trace_texts is not None
        reference = node.prompt_ref
        if not isinstance(reference, PromptReference):
            raise ValueError(f"inferact_synthetic request {node.source_key} has no valid prompt_ref")
        human_content = self.trace_texts.read(reference)
        user_message: dict[str, object] = {"role": "user", "content": human_content}
        if context is None:
            prompt = SyntheticPrompt("", (), (user_message,))
        else:
            messages = list(context.prompt.messages)
            assistant: dict[str, object] = {"role": "assistant", "content": context.assistant_content}
            if context.assistant_reasoning_content is not None:
                assistant["reasoning_content"] = context.assistant_reasoning_content
            messages.extend((assistant, user_message))
            prompt = SyntheticPrompt("", (), tuple(messages))

        assert node.planned_input_tokens is not None
        count = await self.tokenizer.count(prompt)
        return await self._calibrate_current_turn(prompt, node, count)

    async def _calibrate_current_turn(
        self, prompt: SyntheticPrompt, node: ReplayPlanNode, initial_count: int
    ) -> SyntheticPrompt:
        """Fit the latest user suffix to the target, keeping all historical messages intact.

        Descending search keeps the longest source prefix that fits the target.
        It checks character boundaries without assuming monotone token counts.
        Bounded suffix repair then attempts an exact full-template count. Every
        accepted candidate is measured; histories that cannot fit fail explicitly.
        """

        assert node.planned_input_tokens is not None
        target = node.planned_input_tokens
        index = len(prompt.messages) - 1
        original = prompt.messages[index]["content"]
        assert prompt.messages[index]["role"] == "user" and isinstance(original, str)
        candidate, count, kept = prompt, initial_count, original
        history = [count]
        original_ids: tuple[int, ...] | None = None
        empty_ids: tuple[int, ...] | None = None
        original_content_ids: tuple[int, ...] | None = None
        if count != target:
            # Derive a conservative unchanged token prefix from this exact history.
            original_ids = await self.tokenizer.prompt_token_ids(prompt)
            empty = self._replace_calibration_remainder(prompt, index, "")
            empty_ids = await self.tokenizer.prompt_token_ids(empty)
            if len(original_ids) != count:
                raise ValueError(f"Tokenizer count/IDs disagree for {node.source_key}")
        if count > target:
            assert empty_ids is not None
            empty_count = len(empty_ids)
            history.append(empty_count)
            original_content_ids = await self.tokenizer.text_token_ids(original)
            candidate, count, kept, search_history = await self._find_current_user_prefix(
                prompt, target, empty_count, node.source_key
            )
            history.extend(search_history)
        padding_base_count = count
        attempts = requested = 0
        if count < target:
            candidate, count, attempts, requested, repair_history = await self._repair_calibration_suffix(
                candidate,
                index,
                kept,
                f"trace-record:{node.runtime_request_id}",
                target,
                count,
                allow_shorter=True,
            )
            history.extend(repair_history)
        if history[-1] != count:
            history.append(count)
        if count != target:
            raise ValueError(
                f"Current-turn calibration unreachable for {node.source_key}: "
                f"final_tokens={count} target={target}; frozen history preserved"
            )
        preserved = None
        if original_ids is not None:
            assert empty_ids is not None
            preserved = await self._verify_frozen_token_prefix(
                prompt, candidate, original_ids, empty_ids, count, node.source_key
            )
        trimmed_characters = len(original) - len(kept)
        trimmed_tokens = 0
        if trimmed_characters:
            assert original_content_ids is not None
            trimmed_tokens = max(0, len(original_content_ids) - len(await self.tokenizer.text_token_ids(kept)))
        padded = count > padding_base_count
        adjustment = (
            "trim_and_pad"
            if trimmed_characters and padded
            else ("trim" if trimmed_characters else "pad" if padded else "none")
        )
        calibration = PromptCalibration(
            target_tokens=target,
            initial_tokens=initial_count,
            final_tokens=count,
            added_filler_tokens=requested,
            requested_filler_tokens=requested,
            actual_prompt_token_gain=max(0, count - padding_base_count),
            trimmed_filler_tokens=0,
            residual_tokens=count - target,
            target_met=count == target,
            accepted_with_tolerance=False,
            repair_attempts=attempts,
            count_history=tuple(history),
            adjustment=adjustment,
            trimmed_current_user_tokens=trimmed_tokens,
            trimmed_current_user_characters=trimmed_characters,
            preserved_prefix_tokens=preserved,
        )
        logger.info(
            "Current-turn calibration: source_key=%s initial=%d final=%d target=%d "
            "trimmed_user_characters=%d padding_gain=%d preserved_prefix_tokens=%s",
            node.source_key,
            initial_count,
            count,
            target,
            trimmed_characters,
            calibration.actual_prompt_token_gain,
            preserved,
        )
        return replace(candidate, calibration=calibration)

    async def _find_current_user_prefix(
        self, prompt: SyntheticPrompt, target: int, empty_count: int, source_key: str
    ) -> tuple[SyntheticPrompt, int, str, tuple[int, ...]]:
        """Find the longest fitting strict prefix after the original exceeds target."""

        index = len(prompt.messages) - 1
        original = prompt.messages[index]["content"]
        assert prompt.messages[index]["role"] == "user" and isinstance(original, str)
        history: list[int] = []
        # Counts can fall when a longer prefix changes tokenization or the
        # chat template. Check every longer prefix before accepting a shorter
        # one; neither a raw-token hint nor an over-budget empty user bounds
        # the search. Character slices always preserve literal source text.
        for length in range(len(original) - 1, -1, -1):
            text = original[:length]
            trial = self._replace_calibration_remainder(prompt, index, text)
            trial_count = empty_count if length == 0 else await self.tokenizer.count(trial)
            history.append(trial_count)
            if trial_count <= target:
                return trial, trial_count, text, tuple(history)
        raise ValueError(
            f"Current-turn calibration unreachable for {source_key}: "
            f"no current-user prefix fits target={target}; "
            f"frozen history with empty user needs {empty_count} tokens; "
            "historical messages will not be trimmed"
        )

    async def _verify_frozen_token_prefix(
        self,
        original: SyntheticPrompt,
        candidate: SyntheticPrompt,
        original_ids: tuple[int, ...],
        empty_ids: tuple[int, ...],
        count: int,
        source_key: str,
    ) -> int:
        """Validate the calibrated count and frozen tokens, returning the preserved length."""

        boundary_ids = empty_ids
        if original_ids == empty_ids:
            # An empty (or template-stripped) user must not lock the generation suffix.
            boundary = self._replace_calibration_remainder(
                original, len(original.messages) - 1, "AgentInfer boundary probe"
            )
            boundary_ids = await self.tokenizer.prompt_token_ids(boundary)
        locked = 0
        for left, right in zip(original_ids, boundary_ids, strict=False):
            if left != right:
                break
            locked += 1
        final_ids = await self.tokenizer.prompt_token_ids(candidate)
        if len(final_ids) != count or final_ids[:locked] != original_ids[:locked]:
            raise ValueError(f"Current-turn calibration changed frozen token prefix for {source_key}")
        preserved = 0
        for left, right in zip(original_ids, final_ids, strict=False):
            if left != right:
                break
            preserved += 1
        return preserved

    async def _continuation_prompt(
        self,
        node: ReplayPlanNode,
        namespace: str,
        context: PromptExchange,
    ) -> SyntheticPrompt:
        request_private_remainder = await self.tokenizer.token_text(namespace, 1)
        messages = list(context.prompt.messages)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": context.assistant_content}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": request_private_remainder}],
                },
            ]
        )
        if self._add_extra_system_message(node):
            tokens = (
                self.config.replay.lead_continuation_system_tokens
                if node.actor_role == "lead"
                else self.config.replay.subagent_continuation_system_tokens
            )
            family = "lead" if node.actor_role == "lead" else "subagent"
            reminder = await self.tokenizer.token_text(f"shared:{family}:continuation-system", tokens)
            if reminder:
                messages.append({"role": "system", "content": reminder})
        return SyntheticPrompt(
            context.prompt.system,
            context.prompt.tools,
            tuple(messages),
            extra_body=context.prompt.extra_body,
        )

    @staticmethod
    def _calibration_remainder_text(message: dict[str, object]) -> str | None:
        """Return the mutable request-scoped remainder from one User message."""

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    return str(block["text"])
        return None

    @staticmethod
    def _replace_calibration_remainder(
        prompt: SyntheticPrompt,
        index: int,
        text: str,
    ) -> SyntheticPrompt:
        """Replace only the final Text Block used as the calibration remainder."""

        messages = list(prompt.messages)
        message = messages[index]
        content = message.get("content")
        if isinstance(content, str):
            messages[index] = {**message, "content": text}
        elif isinstance(content, list):
            blocks = list(content)
            for block_index in range(len(blocks) - 1, -1, -1):
                block = blocks[block_index]
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks[block_index] = {**block, "text": text}
                    messages[index] = {**message, "content": blocks}
                    break
            else:
                raise ValueError("Synthetic user message has no text block for calibration")
        else:
            raise ValueError("Synthetic user message has no text content for calibration")
        return SyntheticPrompt(prompt.system, prompt.tools, tuple(messages), extra_body=prompt.extra_body)

    async def _repair_calibration_suffix(
        self,
        prompt: SyntheticPrompt,
        index: int,
        content: str,
        namespace: str,
        target: int,
        current_count: int,
        *,
        allow_shorter: bool = False,
    ) -> tuple[SyntheticPrompt, int, int, int, tuple[int, ...]]:
        """Find the closest deterministic request-private suffix.

        Candidate suffixes append to ``content`` without modifying shared Prompt
        fields. Each candidate is counted as a complete Prompt by the Backend.
        The return value contains the selected Prompt/count, attempted candidate
        count, selected nominal filler size, and all candidate count results.
        Exact matches return immediately; otherwise an under-target candidate is
        preferred over an equally close over-target candidate.
        ``allow_shorter`` also checks smaller fillers when a boundary adds tokens.
        """

        delta = target - current_count
        if delta <= 0:
            return prompt, current_count, 0, 0, ()

        best_prompt = prompt
        best_count = current_count
        best_requested_tokens = 0
        best_key = (abs(delta), current_count > target, 0, 0)
        attempts = 0
        count_history: list[int] = []
        offsets = (0, 1, -1, 2, -2, 3, -3) if allow_shorter else range(4)
        for extra_tokens in offsets:
            requested_tokens = delta + extra_tokens
            if requested_tokens <= 0:
                continue
            for variant in range(8):
                filler = await self.tokenizer.token_text(
                    f"{namespace}:repair:{variant}",
                    requested_tokens,
                )
                candidate = self._replace_calibration_remainder(prompt, index, content + filler)
                candidate_count = await self.tokenizer.count(candidate)
                attempts += 1
                count_history.append(candidate_count)
                candidate_key = (
                    abs(target - candidate_count),
                    candidate_count > target,
                    extra_tokens,
                    variant,
                )
                if candidate_key < best_key:
                    best_prompt = candidate
                    best_count = candidate_count
                    best_requested_tokens = requested_tokens
                    best_key = candidate_key
                if candidate_count == target:
                    return candidate, candidate_count, attempts, requested_tokens, tuple(count_history)
        return best_prompt, best_count, attempts, best_requested_tokens, tuple(count_history)

    async def _trim_historical_private_remainder(
        self,
        prompt: SyntheticPrompt,
        target: int,
        current_count: int,
    ) -> tuple[SyntheticPrompt, int, int]:
        """Trim historical request-scoped remainders until the target becomes reachable."""

        calibrated = prompt
        removed_tokens = 0
        user_indices = [
            index
            for index, message in enumerate(calibrated.messages)
            if message.get("role") == "user" and self._calibration_remainder_text(message) is not None
        ]
        # Preserve the current request remainder and trim the nearest historical private remainder.
        for index in reversed(user_indices[:-1]):
            message = calibrated.messages[index]
            content = self._calibration_remainder_text(message)
            assert content is not None
            content_ids = await self.tokenizer.text_token_ids(content)
            if not content_ids:
                continue
            empty = self._replace_calibration_remainder(calibrated, index, "")
            empty_count = await self.tokenizer.count(empty)
            if empty_count > target:
                calibrated = empty
                current_count = empty_count
                removed_tokens += len(content_ids)
                continue

            low = 1
            high = len(content_ids)
            best: tuple[SyntheticPrompt, int, int] | None = None
            while low <= high:
                remove = (low + high) // 2
                trimmed = await self.tokenizer.detokenize_tokens(content_ids[:-remove])
                candidate = self._replace_calibration_remainder(calibrated, index, trimmed)
                candidate_count = await self.tokenizer.count(candidate)
                if candidate_count <= target:
                    best = (candidate, candidate_count, remove)
                    high = remove - 1
                else:
                    low = remove + 1
            if best is not None:
                candidate, candidate_count, remove = best
                return candidate, candidate_count, removed_tokens + remove
        raise ValueError(f"Prompt minimum {current_count} tokens exceeds target {target}")

    async def _calibrate(
        self,
        prompt: SyntheticPrompt,
        namespace: str,
        target: int,
        *,
        context_mode: str,
    ) -> SyntheticPrompt:
        """Calibrate one Prompt to ``target`` using only mutable private text.

        The Backend count endpoint is authoritative. Normal filler is preserved
        for compatibility; repeated counts trigger deterministic suffix repair.
        A non-zero residual is returned only when it is within the configured
        tolerance, and is recorded in :class:`PromptCalibration`.
        """

        count = await self.tokenizer.count(prompt)
        initial_count = count
        calibrated = prompt
        trimmed_tokens = 0
        requested_filler_tokens = 0
        repair_attempts = 0
        count_history = [count]
        if count > target:
            overrun = count - target
            adaptive = self.config.replay.context_adjustment_mode == "adaptive"
            if not adaptive or overrun > self.config.replay.context_micro_trim_limit(target):
                raise ValueError(f"Prompt minimum {count} tokens exceeds target {target}")
            calibrated, count, trimmed_tokens = await self._trim_historical_private_remainder(
                calibrated,
                target,
                count,
            )
            count_history.append(count)
        padding_base_count = count
        stalled = False
        for _ in range(8):
            if count == target:
                break
            delta = target - count
            if delta < 0:
                break
            user_indices = [
                index
                for index, message in enumerate(calibrated.messages)
                if message.get("role") == "user" and self._calibration_remainder_text(message) is not None
            ]
            if not user_indices:
                raise ValueError("Synthetic Prompt has no user text block for calibration")
            index = user_indices[-1]
            content = self._calibration_remainder_text(calibrated.messages[index])
            assert content is not None
            if stalled:
                (
                    calibrated,
                    count,
                    repair_attempts,
                    repair_requested_tokens,
                    repair_history,
                ) = await self._repair_calibration_suffix(
                    calibrated,
                    index,
                    content,
                    namespace,
                    target,
                    count,
                )
                requested_filler_tokens += repair_requested_tokens
                count_history.extend(repair_history)
                if count_history[-1] != count:
                    count_history.append(count)
                break
            filler = await self.tokenizer.token_text(namespace, delta)
            candidate = self._replace_calibration_remainder(calibrated, index, content + filler)
            next_count = await self.tokenizer.count(candidate)
            stalled = next_count in count_history
            count_history.append(next_count)
            if next_count > target:
                (
                    calibrated,
                    count,
                    repair_attempts,
                    repair_requested_tokens,
                    repair_history,
                ) = await self._repair_calibration_suffix(
                    calibrated,
                    index,
                    content,
                    namespace,
                    target,
                    count,
                )
                requested_filler_tokens += repair_requested_tokens
                count_history.extend(repair_history)
                if count_history[-1] != count:
                    count_history.append(count)
                break
            requested_filler_tokens += delta
            calibrated = candidate
            count = next_count

        if count < target and repair_attempts == 0:
            user_indices = [
                index
                for index, message in enumerate(calibrated.messages)
                if message.get("role") == "user" and self._calibration_remainder_text(message) is not None
            ]
            if not user_indices:
                raise ValueError("Synthetic Prompt has no user text block for calibration")
            index = user_indices[-1]
            content = self._calibration_remainder_text(calibrated.messages[index])
            assert content is not None
            (
                calibrated,
                count,
                repair_attempts,
                repair_requested_tokens,
                repair_history,
            ) = await self._repair_calibration_suffix(
                calibrated,
                index,
                content,
                namespace,
                target,
                count,
            )
            requested_filler_tokens += repair_requested_tokens
            count_history.extend(repair_history)
            if count_history[-1] != count:
                count_history.append(count)

        residual_tokens = count - target
        tolerance = self.config.replay.prompt_calibration_tolerance_tokens
        if residual_tokens and abs(residual_tokens) > tolerance:
            raise ValueError(f"Prompt calibration produced {count} tokens for target {target}")
        if residual_tokens:
            logger.warning(
                "Replay Prompt calibration accepted a residual after suffix repair: "
                "namespace=%s target_tokens=%d final_tokens=%d residual_tokens=%d repair_attempts=%d",
                namespace,
                target,
                count,
                residual_tokens,
                repair_attempts,
            )
        padded = requested_filler_tokens > 0
        if context_mode == "reset":
            adjustment = "reset_and_pad" if padded else "reset"
        elif trimmed_tokens and padded:
            adjustment = "trim_and_pad"
        elif trimmed_tokens:
            adjustment = "trim"
        elif padded:
            adjustment = "pad"
        else:
            adjustment = "none"
        return SyntheticPrompt(
            calibrated.system,
            calibrated.tools,
            calibrated.messages,
            PromptCalibration(
                target_tokens=target,
                initial_tokens=initial_count,
                final_tokens=count,
                added_filler_tokens=requested_filler_tokens,
                requested_filler_tokens=requested_filler_tokens,
                actual_prompt_token_gain=max(0, count - padding_base_count),
                trimmed_filler_tokens=trimmed_tokens,
                residual_tokens=residual_tokens,
                target_met=residual_tokens == 0,
                accepted_with_tolerance=residual_tokens != 0,
                repair_attempts=repair_attempts,
                count_history=tuple(count_history),
                adjustment=adjustment,
            ),
            calibrated.extra_body,
        )
