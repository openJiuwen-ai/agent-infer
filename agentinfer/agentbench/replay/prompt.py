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
from dataclasses import dataclass
from typing import Literal

import httpx

from .config import ReplayBenchConfig
from .planner import ReplayPlanNode, ReplayTaskPlan
from .tokenizer_retry import TOKENIZER_REQUEST_MAX_ATTEMPTS, TOKENIZER_RETRY_BACKOFF_SECONDS
from .unified_trace_ir import PromptReference, TraceTextStore, UnifiedTraceIR

logger = logging.getLogger(__name__)

SystemContent = str | tuple[dict[str, object], ...]


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


@dataclass(frozen=True)
class SyntheticPrompt:
    """Represent one calibrated synthetic Prompt in Backend-neutral form."""

    system: SystemContent
    tools: tuple[dict[str, object], ...]
    messages: tuple[dict[str, object], ...]
    calibration: PromptCalibration | None = None
    extra_body: dict[str, object] | None = None

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

    def tokenizer_payload(self, model: str) -> dict[str, object]:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + list(self.messages)
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if self.tools:
            payload["tools"] = list(self.tools)
        return payload


@dataclass(frozen=True)
class PromptExchange:
    prompt: SyntheticPrompt
    assistant_content: str


class TokenizerClient:
    """Build deterministic filler text and count its serialized Prompt tokens."""

    def __init__(self, config: ReplayBenchConfig) -> None:
        self.model = config.backend.model
        self.endpoint = config.backend.endpoint
        self.base_url = config.backend.resolved_tokenizer_base_url.rstrip("/")
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

    async def close(self) -> None:
        await self.client.aclose()

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
        response = await self._post("/tokenize", json=prompt.tokenizer_payload(self.model))
        response.raise_for_status()
        return int(response.json()["count"])

    async def text_token_ids(self, text: str) -> tuple[int, ...]:
        response = await self._post(
            "/tokenize",
            json={"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        response.raise_for_status()
        return tuple(response.json()["tokens"])

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        if not tokens:
            return ""
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

    async def token_text(self, namespace: str, count: int) -> str:
        if count == 0:
            return ""
        cache_key = (namespace, count)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        ids = self._token_ids.get(namespace)
        if ids is None:
            material = hashlib.sha256(namespace.encode()).hexdigest()
            ids = await self.text_token_ids(f" {material}")
            self._token_ids[namespace] = ids
        tokens = [ids[index % len(ids)] for index in range(count)]
        text = await self.detokenize_tokens(tokens)
        self._text_cache[cache_key] = text
        return text


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
        if config.replay.prompt_shape == "trace_record" and trace_ir is None:
            raise ValueError("trace_record Prompt construction requires a validated unified Trace IR")
        self.trace_texts = TraceTextStore(trace_ir) if trace_ir is not None else None

    async def build(
        self,
        task: ReplayTaskPlan,
        node: ReplayPlanNode,
        context: PromptExchange | None,
    ) -> SyntheticPrompt:
        """Build one node Prompt from its recipe, context mode, and prior exchange."""

        if self.config.replay.prompt_shape == "trace_record":
            return await self._trace_record_prompt(node, context)
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
        elif node.prompt_kind == "subagent_first":
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
            session_scope = f"session:{task.runtime_session_id}:lead"
        else:
            system_prefix = self.config.replay.subagent_1st_sys_shared_prefix
            session_system_prefix = self.config.replay.subagent_1st_sys_session_prefix
            tool_prefix = self.config.replay.subagent_tool_1st_shared_prefix
            message_prefix = self.config.replay.subagent_1st_msg_shared_prefix
            family = "subagent"
            session_scope = f"actor:{task.runtime_session_id}:{node.actor_id}"
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
        """Rebuild an unmodified text turn and audit its input token count."""

        assert self.trace_texts is not None
        reference = node.prompt_ref
        if not isinstance(reference, PromptReference):
            raise ValueError(f"trace_record request {node.source_key} has no valid prompt_ref")
        human_content = self.trace_texts.read(reference)
        user_message: dict[str, object] = {"role": "user", "content": human_content}
        if context is None:
            prompt = SyntheticPrompt("", (), (user_message,))
        else:
            messages = list(context.prompt.messages)
            assistant: dict[str, object] = {"role": "assistant", "content": context.assistant_content}
            messages.extend((assistant, user_message))
            prompt = SyntheticPrompt("", (), tuple(messages))

        assert node.planned_input_tokens is not None
        count = await self.tokenizer.count(prompt)
        residual = count - node.planned_input_tokens
        tolerance = self.config.replay.prompt_calibration_tolerance_tokens
        if abs(residual) > tolerance:
            logger.warning(
                "trace_record Prompt exceeds the residual audit tolerance without modifying source text: "
                "source_key=%s target_tokens=%d final_tokens=%d residual_tokens=%d tolerance_tokens=%d",
                node.source_key,
                node.planned_input_tokens,
                count,
                residual,
                tolerance,
            )
        calibration = PromptCalibration(
            target_tokens=node.planned_input_tokens,
            initial_tokens=count,
            final_tokens=count,
            added_filler_tokens=0,
            requested_filler_tokens=0,
            actual_prompt_token_gain=0,
            trimmed_filler_tokens=0,
            residual_tokens=residual,
            target_met=residual == 0,
            accepted_with_tolerance=residual != 0 and abs(residual) <= tolerance,
            repair_attempts=0,
            count_history=(count,),
            adjustment="none",
        )
        return SyntheticPrompt(prompt.system, prompt.tools, prompt.messages, calibration, prompt.extra_body)

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
    ) -> tuple[SyntheticPrompt, int, int, int, tuple[int, ...]]:
        """Find the closest deterministic request-private suffix.

        Candidate suffixes append to ``content`` without modifying shared Prompt
        fields. Each candidate is counted as a complete Prompt by the Backend.
        The return value contains the selected Prompt/count, attempted candidate
        count, selected nominal filler size, and all candidate count results.
        Exact matches return immediately; otherwise an under-target candidate is
        preferred over an equally close over-target candidate.
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
        for extra_tokens in range(4):
            requested_tokens = delta + extra_tokens
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
