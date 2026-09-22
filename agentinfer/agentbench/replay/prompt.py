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
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from .config import ReplayBenchConfig
from .local_tokenizer import LocalTokenizerCounter
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
    trimmed_current_user_tokens: int = 0
    trimmed_current_user_characters: int = 0
    preserved_prefix_tokens: int | None = None


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


class TokenizerClient:
    """Build deterministic filler text and count its serialized Prompt tokens."""

    def __init__(self, config: ReplayBenchConfig, local_tokenizer: LocalTokenizerCounter | None = None) -> None:
        """Create a tokenizer client, optionally reusing one validated during conversion."""

        self.model = config.backend.model
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
            return await asyncio.to_thread(self.local_tokenizer.text_token_ids, text)
        response = await self._post(
            "/tokenize",
            json={"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        response.raise_for_status()
        return tuple(response.json()["tokens"])

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
        """Append a live exchange and exactly calibrate only the newest user text."""

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
                use_literal_boundaries=True,
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
        use_literal_boundaries: bool = False,
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
            fillers: list[str] = []
            # A repeated token-derived filler can merge with the source suffix
            # at a BPE boundary and add zero tokens.  These short deterministic
            # suffixes provide a small set of independently tokenized repair
            # boundaries without changing the frozen history.  They are only
            # needed for the common one-token gap; larger gaps keep the normal
            # token-derived repair path below.
            for variant in range(8):
                fillers.append(
                    await self.tokenizer.token_text(
                        f"{namespace}:repair:{variant}",
                        requested_tokens,
                    )
                )
            if use_literal_boundaries and requested_tokens == 1:
                fillers.extend((".", ",", ":", ";", "!", "?", "0", "…"))
            for variant, filler in enumerate(fillers):
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
