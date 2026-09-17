# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import sys
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.local_tokenizer import LocalTokenizerCounter, discover_local_tokenizer


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [ord(character) for character in text]


class _AdditiveTokenizer:
    chat_template = "test-template"

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        assert add_special_tokens is False
        return _Encoding(text)

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        rendered = "<begin>" + "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        return rendered + ("<assistant>" if kwargs["add_generation_prompt"] else "")


class _HistorySensitiveTokenizer(_AdditiveTokenizer):
    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        normalized = []
        for index, message in enumerate(messages):
            content = str(message["content"])
            if message["role"] == "assistant" and index < len(messages) - 1 and "</think>" in content:
                content = content.split("</think>", 1)[1]
            normalized.append({"role": message["role"], "content": content})
        return super().apply_chat_template(normalized, **kwargs)


def _config() -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "backend": {
                "base_url": "http://backend",
                "tokenizer_base_url": "http://tokenizer",
                "model": "served-alias",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "replay": {"trace_path": "source.jsonl"},
        }
    )


def test_local_counter_selects_incremental_only_for_additive_template() -> None:
    additive = LocalTokenizerCounter(_AdditiveTokenizer(), {})
    history_sensitive = LocalTokenizerCounter(_HistorySensitiveTokenizer(), {})
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]

    assert additive.incremental is True
    assert additive.input_tokens(messages) == len(additive.chat_token_ids(messages))
    assert history_sensitive.incremental is False
    assert history_sensitive.input_tokens(messages) == len(history_sensitive.chat_token_ids(messages))


@pytest.mark.parametrize("message_count", [7, 11, 13])
def test_unprobed_history_lengths_use_full_template(message_count) -> None:
    class LaterHistorySensitiveTokenizer(_HistorySensitiveTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == message_count:
                return _HistorySensitiveTokenizer.apply_chat_template(self, messages, **kwargs)
            return _AdditiveTokenizer.apply_chat_template(self, messages, **kwargs)

    counter = LocalTokenizerCounter(LaterHistorySensitiveTokenizer(), {})
    assert counter.incremental is True
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": "ask" if index % 2 == 0 else "<think>hidden</think>answer",
        }
        for index in range(message_count)
    ]
    expected = len(counter.chat_token_ids(messages))
    assert counter.input_tokens(messages) == expected
    # Demonstrate that extrapolating the offset would produce a wrong target.
    assert counter._incremental_count(messages) != expected


def test_discovery_uses_backend_model_root_and_requires_matching_token_ids(monkeypatch, tmp_path) -> None:
    tokenizer = _AdditiveTokenizer()
    loaded: list[tuple[str, bool]] = []
    model_root = tmp_path / "actual-model"
    model_root.mkdir()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(candidate: str, *, local_files_only: bool) -> _AdditiveTokenizer:
            loaded.append((candidate, local_files_only))
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))

    def get_json(url: str) -> dict[str, object] | None:
        if url == "http://tokenizer/tokenizer_info":
            return {"tokenizer_class": "Fake", "chat_template": "backend-template"}
        if url == "http://tokenizer/v1/models":
            return {"data": [{"id": "served-alias", "root": str(model_root)}]}
        return None

    def backend_tokens(payload: dict[str, object]) -> tuple[int, ...]:
        if "prompt" in payload:
            text = str(payload["prompt"])
        else:
            messages = payload["messages"]
            assert isinstance(messages, list)
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return tuple(ord(character) for character in text)

    counter = discover_local_tokenizer(_config(), get_json=get_json, backend_tokens=backend_tokens)

    assert counter is not None
    assert loaded == [(str(model_root), True)]
    assert tokenizer.chat_template == "backend-template"


def test_discovery_rejects_tokenizer_that_differs_from_backend(monkeypatch) -> None:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(candidate: str, *, local_files_only: bool) -> _AdditiveTokenizer:
            return _AdditiveTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))

    counter = discover_local_tokenizer(
        ReplayBenchConfig.model_validate(
            {"backend": {"model": "different-model"}, "replay": {"trace_path": "source.jsonl"}}
        ),
        get_json=lambda url: None,
        backend_tokens=lambda payload: (999,),
    )

    assert counter is None
