# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise live-history calibration without rewriting cached conversation prefixes."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.prompt import PromptBuilder, SyntheticPrompt


class TextTokenizer:
    """Expose character tokens for exact, inspectable whole-prompt tests."""

    async def prompt_token_ids(self, prompt):
        return tuple(ord(c) for message in prompt.messages for c in message["content"])

    async def count(self, prompt):
        return len(await self.prompt_token_ids(prompt))

    async def text_token_ids(self, text):
        return tuple(map(ord, text))

    async def detokenize_tokens(self, tokens):
        return "".join(map(chr, tokens))

    async def token_text(self, namespace, count):
        return "p" * count


def config(tolerance=0):
    return ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_path": "source.json",
                "prompt_calibration_tolerance_tokens": tolerance,
            },
        }
    )


def calibrate(prompt, target, tokenizer=None, tolerance=0):
    async def run():
        counter = tokenizer or TextTokenizer()
        builder = PromptBuilder(config(tolerance), counter)
        node = SimpleNamespace(planned_input_tokens=target, source_key="turn", runtime_request_id="request")
        return await builder._calibrate_current_turn(prompt, node, await counter.count(prompt))

    return asyncio.run(run())


@pytest.mark.parametrize(
    "target,text,adjustment", [(18, "中文toolpp", "pad"), (13, "中文t", "trim"), (16, "中文tool", "none")]
)
def test_only_current_user_changes(target, text, adjustment):
    """Source clipping and padding preserve all old messages, including old filler/reasoning."""
    prompt = SyntheticPrompt(
        "",
        (),
        (
            {"role": "user", "content": "oldpp"},
            {"role": "assistant", "content": "reply", "reasoning_content": "retained reasoning"},
            {"role": "user", "content": "中文tool"},
        ),
    )
    original = copy.deepcopy(prompt.messages)
    result = calibrate(prompt, target)
    assert prompt.messages == original
    assert result.messages[:-1] == original[:-1]
    assert result.messages[-1]["content"] == text
    assert result.calibration.final_tokens == target
    assert result.calibration.adjustment == adjustment
    assert result.calibration.trimmed_filler_tokens == 0
    assert result.calibration.trimmed_current_user_characters == (3 if adjustment == "trim" else 0)
    if adjustment != "none":
        assert result.calibration.preserved_prefix_tokens >= 10


def test_frozen_history_over_budget_fails_without_erasing_old_padding():
    prompt = SyntheticPrompt(
        "",
        (),
        (
            {"role": "user", "content": "oldpp"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ),
    )
    with pytest.raises(ValueError, match="frozen history with empty user needs 10 tokens"):
        calibrate(prompt, 9)
    assert prompt.messages[0]["content"] == "oldpp"
    assert prompt.messages[-1]["content"] == "new"


@pytest.mark.parametrize("tolerance", [0, 1, 8])
def test_frozen_history_one_token_over_budget_is_always_rejected(tolerance):
    prompt = SyntheticPrompt(
        "",
        (),
        (
            {"role": "user", "content": "oldpp"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ),
    )
    with pytest.raises(ValueError, match="frozen history with empty user needs 10 tokens"):
        calibrate(prompt, 9, tolerance=tolerance)


def test_padding_is_not_assumed_additive_at_text_boundary():
    class MergeTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            text = "".join(message["content"] for message in prompt.messages)
            # A source/filler boundary merges two characters into one token.
            return tuple(map(ord, text.replace("xp", "x")))

    prompt = SyntheticPrompt("", (), ({"role": "user", "content": "x"},))
    result = calibrate(prompt, 4, MergeTokenizer())
    assert result.messages[-1]["content"] == "xpppp"
    assert result.calibration.final_tokens == 4
    assert result.calibration.repair_attempts > 1


@pytest.mark.parametrize("tolerance", [0, 1, 8])
def test_unreachable_token_length_never_accepts_a_residual(tolerance):
    class EvenTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            tokens = await super().prompt_token_ids(prompt)
            return tuple(token for token in tokens for _ in range(2))

    prompt = SyntheticPrompt("", (), ({"role": "user", "content": "a"},))
    with pytest.raises(ValueError, match="Current-turn calibration unreachable"):
        calibrate(prompt, 3, EvenTokenizer(), tolerance=tolerance)


def test_template_rewrite_of_old_prefix_is_rejected():
    class RewritingTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            tokens = await super().prompt_token_ids(prompt)
            if prompt.messages[-1]["content"].endswith("p"):
                return (999,) + tokens[1:]
            return tokens

    prompt = SyntheticPrompt(
        "",
        (),
        (
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ),
    )
    with pytest.raises(ValueError, match="changed frozen token prefix"):
        calibrate(prompt, 13, RewritingTokenizer())


def test_padding_overshoot_can_be_repaired_by_a_shorter_filler():
    class ExpansionTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            tokens = await super().prompt_token_ids(prompt)
            if "p" in prompt.messages[-1]["content"]:
                tokens += (999,)
            return tokens

    prompt = SyntheticPrompt("", (), ({"role": "user", "content": "abc"},))
    result = calibrate(prompt, 6, ExpansionTokenizer())
    assert result.messages[-1]["content"] == "abcpp"
    assert result.calibration.final_tokens == 6


def test_empty_current_user_can_be_padded_without_locking_generation_suffix():
    class FramedTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            text = "".join(m["content"] for m in prompt.messages) + "<assistant>"
            return tuple(map(ord, text))

    prompt = SyntheticPrompt("", (), ({"role": "user", "content": ""},))
    result = calibrate(prompt, 13, FramedTokenizer())
    assert result.messages[-1]["content"] == "pp"
    assert result.calibration.final_tokens == 13


def test_partial_token_decode_never_replaces_source_characters():
    class PartialUnicodeTokenizer(TextTokenizer):
        async def detokenize_tokens(self, tokens):
            return "中\ufffd"

    prompt = SyntheticPrompt("", (), ({"role": "user", "content": "中文测试"},))
    result = calibrate(prompt, 2, PartialUnicodeTokenizer())
    assert result.messages[-1]["content"] == "中文"
    assert result.calibration.trimmed_current_user_characters == 2


@pytest.mark.parametrize(
    "short_count,empty_count,long_count",
    [(5, 0, 4), (4, 0, 4), (5, 5, 4), (5, 0, 3)],
    ids=["over-budget-short-prefix", "exact-short-prefix", "over-budget-empty-user", "longest-prefix-needs-padding"],
)
def test_nonmonotone_counts_keep_longest_fitting_source_prefix(short_count, empty_count, long_count):
    class NonmonotoneTokenizer(TextTokenizer):
        async def prompt_token_ids(self, prompt):
            previous = tuple(ord(c) for message in prompt.messages[:-1] for c in message["content"])
            text = prompt.messages[-1]["content"]
            counts = {"": empty_count, "abc": short_count, "abcdef": long_count, "abcdefgh": 9}
            count = long_count + 1 if text == "abcdefp" else counts.get(text, len(text))
            # Empty-user template tokens differ from the source content tokens.
            token_base = 1000 if text else 2000
            return previous + tuple(token_base + index for index in range(count))

    prompt = SyntheticPrompt(
        "",
        (),
        (
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply", "reasoning_content": "preserved"},
            {"role": "user", "content": "abcdefgh"},
        ),
    )
    original = copy.deepcopy(prompt.messages)
    result = calibrate(prompt, 12, NonmonotoneTokenizer())

    assert prompt.messages == original
    assert result.messages[:-1] == original[:-1]
    assert result.messages[-1]["content"] == "abcdef" + ("p" if long_count == 3 else "")
    assert result.calibration.final_tokens == 12
    assert result.calibration.residual_tokens == 0
    assert result.calibration.trimmed_current_user_characters == 2
    assert result.calibration.trimmed_current_user_tokens == 2
    assert result.calibration.adjustment == ("trim_and_pad" if long_count == 3 else "trim")


@pytest.mark.parametrize("target,kept", [(2, "中文"), (0, "")])
def test_prefix_search_can_be_used_without_suffix_repair(target, kept):
    prompt = SyntheticPrompt("", (), ({"role": "user", "content": "中文测试"},))
    builder = PromptBuilder(config(), TextTokenizer())
    candidate, count, text, history = asyncio.run(builder._find_current_user_prefix(prompt, target, 0, "turn"))
    assert candidate.messages[-1]["content"] == text == kept
    assert count == target == history[-1]
    assert prompt.messages[-1]["content"] == "中文测试"


@pytest.mark.parametrize("old_text,count", [("old", 4), ("old", 5), ("new", 4)])
def test_frozen_prefix_verification_independently_checks_tokens_and_count(old_text, count):
    original = SyntheticPrompt("", (), ({"role": "user", "content": "old"}, {"role": "user", "content": "abc"}))
    candidate = SyntheticPrompt("", (), ({"role": "user", "content": old_text}, {"role": "user", "content": "a"}))
    builder = PromptBuilder(config(), TextTokenizer())

    async def verify():
        return await builder._verify_frozen_token_prefix(
            original, candidate, tuple(map(ord, "oldabc")), tuple(map(ord, "old")), count, "turn"
        )

    if old_text == "old" and count == 4:
        assert asyncio.run(verify()) == 4
    else:
        with pytest.raises(ValueError, match="changed frozen token prefix"):
            asyncio.run(verify())
