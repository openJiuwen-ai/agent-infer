# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Tests for shared Agent identity precedence and framework inference."""

from unittest.mock import patch

import pytest

from agentinfer.scheduling import AgentIdentity, encode_agent_identity
from agentinfer.scheduling.identity import MetadataError, parse_agent_identity

pytestmark = pytest.mark.cpu_test


def test_public_identity_encoder_round_trips_explicit_taskless_context() -> None:
    identity = AgentIdentity(
        program_id="agent-sess-abc123",
        task_id=None,
        session_id="agent-sess-abc123",
        blocks_parent=False,
        expected_resume=True,
    )

    parsed = parse_agent_identity(
        vllm_xargs={"agentic_context": encode_agent_identity(identity)},
        headers={},
    )

    assert parsed == identity


def test_claude_subagent_headers_build_blocking_relationship() -> None:
    identity = parse_agent_identity(
        headers={"X-Claude-Code-Session-Id": "session-a", "X-Claude-Code-Agent-Id": "agent-b"},
    )

    assert identity is not None
    assert identity.program_id == "session-a:agent-b"
    assert identity.task_id == identity.session_id == "session-a"
    assert identity.parent_program_id == "session-a:lead"
    assert identity.blocks_parent is True
    assert identity.expected_resume is False


def test_claude_session_without_agent_defaults_to_lead() -> None:
    identity = parse_agent_identity(headers={"X-Claude-Code-Session-Id": "session-a"})

    assert identity is not None
    assert identity.program_id == "session-a:lead"
    assert identity.agent_role == "lead"
    assert identity.blocks_parent is False
    assert identity.expected_resume is True


def test_codex_session_and_thread_headers_build_minimal_blocking_relationship() -> None:
    identity = parse_agent_identity(
        headers={"session-id": "codex-session", "thread-id": "thread-7"},
    )

    assert identity is not None
    assert identity.program_id == "codex-session:thread-7"
    assert identity.task_id == identity.session_id == "codex-session"
    assert identity.agent_id == "thread-7"
    assert identity.parent_program_id == "codex-session:lead"
    assert identity.blocks_parent is True
    assert identity.expected_resume is False


def test_opencode_session_header_is_currently_session_level_lead_fallback() -> None:
    identity = parse_agent_identity(
        headers={"X-Session-Id": "opencode-session", "x-parent-session-id": "ignored-parent"},
    )

    assert identity is not None
    assert identity.program_id == "opencode-session:lead"
    assert identity.task_id == identity.session_id == "opencode-session"
    assert identity.agent_id == "lead"
    assert identity.parent_program_id is None
    assert identity.blocks_parent is False
    assert identity.expected_resume is True


def test_task_qualified_agent_ids_are_normalized_before_role_inference() -> None:
    lead = parse_agent_identity(
        vllm_xargs={"agentic_context": {"task_id": "task-a", "agent_id": "task-a:lead"}},
        headers={},
    )
    child = parse_agent_identity(
        vllm_xargs={"agentic_context": {"task_id": "task-a", "agent_id": "task-a:child"}},
        headers={},
    )

    assert lead is not None
    assert lead.program_id == "task-a:lead"
    assert lead.agent_id == lead.agent_role == "lead"
    assert lead.parent_program_id is None
    assert lead.blocks_parent is False
    assert lead.expected_resume is True
    assert child is not None
    assert child.program_id == "task-a:child"
    assert child.agent_id == "child"
    assert child.parent_program_id == "task-a:lead"


def test_self_parent_identity_is_rejected() -> None:
    with pytest.raises(MetadataError, match="parent_program_id must differ"):
        parse_agent_identity(
            vllm_xargs={
                "agentic_context": {
                    "task_id": "task-a",
                    "agent_id": "lead",
                    "parent_program_id": "task-a:lead",
                }
            },
            headers={},
        )


def test_mixed_framework_headers_do_not_form_a_cross_framework_identity() -> None:
    identity = parse_agent_identity(
        headers={"X-Session-Id": "opencode-session", "thread-id": "codex-thread"},
    )

    assert identity is not None
    assert identity.program_id == "opencode-session:lead"
    assert identity.agent_id == identity.agent_role == "lead"
    assert identity.parent_program_id is None
    assert identity.blocks_parent is False


def test_complete_framework_headers_use_deterministic_precedence() -> None:
    identity = parse_agent_identity(
        headers={
            "X-Claude-Code-Session-Id": "claude-session",
            "X-Claude-Code-Agent-Id": "claude-agent",
            "session-id": "codex-session",
            "thread-id": "codex-thread",
            "X-Session-Id": "opencode-session",
        },
    )

    assert identity is not None
    assert identity.program_id == "claude-session:claude-agent"
    assert identity.session_id == "claude-session"
    assert identity.agent_id == "claude-agent"


def test_explicit_context_precedes_framework_headers() -> None:
    identity = parse_agent_identity(
        vllm_xargs={"agentic_context": {"task_id": "body-task", "agent_id": "body-agent"}},
        headers={"X-Claude-Code-Session-Id": "header-task", "X-Claude-Code-Agent-Id": "header-agent"},
    )

    assert identity is not None
    assert identity.program_id == "body-task:body-agent"
    assert identity.task_id == "body-task"


def test_explicit_null_task_falls_back_to_session() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": {
                "task_id": None,
                "session_id": "session-a",
                "agent_id": "lead",
            }
        },
        headers={},
    )

    assert identity is not None
    assert identity.task_id == "session-a"
    assert identity.program_id == "session-a:lead"


def test_explicit_program_preserves_intentionally_null_task() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": {
                "program_id": "agent-sess-abc123",
                "task_id": None,
                "session_id": "agent-sess-abc123",
            }
        },
        headers={},
    )

    assert identity is not None
    assert identity.program_id == identity.session_id == "agent-sess-abc123"
    assert identity.task_id is None


def test_explicit_null_primary_fields_allow_alias_fallbacks() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": {
                "task_id": "task-a",
                "agent_id": "child",
                "parent_program_id": None,
                "parent_id": "task-a:parent",
                "blocks_parent": None,
                "blocking_parent": False,
            }
        },
        headers={},
    )

    assert identity is not None
    assert identity.parent_program_id == "task-a:parent"
    assert identity.blocks_parent is False


def test_explicit_null_optional_boolean_uses_inferred_default() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": {
                "task_id": "task-a",
                "agent_id": "child",
                "expected_resume": None,
            }
        },
        headers={},
    )

    assert identity is not None
    assert identity.blocks_parent is True
    assert identity.expected_resume is False


def test_explicit_program_namespace_is_preserved_for_parent() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": (
                '{"program_id":"temp:session-a:child-a","task_id":"task-a",'
                '"session_id":"session-a","agent_id":"child-a"}'
            ),
        },
        headers={},
    )

    assert identity is not None
    assert identity.program_id == "temp:session-a:child-a"
    assert identity.parent_program_id == "temp:session-a:lead"


def test_blocking_subagent_can_explicitly_expect_resume() -> None:
    identity = parse_agent_identity(
        vllm_xargs={
            "agentic_context": {
                "task_id": "task-a",
                "agent_id": "child",
                "blocks_parent": True,
                "expected_resume": True,
            }
        },
        headers={},
    )

    assert identity is not None
    assert identity.expected_resume is True


def test_parallel_subagent_can_explicitly_disable_parent_blocking() -> None:
    identity = parse_agent_identity(
        vllm_xargs={"agentic_context": {"task_id": "task-a", "agent_id": "child", "blocks_parent": False}},
        headers={},
    )

    assert identity is not None
    assert identity.blocks_parent is False
    assert identity.expected_resume is True


def test_agent_hint_uses_session_as_program_without_task_group() -> None:
    identity = parse_agent_identity(
        agent_hint={
            "session_id": "agent-sess-abc123",
            "parent_session_id": "parent-sess-001",
            "cache_control": {"type": "ephemeral"},
        },
        headers={},
    )

    assert identity is not None
    assert identity.program_id == identity.session_id == "agent-sess-abc123"
    assert identity.task_id is None
    assert identity.agent_id is None
    assert identity.parent_program_id == "parent-sess-001"
    assert identity.blocks_parent is True
    assert identity.expected_resume is False


def test_blocking_agent_hint_can_explicitly_expect_resume() -> None:
    identity = parse_agent_identity(
        agent_hint={
            "session_id": "agent-sess-abc123",
            "parent_session_id": "parent-sess-001",
            "expected_resume": True,
        },
        headers={},
    )

    assert identity is not None
    assert identity.blocks_parent is True
    assert identity.expected_resume is True


def test_agent_hint_requires_parent_when_blocking_by_default() -> None:
    with pytest.raises(MetadataError, match="blocking agent_hint requires parent_session_id"):
        parse_agent_identity(
            agent_hint={"session_id": "agent-sess-abc123"},
            headers={},
        )


def test_direct_vllm_xargs_identity_fields_are_ignored() -> None:
    assert parse_agent_identity(vllm_xargs={"program_id": "unsupported-direct-xargs"}, headers={}) is None


def test_non_framework_identity_headers_are_ignored() -> None:
    identity = parse_agent_identity(
        headers={
            "x-agentic-program-id": "unsupported-program",
            "x-task-id": "unsupported-task",
            "x-agent-id": "unsupported-agent",
        },
    )

    assert identity is None


def test_malformed_vllm_xargs_is_rejected() -> None:
    with pytest.raises(MetadataError, match="vllm_xargs must be an object"):
        parse_agent_identity(vllm_xargs="invalid", headers={})


@pytest.mark.parametrize(
    "encoded_context",
    [
        '{"program_id":"' + "x" * (16 * 1024) + '"}',
        '{"program_id":' + "9" * 5000 + "}",
        '{"program_id":"\ud800"}',
    ],
)
def test_encoded_agentic_context_has_bounded_metadata_errors(encoded_context: str) -> None:
    with pytest.raises(MetadataError, match="bounded JSON object"):
        parse_agent_identity(vllm_xargs={"agentic_context": encoded_context}, headers={})


def test_deep_encoded_agentic_context_is_rejected_without_leaking_decoder_errors() -> None:
    encoded_context = '{"program_id":' + "[" * 8000 + '"value"' + "]" * 8000 + "}"

    with pytest.raises(MetadataError):
        parse_agent_identity(vllm_xargs={"agentic_context": encoded_context}, headers={})


def test_decoder_recursion_error_is_translated_to_metadata_error() -> None:
    with (
        patch("agentinfer.scheduling.identity.json.loads", side_effect=RecursionError),
        pytest.raises(MetadataError, match="bounded JSON object"),
    ):
        parse_agent_identity(vllm_xargs={"agentic_context": "{}"}, headers={})


def test_no_identity_returns_none() -> None:
    assert parse_agent_identity(headers={}) is None
