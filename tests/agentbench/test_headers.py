# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from agentinfer.scheduling.headers import (
    CLAUDE_AGENT_HEADER,
    CLAUDE_SESSION_HEADER,
    parse_agent_identity,
    select_router_headers,
)


def test_router_receives_session_and_agent_headers() -> None:
    headers = {CLAUDE_SESSION_HEADER: "session", CLAUDE_AGENT_HEADER: "sub-1", "Authorization": "secret"}

    assert select_router_headers(headers) == {
        CLAUDE_SESSION_HEADER: "session",
        CLAUDE_AGENT_HEADER: "sub-1",
    }
    identity = parse_agent_identity(headers)
    assert (identity.session_id, identity.actor_id, identity.actor_role) == ("session", "sub-1", "subagent")


def test_session_without_agent_is_local_lead() -> None:
    headers = {CLAUDE_SESSION_HEADER.lower(): " session "}
    identity = parse_agent_identity(headers)

    assert identity.session_id == "session"
    assert identity.actor_id == "lead"
    assert identity.actor_role == "lead"
    assert select_router_headers(headers) == {CLAUDE_SESSION_HEADER: "session"}


def test_agent_without_session_is_observed_and_forwarded() -> None:
    headers = {CLAUDE_AGENT_HEADER.swapcase(): " sub-1 "}

    identity = parse_agent_identity(headers)
    assert (identity.session_id, identity.actor_id, identity.actor_role) == (None, "sub-1", "subagent")
    assert select_router_headers(headers) == {CLAUDE_AGENT_HEADER: "sub-1"}


def test_empty_identity_values_are_absent() -> None:
    headers = {CLAUDE_SESSION_HEADER: "  ", CLAUDE_AGENT_HEADER: ""}

    identity = parse_agent_identity(headers)
    assert (identity.session_id, identity.actor_id, identity.actor_role) == (None, "unknown", "unknown")
    assert select_router_headers(headers) == {}
