"""Shared Claude request header parsing and Router forwarding policy."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

CLAUDE_SESSION_HEADER = "X-Claude-Code-Session-Id"
CLAUDE_AGENT_HEADER = "X-Claude-Code-Agent-Id"


@dataclass(frozen=True)
class AgentRequestIdentity:
    """Represent Claude actor identity observed at the benchmark boundary."""

    session_id: str | None
    actor_id: str
    actor_role: Literal["lead", "subagent", "unknown"]


def _value(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    value = next((value for key, value in headers.items() if key.lower() == wanted), None)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def parse_agent_identity(headers: Mapping[str, str]) -> AgentRequestIdentity:
    """Parse Claude session and agent headers for local benchmark tracing."""

    session_id = _value(headers, CLAUDE_SESSION_HEADER)
    agent_id = _value(headers, CLAUDE_AGENT_HEADER)
    if agent_id:
        return AgentRequestIdentity(session_id, agent_id, "subagent")
    if session_id:
        return AgentRequestIdentity(session_id, "lead", "lead")
    return AgentRequestIdentity(None, "unknown", "unknown")


def select_router_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Select non-empty Claude identity headers for upstream forwarding."""

    selected = {}
    for name in (CLAUDE_SESSION_HEADER, CLAUDE_AGENT_HEADER):
        if value := _value(headers, name):
            selected[name] = value
    return selected
