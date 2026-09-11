# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare Claude Code state and launch environment."""

import json
import logging
import os
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)


def bootstrap_claude_state(config_dir: Path) -> None:
    """Write both onboarding fields used across supported Claude versions."""

    (config_dir / ".claude.json").write_text(
        json.dumps(
            {
                "onboardingComplete": True,
                "hasCompletedOnboarding": True,
                "autoUpdateDisabled": True,
                "projectTrustAccepted": True,
                "theme": "dark-2",
            }
        ),
        encoding="utf-8",
    )


def build_claude_settings(
    api_base_url: str,
    model: str,
    *,
    workspace: str,
) -> dict[str, object]:
    """Build the non-secret Claude settings.json payload for a benchmark run.

    The returned payload wires Claude's native bubblewrap Bash sandbox
    (hard-failing if unavailable) and a PreToolUse fence hook that confines
    Write/Edit/MultiEdit/NotebookEdit to ``workspace``. PoC-verified: together
    these block the relative/absolute/symlink Bash and file-write escape paths
    under both ``acceptEdits`` and ``bypassPermissions`` profiles, and
    ``bypassPermissions`` does not disable hooks.
    """

    settings: dict[str, object] = {
        "env": {
            "ANTHROPIC_BASE_URL": api_base_url,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
        "skipDangerousModePermissionPrompt": True,
        "permissions": {
            "allow": ["Bash"],
            "deny": ["WebSearch", "WebFetch"],
        },
    }
    if not workspace:
        raise ValueError("workspace is required for Claude isolation")
    fence_path = Path(__file__).resolve().parent / "fence.py"
    settings["sandbox"] = {
        "enabled": True,
        "failIfUnavailable": True,
        "allowUnsandboxedCommands": False,
    }
    settings["hooks"] = {
        "PreToolUse": [
            {
                "matcher": "Edit|Write|MultiEdit|NotebookEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": (f"ALLOWED_ROOT={shlex.quote(workspace)} python3 {shlex.quote(str(fence_path))}"),
                    }
                ],
            }
        ]
    }
    return settings


def build_claude_env(
    claude_state_dir: Path,
    api_base_url: str,
    model: str,
) -> dict[str, str]:
    """Build environment variables passed to Claude Code."""

    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if token is None:
        token = "smoke"
        logger.warning("ANTHROPIC_AUTH_TOKEN is not set; using the local benchmark placeholder")
    return {
        "ANTHROPIC_BASE_URL": api_base_url,
        "ANTHROPIC_AUTH_TOKEN": token,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CONFIG_DIR": str(claude_state_dir),
    }
