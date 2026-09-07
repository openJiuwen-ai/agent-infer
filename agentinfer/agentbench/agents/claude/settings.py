# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare Claude Code state and launch environment."""

import json
import logging
import os
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


def build_claude_settings(api_base_url: str, model: str) -> dict[str, object]:
    """Build the non-secret Claude settings.json payload for a benchmark run."""

    return {
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
