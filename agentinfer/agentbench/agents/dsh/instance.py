# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare one task-local DeepSeek Harness home and settings document.

Each benchmark task gets a private DSH_HOME so settings, credentials, and
session logs can never leak across tasks. The headless profile auto-initializes
under that home on first boot; the settings document routes the agent's model
to the benchmark request proxy through an OpenAI-compatible provider route.
"""

import json
import os
from pathlib import Path

import yaml

from .bridge import AGENTBENCH_BRIDGE

HOME_DIR_NAME = "dsh-home"
SETTINGS_FILENAME = "settings.yaml"
POLICY_PATCH_FILENAME = "agentbench.patch.yml"
BRIDGE_FILENAME = "agentbench-bridge.mjs"
DEFAULT_CONTEXT_WINDOW = 262144
DEFAULT_MAX_TOKENS = 8192

_HOME_ENV = "DSH_HOME"
_PERMISSION_MODE_ENV = "DSH_PERMISSION_MODE"
_PERMISSION_MODE_VALUE = "workspace-write"
_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"
_API_KEY_PLACEHOLDER = "agentbench"
_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
_ROOT_SESSION_ENV = "AGENTBENCH_ROOT_SESSION_ID"
_TITLE_PATCH = "- id: session-title-llm\n  disabled: true\n"


def build_dsh_settings(api_base_url: str, model: str) -> dict[str, object]:
    """Build settings for DSH's DeepSeek provider route."""

    return {
        "agent-default-model": {"provider": "deepseek-official", "model": model},
        "llm-deepseek": {
            "apiKeyEnv": _API_KEY_ENV_NAME,
            "baseURL": f"{api_base_url.rstrip('/')}/v1",
            "thinking": "disabled",
            "maxTokens": DEFAULT_MAX_TOKENS,
            "retryPolicy": {"mode": "normal", "maxRetries": 0},
            "models": [{"id": model, "name": model, "contextWindow": DEFAULT_CONTEXT_WINDOW}],
        },
    }


class DshInstance:
    """Prepare the isolated home, settings document, and child environment for one task.

    The settings document is the adapter-owned config surface: it selects the
    default model and declares the provider route, while the child environment
    covers the permission mode, credential placeholders, and terminal pinning.
    """

    def __init__(
        self,
        *,
        artifact_dir: Path,
        api_base_url: str,
        model: str,
        session_id: str,
        policy_patch: str = "",
        permission_mode: str = _PERMISSION_MODE_VALUE,
        enforce_plan_mode: bool = False,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.session_id = session_id
        self.policy_patch = policy_patch
        self.permission_mode = permission_mode
        self.enforce_plan_mode = enforce_plan_mode
        self.home_dir = artifact_dir / HOME_DIR_NAME
        self.settings_path = self.home_dir / SETTINGS_FILENAME
        self.policy_patch_path = self.home_dir / POLICY_PATCH_FILENAME
        self.bridge_path = self.home_dir / BRIDGE_FILENAME
        self.artifact_settings_path = artifact_dir / SETTINGS_FILENAME
        self.snapshot_path = artifact_dir / "dsh-settings-snapshot.json"
        self.settings = build_dsh_settings(api_base_url, model)

    def bootstrap(self) -> None:
        """Create the task home and write settings plus redacted evidence files."""

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.home_dir.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(self.settings, allow_unicode=True, sort_keys=False)
        self.settings_path.write_text(serialized, encoding="utf-8")
        self.bridge_path.write_text(AGENTBENCH_BRIDGE, encoding="utf-8")
        # An absolute-path `name:` makes the loader import this file as an
        # ESM module (file:// specifier); relative paths resolve against the
        # headless profile dir, which this task-local home is not. Forward
        # slashes keep the YAML scalar free of backslash escapes on Windows.
        bridge_name = str(self.bridge_path).replace("\\", "/")
        bridge_patch = (
            "\n- insert:\n"
            "    - id: agentbench-bridge\n"
            f"      name: {bridge_name}\n"
            "      config:\n"
            f"        forcePlanMode: {str(self.enforce_plan_mode).lower()}\n"
            f"        autoApprove: {str(self.enforce_plan_mode).lower()}\n"
        )
        self.policy_patch_path.write_text(_TITLE_PATCH + self.policy_patch + bridge_patch, encoding="utf-8")
        self.artifact_settings_path.write_text(serialized, encoding="utf-8")
        environment = self.environment()
        snapshot = {
            "settings": self.settings,
            "environment": {
                key: environment[key]
                for key in (
                    _HOME_ENV,
                    _PERMISSION_MODE_ENV,
                    _BASE_URL_ENV,
                    _ROOT_SESSION_ENV,
                )
            },
            "policy_patch": self.policy_patch + bridge_patch,
        }
        self.snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    def environment(self) -> dict[str, str]:
        """Build the isolated child-process environment for this task."""

        values = {
            _HOME_ENV: str(self.home_dir),
            _PERMISSION_MODE_ENV: self.permission_mode,
            _API_KEY_ENV_NAME: _API_KEY_PLACEHOLDER,
            _BASE_URL_ENV: f"{self.api_base_url}/v1",
            _ROOT_SESSION_ENV: self.session_id,
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
        }
        return {**os.environ, **values}
