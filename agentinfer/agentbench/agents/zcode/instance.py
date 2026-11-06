# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Create task-local ZCode model, runtime and hook configuration."""

import json
import os
import sys
from pathlib import Path

from ..contracts import AgentRunRequest
from .profiles import EXPLORE_TOOL_ROUNDS, explore_name, get_profile

HOOK_EVENTS = ("SessionStart", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop")


def bootstrap(request: AgentRunRequest, bridge_url: str) -> dict[str, str]:
    profile = get_profile(request.profile_name)
    home = request.artifact_dir / "zcode-home"
    storage = home / ".zcode" / "cli"
    storage.mkdir(parents=True, exist_ok=True)
    if profile.planning:
        agents = storage / "agents"
        agents.mkdir(exist_ok=True)
        (agents / "explore.md").write_text(
            f"---\nname: {explore_name(request.session_id)}\ndescription: Read-only repository exploration for AgentBench.\n"
            "tools: [Read, Glob, Grep]\npermissionMode: plan\n---\n"
            "Search first, read at most 200 lines per call, and return concise findings (at most 1000 characters). "
            f"You have at most {EXPLORE_TOOL_ROUNDS} tool rounds; finish with the findings collected so far. "
            "Do not edit files, run commands, or delegate.\n"
        )
    provider = {
        "schemaVersion": 1,
        "config": {
            "providerConfigRules": {
                "providerRules": [
                    {
                        "providerId": "agentbench",
                        "providerName": "AgentBench",
                        "config": {
                            "group": "standard-personal",
                            "access": {"type": "api-key", "apiKey": "agentbench"},
                            "api": {"type": "openai-chat-completions", "baseUrl": f"{bridge_url}/v1"},
                            "personalModelIds": [request.model],
                        },
                    }
                ],
            },
            "modelConfigRules": {"providerModelRules": [], "manualProviderModelRules": []},
            "defaultModelSelection": {
                "providerId": "agentbench",
                "modelId": request.model,
                "options": {"reasoningLevel": "disabled"},
            },
        },
    }
    # Explicit built-in rules pin model limits and disable remote catalogue refresh.
    builtin = {
        "schemaVersion": 1,
        "revision": 0,
        "config": {
            "providerConfigRules": {"templateRules": [], "providerRules": []},
            "modelConfigRules": {
                "modelRules": [
                    {
                        "modelMatch": ".*",
                        "config": {
                            "enabled": True,
                            "properties": {
                                "contextWindow": request.context_window,
                                "requiresMfjsToolSchema": False,
                                "supportsToolCall": True,
                                "supportsJsonSchemaOutput": False,
                                "supportsNativeWebSearch": False,
                                "supportsMidConversationSystem": False,
                                "inputFormat": {
                                    "supportsText": True,
                                    "supportsImage": False,
                                    "supportsVideo": False,
                                    "supportsAudio": False,
                                    "supportsPdf": False,
                                },
                                "outputFormat": {"supportsText": True},
                            },
                            "optionSpecs": {
                                "maxOutputTokens": {
                                    "max": request.max_tokens,
                                    "map": "{'max_tokens': maxOutputTokens}",
                                },
                                "reasoningLevel": {"values": ["disabled"], "map": "{}"},
                            },
                        },
                    }
                ],
                "modelApiRules": [],
                "providerSiteRules": [],
                "templateModelRules": [],
                "builtinProviderModelRules": [],
            },
        },
    }
    hook = {
        "type": "process",
        "command": sys.executable,
        "args": [str(Path(__file__).with_name("hook.py")), f"{bridge_url}/hook"],
        "timeoutMs": 15000,
        "onError": "deny",
    }
    config = {
        "features": {"mcp": False, "subagent": profile.planning, "memory": False, "skill": False},
        "memory": {"use": False},
        "plugins": {"enabled": False},
        "skills": {"enabled": False},
        "hooks": {"enabled": True, "events": {event: [{"hooks": [hook]}] for event in HOOK_EVENTS}},
    }
    provider_path = home / "provider_config.json"
    provider_path.write_text(json.dumps(provider, indent=2) + "\n")
    builtin_path = home / "builtin_config.json"
    builtin_path.write_text(json.dumps(builtin, indent=2) + "\n")
    (storage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (request.artifact_dir / "zcode-config.json").write_text(
        json.dumps({"provider": provider, "builtin": builtin, "runtime": config}, indent=2) + "\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"}
    }
    temporary = home / "tmp"
    temporary.mkdir(exist_ok=True)
    return {
        **environment,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "TMPDIR": str(temporary),
        "ZCODE_DATA_BASE_DIR": str(home),
        "ZCODE_STORAGE_DIR": str(storage),
        "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE": str(provider_path),
        "ZCODE_BUILTIN_PROVIDER_CONFIG_FILE": str(builtin_path),
        "ZCODE_LOG_DIR": str(home / "logs"),
        "ZCODE_SESSION_DB_PATH": str(storage / "db.sqlite"),
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
