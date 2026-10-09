"""Codex installation assets: placement, merge, idempotency and exact restore."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vllm_evolve.install import codex_installer


def test_install_places_local_frontier_codex_assets(tmp_path):
    result = codex_installer.install(tmp_path)
    assert result.status == "installed"
    skill = tmp_path / ".agents" / "skills" / "ve-evolve" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "frontier-evolve" in text and "Do not invoke SSH" in text
    assert (tmp_path / ".agents" / "skills" / "ve-context" / "SKILL.md").is_file()
    assert (tmp_path / ".agents" / "skills" / "ve-design" / "SKILL.md").is_file()
    assert (tmp_path / ".agents" / "skills" / "ve-generate" / "SKILL.md").is_file()
    assert (tmp_path / ".agents" / "skills" / "ve-verify" / "SKILL.md").is_file()
    assert (tmp_path / ".agents" / "skills" / "ve-research" / "SKILL.md").is_file()
    assert (tmp_path / ".codex" / "agents" / "vllm-policy-optimizer.toml").is_file()
    assert (tmp_path / ".codex" / "hooks" / "phase_guard_hook.py").is_file()
    assert (tmp_path / ".codex" / "hooks.json").is_file()
    assert (tmp_path / ".codex" / ".vllm-evolve-install.json").is_file()
    assert "vllm-evolve-codex:start" in (
        tmp_path / "AGENTS.md"
    ).read_text(encoding="utf-8")


def test_installed_agents_requires_persistent_execution_until_hard_boundary(tmp_path):
    codex_installer.install(tmp_path)
    text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Execution persistence" in text
    assert "ends that attempt, not the user's goal" in normalized
    assert "all safe in-scope alternatives have been exhausted" in normalized
    assert "Do not pause to ask whether to continue" in normalized
    assert "Never bypass a phase" in normalized


def test_codex_installer_has_no_claude_asset_dependency():
    source = Path(codex_installer.__file__).read_text(encoding="utf-8")
    assert "assets/claude" not in source
    assert "_claude_assets" not in source


def test_install_merges_existing_hooks_and_is_idempotent(tmp_path):
    hooks = tmp_path / ".codex" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({
        "description": "existing",
        "hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": []}]},
    }), encoding="utf-8")
    codex_installer.install(tmp_path)
    merged = json.loads(hooks.read_text(encoding="utf-8"))
    assert merged["description"] == "existing"
    assert merged["hooks"]["PostToolUse"]
    assert "phase_guard_hook.py" in (
        merged["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert codex_installer.install(tmp_path).status == "already_installed"
    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))


def test_uninstall_exactly_restores_touched_files(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing agent guide\n", encoding="utf-8")
    skill = tmp_path / ".agents" / "skills" / "ve-context" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("existing skill\n", encoding="utf-8")
    hooks = tmp_path / ".codex" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    original_hooks = json.dumps({"hooks": {"Stop": []}}, indent=2)
    hooks.write_text(original_hooks, encoding="utf-8")

    codex_installer.install(tmp_path)
    result = codex_installer.uninstall(tmp_path)

    assert result["status"] == "uninstalled"
    assert agents.read_text(encoding="utf-8") == "# Existing agent guide\n"
    assert skill.read_text(encoding="utf-8") == "existing skill\n"
    assert hooks.read_text(encoding="utf-8") == original_hooks
    assert not (tmp_path / ".agents" / "skills" / "ve-design").exists()
    assert not (tmp_path / ".codex" / "agents").exists()


def test_installed_hook_blocks_codex_apply_patch(tmp_path):
    codex_installer.install(tmp_path)
    hook = tmp_path / ".codex" / "hooks" / "phase_guard_hook.py"
    payload = {
        "tool_name": "apply_patch",
        "tool_input": {
            "patch": "*** Begin Patch\n"
                     "*** Update File: targets/scheduling/seed.py\n"
                     "@@\n-old\n+new\n"
                     "*** End Patch\n",
        },
    }
    result = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "blocked" in result.stderr


def test_invalid_existing_hooks_are_not_clobbered(tmp_path):
    hooks = tmp_path / ".codex" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text("{broken", encoding="utf-8")
    try:
        codex_installer.install(tmp_path)
    except ValueError as exc:
        assert "invalid existing Codex hooks" in str(exc)
    else:
        raise AssertionError("invalid hooks.json should fail")
    assert hooks.read_text(encoding="utf-8") == "{broken"
    assert not (tmp_path / ".agents").exists()
    assert not (tmp_path / ".codex" / "agents").exists()
