from __future__ import annotations

import json
import shlex
import subprocess
import sys

from vllm_evolve.install import codex_installer
from vllm_evolve.install.hook_decision import decide
from vllm_evolve.tools.phase_guard import Phase


def test_check_executes_exact_installed_hook_and_hashes_all_assets(tmp_path):
    codex_installer.install(tmp_path, python=sys.executable)
    health = codex_installer.check(tmp_path)

    assert health["health"] == "healthy", health
    assert health["hook_smoke"]["ok"] is True
    assert health["hook_smoke"]["allow_returncode"] == 0
    assert health["hook_smoke"]["deny_returncode"] == 2
    assert health["content_checks"] and all(row["ok"] for row in health["content_checks"])


def test_check_and_install_report_content_drift_instead_of_false_green(tmp_path):
    codex_installer.install(tmp_path, python=sys.executable)
    skill = tmp_path / ".agents" / "skills" / "ve-evolve" / "SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8") + "\nlocal drift\n", encoding="utf-8")

    health = codex_installer.check(tmp_path)
    assert health["health"] == "drifted"
    assert any("ve-evolve/SKILL.md" in issue for issue in health["issues"])
    assert codex_installer.install(tmp_path).status == "drifted"


def test_installed_hook_fails_closed_on_invalid_payload(tmp_path):
    codex_installer.install(tmp_path, python=sys.executable)
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    result = subprocess.run(
        shlex.split(command),
        input="not-json",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 2
    assert "guard unavailable" in result.stderr


def test_shell_and_exec_command_cannot_write_protected_paths():
    for tool_name, tool_input in (
        ("Bash", {"command": "echo hacked > targets/scheduling/seed.py"}),
        ("exec_command", {"cmd": "sed -i '' s/a/b/ config/targets/scheduling.yaml"}),
    ):
        decision = decide(tool_name, tool_input, Phase.GENERATE)
        assert decision.allow is False
        assert "protected" in decision.reason


def test_force_refresh_preserves_user_content_outside_managed_blocks(tmp_path):
    codex_installer.install(tmp_path, python=sys.executable)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# User guide\n\n" + agents.read_text(encoding="utf-8"), encoding="utf-8")
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks["hooks"]["PostToolUse"] = [{"matcher": "Bash", "hooks": []}]
    hooks_path.write_text(json.dumps(hooks), encoding="utf-8")

    result = codex_installer.install(tmp_path, force=True, python=sys.executable)
    refreshed_hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert result.status == "reinstalled"
    assert agents.read_text(encoding="utf-8").startswith("# User guide")
    assert refreshed_hooks["hooks"]["PostToolUse"]
    managed = [
        entry
        for entry in refreshed_hooks["hooks"]["PreToolUse"]
        if any("phase_guard_hook.py" in hook["command"] for hook in entry["hooks"])
    ]
    assert len(managed) == 1
