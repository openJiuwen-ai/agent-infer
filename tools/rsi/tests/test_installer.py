"""Tests for the `ve init` installer: placement, deep-merge, idempotency,
uninstall restore, and the installed hook's end-to-end behaviour. No GPU."""
from __future__ import annotations

import json
import subprocess
import sys

from vllm_evolve.install import installer


def _settings(tmp):
    return tmp / ".claude" / "settings.json"


def test_install_places_all_assets(tmp_path):
    res = installer.install(tmp_path)
    assert res.status == "installed"
    c = tmp_path / ".claude"
    assert (c / "agents" / "vllm-policy-optimizer.md").exists()
    assert (c / "hooks" / "phase_guard_hook.py").exists()
    assert (c / "settings.json").exists()
    assert (c / ".vllm-evolve-install.json").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    for skill in ["ve-context", "ve-design", "ve-generate", "ve-verify", "ve-bench", "ve-decide"]:
        assert (c / "skills" / skill / "SKILL.md").exists()


def test_install_deep_merges_preserving_existing(tmp_path):
    _settings(tmp_path).parent.mkdir(parents=True)
    _settings(tmp_path).write_text(
        json.dumps({"model": "opus", "permissions": {"allow": ["Bash(git:*)"]}}),
        encoding="utf-8",
    )
    installer.install(tmp_path)
    merged = json.loads(_settings(tmp_path).read_text(encoding="utf-8"))
    assert merged["model"] == "opus"                       # preserved
    assert "Bash(git:*)" in merged["permissions"]["allow"]  # preserved
    assert "Bash(ar:*)" in merged["permissions"]["allow"]   # added
    pre = merged["hooks"]["PreToolUse"]
    assert pre and "phase_guard_hook.py" in pre[0]["hooks"][0]["command"]


def test_idempotent(tmp_path):
    installer.install(tmp_path)
    n_before = len(list((tmp_path / ".claude").rglob("*")))
    res2 = installer.install(tmp_path)
    assert res2.status == "already_installed"
    assert len(list((tmp_path / ".claude").rglob("*"))) == n_before


def test_uninstall_restores_existing_settings_and_claude_md(tmp_path):
    _settings(tmp_path).parent.mkdir(parents=True)
    original = json.dumps({"model": "opus", "permissions": {"allow": ["Bash(git:*)"]}}, indent=2)
    _settings(tmp_path).write_text(original, encoding="utf-8")
    original_md = "# My project\n\nExisting notes.\n"
    (tmp_path / "CLAUDE.md").write_text(original_md, encoding="utf-8")

    installer.install(tmp_path)
    installer.uninstall(tmp_path)

    assert _settings(tmp_path).read_text(encoding="utf-8") == original   # exact restore
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == original_md
    assert not (tmp_path / ".claude" / "agents" / "vllm-policy-optimizer.md").exists()
    assert installer.check(tmp_path)["installed"] is False


def test_uninstall_deletes_when_settings_absent(tmp_path):
    installer.install(tmp_path)
    installer.uninstall(tmp_path)
    assert not _settings(tmp_path).exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / ".claude").exists()


def test_check_reports_status(tmp_path):
    assert installer.check(tmp_path)["installed"] is False
    installer.install(tmp_path)
    chk = installer.check(tmp_path)
    assert chk["installed"] is True
    assert chk["hook_present"] is True
    assert chk["settings_present"] is True


def test_installed_hook_blocks_protected_edit(tmp_path):
    installer.install(tmp_path)
    hook = tmp_path / ".claude" / "hooks" / "phase_guard_hook.py"
    deny = {"tool_name": "Edit", "tool_input": {"file_path": "targets/scheduling/seed.py"}}
    r = subprocess.run([sys.executable, str(hook)], input=json.dumps(deny),
                       text=True, capture_output=True)
    assert r.returncode == 2
    assert "blocked" in r.stderr

    allow = {"tool_name": "Bash", "tool_input": {"command": "git status"}}
    r2 = subprocess.run([sys.executable, str(hook)], input=json.dumps(allow),
                        text=True, capture_output=True)
    assert r2.returncode == 0
