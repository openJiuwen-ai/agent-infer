"""Validate that the shipped Claude Code assets have correct frontmatter."""
from __future__ import annotations

import yaml

from vllm_evolve.install.installer import _assets_dir

ASSETS = _assets_dir()
_SKILLS = ["ve-context", "ve-design", "ve-generate", "ve-verify", "ve-bench", "ve-decide",
           "ve-route", "ve-tune", "ve-port"]


def _frontmatter(path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path} has no YAML frontmatter"
    _, fm, _ = text.split("---", 2)
    return yaml.safe_load(fm)


def test_assets_dir_exists():
    assert ASSETS.is_dir()
    assert (ASSETS / "settings.fragment.json").exists()
    assert (ASSETS / "CLAUDE.snippet.md").exists()


def test_agent_frontmatter_valid():
    fm = _frontmatter(ASSETS / "agents" / "vllm-policy-optimizer.md")
    assert fm["name"] == "vllm-policy-optimizer"
    assert isinstance(fm.get("description"), str) and fm["description"].strip()
    assert "tools" in fm


def test_all_skills_have_valid_frontmatter_matching_dir():
    for name in _SKILLS:
        fm = _frontmatter(ASSETS / "skills" / name / "SKILL.md")
        assert fm["name"] == name, f"{name}: frontmatter name != dir name"
        assert isinstance(fm.get("description"), str) and fm["description"].strip()


def test_settings_fragment_is_valid_json_with_hook_and_permission():
    import json
    frag = json.loads((ASSETS / "settings.fragment.json").read_text(encoding="utf-8"))
    assert "Bash(ar:*)" in frag["permissions"]["allow"]
    pre = frag["hooks"]["PreToolUse"]
    assert pre and pre[0]["matcher"]


def test_claude_snippet_has_managed_markers():
    text = (ASSETS / "CLAUDE.snippet.md").read_text(encoding="utf-8")
    assert "<!-- vllm-evolve:start -->" in text
    assert "<!-- vllm-evolve:end -->" in text
