"""M3 / AC3: the ve-autopt orchestration skill ships and documents the loop — it runs the
deterministic `ve` verbs as TOOLS and spawns the ve-* sub-agents, but the orchestrator never
produces a metric or a verdict itself (the deterministic gate decides). No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

SKILL = SRC / "vllm_evolve" / "assets" / "claude" / "skills" / "ve-autopt" / "SKILL.md"


def _body() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_exists_with_frontmatter():
    text = _body()
    assert text.startswith("---") and "name: ve-autopt" in text


def test_skill_calls_the_deterministic_verbs_as_tools():
    b = _body()
    for verb in ("ve profile", "ve diagnose", "ve bench", "ve verify-gain"):
        assert verb in b, verb


def test_skill_spawns_each_subagent():
    b = _body()
    for agent in ("ve-goal", "ve-research", "ve-diagnose", "ve-author"):
        assert agent in b, agent


def test_skill_declares_the_frozen_invariant():
    b = _body()
    # the VERDICT comes from verify-gain, not the orchestrator/agent
    assert "verify-gain" in b
    assert ("you never produce it" in b.lower()) or ("you do not judge here" in b.lower())
    assert "PROPOSAL" in b or "PROPOSE" in b
    # the accept threshold is NOT taken from the agent
    assert "accept_threshold_pct" in b
    # honest conclusion when nothing wins
    assert "DoD-B" in b


def test_skill_benches_baseline_with_different_runner_and_wires_holdout():
    # a literal reader must NOT bench the candidate twice (self-comparison -> fake ~0% gain), and
    # must wire the holdout legs or verify-gain can never legitimately adopt (skeptic M3 findings).
    b = _body()
    assert "--runner strong_baseline" in b or "--runner vanilla" in b
    assert "--holdout-baseline" in b and "--holdout-candidate" in b


def test_ar_init_ships_ve_autopt_skill(tmp_path):
    from vllm_evolve.install import installer
    installer.install(tmp_path)
    assert (tmp_path / ".claude" / "skills" / "ve-autopt" / "SKILL.md").exists()
