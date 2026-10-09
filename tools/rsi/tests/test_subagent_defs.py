"""M1 / AC2: the four autopt CC sub-agent definitions ship with LEAST-PRIVILEGE tool scopes and a
frozen "untrusted producer" contract. The tool scoping is the anti-fabrication core of this layer:
the INTERPRETERS (goal/research/diagnose) get no Bash and no Write, so they structurally cannot
bench or author anything — only the orchestrator runs the deterministic `ar` verbs. No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

AGENTS = SRC / "vllm_evolve" / "assets" / "claude" / "agents"
SUBAGENTS = ("ve-goal", "ve-research", "ve-diagnose", "ve-author")
_EXPECT_TOOLS = {
    "ve-goal": {"Read"},
    "ve-research": {"Read", "Grep", "Glob"},
    "ve-diagnose": {"Read"},
    # Evolution v2: the author RETURNS a source string; the orchestrator holds the pen, so the
    # author needs no Write at all (strictly less privilege than before).
    "ve-author": {"Read"},
}


def _agent(name: str) -> tuple[dict, str]:
    text = (AGENTS / f"{name}.md").read_text(encoding="utf-8")
    assert text.startswith("---"), name
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm), body


def _tools(meta: dict) -> set[str]:
    return {t.strip() for t in meta["tools"].split(",")}


def test_each_subagent_has_valid_frontmatter():
    for name in SUBAGENTS:
        meta, body = _agent(name)
        assert meta["name"] == name
        assert meta["description"].strip()
        assert meta.get("tools")
        assert body.strip()


def test_tool_scopes_are_least_privilege():
    for name in SUBAGENTS:
        assert _tools(_agent(name)[0]) == _EXPECT_TOOLS[name], name


def test_interpreters_cannot_run_commands_or_author():
    # goal/research/diagnose must not be able to run a command (Bash) or author/edit -> they
    # structurally cannot measure or judge.
    for name in ("ve-goal", "ve-research", "ve-diagnose"):
        tools = _tools(_agent(name)[0])
        assert "Bash" not in tools, name
        assert "Write" not in tools and "Edit" not in tools, name
    # NONE of the four may run Bash — only the orchestrator runs the deterministic ar verbs.
    for name in SUBAGENTS:
        assert "Bash" not in _tools(_agent(name)[0]), name


def test_each_subagent_declares_contract_and_frozen_invariant():
    refs = {"ve-goal": "spec.schema.json", "ve-research": "policy_id",
            "ve-diagnose": "diagnosis.schema.json", "ve-author": "work.py"}
    for name, ref in refs.items():
        assert ref in _agent(name)[1], (name, ref)
    for name in SUBAGENTS:
        body = _agent(name)[1]
        assert "UNTRUSTED PRODUCER" in body and "PROPOSAL" in body, name


def test_evolution_v2_contracts_match_wiring():
    # ve-author: consumes the to_prompt rendering, RETURNS source, the orchestrator writes
    # work.py in GENERATE — the .md must match the evolve_loop wiring exactly (no G1-style drift).
    body = _agent("ve-author")[1]
    assert "to_prompt" in body and "last_errors" in body
    assert "RETURN" in body and "source" in body
    assert "ORCHESTRATOR" in body and "GENERATE" in body
    # ve-research consumes lessons and cites real lesson ids
    research = _agent("ve-research")[1]
    assert "lesson_id" in research and "frontier_sim" in research
    # Research Harness (R7): ve-research also consumes the hypothesis ledger — falsified is a
    # first-class result and citations are the verifiable hypothesis:<event_id> form.
    assert "falsified" in research and "hypothesis:<event_id>" in research
    # ve-diagnose may add free hypotheses but the judged fields stay in the enum
    diagnose = _agent("ve-diagnose")[1]
    assert "free-form hypotheses" in diagnose and "schema enums" in diagnose
    # Research Harness (R7): ve-diagnose can emit a structured {statement, prediction,
    # experiment_spec} execution-channel PROPOSAL routed to `ve experiment run` — proposal only,
    # the deterministic adjudicator (never the agent) returns the verdict.
    assert "experiment_spec" in diagnose and "statement" in diagnose and "prediction" in diagnose
    assert "ve experiment run" in diagnose and "adjudicat" in diagnose
    # ve-goal asks instead of guessing when load-bearing info is missing
    assert "clarifying questions" in _agent("ve-goal")[1]


SKILL = SRC / "vllm_evolve" / "assets" / "claude" / "skills" / "ve-autopt" / "SKILL.md"


def test_orchestrator_skill_routes_the_research_harness():
    # R7 (Research Harness): the orchestrator skill must drive the new flow — route a structured
    # diagnosis proposal through `ve experiment run`, consume the hypothesis ledger (incl.
    # falsified), cite hypothesis:<event_id>, and defer the verdict to the adjudicator.
    body = SKILL.read_text(encoding="utf-8")
    assert "ve experiment run" in body
    assert "experiment_spec" in body and "{statement, prediction, experiment_spec}" in body
    assert "ve experiment list" in body                 # gathers the ledger
    assert "hypothesis:<event_id>" in body
    assert "falsified" in body
    assert "adjudicator" in body                         # the verdict is deterministic
    assert "verify-citations" in body                    # citations are verified before trust
    # the research chain stays PROPOSAL-layer, never the adoption gate
    assert "never feeds the adoption gate" in body.replace("NEVER", "never")


def test_ar_init_ships_the_subagents(tmp_path):
    from vllm_evolve.install import installer
    installer.install(tmp_path)
    for name in SUBAGENTS:
        assert (tmp_path / ".claude" / "agents" / f"{name}.md").exists(), name
    # the orchestrator skill ships too
    assert (tmp_path / ".claude" / "skills" / "ve-autopt" / "SKILL.md").exists()
