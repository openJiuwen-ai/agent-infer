"""M6 / AC6: the deterministic measurement + judgment core is FROZEN — provably agent-free.

No LLM/agent library may appear ANYWHERE in the transitive import closure of the measure/judge
modules (a one-hop scan would miss an LLM imported via an unfrozen helper); the judge path is
deterministic; the accept-gate threshold is a config decision, never the (possibly sub-agent-
produced) Spec; and the PreToolUse hook confines a sub-agent author to the policy file and protects
auto-executed Python hooks. Pure, no GPU.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

_PKG = SRC / "vllm_evolve"

# seed modules of the measure/judgment path; these + everything they reach MUST be agent-free
_FROZEN_SEEDS = [
    "bench/native.py", "bench/dispatch.py", "bench/runner.py", "bench/metrics.py",
    "bench/slo.py", "bench/compare.py", "bench/decision.py", "bench/eval_result.py",
    "bench/config.py", "core/accept.py", "core/verify.py",
]
# LLM / agent-orchestration libraries that must never be IMPORTED into the frozen core. (Substring
# scans false-positive: native.py serves vLLM's *OpenAI-compatible* HTTP API; match imports only.)
_FORBIDDEN = ("anthropic", "openai", "litellm", "langchain", "cohere",
              "google.generativeai", "replicate", "vertexai")
_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(re.escape(x) for x in _FORBIDDEN) + r")(?:[.\s]|$)",
    re.MULTILINE)


def _vllm_imports(rel: str) -> set[str]:
    tree = ast.parse((_PKG / rel).read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm_evolve"):
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names if a.name.startswith("vllm_evolve"))
    return mods


def _rel_of(mod: str) -> str | None:
    p = _PKG.joinpath(*mod.split(".")[1:]).with_suffix(".py")
    return str(p.relative_to(_PKG)).replace("\\", "/") if p.is_file() else None


def _closure(seeds: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for mod in _vllm_imports(rel):
            r = _rel_of(mod)
            if r and r not in seen:
                stack.append(r)
    return seen


def test_frozen_core_transitive_closure_imports_no_llm():
    closure = _closure(_FROZEN_SEEDS)
    # the scan must actually walk deps (not be one-hop): known indirect deps are present
    assert "bench/runtime.py" in closure and "core/schemas.py" in closure, closure
    offenders = []
    for rel in sorted(closure):
        text = (_PKG / rel).read_text(encoding="utf-8")
        offenders += [f"{rel}: imports {m.group(1)}" for m in _IMPORT_RE.finditer(text)]
    assert offenders == [], offenders


# Packages the frozen closure must NEVER reach (untrusted orchestration / routing / agents).
_FORBIDDEN_PKGS = ("autopt/", "intent/", "engine/", "modes/", "assets/", "install/", "ops/",
                   "knowledge/", "trust/", "scaffold/")


def test_frozen_closure_whitelist_only_bench_core_config():
    """P2 whitelist: the frozen judgment closure may reach ONLY ``bench/**``, ``core/**``,
    or the package-root ``config.py`` — and explicitly none of the untrusted layers. This is the
    structural guarantee that the measure/judge core stays agent-free even as code moves around."""
    closure = _closure(_FROZEN_SEEDS)
    stray = sorted(r for r in closure
                   if not (r.startswith(("bench/", "core/")) or r == "config.py"))
    assert stray == [], f"frozen closure escaped bench/core/config: {stray}"
    leaked = sorted(r for r in closure if r.startswith(_FORBIDDEN_PKGS))
    assert leaked == [], f"frozen closure reached an untrusted layer: {leaked}"
    # closure genuinely walked: data models + decision chain are reachable from the core seeds
    for must in ("core/schemas.py", "bench/eval_result.py", "bench/compare.py"):
        assert must in closure, (must, sorted(closure))


def test_bench_layer_never_imports_core():
    """§1 dependency direction: ``core -> bench`` is allowed, ``bench -> core`` is FORBIDDEN — the
    measurement layer must not depend on the adoption-judgment layer. AST scan (abs + relative)."""
    offenders: list[str] = []
    for p in sorted((_PKG / "bench").rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                abs_core = mod.startswith("vllm_evolve.core")
                rel_core = bool(node.level) and mod.split(".")[0] == "core"
                if abs_core or rel_core:
                    offenders.append(f"{p.name}: from {'.' * (node.level or 0)}{mod}")
            elif isinstance(node, ast.Import):
                offenders += [f"{p.name}: import {a.name}" for a in node.names
                              if a.name.startswith("vllm_evolve.core")]
    assert offenders == [], offenders


def test_judge_path_is_deterministic_no_randomness():
    from vllm_evolve.core.accept import accept_vs_strong_baseline
    kw = dict(candidate_effective=True, quality_ok=True,
              holdout_baseline=[5.0, 5.0, 4.9], holdout_candidate=[7.0, 7.0, 6.9])
    v1 = accept_vs_strong_baseline([5.0, 5.1, 4.9], [7.0, 7.1, 6.9], **kw)
    v2 = accept_vs_strong_baseline([5.0, 5.1, 4.9], [7.0, 7.1, 6.9], **kw)
    assert v1.to_dict() == v2.to_dict()        # seeded bootstrap -> identical, no randomness


def test_accept_gate_threshold_is_never_taken_from_the_spec():
    # a ve-goal-produced Spec must NOT be able to lower the bar. The field is GONE from Spec, and
    # NEITHER judgment entrypoint (run_autopt, ve verify-gain) reads spec.accept_threshold_pct.
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine import orchestrate
    assert "accept_threshold_pct" not in Spec.__dataclass_fields__
    assert "spec.accept_threshold_pct" not in inspect.getsource(orchestrate.run_autopt)
    assert orchestrate._DEFAULT_ACCEPT_THRESHOLD_PCT == 20.0
    cli_src = (_PKG / "cli" / "main.py").read_text(encoding="utf-8")
    assert "spec.accept_threshold_pct" not in cli_src      # cmd_verify_gain uses the config default


def test_hook_confines_author_and_protects_auto_executed_hooks():
    from vllm_evolve.install.hook_decision import decide
    from vllm_evolve.tools.phase_guard import Phase

    def _w(path, phase=Phase.GENERATE):
        return decide("Write", {"file_path": path}, phase).allow

    assert _w("targets/scheduling/work.py") is True
    # any OTHER targets/*.py is blocked even in GENERATE (no work2.py / backdoor file)
    assert _w("targets/scheduling/work2.py") is False
    assert _w("targets/scheduling/evil.py") is False
    # auto-executed Python hooks are protected in ANY phase (can't be planted to subvert the gate)
    assert _w("conftest.py", Phase.GENERATE) is False
    assert _w("tests/conftest.py", Phase.DESIGN) is False
    assert _w("sitecustomize.py", Phase.GENERATE) is False
