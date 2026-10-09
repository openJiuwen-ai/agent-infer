"""Packaging honesty: every shipped console script must resolve to a real module + attribute.

This guards against a deleted module lingering in `[project.scripts]` (an install would otherwise
expose a dead command). It would have failed on the stale `vllm-evolve = "vllm_evolve.cli:app"`
entry after `cli.py` was deleted (zero GPU).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10, both supported by pyproject.toml
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))


def _console_scripts() -> dict[str, str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data.get("project", {}).get("scripts", {})


def test_every_console_script_target_resolves():
    scripts = _console_scripts()
    assert scripts, "expected at least one [project.scripts] entry"
    for name, target in scripts.items():
        module_path, sep, attr = target.partition(":")
        assert sep, f"console script {name!r} target {target!r} must be 'module:attr'"
        mod = importlib.import_module(module_path)              # fails if module was deleted
        assert hasattr(mod, attr), f"console script {name!r} -> {target}: no attribute {attr!r}"


def test_ve_is_the_sole_shipped_cli_and_no_legacy_script():
    scripts = _console_scripts()
    # the CLI command was renamed ar -> ve; `ar` must be gone, `ve` is the sole entry
    assert "ve" in scripts and scripts["ve"] == "vllm_evolve.cli.main:_entrypoint"
    assert "ar" not in scripts
    # the legacy console script that pointed at the deleted cli.py (vllm_evolve.cli:app) must be
    # gone — match the legacy module precisely so it doesn't catch the new vllm_evolve.cli.main pkg
    assert "vllm-evolve" not in scripts
    assert not any(t.startswith("vllm_evolve.cli:") for t in scripts.values())


def test_bench_schema_is_in_package_data():
    # Codex review P2: load_schema() reads the schema via importlib.resources, so it MUST ship in
    # the wheel/sdist — i.e. be listed in [tool.setuptools.package-data].
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    globs = data["tool"]["setuptools"]["package-data"]["vllm_evolve"]
    assert any("bench/schemas" in g for g in globs), globs
    # and it must actually be loadable the way the code loads it
    from vllm_evolve.bench.eval_result import load_schema
    assert "outcome_class" in load_schema()["properties"]


def test_codex_agent_skills_and_hook_are_in_package_data():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    globs = data["tool"]["setuptools"]["package-data"]["vllm_evolve"]
    assert any("assets/codex" in pattern for pattern in globs), globs
    assets = SRC / "vllm_evolve" / "assets" / "codex"
    assert (assets / "agents" / "vllm-policy-optimizer.toml").is_file()
    assert (assets / "hooks" / "phase_guard_hook.py").is_file()
    for skill in ("ve-context", "ve-design", "ve-evolve", "ve-generate", "ve-verify"):
        assert (assets / "skills" / skill / "SKILL.md").is_file()


def test_jsonschema_is_a_runtime_dependency():
    # Codex review P1: validate_eval_result() imports jsonschema at RUNTIME on every native bench,
    # so it must be a MAIN dependency (not dev-only), else a normal install crashes after GPU work.
    import importlib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert any(d.split(">=")[0].split("==")[0].strip() == "jsonschema" for d in deps), deps
    importlib.import_module("jsonschema")    # actually importable in this environment
