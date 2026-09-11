# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Guard stable dependency boundaries between AgentBench and production code."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_IMPORTERS = {Path("agentinfer/agentcache/entrypoints/bench.py")}


def _imported_names(node: ast.Import | ast.ImportFrom, relative: Path) -> list[str]:
    """Extract absolute module names imported by an AST import node.

    Returns the base module plus each dotted submodule name, resolving
    relative imports (``level``) against the importing file's package path.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        package = relative.with_suffix("").parts[:-1]
        prefix = package[: len(package) - node.level + 1]
        base = ".".join((*prefix, *(node.module or "").split(".")))
    else:
        base = node.module or ""
    return [base, *(f"{base}.{alias.name}" if base else alias.name for alias in node.names)]


def test_production_modules_do_not_import_agentbench_outside_entrypoint() -> None:
    violations: list[str] = []
    for path in sorted((_ROOT / "agentinfer").rglob("*.py")):
        relative = path.relative_to(_ROOT)
        if relative.parts[:2] == ("agentinfer", "agentbench") or relative in _ALLOWED_IMPORTERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = _imported_names(node, relative)
            if any(name == "agentinfer.agentbench" or name.startswith("agentinfer.agentbench.") for name in names):
                violations.append(f"{relative}:{node.lineno}")

    assert violations == []
