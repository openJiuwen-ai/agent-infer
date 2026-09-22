"""
Code safety and signature validation.

Two public entry points:
- ``check_safety(code)`` — reject dangerous imports and patterns
- ``check_signatures(code, expected)`` — verify expected functions exist
- ``check_reserved_class_redefinitions(code, names)`` — reject policy-owned
  copies of bridge runtime types

``SignatureValidator`` combines both for backward compatibility.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

FORBIDDEN_IMPORTS = frozenset([
    'os', 'sys', 'subprocess', 'socket', 'threading', 'multiprocessing',
    '__import__', 'importlib', 'ctypes', 'cffi',
])

FORBIDDEN_PATTERNS = [
    r'\b__import__\b',
    r'\bexec\s*\(',
    r'\beval\s*\(',
    r'\bopen\s*\(',
    r'\bos\.\w+',
    r'\bsys\.\w+',
    r'\bsubprocess\.',
    r'\bwhile\s+True\b',
    r'\bwhile\s+1\b',
]


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


def check_safety(code: str) -> ValidationResult:
    """L1: reject dangerous patterns and imports. Does NOT check signatures."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return ValidationResult(False, f"Forbidden pattern: {pattern}")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(False, f"SyntaxError: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in FORBIDDEN_IMPORTS:
                    return ValidationResult(False, f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in FORBIDDEN_IMPORTS:
                return ValidationResult(False, f"Forbidden import from: {node.module}")

    return ValidationResult(True)


def check_signatures(code: str, expected_functions: list[str]) -> ValidationResult:
    """Verify all expected function names exist in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(False, f"SyntaxError: {e}")

    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for fn in expected_functions:
        if fn not in defined:
            return ValidationResult(False, f"Missing function: {fn}")

    return ValidationResult(True)


def check_reserved_class_redefinitions(
    code: str,
    reserved_names: set[str] | frozenset[str],
) -> ValidationResult:
    """L2: bridge-owned runtime classes must retain their canonical semantics.

    Candidate policy source is injected into the scheduler plugin module.  A
    same-named class therefore shadows the bridge definition globally, even if
    the candidate only intended to provide typing scaffolding.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(False, f"SyntaxError: {e}")
    redefined = sorted(
        {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name in reserved_names
        }
    )
    if redefined:
        return ValidationResult(
            False,
            "policy redefines bridge-owned runtime class(es): "
            + ", ".join(redefined),
        )
    return ValidationResult(True)


class SignatureValidator:
    """Combined safety + signature check. Kept for backward compatibility."""

    FORBIDDEN_PATTERNS = FORBIDDEN_PATTERNS

    def validate(
        self,
        code: str,
        expected_functions: list[str],
        expected_signatures: dict[str, str] | None = None,
    ) -> ValidationResult:
        result = check_safety(code)
        if not result.ok:
            return result
        return check_signatures(code, expected_functions)
