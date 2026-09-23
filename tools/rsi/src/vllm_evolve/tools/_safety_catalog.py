"""Authoritative catalog of safety / verification rejection categories.

This is the **harness's contract** with users about which classes of unsafe
or malformed code are caught before a policy is ever loaded into the bench
process. Every entry in :data:`STATIC_CATEGORIES` MUST be backed by at
least one fixture in ``tests/fixtures/bad_policies/`` and at least one
rejection check inside ``vllm_evolve.cli.main._verify_code``;
``tests/test_verify_meta.py`` enforces that parity.

The catalog spans three sources, all of which the harness commits to
exercising:

* ``trust/safety.py`` (L1 static):
  ``FORBIDDEN_IMPORTS``, ``FORBIDDEN_PATTERNS``, ``ast.parse`` errors,
  ``check_signatures`` missing-function rejection.
* ``vllm_evolve/ar_cli._verify_code`` (L1/L2 static):
  ``ast``-walk over the policy that adds missing-return-type detection,
  signature parity against ``targets/<target>/skeleton.py``, and a
  forbidden-pattern regex re-run.
* ``vllm_evolve/tools/verify_tool`` (L2 static):
  the nested-loop heuristic for ``O(n^2)`` detection.

The catalog lives outside ``trust/`` so the underlying safety logic in
``trust/safety.py`` stays unchanged. When a new rule is added in a later
round, the developer must:

1. Add the rule's logic (in ``trust/safety.py``, ``ar_cli._verify_code``,
   or a sibling module).
2. Add the rule's category name to :data:`_CATEGORY_ENTRIES` below.
3. Add a fixture in ``tests/fixtures/bad_policies/<category>.py``
   (skipped only if ``runtime_only=True``).

Skipping any of steps 2 or 3 causes the meta-test to fail loudly.

Round-1 hardening (per Codex's Round-0 review): the catalog now
asserts a sanity invariant on ``trust/safety.py`` at import time so a
silent emptying of the forbidden-import / forbidden-pattern set surfaces
as an explicit failure rather than a silently-passing build.
"""

from __future__ import annotations

from typing import Final

from vllm_evolve.trust.safety import FORBIDDEN_IMPORTS, FORBIDDEN_PATTERNS

# Sanity-check coupling to ``trust/safety.py``. The catalog claims that
# ``forbidden_import`` and ``forbidden_pattern`` are real categories; if
# the underlying lists are empty the claim is a lie. Fail at import time
# rather than letting tests pass with no coverage.
assert FORBIDDEN_IMPORTS, (
    "trust.safety.FORBIDDEN_IMPORTS is empty -- catalog claim for the "
    "'forbidden_import' category is no longer backed by real logic."
)
assert FORBIDDEN_PATTERNS, (
    "trust.safety.FORBIDDEN_PATTERNS is empty -- catalog claim for the "
    "'forbidden_pattern' category is no longer backed by real logic."
)


# Each entry: (category_name, expected_issue_layer, runtime_only, summary).
#
# ``expected_issue_layer`` is the prefix used by ``ve verify`` when
# emitting the issue string ("L1" for safety, "L2" for constraints).
#
# ``runtime_only=True`` marks a category whose authoritative check
# happens at bench time rather than ``ve verify``. ``ve verify`` still
# applies a best-effort static heuristic, but the absence of a static
# match does **not** mean the policy is clean -- the runtime check at
# ``ve bench`` time is the authority. The meta-test recognises this:
# runtime-only categories DO require a fixture so the heuristic is at
# least exercised, but the contract about "static rejection" is relaxed
# from "must reject" to "heuristic best-effort".
_CATEGORY_ENTRIES: Final[tuple[tuple[str, str, bool, str], ...]] = (
    (
        "forbidden_import",
        "L1",
        False,
        "Imports a module on trust.safety.FORBIDDEN_IMPORTS (e.g. os, "
        "sys, subprocess).",
    ),
    (
        "forbidden_pattern",
        "L1",
        False,
        "Uses a pattern in trust.safety.FORBIDDEN_PATTERNS (eval, exec, "
        "open, while True, ...).",
    ),
    (
        "signature_mismatch",
        "L2",
        False,
        "Either does not define one of the target's evolvable_functions, "
        "or defines it with arg names that diverge from the matching "
        "skeleton.py signature.",
    ),
    (
        "syntax_error",
        "L1",
        False,
        "Source fails ast.parse; rejected before any signature check.",
    ),
    (
        "quadratic_loop",
        "L2",
        False,
        "Static heuristic detects nested for-loops over the same "
        "iterable (verify_tool.verify_code).",
    ),
    (
        "missing_return_type",
        "L1",
        False,
        "An evolvable_function is defined without a return annotation. "
        "ar_cli._verify_code walks the AST for FunctionDef.returns is None.",
    ),
    (
        "fabricated_ids",
        "L2",
        True,
        "Policy emits a request_id that did not come from the input "
        "objects (e.g. 'FAKE_REQ_001'). The authoritative check happens "
        "at runtime in ve bench when the scheduler is actually invoked; "
        "ve verify only applies a static heuristic that matches suspect "
        "string literals like 'FAKE_*' or 'dummy_*'.",
    ),
)


CATEGORIES: Final[frozenset[str]] = frozenset(
    name for name, _, _, _ in _CATEGORY_ENTRIES
)


STATIC_CATEGORIES: Final[frozenset[str]] = frozenset(
    name for name, _, runtime_only, _ in _CATEGORY_ENTRIES if not runtime_only
)


RUNTIME_ONLY_CATEGORIES: Final[frozenset[str]] = frozenset(
    name for name, _, runtime_only, _ in _CATEGORY_ENTRIES if runtime_only
)


CATEGORY_LAYERS: Final[dict[str, str]] = {
    name: layer for name, layer, _, _ in _CATEGORY_ENTRIES
}


CATEGORY_SUMMARIES: Final[dict[str, str]] = {
    name: summary for name, _, _, summary in _CATEGORY_ENTRIES
}


__all__ = [
    "CATEGORIES",
    "CATEGORY_LAYERS",
    "CATEGORY_SUMMARIES",
    "RUNTIME_ONLY_CATEGORIES",
    "STATIC_CATEGORIES",
]
