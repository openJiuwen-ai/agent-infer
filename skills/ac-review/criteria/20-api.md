# 20 — Public API

## TRIGGER

Applies when the diff modifies `src/agentcache/__init__.py`, anything
exported via `__all__`, or any symbol reachable as `agentcache.<name>`.

## CRITERION A1: public functions/classes have docstrings

- **Severity:** blocker
- **Check:** Every symbol exported from the package top level has a
  docstring with at least a one-line summary.
- **Fix:** Add a docstring summarising what the symbol does and how to call
  it.

## CRITERION A2: public API changes are documented in the PR

- **Severity:** warning
- **Check:** If the diff adds, removes, or changes the signature of a public
  symbol, the PR description calls it out under a "Breaking changes" /
  "API changes" note.
- **Fix:** Add the note to the PR description.

## CRITERION A3: no unintended private-symbol leakage

- **Severity:** warning
- **Check:** Symbols intended to be private are prefixed with `_` and are
  not in `__all__`.
- **Fix:** Prefix the symbol with `_` and remove it from `__all__` if
  present.
