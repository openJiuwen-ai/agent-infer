# 10 — Style

## TRIGGER

Applies to any diff that adds or modifies `*.py` files under `src/` or
`tests/`.

## CRITERION S1: formatting passes ruff format

- **Severity:** blocker
- **Check:** `ruff format --check <changed files>` reports no changes
  needed. (Covered by pre-commit; verify explicitly for style-only
  feedback.)
- **Fix:** `ruff format <changed files>`.

## CRITERION S2: imports sorted and first-party known

- **Severity:** blocker
- **Check:** `ruff check --select I <changed files>` is clean, and
  `agentcache` is treated as first-party (configured in `pyproject.toml`
  `[tool.ruff.lint.isort]`).
- **Fix:** `ruff check --select I --fix <changed files>`.

## CRITERION S3: naming follows snake_case for functions/variables

- **Severity:** warning
- **Check:** Module-level functions and variables use `snake_case`; classes
  use `PascalCase`; constants use `UPPER_SNAKE`.
- **Fix:** Rename the offending symbol and update all call sites.

## CRITERION S4: module-level docstring on every new module

- **Severity:** nit
- **Check:** Every new `.py` file begins with a one-line module docstring
  describing its responsibility.
- **Fix:** Add the docstring as the first statement in the module.
