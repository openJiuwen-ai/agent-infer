# 10 — Style

## TRIGGER

Applies to any diff that adds or modifies `*.py` files under `agentinfer/` or `tests/`.

## CRITERION S1: formatting passes ruff format

- **Severity:** blocker
- **Check:** `ruff format --check <changed files>` reports no changes needed.
  (Covered by pre-commit; verify explicitly for style-only feedback.)
- **Fix:** `ruff format <changed files>`.

## CRITERION S2: imports sorted and first-party known

- **Severity:** blocker
- **Check:** `ruff check --select I <changed files>` is clean, and `agentinfer`
  is treated as first-party (configured in `pyproject.toml`
  `[tool.ruff.lint.isort]`).
- **Fix:** `ruff check --select I --fix <changed files>`.

## CRITERION S3: naming follows snake_case for functions/variables

- **Severity:** warning
- **Check:** Module-level functions and variables use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE`.
- **Fix:** Rename the offending symbol and update all call sites.

## CRITERION S4: module-level docstring on every new module

- **Severity:** nit
- **Check:** Every new `.py` file begins with a one-line module docstring describing its responsibility.
- **Fix:** Add the docstring as the first statement in the module.

## CRITERION S5: function names describe observable behavior

- **Severity:** warning
- **Check:** Function names describe the observable transformation or side
  effect, not the implementation detail. A name should tell callers what changes
  or what value is produced, not which internal algorithm, data structure, or
  temporary mechanism is used.
- **Fix:** Rename the function around the caller-visible behavior and update call
  sites.

## CRITERION S6: every function has a contract docstring

- **Severity:** warning
- **Check:** Every new or materially changed function has a precise docstring.
  The docstring states the contract: what the function accepts, what it returns
  or mutates, and any non-obvious input constraints or side effects.
- **Fix:** Add or tighten the function docstring; if the contract is hard to
  explain, simplify the function boundary first.
