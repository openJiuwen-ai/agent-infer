# 30 — Testing

## TRIGGER

Applies to any diff that adds or modifies `*.py` under `src/` or `tests/`.

## CRITERION T1: cache mutations have a test

- **Severity:** blocker
- **Check:** Any new/changed code path that writes to, evicts from, or
  invalidates the cache has at least one test exercising that behaviour.
- **Fix:** Add a test under `tests/` that drives the mutation and asserts
  the observable effect.

## CRITERION T2: tests do not depend on ordering

- **Severity:** warning
- **Check:** Tests pass when run in any order. No test relies on another
  test having run first.
- **Fix:** Make the test self-contained: set up its own state in a fixture.

## CRITERION T3: smoke test imports succeed

- **Severity:** blocker
- **Check:** For newly added public symbols, a smoke test asserting the
  import succeeds exists (per `ac-bootstrap` step 5).
- **Fix:** Add the import smoke test.
