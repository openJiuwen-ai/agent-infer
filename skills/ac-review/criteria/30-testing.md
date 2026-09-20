# 30 — Testing

## TRIGGER

Applies to any diff that adds or modifies `*.py` under `agentinfer/` or `tests/`.

## CRITERION T1: behavior changes have focused tests

- **Severity:** blocker
- **Check:** Any new or changed cache mutation, scheduling/admission decision,
  lifecycle transition, request adaptation, or benchmark orchestration path has
  a focused test exercising its observable behavior.
- **Fix:** Add a test under the matching suite in `tests/` that drives the changed
  path and asserts the observable effect.

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
