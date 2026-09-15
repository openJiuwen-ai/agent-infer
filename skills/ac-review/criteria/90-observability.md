# 90 — Observability

## TRIGGER

Applies to diffs that add or modify request handling, background jobs,
external API calls, database writes, authentication or authorization decisions,
or other system-boundary operations.

## CRITERION O1: system-boundary operations are logged

- **Severity:** warning
- **Check:** Important events at system boundaries are logged, including requests,
  jobs, external API calls, database writes, and auth decisions.
- **Fix:** Add logs at the boundary where the operation starts, completes, or fails.

## CRITERION O2: logs include useful context

- **Severity:** warning
- **Check:** Logs include enough context to diagnose the event, such as relevant
  IDs, operation name, status, duration, and error type.
- **Fix:** Add structured context fields or message details that identify what
  happened without requiring local reproduction.

## CRITERION O3: logs do not expose secrets or private data

- **Severity:** blocker
- **Check:** Logs do not include passwords, tokens, API keys, full credit card
  numbers, or private user content unless explicitly safe.
- **Fix:** Remove, redact, hash, or replace sensitive values with safe identifiers.

## CRITERION O4: log levels are consistent

- **Severity:** warning
- **Check:** Log levels match the event severity: `DEBUG` for local diagnosis
  details, `INFO` for normal important events, `WARN` for unexpected but
  recoverable conditions, and `ERROR` for failed operations needing attention.
- **Fix:** Adjust the log level to match the event severity.

## CRITERION O5: exceptions include stack traces

- **Severity:** warning
- **Check:** Exception logs preserve stack traces when reporting failed operations.
- **Fix:** Use exception-aware logging or include exception information so the
  traceback is available.

## CRITERION O6: log messages are specific and actionable

- **Severity:** warning
- **Check:** Log messages state the specific operation and outcome instead of
  generic text like `failed`, `error`, or `done`.
- **Fix:** Rewrite the message so an operator can identify the operation, status,
  and next investigation step.

## CRITERION O7: failures are not logged redundantly at every layer

- **Severity:** nit
- **Check:** The same failure is not logged repeatedly at every call layer unless
  each log adds new diagnostic context.
- **Fix:** Keep the log at the boundary or handling layer that has the most useful
  context, and remove duplicate lower-level logs.
