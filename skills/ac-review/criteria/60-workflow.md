# 60 — PR / RFC workflow

## TRIGGER

Applies to diffs that change behavior, public API, benchmark or test utilities,
review criteria, PR/RFC templates, or developer workflow docs. Also applies to
CI configuration diffs when present, but does not assume E2E benchmark CI exists.

`60-workflow.md` owns reviewability and process linkage. Command, artifact,
benchmark, and fallback-smoke evidence belong in `70-regression.md`.

## CRITERION W1: RFC or issue linkage present when needed

- **Severity:** warning
- **Check:** Behavior, design, API, benchmark, or workflow changes link a related
  RFC or issue, or explain why the change is small enough not to need one.
- **Fix:** Add the RFC/issue link to the PR body, or state why no RFC/issue is
  required.

## CRITERION W2: RFC success criteria map to PR evidence

- **Severity:** blocker
- **Check:** If an RFC exists, the PR maps each relevant RFC success criterion to
  implementation evidence, validation evidence, or an explicit out-of-scope
  deferral.
- **Fix:** Add an RFC / Criteria Mapping section that lists requirement,
  evidence, and test or artifact for each criterion.

## CRITERION W3: PR claims match the diff and evidence

- **Severity:** blocker
- **Check:** The PR body does not claim behavior, tests, benchmarks, artifacts,
  or reviewer-visible outcomes that are absent from the diff or provided
  evidence.
- **Fix:** Remove unsupported claims or add the missing evidence.

## CRITERION W4: open questions are resolved or deferred

- **Severity:** warning
- **Check:** Open RFC or reviewer questions are either resolved, explicitly
  deferred to a linked follow-up, or marked out of scope with rationale.
- **Fix:** Update the PR/RFC notes with the decision, follow-up link, or
  out-of-scope explanation.

## CRITERION W5: validation command and result are present

- **Severity:** blocker
- **Check:** The PR includes exact validation commands and relevant output,
  artifact paths, or a justified not-run explanation. Quick local prechecks may
  report this as missing because the PR is not opened yet; full review requires
  reviewer-facing evidence.
- **Fix:** Add the command, result or artifact path, or explain why validation was
  not run and what smaller check was used instead.

## CRITERION W6: CI/workflow edits remain locally checkable

- **Severity:** warning
- **Check:** If `.github/workflows/`, `.pre-commit-config.yaml`, or tool config
  changes, the PR explains how the workflow/config was validated locally or why
  validation is deferred.
- **Fix:** Add the relevant syntax check, pre-commit command, or deferral reason.
