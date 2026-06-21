# 00 — Meta (always-true repo rules)

## TRIGGER

Always applies to every diff that touches the repo.

## CRITERION M1: DCO sign-off present

- **Severity:** blocker
- **Check:** Every commit in the PR carries a `Signed-off-by: Name <email>`
  trailer. CI's `dco` job enforces this; do not bypass it.
- **Fix:** `git commit --amend --signoff` (or rebase with `--signoff`) on
  every offending commit.

## CRITERION M2: pre-commit clean

- **Severity:** blocker
- **Check:** `pre-commit run --all-files` exits 0. Covers ruff check, ruff
  format, typos, markdownlint.
- **Fix:** Run `pre-commit run --all-files` locally, fix every reported
  issue, re-run until clean.

## CRITERION M3: PR template fields filled

- **Severity:** warning
- **Check:** The PR description uses `.github/PULL_REQUEST_TEMPLATE.md` and
  its required sections are filled (not left as placeholder text).
- **Fix:** Edit the PR description to complete every required section.

## CRITERION M4: no secrets or tokens

- **Severity:** blocker
- **Check:** The diff contains no API keys, tokens, or private endpoint
  URLs.
- **Fix:** Remove the secret; rotate it if it was pushed; use environment
  variables / a secrets manager.
