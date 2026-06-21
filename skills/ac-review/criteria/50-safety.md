# 50 — Safety / correctness

## TRIGGER

Applies when the diff touches eviction policy, cache key construction, or
any path that decides whether cached data is returned to a request.

## CRITERION F1: no cross-request cache poisoning

- **Severity:** blocker
- **Check:** The cache key includes every element that affects the response
  (prompt, model, generation params, tenant/owner where relevant). Two
  requests that should get different responses cannot collide to the same
  key.
- **Fix:** Add the missing element(s) to the key; add a regression test
  that asserts distinct responses are not served from the same entry.

## CRITERION F2: eviction does not corrupt in-flight reads

- **Severity:** blocker
- **Check:** Eviction cannot free an entry that a concurrent reader is
  still consuming.
- **Fix:** Add refcounting / read-lock / generation check so eviction
  skips entries in active use; add a concurrency test.

## CRITERION F3: invalidation covers all derived entries

- **Severity:** warning
- **Check:** When an entry is invalidated, any entry that was derived from
  it (e.g. a longer prefix built on a now-stale prefix) is also
  invalidated.
- **Fix:** Walk the derivation chain on invalidation; add a test covering
  the derived-entry case.
