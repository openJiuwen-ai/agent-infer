---
name: ac-bootstrap
description: Use when scaffolding a new AgentCache cache backend, eviction policy, or public entry point; places files per project conventions and runs pre-commit.
---

# ac-bootstrap

Scaffold a new AgentCache component (cache backend, eviction policy, or public
entry point) following the project's conventions, then validate locally before
committing.

## WHEN TO INVOKE

- The user asks to "add a new cache backend" / "add an eviction policy" /
  "add a public API entry point" for AgentCache.
- The user asks to scaffold or stub a new AgentCache module.

Do NOT invoke for: editing an existing component, adding tests only, or
non-AgentCache scaffolding.

## STEPS

1. Identify the kind: **backend**, **policy**, or **entry point**.
2. Read the existing layout under `src/agentcache/`. If `src/agentcache/` does
   not yet exist, this is the first component: create the package skeleton:

   ```bash
   mkdir -p src/agentcache
   cat > src/agentcache/__init__.py <<'PY'
   """AgentCache: efficient cache management for agent workflows."""
   PY
   ```

3. Place the new file by kind:

   - **backend** -> `src/agentcache/backends/<name>.py`
   - **policy** -> `src/agentcache/policies/<name>.py`
   - **entry point** -> `src/agentcache/__init__.py` (add the public export)

   Never put backends and policies in the same module.
4. Add the symbol to the relevant `__init__.py` `__all__` / import so it is
   importable from the package top level (for entry points and public
   backends).
5. Add a smoke test that imports the new symbol:

   - File: `tests/test_<kind>_<name>.py`
   - Minimal body: `from agentcache import <symbol>` inside a test that
     asserts the import succeeds.

6. Run the full pre-commit pass on the changed files:

   ```bash
   pre-commit run --files src/agentcache/** tests/**
   ```

   Fix every reported issue before continuing.
7. Confirm the commit message will carry the DCO `Signed-off-by:` line. The
   repo's `commit-msg` hook adds it automatically if missing; do not strip it.

## DON'T

- Don't add dependencies to `pyproject.toml` without surfacing it for review.
- Don't place backends and policies in the same module.
- Don't skip the `pre-commit` run "to save time".
- Don't commit without the `Signed-off-by:` trailer (DCO check will fail in
  CI).

## AFTER

Once the new component compiles and pre-commit passes, invoke `ac-review` to
validate the diff against repo conventions before opening the PR.
