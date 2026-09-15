---
name: ac-bootstrap
description: Use when scaffolding a new AgentInfer cache backend, policy, scheduling component, integration adapter, or public entry point; discovers current ownership boundaries and runs focused validation.
allowed-tools:
  - Bash
  - Read
  - Write
  - Grep
  - Glob
  - Edit
---

# ac-bootstrap

Scaffold a new AgentInfer cache backend, policy, scheduling component, integration
adapter, or public entry point by following analogous code in the current
checkout, then validate locally before committing.

## WHEN TO INVOKE

- The user asks to add a cache backend, eviction/admission policy, scheduling
  component, integration adapter, or public API entry point for AgentInfer.
- The user asks to scaffold or stub a new AgentInfer module.

Do NOT invoke for: editing an existing component, adding tests only, or non-AgentInfer scaffolding.

## STEPS

1. **Identify the responsibility**: **backend**, **policy**, **scheduling
   component**, **integration adapter**, or **public entry point**.

2. **Discover the current ownership boundary** before creating files. Use `Glob`
   to list Python modules and tests, then `Grep` for analogous interfaces,
   implementations, exports, and tests. Read `pyproject.toml` to confirm the
   installed top-level package; do not assume a layout from another branch.

   In the current `agentinfer` layout, engine-neutral backend and policy
   contracts belong under `agentinfer/scheduling/`, while vLLM-specific cache
   adapters belong under `agentinfer/agentcache/`. If the checkout differs,
   follow its existing package and test structure instead of creating a parallel
   hierarchy.

3. **Place the new file beside analogous code**:

   - **backend or policy contract/implementation** -> matching engine-neutral
     scheduling domain
   - **scheduling component** -> matching scheduling runtime/factor/strategy
     domain
   - **engine integration adapter** -> matching engine-specific integration
     domain
   - **public entry point** -> the owning package's `__init__.py` or registered
     CLI boundary

   Preserve the boundary between engine-specific adapters and engine-neutral
   scheduling contracts.

4. **Add the symbol** to the owning `__init__.py` only when it is intentionally
   public. Keep internal implementation details out of `__all__`.

5. **Add focused tests** beside the analogous suite discovered in step 2. In the
   current `agentinfer` layout, likely suites are:

   - Cache/engine integration behavior: `tests/agentcache/`
   - Scheduling contracts/runtime: `tests/scheduling/`
   - Benchmark behavior: `tests/agentbench/`

   Treat these as current-layout examples, not universal paths. For a public
   symbol, include an import smoke test using the package path confirmed from
   `pyproject.toml`.

6. **Run focused tests and pre-commit on the exact changed files**:

   ```bash
   pytest <matching-test-files> -v
   pre-commit run --files <changed-files>
   ```

   Fix every reported issue before continuing.

7. **Confirm the commit message** will carry the DCO `Signed-off-by:` line. The
   repo's `commit-msg` hook adds it automatically if missing; do not strip it.

## DON'T

- Don't add dependencies to `pyproject.toml` without surfacing them for review.
- Don't mix engine-specific adapters into engine-neutral scheduling modules.
- Don't invent a top-level package or test hierarchy without checking the current
  packaging configuration and analogous code.
- Don't skip focused tests or pre-commit "to save time".
- Don't commit without the `Signed-off-by:` trailer (DCO check will fail in CI).

## AFTER

Once the new component compiles and pre-commit passes, invoke `ac-review` to
validate the diff against repo conventions before opening the PR.
