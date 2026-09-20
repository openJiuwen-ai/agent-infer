# 80 — Architecture

## TRIGGER

Applies to any diff that adds or modifies core Python code under `agentinfer/`,
especially public APIs, engine-neutral scheduling contracts, vLLM adapters,
shared utilities, or files that own request/cache/benchmark behavior.

## Global Architecture Review Gate

Reviewer-level goals to apply across every remaining phase:

- File headers define ownership boundaries and reader entry points.
- Public/core class and function docstrings explain contracts and non-obvious
  inputs.
- No unexplained broad `Any`, `**kwargs`, or untyped dict boundary remains in
  reviewed code.
- Reusable helpers are placed in named utility/domain modules, not buried in
  feature files.
- New key methods on core classes are justified by class responsibility and
  call-site evidence.

## CRITERION AR1: file boundary header present

- **Severity:** warning
- **Check:** Every new or substantially changed file starts with a short header
  defining its ownership boundary: what behavior the file covers, what it does
  not cover, and the core classes / key functions readers should inspect first.
- **Fix:** Add or update the file header so readers can route themselves without
  scanning the whole module.

## CRITERION AR2: clear core contracts

- **Severity:** blocker
- **Check:** Every public or core class/function has a precise docstring.
  Inputs are explicitly typed and explained when non-obvious. Broad `Any`,
  unclear `**kwargs`, and vague dict-shaped parameters are avoided unless a
  documented boundary reason exists.
- **Fix:** Add explicit types and contract docstrings. If flexible payloads are
  unavoidable, document accepted keys or promote them into typed models or
  dataclasses.

## CRITERION AR3: shared helpers live in domain utilities

- **Severity:** warning
- **Check:** Reusable helper behavior is placed in named utility/domain modules
  with clear ownership. Feature files do not accumulate large clusters of
  private helpers that obscure the file's primary responsibility.
- **Fix:** Move reusable or boundary-obscuring helpers into a domain-named module
  such as `prometheus`, `paths`, `jsonl`, or `agent_roles`; avoid vague dumping
  grounds.

## CRITERION AR4: core class surface is intentional

- **Severity:** warning
- **Check:** New methods on key classes are justified by class responsibility and
  call-site evidence. The method set remains intentionally designed rather than
  collecting unrelated behavior.
- **Fix:** Include the architecture review: ownership, call sites, alternatives,
  and why the method belongs on that class. Move behavior elsewhere if it sits
  outside the class boundary.

## CRITERION AR5: public API first, private helpers later

- **Severity:** warning
- **Check:** Modules present their public API and core reader entry points before
  private implementation helpers. Private helpers support the public surface;
  they should not force readers to dig through implementation details first.
- **Fix:** Reorder the module or split helpers into a domain utility so the public
  contract is visible before private machinery.
