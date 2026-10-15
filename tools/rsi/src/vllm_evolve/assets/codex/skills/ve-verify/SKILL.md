---
name: ve-verify
description: Apply the shared L1/L2 gate before any real local Frontier evaluation.
---

# ve-verify — reject unsafe or incompatible source

For a manual round, run `ve phase set VERIFY`, then
`ve verify <policy> scheduling`.

The gate checks AST safety, required annotations, signature parity, nested-loop risk,
fabricated request IDs, and reserved marker forgery. A failed candidate returns its exact errors in
the next `AuthorContext.last_errors`; repair source must pass the same gate before Frontier. Never
replace this with a fixture evaluator or skip directly to simulation.
