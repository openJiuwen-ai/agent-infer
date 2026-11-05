# Contributing

Read the [engine README](README.md), [architecture](docs/architecture.md),
[reproducibility guide](docs/reproducibility.md) and [agent contract](AGENTS.md).
Replace scaffold instructions with this engine's actual commands before use.

## Implement and check

1. Record the governing issue's goal, accepted scope, accountable owner and
   completion criteria. Keep implementation separate from any campaign claim.
2. Use a task branch; preserve other work. Plan the smallest relevant change,
   implement it, run the real language/build checks, and review the diff. Use
   [tests](tests/README.md) for correctness and
   [benchmarks](benchmarks/README.md) for the separately scoped measurement path.
3. Report actual commands/results and skipped checks. Review current navigation,
   local images and real clone/install sources; retain historical source pins.
   An independent reviewer inspects the actual diff and evidence before any
   delegated integration; self-review remains labeled as such.
4. Sign commits using `git commit -s` and the configured author identity. Material
   agent work includes `Agent-assisted-by: Codex (implementation)` when Codex is
   the actual contributor; substitute the actual agent/role for other agents.
   Keep governing issue references in the PR body, outside commit messages.

```bash
<install-development-dependencies-command>
<lint-or-build-command>
<lightweight-test-command>
git diff --check
git commit -s
```

## Evidence and acceptance

Use the four segments described in [evidence](evidence/README.md). Freeze
baseline/candidate identities, workload, required L0–L4 levels, numeric gates
and validators in the acceptance issue and evidence before experiments.
`PASS` requires every required gate and complete evidence; `FAIL` means complete
enough evidence demonstrates a missed gate; incomplete or mismatched validation
is `INCONCLUSIVE`. Microbenchmarks do not prove E2E gains. CPU CI does not replace
model quality or service validation. Do not modify historical evidence to fit a
new implementation, and do not publish unsupported gain numbers.

## Publish a source recipe

The engine maintainer owns implementation and source recipes. Recipe maintainers
own schema, catalog and website. The acceptance owner owns performance verdicts.
Update [recipes](recipes/README.md) for each exact deployment combination; cite
real deployment versions and evidence. Distinguish `validated`, `unverified` and
`unsupported` deployment from `PASS`, `FAIL` and `INCONCLUSIVE` acceptance.

Commit the source recipe and request registration/sync through the
[recipe contribution guide](https://github.com/InferMatrixLab/recipe/blob/main/CONTRIBUTING.md).
The collection PR must verify source repo/path/commit, schema, current commands,
edit links and website presentation. Keep one owner for each recipe ID. Do not
execute recipe commands as part of content sync or site builds.

## Pull requests

Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) to report the governing
issue, artifacts, checks, limits and review state. Do not prefill independent
approval. Delivery, goal acceptance, campaign `PASS`, merge and production
publication are separate outcomes; follow the named owner's authority.
