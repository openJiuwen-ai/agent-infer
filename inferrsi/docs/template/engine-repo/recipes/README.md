# Source recipes

Maintain one YAML source per exact model × engine × backend × hardware/count ×
precision × task combination. These deployment recipes follow
[InferMatrixLab/recipe schema version 1](https://github.com/InferMatrixLab/recipe/blob/main/schemas/recipe.schema.json).
The starting YAML is maintained in
[InferRSI/docs/template](https://github.com/InferMatrixLab/InferRSI/blob/main/docs/template/recipe.yaml).
This format is separate from InferRSI's internal campaign RunSpec recipe.

```text
recipes/<recipe-id>.yaml
```

Replace every placeholder with actual facts: stable unique ID, model/provider
and snake_case task, engine repository and deployment ref, backend/precision,
exact hardware and positive count, environment, reviewed commands, real GitHub
maintainers and update date. Backend is an open string; do not invent support.
Keep every schema field. Requirements, maintainers and limitations are lists of
strings. Benchmark may be null. Baseline/candidate experiment pins and numeric
gates remain in acceptance issues/evidence, outside portable recipe defaults.

Installation, serve and request commands must be nonempty for `unverified` and
`validated`. Set `validated` only with exact-combination deployment evidence and
date. `unsupported` permits null commands and must include at least one specific
blocking limitation; it must not expose deployment/copy controls. A combination
with unknown required facts stays in catalog inventory until complete.

Acceptance is independently `PASS`, `FAIL` or `INCONCLUSIVE`. PASS needs its
acceptance issue and complete evidence links; deployment validation or CPU CI
does not establish it. Preserve imported reports' source and validation scope.
Commands are reviewed display text, never executed by content sync or the site.

Commit source updates with the implementation. Register the source repository,
ref and relative YAML path through the catalog's
[adding-an-engine guide](https://github.com/InferMatrixLab/recipe/blob/main/docs/adding-an-engine.md).
Recipe maintainers sync and validate a complete candidate snapshot, recording
the exact repo/path/commit in the source lock, then review page/edit links. Keep
one content owner per recipe ID; do not create a second catalog-owned copy.
Schema, schema documentation and this template must change together for future
contract revisions.
