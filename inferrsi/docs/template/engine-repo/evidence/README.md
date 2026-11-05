# Evidence

Reuse the InferRSI four-segment contract. Copy
[the existing template](https://github.com/InferMatrixLab/InferRSI/tree/main/evidence/_template)
for each campaign; it is Markdown and does not require the InferRSI runtime.

```text
evidence/<campaign-id>/
  plan/        Hypothesis, immutable RunSpec/environment and frozen gates
  results/     Raw samples, outputs, logs, measured scope and verdict rationale
  patch/       Candidate diff or exact implementation revision pointer
  validator/   Frozen validation code, dependencies, inputs and hashes
```

Pin baseline/candidate revisions, campaign identity and numeric gates in the
acceptance issue and this campaign's artifacts. Freeze validators and workload
before candidate search. Keep raw quality/performance evidence, rejected changes,
errors and incomplete runs. Do not rewrite historical evidence for new defaults.

| Verdict | Meaning |
| --- | --- |
| PASS | Pinned comparison, complete required evidence and every frozen gate met |
| FAIL | Complete enough evidence demonstrates a required gate was not met |
| INCONCLUSIVE | Missing freeze/levels, incomplete run or environment mismatch |

An interrupted run remains INCONCLUSIVE even if partial samples look faster or
slower. L0–L2/operator/microbench evidence does not prove L3 model quality or L4
E2E service gains; apply the campaign's required levels. Green CI is not PASS.

For public claims, cite acceptance issue, exact revisions, workload/environment
and complete artifacts, retaining the original source and scope. A deployment
recipe links this evidence and records deployment status separately from the
acceptance verdict. Catalog import and new CPU checks do not revalidate an old
hardware report. See [reproducibility](../docs/reproducibility.md).
