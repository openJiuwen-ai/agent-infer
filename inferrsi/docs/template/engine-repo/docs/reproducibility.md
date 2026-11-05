# Reproducibility

For each deployment recipe, record the actual model revision, engine release or
commit, backend, precision, exact device model/count/topology, OS, driver/firmware,
runtime/compiler/framework and all required dependencies. Record model access,
input data, output paths and any configuration changes.

```text
Deployment recipe ID / implementation ref: <recipe-id> / <deployment-ref>
Environment and device identity: <recorded-versions-and-device-facts>
Install / serve / minimal request: <reviewed-complete-commands>
Readiness / success / cleanup: <observable-checks-and-cleanup-commands>
Deployment result / date / evidence: <status-and-record-or-unverified>
```

The deployment ref identifies what users install. Baseline/candidate revisions,
campaign ID, comparison workload, required validation levels, validator/input
hashes and numeric gates belong in the campaign acceptance issue and frozen
evidence. Use the existing four-segment contract in [evidence](../evidence/README.md).

Record setup, readiness, warmups and measured work separately. Describe seed,
prompt/input shape, concurrency, request count, precision, device allocation,
timing boundary, units and statistical method. Keep raw output/quality samples,
logs, source identity, unsuccessful trials and process cleanup records. Freeze
the validators before candidate search and use identical comparison controls.

Validation levels are L0 operator correctness, L1 microbenchmark, L2 isolated
scenario, L3 model/modality quality and L4 service/E2E. The campaign selects the
required levels; it supplies its own gates. Missing levels or an environment
mismatch mean `INCONCLUSIVE`. Microbench gains do not justify service claims.

CPU checks validate repository behavior. Target hardware execution must still
run installation, startup, minimal request, health/output checks and required
quality/E2E work. Imported reports retain their original authorship, software,
hardware and measured scope; documenting or collecting them is not a fresh run.

Keep deployment `validated`/`unverified`/`unsupported` separate from acceptance
`PASS`/`FAIL`/`INCONCLUSIVE`. Cite complete deployment evidence/date for validation
and complete issue/evidence links for PASS. Follow [recipes](../recipes/README.md)
when publishing the exact combination to the catalog.
