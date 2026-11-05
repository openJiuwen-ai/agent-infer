# Benchmarks

Document real benchmark entrypoints, install requirements, input preparation,
readiness, warmups, measured work, cleanup, output schema and raw artifact paths.
Separate CPU correctness checks from hardware measurements and model quality.

```bash
<benchmark-command-with-explicit-workload-and-output-directory>
```

Every reported number states unit, precision, exact hardware/count, workload,
timing scope, repeats/statistics and baseline. Freeze numeric gates and
validators in the acceptance issue/evidence before searching candidates. Keep
rejected and failed trials. Use [evidence](../evidence/README.md) and
[reproducibility](../docs/reproducibility.md); microbenchmarks alone cannot justify
an E2E gain or acceptance PASS. Benchmarks are not default CI work.
