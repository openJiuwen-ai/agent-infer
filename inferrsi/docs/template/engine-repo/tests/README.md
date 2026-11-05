# Tests

Replace this guide with the engine's actual correctness checks and test paths.
Choose the language/build system already used by the engine. Document required
dependencies, CPU-only coverage, optional accelerator markers and expected
results; avoid model downloads in lightweight default checks.

```bash
<install-test-dependencies-command>
<cpu-correctness-test-command>
<explicit-hardware-test-command-if-applicable>
```

Write meaningful behavior/contract regressions when implementation changes
require them; do not add tests that merely assert this scaffold's files exist.
Report failures, skips and unavailable hardware accurately. A passing test suite
does not establish model quality, E2E performance or support on other hardware;
follow [reproducibility](../docs/reproducibility.md) and
[evidence](../evidence/README.md) for those claims.
