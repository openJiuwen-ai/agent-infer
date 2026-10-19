# E2E functional tests (`tests/e2e/function`)

Correctness-oriented GPU/NPU checks (repeatability, smoke). **Not** the performance A/B tree under
[`../perf/`](../perf/).

Default CI skips all of `tests/e2e` (`pytest --ignore=tests/e2e`).

## Features

| Feature | Documentation |
| ------- | --------------- |
| Trace replay (Inferact / TraceLab / AgentX) | [`tracereplay/README.md`](tracereplay/README.md) |

Future functional suites can add a row here and their own subdirectory + README.
