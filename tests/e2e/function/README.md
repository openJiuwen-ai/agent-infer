# E2E functional tests (`tests/e2e/function`)

Correctness-oriented GPU/NPU checks (repeatability, smoke). **Not** the performance A/B tree under
[`../perf/`](../perf/).

Ready CPU CI selects `cpu and ready`. AgentBench cases declare `L20`, `cards_2`, and `ready`;
replay cases have no CI lane.

## Features

| Feature | Documentation |
| ------- | --------------- |
| Trace replay (Inferact / TraceLab / AgentX) | [`tracereplay/README.md`](tracereplay/README.md) |
| AgentBench (Claude Code / DSH / JiuwenSwarm / ZCode) | [`agentbench/README.md`](agentbench/README.md) |

Future functional suites can add a row here and their own subdirectory + README.
