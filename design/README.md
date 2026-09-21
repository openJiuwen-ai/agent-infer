# Internal Design Documents

This directory contains development designs, implementation plans, and module invariants for contributors. Stable
user documentation lives under [`docs/`](../docs/README.md).

## Benchmark Subsystem

- [Subsystem overview](module/benchmarking/index.md)
- [BenchKit orchestration](module/benchmarking/benchkit-orchestration.md)
- [Agent runtime adapters](module/benchmarking/agent-runtime-adapters.md)
- [Request proxy and hints](module/benchmarking/request-proxy-and-hints.md)
- [Artifacts and evaluation](module/benchmarking/artifacts-and-evaluation.md)

## Scheduling Subsystem

- [Program state machine](module/scheduling/program-state-machine.md)
- [Progress-TTL scheduling](module/scheduling/progress-ttl-scheduling.md)
- [vLLM runtime integration](module/scheduling/vllm-runtime-integration.md)
- [AgentInfer serve flag](module/scheduling/agentinfer-serve-flag.md) (draft)

## AgentCache Skills

- [Skills design](superpowers/specs/agentcache-skills-design.md)
- [Skills implementation plan](superpowers/plans/agentcache-skills-implementation.md)

## CI/CD

- [CI quality gate and release guide (DRAFT, zh)](ci-cd-quality-gate-setup-guide.md)

## Maintenance Records

- [CI and unit-test fixes](ci-ut-fixes.md)

Documents with draft status describe contributor contracts under review. Check their `last_reviewed`, code paths, and
validation paths before treating them as current implementation requirements.
