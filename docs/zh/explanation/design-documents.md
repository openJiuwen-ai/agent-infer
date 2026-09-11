# 内部设计索引

用户文档位于 `docs/`，开发设计稿位于仓根目录的 `design/`。这里提供发现入口，但设计稿不是用户操作指南，
也不作为稳定公开 API 承诺。

## 基准子系统

- [基准子系统总览](../../../design/module/benchmarking/index.md)
- [BenchKit 编排](../../../design/module/benchmarking/benchkit-orchestration.md)
- [Agent 运行时适配器](../../../design/module/benchmarking/agent-runtime-adapters.md)
- [请求代理和提示状态](../../../design/module/benchmarking/request-proxy-and-hints.md)
- [产物和评估边界](../../../design/module/benchmarking/artifacts-and-evaluation.md)

这些文档记录模块边界、验证路径和 `BENCH-INV-*` 开发不变量。其状态和 `last_reviewed` 元数据决定是否可用于
当前实现评审。

## 调度子系统

- [Program 状态机](../../../design/module/scheduling/program-state-machine.md)
- [Progress-TTL 调度](../../../design/module/scheduling/progress-ttl-scheduling.md)
- [vLLM 运行时集成](../../../design/module/scheduling/vllm-runtime-integration.md)

这些文档定义 Program 生命周期转换、Progress-TTL 的容量与连续性决策，以及 AgentInfer 准入控制和 vLLM 原生调度器之间的边界。

## AgentCache Skills

- [Skills 设计](../../../design/superpowers/specs/agentcache-skills-design.md)
- [Skills 实施计划](../../../design/superpowers/plans/agentcache-skills-implementation.md)

Skills 的当前使用入口仍位于仓根目录的 [`skills/`](../../../skills/README.md)。设计稿解释历史决策，不替代
各 `SKILL.md` 中的现行指令。

## 维护记录

- [CI 与单元测试修复记录](../../../design/ci-ut-fixes.md)
