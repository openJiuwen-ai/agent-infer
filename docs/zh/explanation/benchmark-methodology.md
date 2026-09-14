# 基准方法

AgentInfer 基准用于回答一个受约束的问题：在相同智能体工作负载和部署条件下，候选调度方案相对固定的
vLLM 基线如何改变运行时间、请求和 Prefix Cache 指标。它不把 Agent 进程完成等同于任务正确。

## 实验组

```text
baseline:  Claude Code -> Request Proxy -> vLLM AsyncScheduler
candidate: Claude Code -> Request Proxy -> vLLM AgentCacheAsyncSchedulerBridge
```

两个实验组使用同一 Request Proxy、BenchKit 运行器、数据集选择和 Agent profile。调度差异只存在于 vLLM
启动配置，避免将请求观测或 Agent 驱动差异误判为调度收益。

## 证据层次

Collector 只采集外部原始证据和可用性；Metric 将来源明确的事实转换为标准化数值；Artifact 负责校验并
序列化任务和运行契约；Comparison 只读取已完成的摘要。这些层不互相复制公式。

请求 trace、服务指标、任务结果、补丁、环境和源码状态仍是权威证据。缺失值必须表示为 unavailable 或 not
applicable，不能按成功的零值处理。

## 正确性边界

`completed` 表示 Agent 执行流程完成，不表示生成的补丁正确。SWE-bench 正确性由独立后处理确定：

```text
model.patch
  -> 在 base_commit 检出任务仓库
  -> 应用补丁
  -> 运行 FAIL_TO_PASS 和 PASS_TO_PASS 测试
  -> 记录 resolved 或 unresolved
```

只有运行和缓存比较证据以及外部正确性证据均完整时，才能形成发布结论。

## 冷启动要求

命令按顺序运行不足以证明冷状态。每个实验组必须使用全新的服务进程，并通过服务启动证据或
`evidence/vllm_metrics_start.prom` 证明累计指标和 Prefix Cache 未从前一次运行继承。证据不足时，报告必须
发出警告。

基线结束后应彻底停止 vLLM，再启动候选组。结果目录、tmux 会话和生命周期 socket 也不得在实验间复用。

## 公平比较

每个配对实验必须保持以下条件一致：

- AgentInfer 提交、模型和硬件。
- 任务集合、任务顺序和 Agent profile。
- 并发度、单任务及整次运行超时。
- 张量并行、vLLM 参数和服务端点。
- Request Proxy 和证据采集配置。

一组基线和候选运行可用于方向性烟雾测试。发布决策应使用多次配对冷启动实验，并保留两个实验组的服务
日志、原始指标和任务产物。

执行步骤见[运行基准测试](../how-to/run-benchmark.md)，证据文件见
[运行产物参考](../reference/run-artifacts.md)。

## 多 Agent 与回放边界

BenchKit 支持 Claude Code、JiuwenSwarm 和 DSH，但一个配对实验必须保持运行时和 profile 一致。请求代理透明转发，身份优先来自规范化元数据；DSH 原生会话头仅提供降级识别，不能单独证明子
Agent 角色。

Trace Replay 复原请求负载与依赖，不能声称恢复原始工具语义或补丁正确性；比较时还需固定 trace、抽样种子和合成前缀预算。
