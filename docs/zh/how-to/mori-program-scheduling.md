# MORI Program 调度

`vllm serve MODEL --agentinfer --agentinfer-strategy mori` 选用 MORI。省略策略选项时使用 Progress-TTL。
同步与异步选项选择相同的 AgentCache SchedulerBridge。当前 MORI 面向单 EngineCore、单 DP rank。

MORI 需要宿主提供 `ProgramSchedulingHost`，由宿主报告有效 HBM、DRAM 和健康状态，
并应用有序的 Program 放置与释放操作。该能力缺失时服务启动会报错。
宿主实现可以独立于 Program 策略交付。

## 调度路径

- vLLM API 在 Chat 模板处理和分词前提取字符数、前缀分组和 Program 身份。
  SchedulerBridge 通过 vLLM 内部 IPC 返回准入结果。
- API 的 SSE 进度与 usage 进入同一个 Program 调度运行时。定时唤醒通过 IPC
  进入 EngineCore。Bridge 从本地宿主读取容量，HTTP 指标服务独立用于观测。
- MORI 在一次周期检查中依次执行空闲置换、WARM 准入、WAITING 准入、HBM 溢出处理和 DRAM 溢出处理。
  Program 放置决策通过宿主协议应用。

入口与协议参见 [Bridge](../../../agentinfer/agentcache/core/program_scheduling.py)、
[IPC](../../../agentinfer/agentcache/core/scheduling_ipc.py) 和
[策略](../../../agentinfer/scheduling/mori/strategy.py)。
