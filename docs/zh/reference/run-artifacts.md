# 运行产物参考

BenchKit 将每次运行写入一个全新的结果目录。原始请求事实、服务采集、任务结果、补丁和环境证据是权威数据；
`summary.json` 是这些事实的标准化汇总。

## 目录结构

```text
<result_dir>/
  manifest.json
  summary.json
  requests.jsonl
  evidence/
    environment.json
    source_control.json
    vllm_metrics_start.prom
    vllm_metrics_end.prom
  tasks/<instance_id>/
    result.json
    model.patch
    transcript.jsonl
    claude-settings.json
    claude-launch-command.txt
    terminal-*.log
  workspaces/<instance_id>/
```

无法采集的证据会在 `manifest.json` 或 `summary.json` 中标记为 unavailable 或 not applicable。对应的可选文件
可能不存在；缺少证据不会被静默转换为数值零或成功状态。

## 顶层文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 解析后的运行配置、CLI 元数据、生命周期状态和证据可用性。 |
| `summary.json` | 标准化的任务、请求、vLLM、正确性和源码健康指标。 |
| `requests.jsonl` | 每个代理请求一行的不可变请求事实。 |

`compare` 只读取已完成且兼容的摘要。未完成、错误配对或模式不兼容的基线和候选摘要会被拒绝。

## `evidence/`

| 文件 | 内容 |
| --- | --- |
| `environment.json` | 主机、Python 和运行环境证据。 |
| `source_control.json` | 分支、提交和工作区状态。 |
| `vllm_metrics_start.prom` | 工作负载开始前采集的原始 vLLM Prometheus 指标。 |
| `vllm_metrics_end.prom` | 工作负载结束后采集的原始 vLLM Prometheus 指标。 |

冷启动结论必须由服务启动证据或起始指标支持。无法证明冷状态时，比较报告会发出警告，不应将该运行用于
发布门禁结论。

## `tasks/<instance_id>/`

| 文件 | 内容 |
| --- | --- |
| `result.json` | Agent 结果、终止原因、拓扑和补丁状态。 |
| `model.patch` | Agent 生成的补丁；未生成补丁时可能不存在。 |
| `transcript.jsonl` | 标准化的 Agent 对话事件。 |
| `claude-settings.json` | 本次任务使用的非敏感 Claude Code 设置。 |
| `claude-launch-command.txt` | 已脱敏的启动命令。 |
| `terminal-*.log` | 周期性终端采集和最终输出。 |

凭据通过启动环境传入，不得出现在设置、命令或日志产物中。发现敏感值时，不应共享该结果目录，并应立即
轮换相关凭据。

## 工作区与正确性

`workspaces/<instance_id>/` 保存任务检出和 Agent 工作目录。`completed` 仅表示 Agent 执行契约完成；
`model.patch` 是否解决 SWE-bench 任务，需要在任务的 `base_commit` 上应用补丁并执行 `FAIL_TO_PASS` 和
`PASS_TO_PASS` 测试后确定。

评估边界和公平比较要求见[基准方法](../explanation/benchmark-methodology.md)。
