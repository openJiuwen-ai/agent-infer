# 接入 vLLM

本指南将当前 AgentInfer 服务栈接入已有的 vLLM 0.23.0 部署。AgentInfer 必须与 vLLM 安装在同一 Python
环境中。

## 将 AgentInfer 安装到 vLLM 环境

源码开发时，激活 vLLM 服务使用的环境，并以可编辑模式安装 AgentInfer：

```bash
python -m pip install -e .
```

发布部署时，下载 [AgentInfer 0.1.0 wheel][release-wheel]，然后执行：

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

确认两个包由同一解释器加载：

```bash
python -c "import agentinfer, vllm; print(vllm.__version__)"
```

命令应输出 `0.23.0`。

## 配置生命周期传输

为生命周期观测选择一个唯一的本地 Unix socket。缺少该变量时，生命周期中间件会拒绝启动：

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle.sock
```

不得复用其他服务正在使用的 socket。只有确认原进程已经停止后，才能删除遗留 socket。

## 启动 AgentInfer 服务路径

使用异步调度桥、两个 API 中间件和 Progress-TTL 控制器工厂启动 vLLM：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --async-scheduling \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
  --additional-config \
  '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'
```

根据部署需要追加张量并行、端口、模型专用工具解析和 Prefix Cache 等标准 vLLM 参数。上游参数见
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。

### 配置 Progress-TTL 策略参数

策略参数位于 `additional_config.agentcache.progress_ttl`。下面的示例展示主要的部署标定参数和当前策略使用的工作量前视防饥饿边界：

```bash
--additional-config '{
  "agentcache": {
    "controller_factory": "agentinfer.agentcache.core.factory.build_progress_ttl_controller",
    "progress_ttl": {
      "ttl_min_seconds": 0.05,
      "ttl_max_seconds": 32,
      "resume_capacity_ratio": 1.0,
      "resume_order": "mru",
      "ttl_prefill_model_intercept_seconds": 0.042935,
      "ttl_prefill_model_linear_seconds_per_1k_tokens": 0.080027,
      "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared": 0.00220962,
      "decode_step_fixed_seconds": 0.012112,
      "decode_step_seconds_per_request": 0.0006939,
      "decode_step_seconds_per_context_token": 2.2516e-7,
      "force_resume_timeout_scale": 3.0,
      "force_resume_timeout_min_seconds": 30,
      "force_resume_timeout_max_seconds": 300
    }
  }
}'
```

Prefill 和 Decode 系数与部署环境相关，应通过标定获得。请求首次进入等待状态时，Force-resume 会冻结
deadline：用请求前方剩余轮数除以近期聚合请求吞吐，乘以 `force_resume_timeout_scale`，再限制到配置的
最小值和最大值之间。在有界吞吐窗口尚未完整时，策略保守地使用最大 timeout。

策略会拒绝已移除的历史字段，而不是静默忽略。请将
`ttl_prefill_seconds_per_1k_uncached_tokens` 替换为三个二次 Prefill 系数，并将
`force_resume_timeout_seconds` 替换为上述 scale、最小值和最大值。请删除
`target_min_segment_rounds`、`privileged_lookahead_rounds`、`pause_capacity_lookahead_rounds`、
`resume_fairness_weight`、`resume_resource_penalty_weight`、`capacity_safety_margin_tokens`、
`ttl_impact_multiplier`、`uncached_ratio_default` 和可配置的 `decode_buffer_tokens`。

## 了解 CLI 委托行为

AgentInfer 会安装一个委托型 `vllm` 控制台脚本：

- 显式的 `vllm bench serve --agentinfer` 命令进入 AgentBench。
- 其他命令委托给上游 vLLM，未设置 `--scheduler-cls` 时使用 AgentInfer 默认调度器。
- 显式传入的 `--scheduler-cls` 会被保留，如上面的服务启动命令所示。

旧版 `AgentAwareScheduler` 和 `agentinfer.LLM` 兼容接口仍然可用，但不会启用这里展示的显式
Progress-TTL 服务路径。

## 验证接入

模型加载完成后，在另一个终端查询服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

组件职责和选择关系见[架构概述](../explanation/architecture.md)，类签名见
[Python API 参考](../reference/python-api.md)。如需进行可复现的基线和候选组比较，请阅读
[运行基准测试](run-benchmark.md)。

[release-wheel]: https://gitcode.com/openJiuwen/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
