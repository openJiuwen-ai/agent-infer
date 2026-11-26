# MORI Program 调度

`vllm serve MODEL --agentinfer --agentinfer-strategy mori` 选择 MORI。省略策略选项时使用 Progress-TTL。
MORI 的 ProgramSchedulingBridge 在单个 EngineCore 内负责 Program 准入。
MoriScheduler 和 MoriAsyncScheduler 共用请求选批逻辑，由请求 KV manager 的 HiRadix 记录 HBM/RAM 副本。
公开服务通过一个端口接收 Chat 和 Program 释放。

在运行 vLLM 的环境中用 `python -m pip install -e .` 安装此检出。已验证 vLLM 0.31.0。启动时检查实际 ABI。
当前支持单 DP rank、一个 dense full-attention KV 组、FCFS、哈希和分配页相同、非推测解码。
以下示例给 Host KV 分配 1 GiB，实际可按模型和可用 RAM 调整。

```bash
vllm serve "$MODEL" --agentinfer --agentinfer-strategy mori \
  --kv-transfer-config '{
    "kv_connector": "MoriCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 1073741824, "kv_offload_backend": "cpu", "lazy_offload": false}
  }'
```

客户端在 Chat 请求体中提供 `program_id`，结束时向同一地址的 `/programs/release` 提交 `{"program_id":"…"}`。
启用 API key 时，Chat、释放和查询请求均须携带 vLLM 的 `Authorization: Bearer …`。
同 ID 在释放成功后重用会创建新代次。
`GET /programs` 查询 Program 状态，`GET /agentinfer/mori/metrics` 查询容量，
`GET /agentinfer/mori/cache` 查询每个 Program 代次的页归属。
默认异步调度，也可显式使用 `--no-async-scheduling` 选择同步调度。
需要对齐 SGLang 文本模板时设置 `agentcache.mori.sglang_text_chat`；
`sglang_unseeded_chat` 同时启用文本归一化、忽略请求 seed 和联合采样。
服务自动启动 Core 调度时钟。可以通过 `additional_config.agentcache.mori.interval_seconds` 设置间隔，默认 5 秒。

## 和 ThunderAgent 的对应关系

| 阶段 | 外置 ThunderAgent | 内嵌 SchedulerBridge |
| --- | --- | --- |
| Chat 到达 | Router 解析原始 HTTP JSON，从 Program ID、字符数和前缀分组决定准入。 | vLLM API middleware 在相同时点取值，发送 `Begin`，等 Core 准入后再进入原生 Chat。 |
| 生成进度 | Router 观察下游 SSE，每 20 条合格记录更新一次 Program token 估算。 | API middleware 观察返回的 SSE，用 `Progress` 上报同样的计数。 |
| 请求完成 | Router 消费 usage，将 Program 记为 ACTING，等待下一轮。 | API 把 usage 或无 usage 的响应结束上报为 `Finish`，Runtime 更新相同的调度状态。 |
| Program 结束 | 客户端显式请求 `/programs/release`，Router 清理队列及 KV 状态。 | 同一 HTTP 接口通过 `Release` 进入 Core。API 断开时用 `Cancel` 清理该请求的准入等待。 |
| 周期调度 | Router 自己计时，读取后端容量后执行五个阶段。 | Core 内的 MORI 计时器投递本地唤醒，Core 读取有效容量并执行同一顺序。 |

`Begin` 发生在模板处理和分词前。原生请求携带 attempt ID 供 Bridge 绑定已通过的准入。
即使两个请求具有相同的分词结果，原始 Chat 的 JSON 长度仍可能使 MORI 作出不同决定。
生成 token 数和客户端 SSE 记录数也各有采样规则。
因此请求事件沿用 API 的观察点，通过 vLLM 原有的 utility IPC 投递。
周期检查和 HBM／DRAM 容量全部留在 Core 内。

## 计时和本地缓存接口

MORI 用异步计时器向 EngineCore 的本地输入队列投递唤醒。
Core 空闲等待输入时可以被唤醒。Core 正等待 GPU batch 的 future 时，唤醒保留在队列，
等 owner 解阻后再执行决策。同步和异步 Scheduler 都由这个 Core owner 提交迁移。

`ProgramCacheAdapter` 提供本地健康及有效 HBM／DRAM，并提交有序的 Program 放置与释放。
它对应外置 ThunderAgent 的后端容量读取和 KV 操作接口。
策略和 Bridge 测试注入独立的记录型 adapter，生产服务由所选 KV 后端提供实现。
MORI 引擎在构造 HiRadix 后装配 adapter。用 `--kv-transfer-config` 指定 CPU 卸载和 `kv_role=kv_both`，有效容量取两级池扣除保留页后可容纳的 token 数。

对暂时失败的 placement，Bridge 保留有序操作并重试，相关准入等待确认。
adapter 应允许按同一 Program 代次重放操作。
显式 release 失败时返回 `released: false`，保留 Program 以便重试。
排队中的 Program 释放成功后，其尚未获准的请求结束等待。
已经获得准入许可的原生请求可以完成派发。

原版规则的对应方法标在 [策略](../../../agentinfer/scheduling/mori/strategy.py)、
[预算](../../../agentinfer/scheduling/mori/ledger.py)、
[API 观察](../../../agentinfer/agentcache/core/mori_api.py) 和
[IPC](../../../agentinfer/agentcache/core/scheduling_ipc.py) 的函数注释中。
