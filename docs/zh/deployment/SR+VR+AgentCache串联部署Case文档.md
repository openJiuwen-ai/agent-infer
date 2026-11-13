# SR + vLLM-Router + AgentCache 串联部署 Case 文档

> 目标：团队把 **Semantic-Router（SR，模型级路由）+ vLLM-Router（VR，副本级负载均衡）+ AgentCache（vLLM 引擎插件）** 串成一个解决方案，部署在**昇腾 A3** NPU 上做验证。本文讲清楚三件东西：①三组件各自怎么部署、②VR 怎么部署模型实例/worker（含 PD 分离）、③模型组合最少需要几张卡。
>
> 代码基线（2026-11-10 同步）：
> | 组件 | 仓库 | 基线 |
> |---|---|---|
> | SR | vllm-project/semantic-router | main=bc5c6bfb7（2026-11-10，config schema 已演进至 v0.5） |
> | VR | vllm-project/router | main=29ac71f（2026-11-10，含 WASM host #251+ 与 `--intra-node-data-parallel-size`） |
> | AgentCache | 本仓 openJiuwen/agent-infer | main=efd66f9（含 0.2.0 tag） |

---

## 1. 三组件分工与数据流

```
Agent 业务方
    │  OpenAI 协议，model=auto（或显式模型名），可携带 agent 会话标识
    ▼
┌─────────────────────────────────────────────────────┐
│ SR 栈（vllm-sr serve 拉起：envoy + router + 模型运行时）│
│ 模型级路由：按关键词/信号/决策规则选档位                │
└──────────────┬──────────────────┬───────────────────┘
        高档流量 │                  │ 低档流量
                ▼                  ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ VR-A（每模型一个 VR 实例） │  │ VR-B                     │
│ --policy cache_aware     │  │ --policy cache_aware     │
│ --wasm-middleware        │  │ --wasm-middleware        │
│   agent_hint_affinity    │  │   agent_hint_affinity    │
└──────┬─────────┬────────┘  └───────────┬─────────────┘
       │ 会话亲和→副本1       │ 会话亲和→副本2        │
       ▼                     ▼                       ▼
┌──────────────┐      ┌──────────────┐        ┌──────────────┐
│ vLLM 引擎 #1  │      │ vLLM 引擎 #2  │        │ vLLM 引擎 #N  │
│ --agentinfer  │      │ --agentinfer  │        │ --agentinfer  │
│ （注入 AgentCache│     │ （注入 AgentCache│      │ （注入 AgentCache│
│  scheduler bridge│   │  scheduler bridge│     │  scheduler bridge│
│  + middleware）│      │  + middleware）│       │  + middleware）│
│ NPU 0,1,2,3   │      │ NPU 4,5,6,7  │        │ NPU ...       │
└──────────────┘      └──────────────┘        └──────────────┘
```

**串联的核心粘合点（为什么这三个能构成一个方案）**：
1. **SR** 决定"这个请求用哪一档模型"（模型级），并把流量发到该档位的 VR 入口；
2. **VR** 用 `--policy cache_aware` 做副本级负载均衡，同时加载 AgentInfer 提供的 **WASM `agent_hint_affinity` 插件**（把请求里的 `agent_hint.session_id` 复制到 `session_params.session_id`）——同一个 agent 会话的请求被**钉在同一个 vLLM 副本**上；
3. **AgentCache** 以插件形式注入每个 vLLM 引擎（scheduler bridge + middleware + Progress-TTL 控制器），配合会话亲和实现 agent 场景的 KV cache 复用/调度管理。

> WASM 亲和插件的构建与运行见本仓 `agentinfer/agentrouter/README.md`（WASM guest：`agent_hint_affinity/`；VR 侧 host 能力 `--wasm-middleware` 来自 vllm-project/router #251+，已包含在所依赖的 main 版本中）。

---

## 2. VR 怎么部署模型实例 / worker

### 2.1 三种接入方式（README.md L61-152 佐证）

| 方式 | 关键参数 | 机制 |
|---|---|---|
| ① 静态 worker 列表 | `--worker-urls http://w1:8000 http://w2:8000` | 你自己起 N 个 vLLM 实例（各自占一组卡），VR 启动时全部纳入并健康检查 |
| ② 数据并行（DP） | `--worker-urls ... --intra-node-data-parallel-size 8` | **VR 自动把每个 worker URL 展开为 N 个 DP-aware worker**（README L63-83）——不用手工起 N 个副本 |
| ③ PD 分离 | `--vllm-pd-disaggregation` + connector 参数 | prefill / decode 分开部署，VR 按角色调度 |

**回答"能不能 PD 分离"：能，且支持三种 KV connector**（README L108-152）：

| Connector | 参数形态 | 特点 |
|---|---|---|
| NIXL | `--prefill http://...:8081 --prefill ... --decode http://...:8083 --decode ...` | P/D 地址静态列出 |
| NCCL | `--vllm-pd-disaggregation --vllm-discovery-address 0.0.0.0:30001`（+ `--prefill-policy` / `--decode-policy`） | ZMQ 服务发现，vLLM worker 自注册（与 v0.1.15 手册的形态一致） |
| Mooncake | `--kv-connector mooncake --prefill ... --decode ...` | 启动时查询各 prefill 节点 bootstrap server，自动注入 kv_transfer_params |

### 2.2 多模型：一个 VR 还是多个 VR？

- VR 支持**单实例多模型**：`src/routers/router_manager.rs`（L110-152）按请求的 `model_id` 从 worker registry 查出该模型的 worker 子集，再按子集选择 `http-pd` / `http-regular` 路由器分发。
- **本 case 采用"每模型一个 VR 实例"**：①一个实例只能一种部署形态（常规 vs PD 是实例级开关）；②换卡/重启互不影响；③SR 侧每模型一个 backend_ref 一一对应。

### 2.3 负载均衡策略

`--policy` 可选：`round_robin` / `random` / `consistent_hash` / `power_of_two` / `cache_aware`（README L228-248）。**本 case 固定 `cache_aware`**——配合 WASM 亲和插件，命中同一 agent 会话的 KV cache。

---

## 3. AgentCache 怎么注入 vLLM 引擎

依据本仓 `docs/en/how-to/integrate-vllm.md` 与 `agentinfer/agentcache/entrypoints/cli/serve_profile.py`：

**前置**：AgentInfer 0.2.0 与 vLLM **0.23.0** 装进**同一个 Python 环境**（wheel 或源码 `pip install -e .`）；每个实例分配**唯一的 lifecycle socket**。

```bash
# ① 方式一：一键注入（推荐）
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle-<实例编号>.sock

vllm serve <模型路径> \
  --served-model-name <模型名> \
  --tensor-parallel-size 4 \
  --port 8100 \
  --agentinfer
```

`--agentinfer` 会自动注入（stderr 打印 `[agentinfer] injected: ...`）：
- `--scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge`（或 `--no-async-scheduling` 时为 Sync 版）
- `--middleware AgentCacheIdentityMiddleware` + `--middleware AgentCacheLifecycleMiddleware`
- 嵌入式 Progress-TTL 控制器（策略经 `--additional-config '{"agentcache":{"progress_ttl":{...}}}'` 校准，默认值见 integrate-vllm.md L79-97）

```bash
# ② 方式二：显式长命令（等价，便于精细控制）
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle-<实例编号>.sock

vllm serve <模型路径> \
  --async-scheduling \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware
```

注意：`--agentinfer` 不能与 `--scheduler-cls` 同时给；同机多实例 socket 不能复用；stale socket 先确认旧进程已停再删。

---

## 4. SR 接入

最新 schema（`src/semantic-router/pkg/configschema/router-config-v0.5.schema.json`，config/README.md）：

```yaml
providers:
  models:
    - name: glm53-flash            # 对 Agent 暴露的模型别名
      provider_model_id: glm-5.3-flash   # ★ 必须与 vLLM 的 --served-model-name 一致
      api_format: openai
      backend_refs:
        - name: vr-a
          endpoint: "<VR-A的IP>:31000/v1"   # ★ 新字段名 endpoint（host:port 或完整 URL），不再是 base_url
          protocol: http
          provider: vllm
          auth:
            header: Authorization            # 默认值，可省
            prefix: Bearer
            api_key_env: OPENAI_API_KEY      # ★ key 走环境变量，不写明文（config/README.md L116 边界）
          weight: 100
    - name: qwen36-35b
      provider_model_id: qwen3.6-35b
      api_format: openai
      backend_refs:
        - name: vr-b
          endpoint: "<VR-B的IP>:31010/v1"
          protocol: http
          provider: vllm
          auth:
            api_key_env: OPENAI_API_KEY
          weight: 100
```

其他要点（config/README.md）：
- 同一别名的多个 `backend_refs` = **同构副本**（weight 加权，HTTP 可按 host/port/weight 变化）——把"SR 直连多个 vLLM 副本"与"经 VR"都覆盖了；本 case 经 VR；
- 新增 `recipes`（路由配置档）与 `entrypoints`（入口映射）概念，最简单的单档场景可以不用；
- 配置校验：`vllm-sr config validate --config config.yaml` → `vllm-sr serve --config config.yaml`；
- SR 自带的小模型运行时（307M 信号编码器 + decision 模型）**可跑 NPU**（`--device npu:0`，已在 Ascend 910B1/950DT 上实测，精度对齐记录见 vllm-project/semantic-router 仓 `src/model-runtime/docs/records/npu-parity-ascend910b1.md`），但 `vllm-sr serve` 编排自建容器**不带 NPU 挂载**：要么外部运行时方式把 Router 的 deployment 指向自备容器，要么整栈宿主机跑。**case 初期建议 SR 模型运行时留 CPU**（省一张卡），后续再切 NPU。

---

## 5. NPU 场景：昇腾 A3 卡数估算

### 5.1 硬件与模型规格

| 项 | 规格 |
|---|---|
| 昇腾 910C 单卡 | FP16 800 TFLOPS，HBM **128GB**，3.2TB/s |
| Atlas 800I A3 Server | 单机 **8×910C = 1024GB** HBM |
| GLM-5.3 | 稀疏 MoE，**744B 总参**，1M 上下文，MIT 开源 |
| GLM-5.3-Flash | MoE，**320B 总参 / 18B 激活**，线性+稀疏混合注意力，KV 缓存比 GLM-5.3 小 **4.44 倍**，1M 上下文 |
| qwen3.6 35B | dense 35B（按型号名） |

（模型规格来源：智谱开放文档 docs.bigmodel.cn、SiliconFlow 模型页；910C 规格来源：华为芯片路线图公开报道。）

### 5.2 W8A8 量化说明（本 case 全部按此口径）

**W8A8 = Weight 8-bit + Activation 8-bit**：把权重和激活值从 bf16/fp16（2 字节）压缩到 INT8（1 字节）的量化方案，vLLM 原生支持，是大模型生产推理的主流形态。

| 项 | 结论 |
|---|---|
| 精度劣化 | 有，但通常 **<1%**（多数基准落在评测噪声范围）；数学精度敏感场景建议上线前 A/B 对比 |
| 性能收益 | 权重显存减半 → 同卡数下 KV cache 空间翻倍；解码是显存带宽瓶颈，每 token 读取量减半 → **吞吐更高**；910C 的 INT8 算力高于 FP16 |
| 权重来源 | 优先用官方/社区发布的量化权重（如 HuggingFace 上 GLM 系列的量化版）；否则用 llm-compressor 等工具自行量化 |
| 是否可选 | **对 GLM-5.3 不是可选项**：bf16 ~1.5TB 单机 1024GB 放不下，只有 W8A8（~744GB）才能单机部署。本文所有卡数结论均基于 W8A8 |

### 5.3 显存与 TP 估算（权重 = 参数 × 1Byte(W8A8) / 2Byte(bf16)，再留 KV+激活余量）

| 模型 | bf16 权重 | W8A8 权重 | 最小 TP（W8A8） | 说明 |
|---|---|---|---|---|
| GLM-5.3 | ~1.5TB（单机放不下） | ~744GB | **TP8 = 8 卡**（1024GB，余 ~280GB 给 KV/激活） | 1M 上下文高并发时 KV 吃紧，长场景需关注；bf16 需 ≥16 卡（跨机） |
| GLM-5.3-Flash | ~640GB | ~320GB | **TP4 = 4 卡**（512GB，余 ~190GB） | 混合注意力 KV 省 4.44 倍，长上下文余量好；bf16 需 TP8 整机。⚠️ **TP2 不可行**（每卡 160GB > 128GB 卡容量）→ Flash 无降 TP/多副本选项 |
| qwen3.6 35B | ~70GB | ~35GB | **TP1 = 1 卡**（bf16 单卡也放得下） | 2 卡可起 2 副本做 DP |

### 5.4 组合最少卡数（结论）

**前提假设**：**W8A8 量化权重（见 §5.2）**、常规部署（不启用 PD）、每模型 1 个 VR + ≥1 实例。

| 组合 | 卡数构成 | 最少卡数 | 硬件形态 |
|---|---|---|---|
| **两档：GLM-5.3-Flash + qwen3.6 35B** | Flash TP4（4卡）+ qwen TP1（1卡） | **5 卡** | 1 台 A3（8 卡）即可，余 3 卡 |
| 两档推荐形态 | Flash TP4 + qwen TP1×2 副本（DP=2） | 6 卡 | 1 台 A3，余 2 卡；qwen 有均衡对象 |
| **三档：GLM-5.3 + GLM-5.3-Flash + qwen3.6 35B** | GLM-5.3 TP8（8卡，独占一台）+ Flash TP4（4卡）+ qwen TP1（1卡） | **13 卡 → 至少 2 台 A3（16 卡）** | 2 台：机1=GLM-5.3 整机；机2=Flash+qwen，余 3 卡 |
| 三档若全部 bf16 | GLM-5.3 ≥16 卡 + Flash 8 卡 + qwen 1 卡 | ≥25 卡 → 3 台 | 不推荐，除非必须 bf16 |

**PD 分离对卡数的影响**：PD 要求 P、D 各自放下全量权重，卡数约翻倍（如 Flash PD = P TP4 + D TP4 = 8 卡）。所以"最少卡数"场景**不启用 PD**；PD 作为后续长上下文/高吞吐扩展项（VR 三种 connector 都可试，昇腾上优先验证 NIXL/Mooncake 的 vllm-ascend 适配）。

---

## 6. 部署 Case 集合（昇腾 A3 四套拓扑）

> 四套 case 递进：**Case A** 跑通最小链路 → **Case B** 单机 DP 用满 8 卡 → **Case C** 三档双机 → **Case D** 四机多副本，VR 全面发力。所有引擎命令都含 `--agentinfer`（AgentCache 注入），lifecycle socket **每实例唯一**；VR 都加载 WASM 亲和插件（构建方式见 §7 步骤 2）。`ASCEND_RT_VISIBLE_DEVICES` 圈卡，卡不够时改编号重建容器即可，VR/SR 配置不动。

### Case A：最小部署（两档，5 卡，1 台 A3）

**卡分配**：

| NPU | 用途 |
|---|---|
| 0-3 | GLM-5.3-Flash，TP4，:8100 |
| 4 | qwen3.6 35B，TP1，:8110 |
| 5-7 | 空闲（后续扩展） |

```bash
# 引擎 ×2
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-a1.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8100 --agentinfer

export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-a2.sock
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /models/qwen3.6-35b \
  --served-model-name qwen3.6-35b --port 8110 --agentinfer

# VR ×2（--worker-urls 静态接入；metrics 端口错开）
vllm-router --host 0.0.0.0 --port 31000 \
  --worker-urls http://<本机IP>:8100 \
  --policy cache_aware \
  --wasm-middleware ./agent_hint_affinity.component.wasm --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions \
  --prometheus-host 0.0.0.0 --prometheus-port 29000

vllm-router --host 0.0.0.0 --port 31010 \
  --worker-urls http://<本机IP>:8110 \
  --policy cache_aware \
  --wasm-middleware ./agent_hint_affinity.component.wasm --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions \
  --prometheus-host 0.0.0.0 --prometheus-port 29001
```

SR 的 `backend_refs` 只需两个：`glm53-flash → <本机IP>:31000/v1`、`qwen36-35b → <本机IP>:31010/v1`（写法见 §4）。

### Case B：单机用满 8 卡（两档 + DP）

**主推方案**：Flash 保持 TP4 单副本（单请求吞吐最大），低档 qwen 用 **vLLM 原生数据并行 DP4** 吃掉剩余 4 卡：

| NPU | 用途 |
|---|---|
| 0-3 | GLM-5.3-Flash，TP4 ×1 副本，:8100 |
| 4-7 | qwen3.6 35B，**--data-parallel-size 4**（4 个 rank，每 rank 1 卡），:8110 |

```bash
# 引擎：qwen 以 DP 模式单进程起 4 rank（vLLM 内部 1 个 API server 端口）
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-b1.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8100 --agentinfer

export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-b2.sock
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve /models/qwen3.6-35b \
  --served-model-name qwen3.6-35b --data-parallel-size 4 --port 8110 --agentinfer

# VR：qwen 侧声明 --intra-node-data-parallel-size 4，VR 自动创建 DP-aware workers
# 并做 rank 级分发（router README L63-83；worker 端 --data-parallel-size 与该参数保持一致）
vllm-router --host 0.0.0.0 --port 31000 \
  --worker-urls http://<本机IP>:8100 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29000

vllm-router --host 0.0.0.0 --port 31010 \
  --worker-urls http://<本机IP>:8110 \
  --intra-node-data-parallel-size 4 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29001
```

**变体（长上下文优先）**：qwen 改 **TP2×2 副本**（NPU 4,5 / 6,7，:8110/:8111，各自独立 `--agentinfer` 进程、socket 不同），VR-B 的 `--worker-urls` 列两个地址（去掉 `--intra-node-data-parallel-size`）——qwen 的 KV 池翻倍、单请求时延更低，代价是并发路数减半。⚠️ 注意 Flash **不能**改 TP2×2 副本：W8A8 下 TP2 每卡需 160GB > 128GB 卡容量，Flash 的 4 张卡锁死 TP4 单副本，其并发能力由 vLLM 引擎内的 continuous batching 承担。选择依据：qwen 高并发优先 → 主推 DP4；qwen 长上下文/单请求质量优先 → 变体。

### Case C：高中低三档，双机部署（2×A3 = 16 卡，含 DP）

**约束回顾**：GLM-5.3 W8A8 权重 ~744GB，TP4（512GB）放不下 → **只能 TP8 单副本**，机 1 没有 DP 空间；DP 空间全部留给机 2 的低档。

**机 1（GLM-5.3 高档，整机）**：

| NPU | 用途 |
|---|---|
| 0-7 | GLM-5.3，W8A8，TP8 ×1 副本，:8200 |

**机 2（Flash 中档 + qwen 低档）**：

| NPU | 用途 |
|---|---|
| 0-3 | GLM-5.3-Flash，TP4 ×1 副本，:8100 |
| 4-7 | qwen3.6 35B，**DP4**（4 rank），:8110 |

```bash
# ── 机 1 ──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-c1.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /models/glm-5.3 \
  --served-model-name glm-5.3 --tensor-parallel-size 8 --port 8200 --agentinfer

vllm-router --host 0.0.0.0 --port 31020 \
  --worker-urls http://<机1IP>:8200 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29020
# 注意：单副本没有均衡对象，VR 仍提供统一入口/健康检查/指标

# ── 机 2 ──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-c2.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8100 --agentinfer

export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-c3.sock
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve /models/qwen3.6-35b \
  --served-model-name qwen3.6-35b --data-parallel-size 4 --port 8110 --agentinfer

vllm-router --host 0.0.0.0 --port 31000 \
  --worker-urls http://<机2IP>:8100 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29000

vllm-router --host 0.0.0.0 --port 31010 \
  --worker-urls http://<机2IP>:8110 \
  --intra-node-data-parallel-size 4 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29001
```

**SR 侧**：SR 栈部署在机 2（或独立管理节点），三个 `backend_refs` 跨机指向：

```yaml
      backend_refs:
        - name: vr-high-glm53
          endpoint: "<机1IP>:31020/v1"
          protocol: http
          provider: vllm
          auth:
            api_key_env: OPENAI_API_KEY
          weight: 100
      # glm53-flash → <机2IP>:31000/v1、qwen36-35b → <机2IP>:31010/v1（同上格式）
```

routing 决策示例：复杂/长程任务 → `glm-5.3`（高档），代码/中等任务 → `glm-5.3-flash`（中档），简单问答/高并发 → `qwen3.6-35b`（低档）。

**机 2 变体**：低档长上下文要求高时改 qwen **TP2×2 副本**（NPU 4,5 / 6,7）——KV 池翻倍、单请求时延更低，静态 `--worker-urls` 接入；⚠️ Flash 不能降 TP（W8A8 TP2 每卡 160GB > 128GB），保持 TP4 单副本。默认主推方案保留 Flash TP4 + qwen DP4。

### Case D：四机完整形态（4×A3 = 32 卡，三档全部多副本，VR 全面发力）

> Case A-C 里 VR 存在感弱是有原因的：单 worker 时 VR 没有均衡对象（只剩入口/健康/指标），DP4 时也只在低档做 rank 级分发。**Case D 让三档都有 ≥2 个 worker**，VR 的 cache_aware 前缀亲和、副本容错、健康摘除全部用上——这是贴近生产的形态。

**卡分配（32 卡全满）**：

| 机器 | NPU | 用途 |
|---|---|---|
| 机 1 | 0-7 | GLM-5.3 **副本 1**，TP8，:8200 |
| 机 2 | 0-7 | GLM-5.3 **副本 2**，TP8，:8200（高档终于有了均衡对象） |
| 机 3 | 0-3 / 4-7 | GLM-5.3-Flash **副本 1 / 副本 2**，TP4，:8100 / :8101 |
| 机 4 | 0-7 | qwen3.6 35B，**DP8**（8 rank，每 rank 1 卡），:8110 |

```bash
# ── 机 1 / 机 2：GLM-5.3 两个副本（命令相同，仅机器和 socket 不同）──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-d1.sock   # 机 2 用 d2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /models/glm-5.3 \
  --served-model-name glm-5.3 --tensor-parallel-size 8 --port 8200 --agentinfer

# ── 机 3：Flash 两个副本 ──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-d3.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8100 --agentinfer
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-d4.sock
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8101 --agentinfer

# ── 机 4：qwen DP8 ──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-lc-d5.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /models/qwen3.6-35b \
  --served-model-name qwen3.6-35b --data-parallel-size 8 --port 8110 --agentinfer

# ── VR ×3（部署在机 1-3 各自或独立管理节点；worker-urls 跨机列出）──
# VR-high：GLM-5.3 双副本，cache_aware 按前缀哈希在机 1/机 2 之间分发
vllm-router --host 0.0.0.0 --port 31020 \
  --worker-urls http://<机1IP>:8200 http://<机2IP>:8200 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29020

# VR-mid：Flash 双副本
vllm-router --host 0.0.0.0 --port 31000 \
  --worker-urls http://<机3IP>:8100 http://<机3IP>:8101 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29000

# VR-low：qwen DP8 rank 级分发
vllm-router --host 0.0.0.0 --port 31010 \
  --worker-urls http://<机4IP>:8110 \
  --intra-node-data-parallel-size 8 \
  --policy cache_aware --wasm-middleware ... --prometheus-port 29001
```

SR 侧三个 `backend_refs` 分别指向 `<VR-high IP>:31020/v1`、`<VR-mid IP>:31000/v1`、`<VR-low IP>:31010/v1`（写法同 §4 / Case C）。

**VR 在这个 case 里干了什么（对照 Case A-C 的"没劲儿"）**：

| VR 能力 | 在 Case D 的体现 |
|---|---|
| cache_aware 前缀亲和 | 同一 agent 会话/同一长 system prompt 的请求钉在同一个副本上，KV cache 命中率最大化（配合 WASM `agent_hint_affinity` 的 session_id 亲和） |
| 副本级负载均衡 | 高档 GLM-5.3 双副本按负载/前缀分发；Flash 双副本同理 |
| 故障容错 | 机 1 整机故障 → VR 健康检查自动摘除，高档流量全落机 2，服务不中断（降级而非宕机） |
| rank 级 DP 分发 | qwen 8 rank 由 VR-low 统一分发，对外仍是一个入口 |
| 统一观测 | 三个 VR 各自的 Prometheus 指标（:29020/:29000/:29001）+ SR 的 x-vsr-* 观测头，两层都能看流量分布 |

**可选扩展**：①机 4 的 qwen 改 TP2×4 副本（长上下文优先）；②GLM-5.3 两台机改 PD 形态（P TP8 + D TP8）——吞吐形态从"两副本"变成"一条 PD 流水线"，适合超长上下文重 prefill 场景，但失去副本容错，建议作为对照实验而非默认。

### Case 选择速查

| Case | 卡数 | 适用 | 特点 |
|---|---|---|---|
| A | 5（余 3） | 初期联调、功能验证 | 链路最短，最快跑通三组件串联；VR 仅入口/观测 |
| B | 8（满） | 单机性能摸底、并发压测 | 引入 DP（rank 级分发），VR 均衡在低档起作用 |
| C | 16（2 台） | 三档完整方案演示 | GLM-5.3 上场；SR 跨机 backend_refs；可做档位路由 E2E |
| D | 32（4 台） | **贴近生产的完整形态** | 三档全部多副本；cache_aware 亲和、副本容错、健康摘除全面验证；VR 价值最大化 |

---

## 7. 端到端部署步骤（case 执行顺序）

```bash
# ── 步骤 0：环境 ──
# vllm-ascend 环境（A3/910C 需 vllm-ascend 对应版本 + CANN）+ pip install agentinfer-0.2.0 wheel（与 vllm 同环境）
python -c "import agentinfer, vllm; print(vllm.__version__)"   # 期望 0.23.0

# ── 步骤 1：起 vLLM 引擎（每个实例注入 agentcache，socket 唯一）──
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle-a1.sock
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/glm-5.3-flash \
  --served-model-name glm-5.3-flash --tensor-parallel-size 4 --port 8100 --agentinfer

export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle-b1.sock
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /models/qwen3.6-35b \
  --served-model-name qwen3.6-35b --port 8110 --agentinfer
# （可选第二副本：ASCEND_RT_VISIBLE_DEVICES=5 --port 8111）

# ── 步骤 2：构建并起 VR（每模型一个实例，cache_aware + WASM 亲和插件）──
# 构建 affinity guest：agentinfer/agentrouter/agent_hint_affinity/build.sh
DIGEST=$(sha256sum agent_hint_affinity.component.wasm | awk '{print $1}')

vllm-router --host 0.0.0.0 --port 31000 \
  --worker-urls http://<本机IP>:8100 \
  --policy cache_aware \
  --wasm-middleware ./agent_hint_affinity.component.wasm \
  --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions \
  --prometheus-host 0.0.0.0

vllm-router --host 0.0.0.0 --port 31010 \
  --worker-urls http://<本机IP>:8100 [...qwen 副本列表...] \
  --policy cache_aware \
  --wasm-middleware ./agent_hint_affinity.component.wasm \
  --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions \
  --prometheus-host 0.0.0.0

# ── 步骤 3：起 SR 栈（v0.5 config，backend_refs 指向两个 VR）──
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml     # SR 的 providers 配置见 §4

# ── 步骤 4：验证 ──
# 4.1 每实例直发（跳过 VR）确认引擎+插件正常，stderr 应有 [agentinfer] injected 行
# 4.2 经 VR 发同一 agent 会话多次请求 → 确认钉在同一副本（cache_aware + session_id 亲和）
# 4.3 经 SR 发 model=auto → 确认档位路由 + 观测头
```

---

## 8. 风险与待确认项

| # | 风险/待确认 | 影响 | 建议 |
|---|---|---|---|
| 1 | vllm-ascend 对 910C + GLM-5.3-Flash 混合注意力（线性+稀疏）的支持版本 | Flash 能否上 A3 的前提 | 部署前核对 vllm-ascend release notes 与模型支持列表；必要时先在 910B 验证 |
| 2 | AgentInfer 0.2.0 绑定 vLLM 0.23.0；vllm-ascend 是 vLLM 的衍生分支，scheduler 接口可能有差异 | `--agentinfer` 注入可能在 vllm-ascend 环境失败 | 先在 CPU/x86 vLLM 0.23.0 验证插件功能，再试 vllm-ascend 环境；不行则排查 scheduler bridge 的接口差异 |
| 3 | GLM-5.3 TP8 W8A8 的 KV 余量（~280GB）在 1M 上下文高并发下吃紧 | 长上下文场景吞吐受限 | 用 Flash 承接长上下文流量（KV 省 4.44 倍）；或对 GLM-5.3 评估 W4 量化 |
| 4 | SR 模型运行时的 NPU 化：vllm-sr serve 编排容器不带 NPU 挂载 | SR 信号模型跑 CPU（初期可接受） | 后续按 ascend-npu.md 的"外部运行时"方式切换 |
| 5 | VR 镜像架构：Atlas 800I A3 是 aarch64 | 需 arm64 构建 VR 镜像 | 按 router 仓 Dockerfile 在鲲鹏机上构建 |
| 6 | WASM 亲和插件要求 VR ≥ #251（已满足，main=29ac71f） | 无 | 直接用 |
