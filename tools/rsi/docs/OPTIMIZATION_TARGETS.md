# vllm-evolve 优化目标全清单

> 68个优化点，6个维度。框架的最终目标是全部覆盖。

> **⚠️ 当前实现范围(以 AGENTS.md / CLAUDE.md 为准)**:本仓库**唯一接到真实后端的支持目标是
> `scheduling`**;下面这份 68 点清单是**长期路线图/愿景**,不是当前已实现状态。**模拟器(DES)已删除**
> —— 所有评测只走 `ar bench`(真实 vLLM,GPU 门控),绝不伪造。因此下表"状态说明"里凡提到"模拟器"的
> 措辞均为历史遗留;真实状态请以 `config/targets/` 下实际接线的 target + `targets/scheduling/` 为准。

## 当前真正支持的目标(权威来源:AGENTS.md / CLAUDE.md)

> **唯一接到真实 vLLM 后端、可经 `ar bench` 真跑的支持目标是 `scheduling`**(见 `targets/scheduling/`)。
> `targets/` 下其它目录均为**实验性、未接真实后端**。**DES 模拟器已删除** —— 所有评测只走真实 vLLM
> (GPU 门控),绝不伪造。

## 图例(⚠️ 路线图收录状态,**不是**当前可运行支持)

> 下表每行的标记表示该优化点在**长期路线图清单**里的收录情况,**不代表已接到真实后端可跑**。当前可跑的
> 只有 `scheduling`(见上)。

- ✅ 已收录进路线图(设计/搜索空间层面已规划)
- 🔧 部分收录(有 YAML 定义或搜索空间雏形)
- ❌ 未收录

## 维度一：请求生命周期 (18个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 1 | Admission control | 请求准入/拒绝/排队 | Code | P0 | ✅ 在线admission模拟 |
| 2 | Request scheduling | 调度batch中prefill/decode的选择和顺序 | Code | P0 | ✅ |
| 3 | PD disaggregated routing | P/D分离场景的路由 | Code | P0 | ✅ |
| 4 | Multi-instance load balancing | 多副本间请求分发 | Code | P0 | ✅ |
| 5 | Prefix matching strategy | 前缀缓存匹配策略 | Code | P1 | ✅ |
| 6 | Prefill vs decode priority | 步内prefill/decode优先级 | Code | P0 | ✅ (scheduling内) |
| 7 | **Chunked prefill strategy** | 长prompt分块策略 | Code+Param | P1 | ✅ |
| 8 | Token budget allocation | 每步prefill/decode token预算分配 | Code | P0 | ✅ (scheduling内隐含) |
| 9 | KV cache eviction | 前缀缓存块淘汰 | Code | P0 | ✅ |
| 10 | KV cache allocation | 块分配策略(locality/NUMA-aware) | Code | P2 | ✅ |
| 11 | Preemption victim selection | 抢占对象选择 | Code | P0 | ✅ (preemption_fn) |
| 12 | **Preemption trigger threshold** | 抢占触发时机 | Param | P1 | ✅ scheduling target已通过preempt_ids支持主动抢占 |
| 13 | Continuous batching granularity | 重调度频率 | Param | P2 | ✅ scheduling内token budget控制 |
| 14 | **Speculative decoding strategy** | 推测解码草稿选择、验证策略 | Code+Param | P1 | ✅ |
| 15 | Spec decode tree structure | 推测树形状 | Param | P2 | ✅ |
| 16 | **Request priority scoring** | 动态优先级计算 | Code | P1 | ✅ |
| 17 | **Output length prediction** | 输出长度预估 | Code | P1 | ✅ |
| 18 | Request coalescing | 相同前缀请求合并prefill | Code | P2 | ✅ |

## 维度二：资源管理 (11个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 19 | **Memory partitioning** | GPU显存分配(KV/权重/激活) | Param | P0 | ✅ gpu_memory_utilization在搜索空间+simulate |
| 20 | **TP mapping** | tensor并行度+层分配 | Param | P0 | ✅ TP/EP在搜索空间，timing model含通信开销 |
| 21 | PP partitioning | pipeline并行分割策略 | Param | P1 | ✅ PP在搜索空间 |
| 22 | **EP degree (MoE)** | expert并行度 | Param | P0(MoE) | ✅ |
| 23 | Kernel launch ordering | GPU kernel执行顺序 | Code | P3 | ✅ |
| 24 | All-reduce algorithm | 集合通信算法选择 | Param | P2 | ✅ |
| 25 | **CPU offload/swap strategy** | KV块CPU换出策略 | Code | P1 | ✅ |
| 26 | CUDA graph capture sizes | 预捕获batch size列表 | Param | P2 | ✅ |
| 27 | **Graph compilation (Ascend)** | 图编译选项、算子融合边界 | Param | P1(Ascend) | ✅ |
| 28 | **Attention backend selection** | 注意力后端选择 | Param | P1 | ✅ |
| 29 | Block size selection | KV缓存块大小 | Param | P2 | ✅ |

## 维度三：工作负载特化 (11个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 30 | MoE expert routing | token到expert分配 | Code | P0(MoE) | ✅ |
| 31 | **MoE expert replication** | 热门expert副本放置 | Param | P1(MoE) | ✅ |
| 32 | MoE capacity factor | overflow/drop比率 | Param | P1(MoE) | ✅ |
| 33 | **Long-context attention** | 滑动窗口/稀疏注意力配置 | Param+Code | P1 | ✅ |
| 34 | **Long-context memory layout** | 128K+上下文的分层KV存储 | Code | P1 | ✅ |
| 35 | **Multi-modal token priority** | 视觉/音频token的prefill优先级 | Code | P1 | ✅ |
| 36 | Multi-modal encoder batching | 编码器批处理策略 | Code | P2 | ✅ |
| 37 | Structured output scheduling | 受约束解码不阻塞batch | Code | P2 | ✅ |
| 38 | **Tool-use suspend/resume** | 工具调用暂停decode保留KV | Code | P2 | ✅ |
| 39 | **LoRA adapter scheduling** | 多LoRA适配器加载/切换/亲和 | Code | P1 | ✅ |
| 40 | Embedding coscheduling | embedding与生成混排 | Code | P2 | ✅ |

## 维度四：系统级配置 Day-0 (16个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 41 | max_num_batched_tokens | 每步token预算 | Param | P0 | ✅ |
| 42 | max_num_seqs | 最大并发序列 | Param | P0 | ✅ |
| 43 | gpu_memory_utilization | KV缓存占显存比 | Param | P0 | ✅ |
| 44 | enable_chunked_prefill | 分块prefill开关 | Param | P1 | ✅ (搜索空间) |
| 45 | chunked_prefill_size | 分块大小 | Param | P1 | ✅ (搜索空间) |
| 46 | TP/PP/EP config | 并行度配置 | Param | P0 | ✅ |
| 47 | Quantization choice | 量化方案 | Param | P0 | ✅ (搜索空间) |
| 48 | KV cache dtype | fp8/fp16 | Param | P1 | ✅ (搜索空间) |
| 49 | Scheduling policy | fcfs/priority | Param | P1 | ✅ (搜索空间) |
| 50 | enable_prefix_caching | 前缀缓存开关 | Param | P1 | ✅ (搜索空间) |
| 51 | PD instance ratio | P/D实例比 | Param | P0(PD) | ✅ (搜索空间) |
| 52 | max_model_len | 最大序列长度 | Param | P1 | ✅ |
| 53 | swap_space_gb | CPU换出空间 | Param | P2 | ✅ |
| 54 | num_gpu_blocks_override | 手动KV块数 | Param | P2 | ✅ |
| 55 | enforce_eager | 禁用CUDA图 | Param | P2 | ✅ |
| 56 | distributed_executor_backend | Ray/multiprocessing | Param | P2 | ✅ |

## 维度五：多节点/集群 (9个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 57 | **Request load balancing** | 多副本请求分发 | Code | P0 | ✅ |
| 58 | Model placement | 异构集群模型放置 | Param | P1 | ✅ |
| 59 | **Auto-scaling trigger** | 扩缩容触发策略 | Code+Param | P1 | ✅ |
| 60 | Health check/failover | 故障检测和切换 | Code | P1 | ✅ |
| 61 | **KV cache migration** | PD间KV迁移策略 | Code | P1(PD) | ✅ |
| 62 | Cross-instance prefix sharing | 跨实例前缀缓存 | Code | P2 | ✅ |
| 63 | **Heterogeneous GPU scheduling** | 按负载路由到不同GPU类型 | Code | P1 | ✅ |
| 64 | Quota/rate limiting | 多租户配额限流 | Code+Param | P1 | ✅ |
| 65 | Warm pool management | 预加载实例池管理 | Param | P2 | ✅ |

## 维度六：质量/正确性权衡 (3个)

| # | 优化点 | 描述 | 演化对象 | 优先级 | 状态 |
|---|--------|------|---------|--------|------|
| 66 | Spec decode acceptance tuning | 推测解码接受阈值 | Param | P2 | ✅ |
| 67 | FP8 KV error budget | 按序列敏感度选KV精度 | Code | P2 | ✅ |
| 68 | Quantization-aware routing | 质量敏感请求路由 | Code | P2 | ✅ |

---

## 统计

> **下表统计的是路线图收录数,不是"已支持/可跑率"。** 当前接到真实后端的只有 `scheduling` 一个目标。

| 维度 | 总数 | ✅路线图收录 | 🔧 | ❌ |
|------|------|---|---|---|
| 请求生命周期 | 18 | 18 | 0 | 0 |
| 资源管理 | 11 | 11 | 0 | 0 |
| 工作负载特化 | 11 | 11 | 0 | 0 |
| 系统级配置 | 16 | 16 | 0 | 0 |
| 多节点/集群 | 9 | 9 | 0 | 0 |
| 质量权衡 | 3 | 3 | 0 | 0 |
| **合计** | **68** | **68 收录(≠支持率)** | **0** | **0** |
| **当前真实后端支持** | — | **仅 `scheduling`** | — | 其余实验性/未接线 |

---

## 迭代路线图

> **⚠️ 历史说明**:下面这份路线图写于 **DES 模拟器时代**,文中所有"DES 模拟器/模拟器组件/多实例 DES/
> 代价模型"措辞均为**历史遗留**。**模拟器已删除**,当前与未来的唯一路径是**把每个目标接到真实 vLLM 后端**
> (像 `scheduling` 那样,经 `ar bench` 真跑、真实指标、绝不伪造)。请把下文的"扩展 DES"一律理解为
> "新增真实 vLLM target 接线 + 真机评测"。

### Phase 1: 补全请求生命周期核心(路线图)

聚焦 P0/P1,**接真实 vLLM target**(不再扩展已删除的 DES):

1. admission — 接真实 vLLM target(扩展 scheduler 准入决策)
2. pd_router — 多实例真机路由 target
3. chunked prefill — 扩展scheduler支持分块
4. output length prediction — 预估vs实际对比
5. preemption trigger threshold — 参数化可演化
6. request priority scoring — 独立target

### Phase 2: 资源管理 + 工作负载特化(路线图)

需要把这些目标接成新的真实 vLLM target(历史措辞为"新模拟器组件",已不适用):

7. CPU offload/swap — 真机评测 swap 延迟
8. LoRA adapter scheduling — 真机评测 adapter 切换代价
9. MoE expert routing — expert级延迟模型
10. long-context memory layout — 分层KV模型
11. multi-modal token priority — workload增加modality字段

### Phase 3: 集群级 + 配置闭环(路线图)

需要集群级真机或 roofline/profile 代价模型(历史措辞为"集群级模拟",已不适用):

12. 多副本load balancing — 多实例真机
13. auto-scaling trigger — 带实例增减的真机
14. heterogeneous GPU scheduling — 多pool 真机
15. TP/PP代价模型 — roofline或profile数据
16. graph compilation (Ascend) — NPU profile数据

### Phase 4: 长尾 + 质量权衡(路线图)

17. speculative decoding — token级精度模型
18. structured output — 约束解码延迟模型
19. tool-use suspend/resume — 暂停事件
20. quantization-aware routing — 质量benchmark数据
