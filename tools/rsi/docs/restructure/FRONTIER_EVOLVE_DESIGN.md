# FRONTIER_EVOLVE_DESIGN — 让 Frontier 执行演化策略(Phase D 桥设计)

> 状态:D0 完成(侦察 + 设计;以下全部基于本地克隆源码逐行确认)。
> 教义:**sim 分数 = 搜索排序信号,真 vLLM = 采纳唯一判官**。四锁与 `real_source_block` 不动;
> 最终产物是 sim-winner PROPOSAL,绝不是 keep/gain/AC6。

## 1. Frontier 侧事实(已确认)

- **注册机制**:`frontier/scheduler/replica_scheduler/replica_scheduler_registry.py` 显式
  `ReplicaSchedulerRegistry.register(ReplicaSchedulerType.X, Cls)`;类型枚举在
  `frontier/types/replica_scheduler_type.py`(`VLLM_V1 = 6`);CLI 字符串 `vllm_v1` 经 config 多态
  dataclass(`config.py: VllmV1SchedulerConfig.get_type() -> ReplicaSchedulerType.VLLM_V1`)解析。
- **唯一排序决策点**:`VLLMv1EngineReplicaScheduler._get_sorted_waiting_queue() -> List[Request]`
  (vllm_v1_engine_replica_scheduler.py:2938)。它合并 `_preempted_requests + _request_queue`,按
  FCFS/priority 排序返回;**容量/token 预算/KV 接纳在下游 `_schedule_waiting_requests` 单独强制**。
  ⇒ 子类只覆写这一个方法,策略便控制"谁先谁后",而约束仍由父类保证。
- **Request 属性**(entities/request.py):`id`、`arrived_at`、`num_prefill_tokens`、
  `num_decode_tokens`、`num_processed_tokens`、`priority`、`_preempted`。

## 2. 我方策略契约(targets/scheduling/skeleton.py)

```python
schedule_batch(waiting_requests: list[RequestInfo], running_requests: list[RequestInfo],
               max_num_batched_tokens: int, max_num_seqs: int,
               available_kv_blocks: int, prefix_cache_hit_rate: float) -> ScheduleDecision
# ScheduleDecision.prefill_batch: list[request_id]  ← 桥消费这个(接纳顺序)
# RequestInfo: request_id/num_prompt_tokens/num_computed_tokens/num_output_tokens/
#              arrival_time_s/is_prefill/kv_blocks_used/prefix_cached_tokens/num_preemptions
```

## 3. 字段映射表(Frontier Request → RequestInfo)

| RequestInfo 字段 | Frontier 来源 | 备注 |
|---|---|---|
| `request_id` | `str(req.id)` | |
| `num_prompt_tokens` | `req.num_prefill_tokens` | |
| `num_computed_tokens` | `req.num_processed_tokens`(getattr,缺省 0) | |
| `num_output_tokens` | `req.num_decode_tokens` | |
| `arrival_time_s` | `req.arrived_at` | sim 秒 |
| `is_prefill` | `True`(桥只译等待队列) | |
| `kv_blocks_used` | 0(等待中未占) | |
| `prefix_cached_tokens` | getattr(req,'num_cached_prefill_tokens',0) | |
| `num_preemptions` | `1 if getattr(req,'_preempted',False) else 0` | |

策略另两个输入:`max_num_batched_tokens` ← config.max_tokens_in_batch、`max_num_seqs` ←
config.batch_size_cap;`available_kv_blocks` ← 大数占位(KV 接纳由父类下游强制,策略无需真值,
传 `2**31`,文档明示);`prefix_cache_hit_rate` ← 0.0(无源,明示)。
`running_requests` ← `self._running_requests` 同表翻译(`is_prefill=False`)。

## 4. 返回应用规则(约束安全)

策略返回 `ScheduleDecision.prefill_batch`(有序 id 列表)。桥的应用:
1. 按 `prefill_batch` 顺序取出现存等待请求(id 匹配;未提及的请求按原 FCFS 序**追加在后**——
   策略漏写≠饿死);
2. 返回该有序列表给 `_get_sorted_waiting_queue` 调用方;**接纳数量/token 预算/KV 仍由父类
   `_schedule_waiting_requests` 强制**——策略只能改顺序/优先级,不能越权超发。
3. `decode_batch`/`preempt_ids` 本版**忽略**(running 集与抢占由 Frontier 原逻辑管;v2 增量)。
4. 策略抛异常 → 当次回退父类原序,`fallbacks += 1`;策略文件加载失败 → **启动即非零退出**。

## 5. marker 协议(诚实)

运行结束(`on_simulation_end` 或析构钩)写 `ve_policy_marker.json` 到 metrics 输出目录:
`{"policy_sha256", "invocations", "fallbacks", "scheduler": "ve_policy"}`。
我方 `parse_frontier_metrics` 读取;**要求跑策略但 marker 缺失或 invocations==0 或
fallbacks==invocations → 该 trial 记失败 eval_result**(绝不把 baseline 行为当 candidate 分数)。

## 6. Frontier 补丁清单(最小)

| 文件 | 改动 |
|---|---|
| `frontier/types/replica_scheduler_type.py` | 加 `VE_POLICY = <下一个空闲值>` |
| `frontier/scheduler/replica_scheduler/ve_policy_replica_scheduler.py` | **新文件**:`VePolicyReplicaScheduler(VLLMv1EngineReplicaScheduler)`,覆写 `_get_sorted_waiting_queue` + marker 落盘;env `VE_POLICY_PATH` 加载策略(importlib.spec,文件不存在→raise SystemExit(3)) |
| `frontier/scheduler/replica_scheduler/replica_scheduler_registry.py` | 注册 VE_POLICY |
| `frontier/config/config.py` | `VePolicySchedulerConfig(VllmV1SchedulerConfig)`,`get_type()->VE_POLICY`(CLI 即可 `--replica_scheduler_config_type ve_policy`) |

补丁 + 应用脚本存 `integrations/frontier/`(vllm-evolve 仓库);Frontier 本地仓库单独 commit。

## 7. vllm-evolve 侧接线(D2/D3 概要)

- `build_frontier_argv`:`runner_kind=candidate 且 policy_path 存在` → scheduler type `ve_policy`
  + 子进程 env `VE_POLICY_PATH=<abs path>`;baseline 仍 `vllm_v1`。
- eval_result 新增 informational:`sim_policy_sha`/`sim_marker`(四锁不动)。
- orchestrate 的 `code:schedule_batch` 腿:evolve_fn 产变体 → `core.verify`(L1/L2)→ frontier_sim
  评估 → 按 `sim_score`(=primary_value,**search-only**)排序 → sim-winner PROPOSAL
  (`runs/<id>/sim_winner/{work_variant.py, evidence.json, REQUIRES_REAL_VLLM_VERIFICATION}`)。

## 8. 范围外(明示)

- 抢占/decode 集控制(v2);KV 真值喂给策略(父类强制,占位即可);ve-author 子代理生成变体
  (先用确定性变体 FCFS→SJF/LJF/age-priority);upstream PR(本地补丁即可)。

---

## 9. 终态(D4 端到端验证通过)

真跑命令(本机纯 CPU):
`ve autopt "maximize goodput" --backend frontier_sim --evolve --model meta-llama/Llama-2-7b-hf
--n-requests 100 --concurrency 18 --max-num-seqs 1 --workload-spec
'{"length_dist":"uniform","min_tokens":64,"max_tokens":1024,"prefill_to_decode_ratio":4.0,
"slo":{"ttft_ms":800}}'`

| 断言 | 结果 |
|---|---|
| ① 诊断 scheduling_queue | ✓ suspected(waiting=8, duty=0.8646 真实证据) |
| ② code:schedule_batch 真演化 | ✓ 3 变体生成→L1/L2→Frontier 内真实执行(evals_used=3) |
| ③ 变体 sim 分数分化 | ✓ SJF **18.055** / LIFO 18.055 / LJF **17.326** goodput_req_s |
| ④ sim-winner proposal | ✓ runs/sim_winner/{work_variant.py, evidence.json, REQUIRES_REAL_VLLM_VERIFICATION} |
| ⑤ 隔离不破 | ✓ outcome=dod_b, adopted=None, no_verified_candidate |

**方向合理性**:SJF > LJF 完全符合调度理论(短作业优先让更多请求满足 TTFT≤800ms)。

**D4 调参史(全程真探测,零造数)**:
1. qps10+uniform → 均载下降,诊断如实判 under_saturated(duty 0.45)→ 未触发演化(定位链正确);
2. qps18 n100 进入 scheduling_queue,但目标 tok_s **工作守恒**(cap=1 下换序不改 makespan)→
   三变体同分——真实物理性质,不是 bug;
3. 改目标 "maximize goodput" + ttft SLO(排序真正影响的指标)→ 分化成立。

教义落地:**sim 分数=搜索排序信号(选变体),真 vLLM=采纳唯一判官**。
