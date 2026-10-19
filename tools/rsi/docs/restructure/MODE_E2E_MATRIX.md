# MODE_E2E_MATRIX — frontier_sim 后端 × agent 模式 端到端验证

验证"加了 Frontier 后端(`--backend frontier_sim`)后,系统在不同需求下、不同 agent 模式都能
**无硬件端到端闭环**"。全程 fixture stub(不实跑 Frontier),结论只到 **DoD-B**(策略盲,永不显示 win)。
历史验证命令：`pytest -q -p no:cacheprovider` + `ruff check src tests`。
当前支持的运行环境和安装步骤见 [README](../../README.md#quick-start)。

---

## 阶段 A — 实现 frontier_sim 后端(全绿)

| 步骤 | 内容 | 状态 | 证据 |
|------|------|------|------|
| A-F0 | 两份 schema source 枚举 +frontier_sim、outcome_class +simulator_nonqualifying;enum-parity 测试 | ✅ | commit a1d7704 |
| A-F1 | `bench/frontier_sim.py` 适配器(进程外、四锁、手搓 eval_result)+ 单测 | ✅ | commit 7c5531d;`tests/bench/test_frontier_sim.py` 4 passed |
| A-F2 | `cli/main.py` `--backend frontier_sim` 贯通 cmd_bench+3 seam;`engine/profile.py` collect_profile_frontier | ✅ | commit a784f9d |
| A-F3 | `tests/test_frontier_quarantine.py` 端到端串联 + 冻结闭包断言 | ✅ | 本次;4 passed,frontier_sim 不在冻结闭包 |

A-F3 已证:整条 round(READ_CONTEXT→…→BENCHMARK `ve bench --backend frontier_sim`→compare→discard)
**无硬件闭环**,`source=frontier_sim`/`outcome_class=simulator_nonqualifying`,compare/verify-gain/keep
全部 `non_real_source_blocked` → **DoD-B**;`frontier_sim` 不在 `test_frozen_core` 冻结闭包。

---

## 阶段 B — 需求 × agent 模式矩阵(待跑)

| 格 | 模式 | 命令 | 判据 | 状态 |
|----|------|------|------|------|
| B1 | autopt(目标驱动) | `ve autopt "maximize throughput" --backend frontier_sim` | ve-goal→research→diagnose→author + orchestrate loop 跑完;eval_result source=frontier_sim 过 schema;只到 DoD-B 不崩 | ✅ `tests/test_frontier_modes.py` 2 passed:run_autopt L1→L4 + CLI 均 outcome=dod_b/adopted=None,exercise optimize(searched_target) |
| B2 | tune(指定策略调优) | `ve tune <scheduling 策略> --backend frontier_sim` | 走既有 gate 不走 admin 旁路 → DoD-B | ✅ `test_frontier_modes.py` mode=tune/backend=frontier_sim,从不 adopt、无 kept/committed |
| B3 | port(版本/硬件迁移) | `ve port <跨 cell> --backend frontier_sim` | 逐 cell 闭环 → DoD-B | ✅ 2 版本×1 硬件矩阵,每 cell source=frontier_sim、`real_cells==0` |
| B4 | 意图路由 | `intent/router.py` `route(text)` | throughput→autopt;"优化策略"→tune;"适配版本/硬件"→port | ✅ 5 例参数化全对(throughput→autopt;调优/given policy→tune;适配/版本/硬件/across→port) |

每格 PASS = ① 入口/路由到正确模式 ② 闭环不崩 ③ eval_result 过 schema 且 source=frontier_sim
④ 四锁生效、非真实源被拒 → DoD-B ⑤ 相关测试 + 全量 gate 绿。

---

## 最终汇总 ✅ 全部成立

**结论:加了 `frontier_sim` 后端后,系统在不同需求下、走不同 agent 模式,都能无硬件端到端闭环。**

- **autopt(目标驱动自动 research)**:`ve autopt "…" --backend frontier_sim` 走 ve-goal→research→
  diagnose→author + orchestrate loop,真实 exercise optimize,结论 **DoD-B**(从不 adopt)。
- **tune(指定策略调优)**:`ve tune <策略> --backend frontier_sim` 走既有 gate(不走 admin 旁路),
  **DoD-B**,无 kept/committed。
- **port(版本×硬件迁移)**:`ve port … --backend frontier_sim` 逐 cell 闭环,每 cell `source=frontier_sim`、
  `real_cells==0`(永不成为真实增益)。
- **意图路由**:`route(text)` 把自然语言需求正确分解到 autopt/tune/port。

**不变量全程守住**:① frontier_sim 进程外解耦(`VE_FRONTIER_*`,未配置干净报错);② 四锁同
`local_smoke`,任何模式产物都被 `real_source_block` 拒于 compare/verify-gain/keep 之外,只能 **DoD-B**;
③ Frontier 策略盲(candidate==baseline,永不显示 win)——它验证的是**管路 + 端到端能力**,不是策略效果;
④ 冻结判定核心保持绿,`frontier_sim` 不在 `test_frozen_core` 冻结闭包。

**全程无硬件、无实跑**(Frontier 子进程由 fixture 桩;列名取自 Frontier `metrics/constants.py` 源码)。
真跑集成(`@requires_frontier`,F4)为延后的可选钩子,默认 skip,本期不做。

Gate:`pytest -q` **546 passed / 3 skipped**,`ruff` 干净。提交:F0 `a1d7704` → F1 `7c5531d` →
F2 `a784f9d` → F3 `0304f01` → B1 `649d3da` → B2–B4(本提交)。

---

# Phase C — 信号保真:模拟器内"定位瓶颈→选优化点"由真实信号驱动

> 目标:修掉 gap①(诊断信号是占位样本)与 gap②(每个 trial 仿真相同),然后**真跑对照实验**
> 证明"不同负载 → 诊断不同瓶颈 → 自动选不同优化点"在 Frontier 模拟器里成立。
> 本阶段允许真跑(纯 CPU,本机 Frontier venv);四锁与 DoD-B 语义不变。

## C1 信号映射表(Frontier 真实输出 → Profile → 哪条诊断规则消费)

| Frontier 真实输出 | 提取方式 | Profile 字段 | 消费的诊断规则 |
|---|---|---|---|
| `system_metrics.json: memory_utilization_percent` | max(各 cluster)/100 | `vllm.kv_util` | kv_capacity / 各规则的 KV 排除项 |
| `system_metrics.json: preemption_statistics.total_preemption_events` | 直读 | `vllm.preempt` | kv_capacity;scheduling_queue 的排除项 |
| `request_metrics.csv: request_inter_arrival_delay + request_first_scheduling_delay` | 重建到达时刻→等待区间→最大重叠 | `vllm.waiting`(排队深度) | scheduling_queue / under_saturated / kv_capacity |
| `frontier_stage_batch_ledger.jsonl: stage_start/end_ts` | 忙窗并集 ÷ 总时长 | `gpu.duty_cycle`(引擎占空比) | under_saturated / scheduling_queue(duty 证据分支)/ compute 排除 |
| `ledger: request_ids` | 每批最大请求数 | `vllm.running` | (信息性) |
| **无源信号** | — | `sm_util / mem_bw_util / nccl_frac` = **缺省** | 规则引擎自带 limitation 降级路径(绝不造数) |

另修真跑发现的 **1000× 单位 bug**:Frontier CSV 延迟列是毫秒(与 system stats 一致),原 goodput 计算误当秒。

## C2 knob 映射(每个 trial 的仿真真的不同)

| BenchConfig knob | Frontier flag(`frontier.main --help` 验证存在) |
|---|---|
| `engine.max_num_seqs` | `--vllm_v1_scheduler_config_batch_size_cap`(Frontier 自述 "max_num_seqs in vLLM")|
| `engine.max_num_batched_tokens` | `--vllm_v1_scheduler_config_max_tokens_in_batch` |
| `workload.arrival_rate_qps`(优先)/ `concurrency`(1:1 开环近似,文档化)| `--poisson_request_interval_generator_config_qps` |
| 无 Frontier 对应的 knob(quantization、gpu_memory_utilization…) | **不映射,不硬凑** |

## C4a 判定侧泛化(非造信号)

- `scheduling_queue` 规则新增 **duty 证据分支**:SM 无源时,`duty<0.9 + waiting>4 + KV 不满 + 无抢占`
  → suspected(弱证据封顶);`duty>=0.9` 不触发(已钉测试)。同样惠及未装 nvidia-smi 采样的真实跑。
- dummy 步长 1.0ms→0.001ms:原值下单请求占引擎 ~58s,任何 qps≥1 都饱和;现在 qps≥1 可真实覆盖
  idle → queued → overloaded 三个 regime。

## C4b 真跑对照(本机真 Frontier,纯 CPU;同一命令,只改负载)

| | Run A(低压) | Run B(排队压) |
|---|---|---|
| 命令差异 | `--n-requests 8`(qps 1) | `--n-requests 60 --concurrency 10 --max-num-seqs 1` |
| **诊断** | `under_saturated` **confirmed** | `scheduling_queue` **suspected** |
| **真实证据** | `duty_cycle=0.0798, waiting=1` | `waiting=6, duty_cycle=0.7249, kv_util=0.1508, preempt=0` |
| **自动选靶** | `config:max_num_seqs`(空间 128/256/512)→ `config:concurrency` | `config:max_num_seqs`(空间 128/256/384)→ `code:schedule_batch`(未接线,跳过) |
| 结论 | `outcome=dod_b, adopted=None` | `outcome=dod_b, adopted=None` |

轨迹:`runs/frontier_traj_lowload.json` / `runs/frontier_traj_highload.json`。

**调参根因记录**(诚实排障,非造信号):B 第一版(qps 12,默认批容量)判 `unknown`——vLLM v1 默认
`batch_size_cap` 大,所有请求被立即接纳,`first_scheduling_delay≈0`→`waiting=1`,队列根本不积压;
收紧 `max_num_seqs=1` 后才出现"保守接纳→排队但引擎有空闲"的真实 regime(探测 `waiting=6, duty=0.72`)。

## 遗留(明示)

- `sm_util / mem_bw_util / nccl_frac` 无 Frontier 源 → 永远缺省;依赖它们的 compute/bandwidth/comm
  规则在本后端不可达(诚实降级)。
- `kv_capacity` 在 dummy 模式不可达:`memory_utilization_percent` 是静态权重占比(15%),不随 KV 增长。
  (C3 离线 stub 测试已证明该规则路径本身工作。)
- `code:schedule_batch` 演化未接线(legacy GA 移除后的既有状态),与本阶段无关。
- 隔离不变:两条轨迹均 DoD-B;trial 分数因 LOCK C 不可 verified,采纳门槛纹丝未动。

**Phase C 结论:模拟器内"profiling 指标差 → 真实信号签名 → 定位瓶颈 → 自动选优化点"链路成立。**
Gate:551 passed / 3 skipped,ruff 干净。提交:C1 `6f5287f` → C2 `81e435b` → C3 `daf5543` → C4a `05ad0b2` → C5(本提交)。

---

# Phase D — 演化闭环:Frontier 真实执行演化策略,sim 分数驱动搜索

> 教义:**sim 分数 = 搜索排序信号(选变体),真 vLLM = 采纳唯一判官**。四锁/real_source_block 零改动;
> 最终产物 sim-winner PROPOSAL(带 REQUIRES_REAL_VLLM_VERIFICATION 戳),永远不是 keep/gain/AC6。

## 桥架构(详见 FRONTIER_EVOLVE_DESIGN.md)

Frontier 本地补丁(4 文件,`integrations/frontier/ve_policy.patch` 可复现):`ve_policy` 调度器
= vllm_v1 子类,只覆写唯一排序决策点 `_get_sorted_waiting_queue`——策略(env `VE_POLICY_PATH`)
控制"谁先谁后",容量/token 预算/KV 接纳仍由父类强制;未提及请求按 FCFS 追加(不饿死);异常响亮
回退计数;结束写 marker(policy_sha256/invocations/fallbacks)。我方守卫:marker 缺失/零调用/
全回退 → 该 trial 记失败,**绝不把 baseline 行为当 candidate 分数**。

## D1 桥验证(真跑):FCFS vs LIFO,cap=1, qps=10 —— 两策略各被真实调用 **1080 次**(0 回退),
7/12 请求 TTFT 分化(请求 1:56.3ms vs 173.0ms)。

## D4 端到端演化(真跑,五断言全过)

命令:`ve autopt "maximize goodput" --backend frontier_sim --evolve … --concurrency 18
--max-num-seqs 1 --n-requests 100 --workload-spec uniform(64-1024)+slo(ttft_ms=800)`

| 变体 | goodput_req_s(sim,真实逐请求延迟+SLO 计算) |
|---|---|
| **SJF(winner)** | **18.055** |
| LIFO | 18.055 |
| LJF | 17.326 |

- 诊断 `scheduling_queue`(waiting=8, duty=0.86 真实证据)→ 自动升级到 `code:schedule_batch` 演化;
- 3 变体经 L1/L2 → **在 Frontier 内真实执行**(marker 验证)→ goodput 排序 → SJF 胜(符合调度理论);
- `runs/sim_winner/` proposal 落盘;`outcome=dod_b, adopted=None`(锁全活)。

**D4 调参史(真探测,零造数)**:① qps10 均载低→如实判 under_saturated 未触发演化;② tok_s 工作
守恒(cap=1 换序不改 makespan)三变体同分→真实物理性质;③ 改 goodput+TTFT SLO(排序真正影响的
指标)→ 分化成立。

**Phase D 结论:演化闭环成立——模拟器真实执行演化出的策略、变体真实分化、自动选出 winner、
采纳门槛纹丝不动。** 提交:D0 737692e → D1 a125279(Frontier 本地 32a48ca)→ D2 99ed184 →
D3 12482e4 → D4a 6be6111 → D4b ac6c6bf → D5(本提交)。

---

# Phase E — Evolution Engine v2:真正的世代演化(真跑验收 10/10)

> 演化引擎的约束不变:sim 分数=搜索排序,
> 采纳只认真 vLLM;四锁/冻结核零改动(全程 gate 绿,570 passed)。

## 提交表(B→D→A→C→F→G→H→E→e2e→J)

| 步 | commit | 内容 |
|---|---|---|
| B | 1a66a07 | AuthorContext/Lineage/Lesson/EvolutionResult;to_prompt() 全字段+教义快照 |
| D | e08f3bd | 0005_lessons:不可变证据行+lessons_latest 去重视图+limit 硬上限 |
| A | 52e0a8a | evolve_loop:遗传通道/预算钉 eval 边界/repair/精英+去重/run 末写 lessons |
| C | ea887be | template_author_fn:消费父代分数与报错的确定性变异作者 |
| F | 1976fb2 | orchestrate 接世代引擎;EvolutionResult→Candidate 轨迹键全保留 |
| G | a703a96 | 4 agent .md 与接线一字不差(ve-author tools={Read},编排者执笔);资产测试原子改 |
| H | 9d88898 | CLI --generations/--population/--repair-limit;引擎内硬上限 24;SKILL 同步 |
| E | 104ba5a | research gather() 合并 lessons;verify_citations 支持 lesson:id |

## e2e 真跑(本机纯 CPU 真 Frontier;断言 10/10 全过)

命令:`ve autopt "maximize goodput" --backend frontier_sim --evolve --generations 3 --population 4
…(排队 regime qps18/cap1/n100 + uniform 长度 + ttft≤800ms SLO)`

**世代史(每个分数 = 一次真实 Frontier 仿真,策略被真实执行)**:

| 代 | 内容 | best goodput_req_s |
|---|---|---|
| 0 | 四个家族种子(fcfs/sjf/ljf/lifo) | **18.2376(FCFS)** ← 全局赢家 |
| 1 | 基于最优父代的 aging 变异 ×4 | 18.0552 |
| 2 | 同空间(sha 去重防重评) | 18.0552 → 连续两代无改进,诚实早停 |

8 次评估 / 8 个不同 sha;lessons 1–6 落库(样例:`run summary: 8 variants over 3 generation(s);
winner gen0 goodput_req_s=18.2376; terminated=no_improvement`,source=frontier_sim);
`verdict=no_verified_candidate`、`outcome=dod_b`、`adopted=None`(锁全活)。

**诚实发现**:该负载下 FCFS 家族胜过 aging-SJF 变体——队列不深时到达序近优,这是真实结果而非缺陷;
引擎正确保留赢家并早停。执行期修订一条(断言 d 的"末代≥首代"推理瑕疵→改为"赢家不丢失",
记 EVOLVE_V2_CODEX_LOG)。

## 与 v1(三选一)的行为对比

| | v1(Phase D) | v2(本期) |
|---|---|---|
| 生成 | 3 固定模板,一锤子 | 世代循环:父代分数+诊断+lessons+报错喂给作者 |
| 反馈 | 无 | 遗传通道有测试证明(假作者只有看到父代分数才能产高分子代) |
| 失败 | verify 不过即丢 | 错误文本回传 repair(重新 verify,无旁路) |
| 记忆 | run 结束即忘 | lessons 不可变证据行,跨 run 被 research 引用(id 可验真) |
| 预算 | 无 | 钉在 eval 调用边界,硬上限 24 |
| 作者权限 | Write work.py | **只读**,返回源码字符串,编排者执笔 |

---

# Phase F — 研究 harness:锚定假设 H* 全链闭环(本机纯 CPU 真 Frontier)

研究 harness 把"自由提假设→确定性裁决真伪"接通(详见 DESIGN.md §9)。Phase F 验收锚定假设
**H\***:"突发多租户负载下,prefix-cache 抖动是 TTFT 尾延迟主因;按租户分组接纳改善 p99 TTFT。"
**通过标准 = 流程闭环,而非假设为真**——verdict 无论 `supported`/`falsified`/`inconclusive` 都算
成功,前提是测量真实、citation 可验、且 B 臂机制(分组接纳 + defer 通道)经 marker 证明确实执行。

## 全链(`tests/test_h_star_e2e.py`,Frontier-gated 真跑,**非 stub**)

合成突发多租户 trace → 注册预测(预测先行)→ A=`vllm_v1`(FCFS)对 B=`tests/fixtures/
policy_group_admission.py`(按 `session_id` 分组、非当前租户进 `defer_ids`)→ 经单一 `MetricExpr`
求 grouped p99 TTFT(`{column:ttft, agg:p99, group_by:request_session_id, group_reduce:max}`)→
确定性裁决 → 台账落库(`prediction→experiment→adjudication` 只插不改)→ `research.gather()` 以
`hypothesis:<event_id>` 复用并可验真。

## 运行环境(缺一即 **off-box skip**,绝不伪造)

| 需求 | 值 |
|---|---|
| `VE_FRONTIER_REPO` | 本机 Frontier checkout(含 `frontier/`) |
| `VE_FRONTIER_PYTHON` | 该 checkout 的解释器(独立 venv,**不装 vllm-evolve**) |
| ve_policy 桥 | `integrations/frontier/ve_policy.patch` 已 apply 到 checkout(`apply_patch.ps1`) |
| trace flag | Frontier 原生 `--trace_request_generator_config_trace_file`(TRACE_REPLAY 的 config 类是 `TraceRequestGeneratorConfig`) |

未配置 → 测试 `skipif` 干净跳过(就像 `@requires_gpu`);求值器/桥逻辑/argv 守卫离线常绿。

## 产物位置

- 每臂目录:`<base_out_dir>/<experiment_id>/arms/<arm>_<seed>/`(e2e 中 B 臂 = `.../h_star/arms/B_0/`)。
- B 臂诚实 marker:`.../arms/B_0/ve_policy_marker.json` = `{invocations, fallbacks, defers}`。
- 台账:SQLite store(`hypothesis_events`,只插不改)。

## 真跑证据(本机纯 CPU,1 passed)

| 项 | 值 |
|---|---|
| B 臂 marker | `invocations=3, fallbacks=0, defers=8` ← 分组策略 **与 defer 通道**真实执行 |
| 列齐 | `request_session_id` + `ttft` 均在 catalog |
| 双臂测量 | A==B==**901.3 ms** grouped p99-max(qps=1 引擎未饱和,接纳序不动尾延迟——真实诚实结果) |
| verdict | **`inconclusive`**(diff 0.0 在 margin 内;**闭环成功**,非假设为真) |
| citation | `hypothesis:<adj_event_id>` 验真 `all_real=True`;`research.gather()` 复用到 |
| outcome | 全程 **DoD-B**(台账/实验数据停留 PROPOSAL 层,绝不入采纳门) |

## DoD-B 成功判据(精确)

`supported` / `falsified` / `inconclusive` 三者皆过,**当且仅当**:(1) 双臂均产出非 `None` 的
grouped 指标(真实测量,非空目录);(2) `hypothesis:<event_id>` citation 可验真;(3) B 臂 marker 满足
`invocations>0 ∧ fallbacks==0 ∧ defers>0`(机制真跑,FCFS 回退不得冒充 B 臂)。`test_h_star_e2e.py`
断言全部三条 + 台账链顺序 + `research.gather()` 复用。

**Phase F 结论:研究 harness 全链(表达→测量→执行→裁决→入账→复用)在真 Frontier 上闭环成立;
安全架构零改动(冻结核 / 四锁 / verify 全绿),台账证据永不采纳。**
