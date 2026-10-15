# vllm-evolve 重构前审计

**审计时间**：2026-05-15
**审计分支**：`feature/full-coverage-68-targets` @ `bcc1e6a`
**目的**：项目以 LLM 模板化方式快速膨胀到"68 目标全覆盖"，需要在重构前把"宣传 vs 实际"的差距、可信资产、模板灌水的痕迹固化下来，作为重构的事实基线。

---

## 1. 致命问题（必须在重构里修掉）

### 1.1 38 个 evaluator 中 35 个完全绕过核心模拟器

实测：`grep -L "PDSimulator\|SchedulerSimulator\|run_single_sim" src/vllm_evolve/sim/evaluators/*.py` → 38 个文件中 **35 个不引用任何核心 DES 引擎**。

它们各自现写 ad-hoc 状态机 + 硬编码常数当作"模拟器"，例如：
- `moe_routing.py`：`base_latency_ms=0.5`、`per_token_cost_ms=0.02`、`comm_overhead_ms=0.3`
- `autoscale.py`：硬编码三段流量曲线 `if t < duration*0.15: ... elif t < ...*0.30:`
- `chunked_prefill.py`：自起 `hw_profiles = {A6000:..., A100:..., H100:...}`，与 `config/hardware/*.yaml` 完全无关

**后果**：演化策略只是在拟合 evaluator 作者拍脑袋的常数，与真实 vLLM 行为无对应关系。整个项目的科学性在此处崩塌。

### 1.2 Hardware profile 99% 是装饰

`grep -c "from_hardware_profile" sim/evaluators/*.py` → 38 个 evaluator 中**只有 2 个**（`admission`、`lora_scheduling`）真正调用 `GPUTimingModel.from_hardware_profile`。其余 36 个或无视参数，或自起 hw 字典。

但 `config/hardware/{ascend_a3,ascend_a5,nvidia_h100}.yaml` 三套 profile 仍然存在，README 仍宣称"different hardware produces different optimal policies"——目前为虚假承诺。

### 1.3 Trust Chain L3 / L4 实际不生效

- `TrustChain.check_overfit()` 在 `trust/chain.py:27` 定义，但 **`tools/simulate.py` 全文未调用**
- L4 fallback 仅在 `engine/adapter.py` 导出 vLLM plugin 时拼到模板里，sim 阶段没有

DESIGN.md §7 把"4 层防御"作为关键卖点，实际生效的只有 L1 (AST) + L2 (O(n²) 正则)，正好是最弱的两层。L3 是承诺，未兑现。

### 1.4 同一 target → 函数名映射有三处手写真理源

| 位置 | 形态 |
|---|---|
| `config/targets/{name}.yaml` | `spec.evolvable_functions[].name` |
| `src/vllm_evolve/sim/evaluators/__init__.py` | `_EVALUATORS` 字典（38 行） |
| `src/vllm_evolve/tools/simulate.py` | `_FN_NAMES` 字典（38 行） |

任何新增/改名要改三处。是 LLM 写完一段忘记上一段的典型痕迹。

---

## 2. 模板灌水的结构性证据

| 现象 | 量化数据 |
|---|---|
| `_load_function` 复制粘贴 | 38 份字面相同的代码 |
| `scenario_cfgs = {moderate_load, high_load_kv_pressure, extreme_bimodal}` 字典 | 35 个 evaluator 重复同一三元开关 |
| `_get_scheduling_scenarios` / `_get_sim_config` | 在 `scheduling.py` / `kv_eviction.py` 复制粘贴 |
| skeleton+seed 平均长度 | 100 行 (~1/3 是装饰性 docstring) |
| `scaffold/mock.py` 的 5 条 diff | **只针对 `targets/scheduling/seed.py`**，跑别的 target 必然失败 |
| `sim/scenario_registry.py` 设计良好的 YAML 加载 | 35 个 evaluator 集体绕过它，自己 hardcode |

`scenario_registry` 失效尤为典型：YAML 抽象层存在，但每个 evaluator 自己重新解释 scenario 字符串，使统一抽象沦为字符串 switch。

---

## 3. 可信资产（重构应保留并以此为核心）

| 模块 | 评价 |
|---|---|
| `sim/pd.py` (585 行) | 真正的事件驱动 DES，`GPUTimingModel` 声称按 A6000 校准，结构清晰 |
| `sim/scheduler.py` (672 行) | 完整 vLLM v1 风格 prefill/decode/preempt，可挂 `schedule_fn` / `preemption_fn` / `eviction_fn` |
| `sim/kv_cache.py` (415 行) | KV 块管理，看上去认真 |
| `sim/workload.py` | Poisson 到达 + prompt/output 长度采样 + prefix 重用 |
| `store/db.py` | SQLite WAL + 内容寻址，靠谱 |
| `trust/safety.py` (L1) | AST + 正则，简单有效 |
| `targets/scheduling/` 全套 | **唯一真正接进核心模拟器的样板**，写得相对扎实 |
| `engine/adapter.py` | 仅适用于 scheduling，但是真能落地的 vLLM SchedulerPlugin 导出 |

**结论**：项目真正能用的部分只有 `scheduling` 这一个 target + 它的整套配套。其他 37 个目标"看起来支持，实际是壳"。

---

## 4. 宣传 vs 真实状态对照

| 文档声称 | 实际 |
|---|---|
| 68 optimization targets, 100% covered | 38 个 evaluator 文件，35 个是凭空写的 ad-hoc 模型 |
| 14 evolution skills | 14 份 .md，但 mock provider 只生成 scheduling diff，其它 skill 在 mock 模式下跑不通 |
| 4-layer trust chain | L1+L2 生效；L3 代码存在但 simulate 从未调用；L4 仅在 plugin 导出时存在 |
| Multi-hardware (Ascend / NVIDIA) | 36/38 evaluator 完全无视硬件 profile |
| 158 tests | 实际 161 collected；14 个在 Windows 上挂（缺 `encoding="utf-8"`）；测试主要是文件存在性而非行为正确性 |
| YAML-driven, zero-code targets | 加 target 仍要写 skeleton + seed + evaluator + `_FN_NAMES` + `_EVALUATORS`，至少 5 处 |
| Skills are .md, any agent can use | 大部分 skill 只在 scheduling 上演示过 |

---

## 5. 次级问题清单

- **Fitness 计算单一**：所有 target 共用 `WeightedRatioFitness`，但不同 target 的 metric 量级（0–1 比例 vs 数百 ms 延迟）差异巨大；无 target 覆盖 `compute_fitness`。
- **Seed baseline 重复计算**：每个 evaluator 自己跑一遍 seed，浪费 ~50% sim 时间且无缓存。
- **Windows 编码兼容**：14 个测试因 `Path.read_text()` 未指定 `encoding="utf-8"`，在中文 Windows 默认 GBK 下失败。
- **README/AGENTS 数字漂移**：14 targets / 158 tests / 119 tests 等多处数字和实际不一致。
- **仓库残留**：根目录有 `vllm_evolve.db-shm` / `db-wal`（SQLite WAL 残留），未 `.gitignore`。
- **`evaluators/__init__.py` 未实现自动发现**：38 个文件文件名几乎等于 target 名，却仍然手写注册表。

---

## 6. 重构方向（建议，待对齐）

### A. 收缩范围
保留 5–7 个能用 `PDSimulator + SchedulerSimulator + KVCacheSimulator` 表达的 target：
`scheduling`、`kv_eviction`、`admission`、`chunked_prefill`、`pd_router`/`multi_instance`、`output_prediction`、`preemption_victim`。
其余 (autoscale / moe_routing / health_check / warm_pool / kernel_launch_order / fp8_kv_budget …) 移入 `targets/experimental/`，README 写明"未做物理建模"。

### B. 一个引擎，多个 view
建立 `sim/EvaluationHarness`：统一 `SimulationContext`（PDSimulator + KVCache + workload + timing + scenario YAML），每个 target 只提供 hook + metric extractor，不再自己写 DES。Hardware profile / scenario / seed baseline 缓存 / L3 anti-overfit 全部集中在 harness 内强制执行。

### C. 单一真理源
target 元数据**只**写在 `config/targets/{name}.yaml`（含 `evolvable_function_name`）。Evaluator 按文件名自动发现。删掉 `_FN_NAMES` / `_EVALUATORS` 两个字典。

### D. Trust chain 真接上
`simulate()` 内强制 L1 → L2 → 跑所有 scenario → worst-case → L3 (`check_overfit`) → 写 store。增加单元测试断言"一个场景超好、另一个差到 70%"的策略被 L3 reject。

### E. Mock provider per-target
按 target 名加载 `targets/{name}/mock_diffs.py`，或对未配置 mock 的 target 显式 fail（而非静默产生垃圾结果）。

### F. 文档诚实化
README 写"7 supported / 30 experimental"，DESIGN.md 写清 L3/L4 真实状态，`OPTIMIZATION_TARGETS.md` 把"全 ✅"改回现实状态。

---

## 7. 待与用户对齐的决策点

1. **范围抉择**：砍到 5–7 个真 target 把核心做扎实（1–2 周可见底）vs 维持 68 目标方向但全部基于统一引擎重写 evaluator（1–2 个月起）？
2. **核心抽象**：是否同意以 `PDSimulator + SchedulerSimulator + KVCache` 为唯一引擎，所有 target 通过 hook/extractor 接入？
3. **演化技能精简**：14 个 skill 中实际会用的是哪几个？（如只用 `/evolve` + `/evolve-eoh`，其余可一并精简）
4. **Trust L3/L4**：必须真实落地，还是先在文档里降级为"planned"再修其它？

---

## 8. 不在本次审计范围（但需提及）

- `engine/controller.py` (464 行) / `engine/population.py` (464 行) / `engine/checkpoint.py` (257 行) 的实际使用情况未深入审计；与 evaluator 的核心问题相比优先级低。
- `openevolve/` 目录未审计。
- 14 个 skill .md 的内容质量未逐份审计。
- `tests/` 仅运行了 pytest 概览，未逐项分析覆盖率。

---

# 附录 A — 第二轮思考：参考 AscendOpGenAgent 后的修正

参考 `Just-it/AscendOpGenAgent`（Ascend NPU 算子生成 + 自动迭代优化框架）的实现，发现第一轮重构方向虽对了一半，但**漏掉了最关键的设计原则**。本附录是对第 6/7 节的**重大修正**。

## A.1 AscendOpGenAgent 的核心设计要点

| # | 要点 | 实现位置 | 价值 |
|---|---|---|---|
| 1 | **真硬件评测，没有自己的模拟器** | `eval_kernel.py` 直接跑 NPU + benchmark；baseline 是 reference 实现的真实测量值 | Ground truth 永远是真机；模拟器只是 cheap pre-filter |
| 2 | **Phase Machine 全程管控** | `phase_machine/phase_policy.py`：`INIT → GENERATE_REF → GENERATE_KERNEL → BASELINE → PLAN → EDIT → DIAGNOSE → FINISH` | Agent 不能任意调用工具，每个 phase 只允许特定 bash/edit |
| 3 | **PreToolUse / PostToolUse Hook 强 block** | `hook_guard_bash.py`、`hook_guard_edit.py`、`hook_guard_task.py` | 错误 phase 调错命令直接拦截，不靠 agent 自觉 |
| 4 | **Keep/Discard 是闭环核心** | `keep_or_discard.py` → `workflow/round.py:record_round()`：correctness/constraint/metric/improvement 四道关卡，KEEP→`git commit`，FAIL→rollback；`consecutive_failures` 累计触发终止 | 每轮强制审判，差的代码不进 git |
| 5 | **指标策略极简** | `task_config/metric_policy.py`：单 primary metric + relative threshold (%) + `lower_is_better` 方向 + 其它指标作为 hard constraints | 一个清晰的 ratio 比较，不是 14 个 metric 加权 |
| 6 | **数值精度按 dtype 严格** | `correctness.py`：fp32/fp16/bf16 各自的 rtol/atol；NaN/Inf 位置精确匹配；bool/int 严格相等 | 正确性是 hard gate，不是软指标 |
| 7 | **Skill = 工作流阶段，不是 mutation 策略** | `skills/triton/{op-task-extractor, kernel-designer, kernel-generator, kernel-verifier, latency-optimizer}` 5 个串行子技能，每个负责一个阶段 | 不是 14 种平行策略，而是 5 步管道 |
| 8 | **Skill 内置真实硬件知识** | `skills/triton/kernel-generator/references/hw-ascend910b{1,2,3,4}.md`、`hw-ascend910-9362.md` …… 共 11 份 hw md | LLM 有真实硬件文档可读，硬件区分不是装饰 |
| 9 | **archive_tasks 是真跑过的成功对照库** | `archive_tasks/{avg_pool3_d, flash_attention, matmul_leakyrelu, rms_norm, sort, ...}/` 每个含 `design/` + `kernel/` + `model.py` | "Lineage" 是文件系统里的真实代码，不是 SQLite 里的一行 |
| 10 | **批量评测独立子模块** | `scripts/batch/{discover, manifest, prepare, run, summarize, verify}.py` | 单个任务的工作流和 KernelBench 批量评测分离 |
| 11 | **错误分类 A/B/C** | A=代码可修；B=基础设施致命；C=同 A 错连续 ≥3 次终止 | 明确的 fallback 与终止条件 |
| 12 | **Verify + Profile 同进程复用 JIT cache** | `eval_kernel.py` 把 verify 和 profile 合并到一个 subprocess | 评测开销不浪费 |

## A.2 vllm-evolve 的根性错位（修正第一轮诊断）

| 第一轮诊断（仍正确） | 第二轮新认知（更根本） |
|---|---|
| 35/38 evaluator 自起 ad-hoc 模拟器 | **错位的根源是把"模拟器"当 ground truth；任何模拟器都不是 ground truth** |
| Hardware profile 是装饰 | **没有真 vLLM 集成时，hardware profile 本来就无处落地** |
| L3 anti-overfit 没接 | **比 L3 更严重的问题是没有 keep/discard 闭环 —— 不管好坏都进 SQLite** |
| 14 个 skill 是 mutation 策略灌水 | **skill 切分维度本身错了：应该按工作流阶段切，不是按 mutation 策略切** |
| target 元数据三处真理源 | **更深层：没有 phase machine，agent 无序调用工具是必然结果** |

**核心修正**：vllm-evolve 把自己定位成"evolution framework"，结果做了一堆 mutation 策略 + evaluator 模板灌水；但真正应该做的是 **"vLLM serving policy 优化的 agent harness"** —— 参考 AscendOpGenAgent，每一步评测都应该最终落到真 vLLM 上，模拟器只是 cheap pre-filter。

## A.3 修正后的重构原则（替代第 6 节）

### A.3.1 双层评测：cheap simulator + real vLLM
- **L0 (cheap)**: 保留 `PDSimulator`，~0.3s 跑完，作为快速 pre-filter
- **L1 (real)**: 必须有真 vLLM benchmark（最简形态：单机 `vllm serve` + locust/wrk 短跑），作为最终 ground truth
- 任何 KEEP 决策必须经过 L1
- 现状是 vllm-evolve 完全没有 L1，所以演化结果在科学上无法验证

### A.3.2 Phase Machine + Hook Guard 强制流程
替代当前"agent 自由调用 simulate/verify/store"。最简的 phase 序列：
```
INIT → READ_CONTEXT → DESIGN → GENERATE → VERIFY (L1 AST+L2)
   → SIMULATE (L0 sim) → BENCHMARK (L1 真 vLLM) → KEEP_OR_DISCARD
   → COMMIT / ROLLBACK → (loop or FINISH)
```
PreToolUse hook 在每个 phase block 不允许的 bash/edit；连续 3 次同类失败 → 终止整轮。

### A.3.3 Keep/Discard 闭环 + git commit/rollback
`record_round(eval_result)` 接口（直接照搬 AscendOpGenAgent 形态）：
1. correctness 失败 → FAIL + rollback
2. hard constraint 违反 → FAIL + rollback
3. primary metric 缺失/NaN → FAIL + rollback
4. 第一轮 → KEEP + git commit
5. 比 best 提升 > threshold → KEEP + git commit；否则 DISCARD + rollback

policy 文件作为 git 分支历史；`consecutive_failures` 累计触发 phase 回退或终止。

### A.3.4 指标极简：1 个 primary + N 个 hard constraints
不要 `WeightedRatioFitness` 14 维加权。每个 target：
```yaml
primary_metric:
  name: p99_ttft_ms
  direction: minimize  # lower_is_better
  improvement_threshold_pct: 2.0  # 必须 >2% 才算提升
constraints:
  - {metric: throughput_tok_s, op: ">=", threshold: seed * 0.95}  # 不能掉超过 5%
  - {metric: kv_hit_rate,      op: ">=", threshold: seed * 0.90}
correctness:
  - schedule_decision_validity  # 所有 ID 必须来自输入
```
`is_improvement()` 直接照搬 AscendOpGenAgent 的实现：相对百分比 + threshold + 方向。

### A.3.5 Skill = 工作流阶段，不是 14 种 mutation
砍掉 14 个并行 skill（它们其实是 14 个 prompt template），按 AscendOpGenAgent 的 5-skill 模式重组：

| AscendOpGenAgent | vllm-evolve 对应 | 职责 |
|---|---|---|
| `op-task-extractor` | `target-context-builder` | 读 prompt_hints + seed + git history + 当前 best |
| `kernel-designer` | `policy-designer` | 高层方案选择（FCFS / SRPT / KV-pressure / cache-aware …），输出 design.md |
| `kernel-generator` | `policy-coder` | 写 schedule_batch 的 Python 实现 |
| `kernel-verifier` | `policy-verifier` | L1 AST + L2 constraints + correctness 测试 |
| `latency-optimizer` | `policy-optimizer` | 在已 KEEP 的策略上做局部优化迭代 |

mutation 策略（EoH/ReEvo/MCTS/AVO/...）作为 `policy-designer` 内部的 prompt 选项，不是 14 个独立 skill。

### A.3.6 archive_tasks 形态：每个 KEEP 的 policy 留下三件套
```
archive_policies/scheduling/run_2026_05_15_kv_pressure_srpt/
├── policy.py             # 最终代码
├── design.md             # agent 写的设计思路
├── eval_result.json      # L0 + L1 metrics + 与 baseline 的对比
└── trace.md              # 该轮的 phase 历程 + 失败/恢复记录
```
SQLite 只索引这些目录，不存代码（policy.py 是 git 跟踪的真理源）。

### A.3.7 真硬件知识文档
照搬 AscendOpGenAgent 的 `hw-*.md` 形态，给每个硬件 profile 配一份 LLM 友好的参考文档：
- `references/hw/h100_pd_decode_bandwidth.md`
- `references/hw/ascend910b_kv_block_size.md`
- `references/hw/a6000_chunked_prefill_window.md`

让 agent 在 design 阶段真能看硬件特性，而不是只看一个 yaml 数字。

### A.3.8 范围彻底收缩：1 个 target 跑通完整闭环
**不是** 5–7 个 target，**先做透 1 个 target（scheduling）的完整工作流**，包括真 vLLM 集成、phase machine、hook guard、keep/discard、git 闭环。
跑通后再考虑：
- 横向复制到其它 target（kv_eviction、admission、chunked_prefill）
- 批量评测（参考 `scripts/batch/`）

## A.4 修正后的目录形态（草案）

```
vllm-evolve/
├── .ar/                          # 类比 AscendOpGenAgent 的 .autoresearch/
│   ├── config.yaml
│   ├── hooks/
│   │   ├── guard_bash.py
│   │   ├── guard_edit.py
│   │   └── post_round.py
│   ├── phase_machine/
│   │   ├── phase_policy.py       # phase → 允许的 bash/edit 表
│   │   ├── transition.py         # phase 推进规则
│   │   ├── validators.py
│   │   └── state_store.py
│   ├── workflow/
│   │   ├── round.py              # record_round() — keep/discard 核心
│   │   ├── baseline.py           # 跑 reference policy 建立 baseline
│   │   ├── planning.py
│   │   └── transition.py
│   └── scripts/
│       ├── eval_sim.py           # L0：跑 PDSimulator
│       ├── eval_vllm.py          # L1：跑真 vLLM benchmark
│       ├── correctness.py        # decision validity 检查
│       └── keep_or_discard.py    # CLI 包装
├── agents/
│   └── vllm-policy-optimizer.md  # 类比 ascend-kernel-developer.md
├── skills/
│   └── scheduling/                # 按 target 分子目录
│       ├── target-context-builder/SKILL.md
│       ├── policy-designer/
│       │   ├── SKILL.md
│       │   └── references/{eoh.md, reevo.md, mcts.md, ...}  # 14 策略下沉为参考资料
│       ├── policy-coder/SKILL.md
│       ├── policy-verifier/SKILL.md
│       └── policy-optimizer/SKILL.md
├── targets/
│   └── scheduling/
│       ├── skeleton.py
│       ├── seed.py
│       ├── prompt_hints.md
│       ├── primary_metric.yaml   # 替代 config/targets/scheduling.yaml
│       ├── constraints.yaml
│       └── references/hw/{h100.md, ascend910b.md, ...}
├── archive_policies/             # 类比 archive_tasks/，每个 KEEP 留下三件套
│   └── scheduling/run_<id>/{policy.py, design.md, eval_result.json, trace.md}
├── benchmarks/
│   └── vllm_microbench/          # L1 benchmark 脚本
├── src/vllm_evolve/
│   ├── sim/                      # 保留 PDSimulator + Scheduler + KVCache + Workload
│   └── store/                    # 保留 SQLite，但只索引 archive_policies
└── docs/
```

砍掉的：
- 38 个 evaluator → 缩成 1 个统一的 L0 evaluator + 1 个 L1 vLLM evaluator
- 14 个 skill → 5 个工作流阶段 skill
- 38 个 target 目录 → 1 个 scheduling 跑通后再扩
- `engine/controller.py / population.py / plateau.py / assembler.py` 中相当一部分 → phase machine 取代

## A.5 修正后的待对齐决策点（替代第 7 节）

1. **是否接受"必须有真 vLLM L1 benchmark"作为前提**？没有 L1 时整个项目科学性立不住。最简形态：单机 `vllm serve` 一个小模型 + 短压测脚本即可起步。
2. **是否接受 Phase Machine + Hook Guard 模式**？这是项目从"工具集"变成"agent harness"的关键，但意味着用户必须用 Claude Code（hook 机制）才能用 vllm-evolve。
3. **是否接受 Skill 由"14 种 mutation 策略"重切为"5 个工作流阶段"**？mutation 策略下沉为 designer skill 内部的参考资料。
4. **是否同意第一阶段只做 1 个 target（scheduling）的完整闭环**？先做透 1 个，再横向复制。
5. **archive_policies 目录 + git commit/rollback 是否可接受**？policy 进 git，意味着仓库会持续增长（参考 AscendOpGenAgent 的 archive_tasks）。

## A.6 第一轮 vs 第二轮对照速查

| 维度 | 第一轮建议 | 第二轮修正 |
|---|---|---|
| Ground truth | 统一引擎跑 5–7 个 target | **真 vLLM benchmark 是 ground truth**，sim 只是 pre-filter |
| 流程 | 让 simulate() 内强制 L1→L2→L3 | **Phase Machine + Hook Guard**，agent 不能跳步 |
| 选择 | 用 store best 排名 | **每轮强制 keep/discard + git commit/rollback** |
| 指标 | 保留 `WeightedRatioFitness` | **极简：1 primary + N constraints + relative threshold** |
| Skill | 精简 14 个但仍按 mutation 策略切 | **按工作流阶段重切为 5 个**，mutation 策略下沉 |
| 范围 | 5–7 个 target | **先做透 1 个 target（scheduling）的完整闭环** |
| 历史 | SQLite 存 lineage | **archive_policies/ + git** 是真理源，SQLite 只索引 |
| 硬件 | 接 GPUTimingModel | **加 hw-*.md 真实硬件参考文档**（agent 看的） |

---

# 附录 B — 第三轮思考：从"为 Claude Code 编程"的角度重新审视

> "我们本质是在 code agent 编程"

这个角度切中了项目最根本的身份错位。前两轮我都还在"把 vllm-evolve 当成一个独立运行的框架"来诊断；第三轮的核心修正是：**vllm-evolve 不是框架，是 Claude Code（或同类 code agent）的 harness**。Claude Code 才是 runtime，我们写的所有东西都是"给 agent 用的"。

## B.1 当前现状的事实清单（关键证据）

实测当前项目对 Claude Code 的"工具化支持"是**几乎不存在**的：

| 项 | 现状 | 应该是 |
|---|---|---|
| `.claude/skills/` 中的 SKILL.md | **只有 1 个**（`evolve`，56 行），且内容是"找 `prompts/prompt_NNNN.txt` 等待 controller 喂"——把 Claude Code 当被外部 controller 调用的 LLM API | 每个工作流阶段一个 SKILL.md，agent 可触达 |
| `skills/*.md`（14 份） | 普通 markdown 文档，**没有 SKILL frontmatter**，Claude Code 根本不会发现它们 | 全部移入 `.claude/skills/<name>/SKILL.md` 形态 |
| `.claude/settings.json` | **不存在** —— 没有配置任何 hook | PreToolUse / PostToolUse / Stop hook 配置 phase guard |
| `agents/` 目录 | **不存在** —— 没有定义任何 sub-agent | `agents/vllm-policy-optimizer.md`（类比 `ascend-kernel-developer.md`） |
| `src/vllm_evolve/scaffold/` (4 个 adapter) | `claude_code` / `opencode` / `mock` / `cli_agent` —— **把 LLM 当 API 来调** | **整体删除**：Claude Code 是 runtime，不是被调用的 provider |
| `src/vllm_evolve/engine/controller.py` (464 行) | 自己跑 evolution loop，让 LLM 通过 file IPC 回应 prompt | **整体删除**：Claude Code 自己就是 controller |
| `engine/population.py` (464 行) / `assembler.py` / `plateau.py` | 框架内自管 population / 上下文拼装 / 平台期检测 | 大部分删除；上下文拼装由 SKILL.md + agent 自己读 |
| 命令名：`vllm-evolve simulate / verify / store / context / configure / export-plugin` | CLI 长名 + 6 个动词，agent 容易忘 | 短名 + 一致动词；写错时 hook 给纠正提示 |
| 工具输出 | 大量 print 文本 + 部分 JSON | 全部结构化 JSON（agent 容易解析） |
| AGENTS.md | 给 agent 看的"指南"型文档（人类思维） | 应该是 phase 表 + tool 表 + 错误恢复表（agent 思维） |
| `.opencode/skills/evolve/SKILL.md` | `.claude/` 的复制品 | 删除——重构后单一来源即可 |

**结论**：当前项目的整个"engine + scaffold + file IPC"层都是建立在"Claude Code 是被外部 controller 调用的 LLM API"这个错误定位上的。这个定位一旦反转，**几千行代码会瞬间失去存在理由**。

## B.2 为什么 AscendOpGenAgent 的设计是正确的（再读一遍，从这个角度）

回头看 AscendOpGenAgent 的目录形态，每一项都对应"为 Claude Code 编程"的一个具体决策：

| 文件/目录 | 是给谁用的 | 对应 Claude Code 概念 |
|---|---|---|
| `agents/ascend-kernel-developer.md` | Claude Code 启动时加载 | **Sub-agent 定义**（system prompt） |
| `skills/triton/{op-task-extractor, kernel-designer, ...}/SKILL.md` | Claude Code 在 phase 内调用 | **Skill**（带 frontmatter 才能被发现） |
| `skills/*/references/*.md` | Skill 内部 agent 读的参考资料 | **Skill resources**（懒加载，不进 context） |
| `skills/*/scripts/validate_*.py` | Skill 调用的脚本 | **Skill scripts**（agent 用 Bash 工具跑） |
| `.autoresearch/scripts/hook_guard_bash.py` | 拦截 agent 的 Bash 调用 | **PreToolUse hook**（settings.json 配置） |
| `.autoresearch/scripts/hook_post_*.py` | agent 工具调用后触发 | **PostToolUse hook** |
| `.autoresearch/scripts/hook_stop_save.py` | agent 完成一轮后触发 | **Stop hook** |
| `.autoresearch/scripts/phase_machine/` | 给 hook 用的判定逻辑 | hook 调用的纯函数 |
| `.autoresearch/scripts/keep_or_discard.py` | Stop hook 调它 | bash 命令，可被 agent 直接调用，也可被 hook 自动调用 |
| `.autoresearch/scripts/eval_kernel.py` | agent 在 BENCHMARK phase 调用 | bash 命令 |
| `archive_tasks/<op>/` | agent 写入的产物 | 工作目录（Edit 工具的输出） |

**Claude Code 是 runtime，AscendOpGenAgent 提供的是：sub-agent 定义 + skills + hooks + bash 工具。** 整个项目几乎没有"自己写 controller"的概念。

## B.3 第三轮重构原则（对附录 A 的再修正）

### B.3.1 删掉所有"把 Claude Code 当 API"的代码

明确删除清单：
- `src/vllm_evolve/scaffold/` 整个目录（4 个 adapter）
- `src/vllm_evolve/engine/controller.py`（464 行 EvolutionController）
- `src/vllm_evolve/engine/population.py` (464 行)、`assembler.py` (253 行)、`plateau.py` (120 行) 中由 controller 驱动的部分
- `cli` 中 `vllm-evolve run` 子命令（启动 controller 跑 file IPC 的入口）
- `prompts/prompt_NNNN.txt` / `response_NNNN.py` IPC 协议
- `mock provider`

预计删除 1500+ 行"假 evolution framework"代码。

### B.3.2 一切核心能力必须暴露为 Claude Code 第一公民

| Claude Code 概念 | vllm-evolve 实现 | 落地位置 |
|---|---|---|
| **Sub-agent** | `vllm-policy-optimizer`：定义 7 个 phase + 错误分类 + 工具表 + 输出规范 | `agents/vllm-policy-optimizer.md` |
| **Skill** | 5 个工作流阶段 skill，每个带 frontmatter | `.claude/skills/{target-context-builder, policy-designer, policy-coder, policy-verifier, policy-optimizer}/SKILL.md` |
| **Skill references** | 14 个 mutation 策略下沉为 designer skill 的参考 | `.claude/skills/policy-designer/references/{eoh.md, reevo.md, ...}` |
| **Skill references** | hw-*.md 真实硬件文档 | `.claude/skills/policy-coder/references/hw/{h100.md, ascend910b.md, ...}` |
| **Skill scripts** | 各阶段验证脚本 | `.claude/skills/policy-verifier/scripts/check_decision_validity.py` 等 |
| **PreToolUse hook** | Bash/Edit 在错 phase 被 block | `.claude/hooks/guard_bash.py` + `settings.json` |
| **PostToolUse hook** | Edit 后自动跑 verify、Bash 后记录到 phase state | `.claude/hooks/post_edit.py`、`post_bash.py` |
| **Stop hook** | agent 完成一轮自动触发 keep/discard + git commit/rollback | `.claude/hooks/stop_save.py` |
| **Bash 工具** | `ar sim` / `ar bench` / `ar verify` / `ar keep` / `ar context` | 短前缀 `ar` + 单动词；输出 JSON |
| **settings.json** | 配置 hooks + 默认 model + 必要 permissions 白名单 | `.claude/settings.json` |

### B.3.3 工具命令的"agent 友好"重设计

当前命令长且不一致（`vllm-evolve simulate / verify / export-plugin / configure search ...`），从 agent 视角看几个问题：
- 每个命令多 token、容易拼错
- 命名不一致：`simulate` 是动词、`store` 是名词、`context` 是名词
- 输出大量人类可读文本，agent 解析成本高

重设计原则：
- **统一前缀** `ar`（autoresearch / agent-runtime，照搬 AscendOpGenAgent 习惯）
- **单动词**：`ar sim` / `ar bench` / `ar verify` / `ar keep` / `ar context` / `ar phase`
- **统一 JSON 输出**：`--json` 默认开，最后一行是结构化 JSON，前面才是人类可读
- **错误返回结构化"下一步指引"**：`{"ok": false, "phase_required": "VERIFY", "hint": "Run: ar verify policy.py"}`，agent 直接照做
- **hook block 给具体可执行的纠正命令**，照搬 AscendOpGenAgent 的"alias 表"机制（`eval.py: eval_wrapper.py`）—— agent 写错命令 hook 给指 alias

### B.3.4 Phase Machine 是 hook，不是运行时类

把 phase machine 从"框架内的一个 Python 类"改成"hook 调用的纯函数"：
- `phase_machine/phase_policy.py` 提供 `check_bash(phase, cmd) -> Decision` / `check_edit(phase, file) -> Decision`
- `.claude/hooks/guard_bash.py` 是薄壳：读当前 phase（从 `.ar_state/phase.txt`） → 调 `check_bash` → block 或放行
- Phase 状态机的转移由 `record_round()` 写入 `.ar_state/`，下一次 hook 读到的就是新 phase

这跟"框架自己跑 controller"的差别本质上是：**控制流的 owner 是 Claude Code，hook 是边界守卫，不是中央调度**。

### B.3.5 Sub-agent definition 是项目入口

照搬 `agents/ascend-kernel-developer.md` 的形态写 `agents/vllm-policy-optimizer.md`：
```markdown
---
name: vllm-policy-optimizer
description: 端到端优化 vLLM serving policy（scheduling/kv_eviction/...），
             经过 design → code → verify → simulate → benchmark → keep/discard 完整闭环
tools: [Bash, Read, Edit, Write, Grep, Glob]
---

# vLLM Policy Optimizer

你的任务是迭代优化一个 vLLM serving policy ……

## Phase 0：参数确认与硬件检测
... 必须运行 `ar phase init --target scheduling --hardware h100` ...

## Phase 1：读取上下文
... 调用 target-context-builder skill ...

## Phase 2：设计方案
... 调用 policy-designer skill；输出 design.md ...

## Phase 3：实现
... 调用 policy-coder skill；最多 5 次迭代 ...

## Phase 4：验证（L0 simulate + L1 vLLM benchmark）
... 强制顺序：先 ar verify → ar sim → 仅当 sim fitness 提升 > 2% 才允许 ar bench ...

## Phase 5：Keep / Discard
... `ar keep <eval.json>` 自动决定 git commit / rollback ...

## 错误分类
- A 类：可修代码错（最多 3 次同类）
- B 类：基础设施致命（vLLM 起不来等），终止
- C 类：连续 3 次同类 A 错 → 终止
```

用户启动 Claude Code 后，只需要：`> Use the vllm-policy-optimizer agent to improve scheduling for Qwen2.5-72B on H100`，agent 自己走完整 7 phase。

### B.3.6 settings.json 是项目"安全壳"配置

最小可用的 `.claude/settings.json`（草案）：
```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "python .claude/hooks/guard_bash.py"}]},
      {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "python .claude/hooks/guard_edit.py"}]}
    ],
    "PostToolUse": [
      {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "python .claude/hooks/post_edit.py"}]}
    ],
    "Stop": [
      {"hooks": [{"type": "command", "command": "python .claude/hooks/stop_save.py"}]}
    ]
  },
  "permissions": {
    "allow": [
      "Bash(ar:*)",
      "Bash(git status)",
      "Bash(git log:*)",
      "Bash(git diff:*)"
    ],
    "deny": [
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git reset --hard)"
    ]
  }
}
```
- `git commit` 由 Stop hook 自动跑（KEEP 时），agent 不允许直接调
- 任何破坏性 git 命令在 deny 名单里
- `ar` 子命令默认放行，普通 bash 走 hook 判定

### B.3.7 SKILL.md 必须遵守 Claude Code skill 规范

每个 skill 是一个目录：
```
.claude/skills/policy-designer/
├── SKILL.md                   # 带 frontmatter（name + description）
├── references/
│   ├── eoh.md                 # 14 mutation 策略下沉到这里
│   ├── reevo.md
│   ├── mcts.md
│   ├── hint-mode.md           # 类比 AscendOpGenAgent 的 hint-mode.md
│   └── cases/                 # 历史成功案例
│       ├── kv-pressure-srpt.md
│       └── cache-aware-scoring.md
└── scripts/
    └── render_design_template.py
```
SKILL.md frontmatter 必须有：
```yaml
---
name: policy-designer
description: |
  Use when an agent needs to choose a high-level optimization
  strategy (FCFS / SRPT / KV-pressure-gated / cache-aware / ...)
  for a vLLM serving policy. Reads target hints + current best +
  past attempts, outputs design.md.
---
```

`description` 措辞决定 Claude Code 能否在合适场景**自动选中** skill —— 这是 SKILL 规范里最容易被低估的设计点。

### B.3.8 archive_policies + git + Stop hook 完成闭环

每轮闭环的 owner 不是 controller，是 Stop hook：
1. Agent 完成一轮（修改 policy.py + 跑 ar bench 写出 eval.json）
2. **Stop hook 自动触发** `python .claude/hooks/stop_save.py`
3. Stop hook 内部跑 `record_round(eval.json)` →
   - KEEP：`git add archive_policies/<run_id>/ && git commit -m "..."`
   - DISCARD：`git restore archive_policies/<run_id>/`
   - FAIL：rollback + 累计 `consecutive_failures`
4. 写入 `.ar_state/phase.txt` 转移 phase

Agent 不需要"记得提交"，也不能绕过提交。

## B.4 修正后的目录形态（对附录 A.4 的再修正）

```
vllm-evolve/
├── .claude/                              # ★ 第一公民
│   ├── settings.json                     # hooks + permissions
│   ├── hooks/
│   │   ├── guard_bash.py
│   │   ├── guard_edit.py
│   │   ├── post_edit.py
│   │   └── stop_save.py
│   └── skills/                           # 每个 phase 一个 skill
│       ├── target-context-builder/SKILL.md
│       ├── policy-designer/
│       │   ├── SKILL.md
│       │   ├── references/
│       │   │   ├── eoh.md                # 14 mutation 策略下沉
│       │   │   ├── reevo.md
│       │   │   └── ...
│       │   └── scripts/
│       ├── policy-coder/
│       │   ├── SKILL.md
│       │   └── references/hw/{h100.md, ascend910b.md, a6000.md}
│       ├── policy-verifier/
│       │   ├── SKILL.md
│       │   └── scripts/check_decision_validity.py
│       └── policy-optimizer/SKILL.md
├── agents/
│   └── vllm-policy-optimizer.md          # ★ 主 agent 系统提示
├── .ar/                                  # 类比 .autoresearch/
│   ├── config.yaml
│   ├── state/                            # phase / consecutive_failures / progress
│   ├── phase_machine/
│   │   ├── phase_policy.py               # 纯函数，给 hook 调
│   │   └── transition.py
│   └── workflow/
│       └── round.py                      # record_round() 实现
├── bin/                                  # ★ ar 命令实现（包装现有 src/）
│   └── ar                                # 单一 entry，子命令 sim/bench/verify/keep/context/phase
├── targets/
│   └── scheduling/
│       ├── skeleton.py
│       ├── seed.py
│       ├── prompt_hints.md
│       └── primary_metric.yaml
├── benchmarks/
│   └── vllm_microbench/                  # L1 真 vLLM 评测脚本
├── archive_policies/                     # ★ 每个 KEEP 的 policy 三件套
│   └── scheduling/run_<id>/
│       ├── policy.py
│       ├── design.md
│       ├── eval_result.json
│       └── trace.md
├── src/vllm_evolve/                      # 大幅瘦身
│   ├── sim/                              # 保留 PDSimulator + Scheduler + KVCache + Workload
│   ├── trust/                            # 保留 L1 safety + L2 constraints
│   ├── store/                            # 保留 SQLite，但只做索引
│   └── targets/                          # 保留 plugin loader
└── docs/
```

砍掉的（相比附录 A.4 又少了一大块）：
- `src/vllm_evolve/scaffold/`（整体）
- `src/vllm_evolve/engine/{controller, population, assembler, plateau, checkpoint}.py`（绝大部分）
- `src/vllm_evolve/cli.py` 中 `run` 子命令
- `prompts/` IPC 目录
- `.opencode/`（如果只服务 Claude Code）
- 14 个 evolve_*.md skill（下沉为 designer skill 的 references/）

## B.5 第三轮新增的待对齐决策点

1. **是否绑定 Claude Code 作为唯一 runtime**？如果要支持 OpenCode/Cursor 等，hook 机制不通用，需要保留某种 IPC 后路（成本极大）。本附录默认"只服务 Claude Code"。
2. **是否接受 `agents/` + `.claude/skills/` + `.claude/hooks/` + `.claude/settings.json` 作为项目第一公民**？这意味着仓库变成"Claude Code project template"。
3. **是否接受工具命令统一改为 `ar <verb>` 短前缀**？涉及破坏性的 CLI 重构，但对 agent 体验影响巨大。
4. **是否接受删除 `engine/controller.py` + `scaffold/` 全部**？这是把"框架"还原成"harness"的关键动作，删 1500+ 行代码。
5. **第一里程碑定义**：让 `vllm-policy-optimizer` agent 在 Claude Code 里跑通 1 轮"scheduling 的 design → code → verify → sim → bench → keep" 完整闭环（不需要演化多轮，只需要闭环跑通）。这个里程碑是否合理？

## B.6 三轮思考的演进总结

| 轮次 | 核心问题 | 关键修正 |
|---|---|---|
| 第一轮 | 模板灌水 + 三处真理源 + L3/L4 没接 | 砍到 5–7 个 target，统一引擎，L3 真接上 |
| 第二轮（附录 A） | 没有真 vLLM = 演化结果无法验证 | L0 sim + L1 真 vLLM；Phase Machine + Hook + keep/discard + git；archive_policies 替代 SQLite 主存；先做透 1 个 target |
| 第三轮（附录 B） | **身份错位：把 Claude Code 当 API 而不是 runtime** | 删 scaffold + controller；改造为 sub-agent + skills + hooks + settings.json 的 Claude Code harness；工具命令 `ar <verb>` 短前缀；SKILL.md 严格遵循规范 |

最简而本质的一句话：

> **vllm-evolve 不应该有 controller，因为 Claude Code 就是 controller；vllm-evolve 应该是一组让 Claude Code 在受控的 phase 里安全地优化 vLLM policy 的 skills + hooks。**
