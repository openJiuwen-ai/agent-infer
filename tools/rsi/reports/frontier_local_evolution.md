# Codex + 本地 Frontier 世代演化报告

公开版本已将原始本机路径替换为 `/path/to/workspace`；测量值、策略源码和哈希保持原样。
这是历史实验记录，当前安装步骤以 [README](../README.md#quick-start) 为准。

日期：2026-07-26  
分支：`codex/frontier-local-evolve`  
功能基线：`origin/feat/autopt-subagents`  
结论：**本地真实 Frontier 模拟验收通过；产物仅为 `sim_winner`，尚未经过真实 vLLM。**

## 1. 结论摘要

本次实现把 auto-subagents 的世代演化架构、Codex 最小适配、可执行的 Frontier
`ve_policy` bridge、官方 BurstGPT 连续片段和合成压力场景连成了本地闭环。全程没有
SSH、远程 GPU 或真实 vLLM 调用。

最终候选是 **D3Q（Decode-Dispersion-aware Dynamic Queue）**：

- 源码 SHA256：
  `d4b3a3ee8c98f82c1d900e7faaf311349f2c61797e551a631b70f9b38d80c327`
- 源码：`reports/frontier_local_evolution/sim_winner/work_variant.py`
- held-out 相对每个场景最强基线的 goodput 中位提升：**13.7196%**
- 正收益场景：**2/3**
- 最差场景：**-0.2258%**，高于 -2% 下限
- 完成请求数：三个场景均不减少
- 候选 marker：全部有效，合计 `invocations=109866`、`fallbacks=0`
- defer / forced：均为 0；本候选不依赖 defer，不存在无限 defer 风险
- 去掉 decode-aware 生命周期机制后，D3Q 的场景收益中位数为 **70.2364%**，2/3
  场景为正，消融门槛通过

门槛逐项结果：

| 验收项 | 要求 | 结果 |
|---|---:|---:|
| held-out goodput 聚合中位提升 | ≥3% | **13.7196%** |
| 正收益场景 | ≥2/3 | **2/3** |
| 单场景最大回退 | 不低于 -2% | **-0.2258%** |
| 完成数 | 不减少 | **通过** |
| marker | SHA 相等、invocations>0、非全 fallback | **通过** |
| 机制消融 | 支持新增机制 | **通过** |

这证明的是：D3Q 在固定 Frontier 版本、模型和这些 trace/config 上产生了可复现的
**模拟收益**。它不证明真实 GPU/vLLM 收益，也不允许进入 `ve keep`。

## 2. 端到端软件流程

```text
Codex / ve-evolve skill
  │
  ├─ 读取研究事实、诊断、历史 lessons
  ├─ AuthorContext.to_prompt()
  │    └─ 诊断 + 父代源码/分数 + peers + 失败原因 + lessons + 预算
  ├─ author 返回完整源码字符串
  ├─ orchestrator 写 variants/g<generation>_<child>_<sha>.py
  ├─ L1 AST/接口检查 + L2 动态安全检查
  ├─ 对 train/validation 场景调用真实 Frontier
  │    └─ VE_FRONTIER_PYTHON -m frontier.main
  │         └─ ve_policy → candidate.schedule_batch()
  │              └─ ve_policy_marker.json
  ├─ 分数进入 Archive，下一代收到父代、分数和错误反馈
  ├─ 冻结 winner source + SHA
  ├─ 冻结后才物化最终 test
  ├─ paired seeds 0/1/2：
  │    FCFS/SJF/LJF/LIFO/current seed → 每场景最强基线
  │    frozen winner
  │    两个机制消融
  └─ acceptance → sim_winner/evidence.json + report
```

本地演化入口 `ve frontier-evolve` 不导入远程 workspace，也不调用 SSH。远程 real-vLLM
路径仍保留在原架构中，但不是本目标默认路径。

## 3. 统一基线与旧 Codex 提交迁移

没有整体 merge `codex/codex-adapter`；以下五个提交逐项审查后语义迁移到当前
`cli/main.py + engine/evolve_loop.py + engine/*` 架构：

| 旧提交 | 处置 | 当前落点与理由 |
|---|---|---|
| `57838c2` Add Codex local-to-remote evolution workflow | 部分迁移、远程部分跳过 | `.codex` agent/hook、`.agents/skills`、Codex installer、AGENTS workflow 迁入 `assets/codex` 与 `install/codex_installer.py`；旧 `ar_cli.py`、`autopt/` 和 SSH workspace 不恢复 |
| `5f1ae40` Implement evidence-bound Codex evolution loop | 语义替代 | 完整 author context、源码返回、预算、lineage、验证反馈和证据绑定由现有 `AuthorContext`、`evolve_loop`、`evolve_target` 和 `local_frontier_evolve` 承担 |
| `c45e732` Revalidate bound environment evidence on keep | 由更强隔离替代 | Frontier 永远是 `simulator_nonqualifying`，不能进入 keep；marker SHA、真实调用次数和命令/环境证据在模拟边界即校验 |
| `816ed06` Harden reproducible acceptance evidence | 语义迁移 | 固定数据/版本/hash、精确 argv、paired seeds、每场景最强基线、冻结后 held-out、原始指标路径和失败结果均落盘 |
| `b571b70` Revalidate runtime effectiveness and quality | 语义迁移，远程 runtime 跳过 | `policy_sha256 + invocations + fallbacks` 证明候选实际执行；goodput/p50/p99/throughput/completion/消融验证有效性；SSH/GPU runtime effectiveness 本目标不执行 |

因此保留了 auto-subagents 分支上的 `ve` CLI、当前 engine、research harness、lessons、
Frontier bridge 和世代 Archive，没有复活旧 `ar_cli/autopt` 目录结构。

## 4. Codex 最小适配

包内源码位于：

- `src/vllm_evolve/assets/codex/agents/vllm-policy-optimizer.toml`
- `src/vllm_evolve/assets/codex/hooks.fragment.json`
- `src/vllm_evolve/assets/codex/hooks/phase_guard_hook.py`
- `src/vllm_evolve/assets/codex/skills/ve-evolve/SKILL.md`
- `src/vllm_evolve/assets/codex/AGENTS.snippet.md`
- `src/vllm_evolve/install/codex_installer.py`

`ve init --client codex` 将它们物化为 `.codex/`、`.agents/skills/` 和受界定的
`AGENTS.md` 片段；支持 check、force、uninstall，且不覆盖用户已有配置。Codex author
只返回源码，由编排器写文件，不依赖 Claude 子代理或 Claude 专属工具。

自动化测试覆盖安装/卸载/幂等、skill 发现、hook 判定、完整 prompt 字段和本地演化
acceptance。

## 5. 数据与防泄漏

### 5.1 官方 BurstGPT

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/HPMLL/BurstGPT` |
| commit | `d895a53bb7b8ec137d0d2fe203b335835a78c10a` |
| 文件 | `data/BurstGPT_1.csv` |
| CSV SHA256 | `46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a` |
| 解析 | streaming；同时支持官方 6/8 列格式 |
| Frontier 转换 | `arrived_at, num_prefill_tokens, num_decode_tokens, session_id, block_hash_ids` |
| tenant/prefix 诚实性 | 固定 neutral `session_id=0`，`block_hash_ids=""`，不开 prefix caching |

仓库中的 `tests/bench/fixtures/burstgpt_fragments.csv` 只做 parser/split 单测，从未用于本报告
的收益数字。

### 5.2 连续时间划分

先过滤非正 duration，再按原序的连续带划分；不 shuffle：

| 用途 | 时间带 | 请求 ID | source 时间 | Frontier rows SHA256 |
|---|---|---|---|---|
| train | first third | 38721–38752 | 822482–822483 s | `d2828f627ae201976f8464a378c0785501f5e667333f4d6a8328cc8de2c70dbd` |
| validation | middle third | 630324–630355 | 1972957–1972958 s | `908058f47af797182b358b6b89dfac4d4570a1ce2d92ab805ea4f73382216442` |
| validation regression | first half of final third | 1089830–1089861 | 4012044–4012045 s | `cd810d39bb0d3ec422b3631d96f4e75fb68005a34cee3ebe6c42273a2a2d3ea2` |
| final test | last sixth | 1248788–1248819 | 4901571–4901575 s | `836d4e28cdf46797f862627b43eb3255f7520db30f27432091080769ca41898f` |

每段 32 请求。搜索只物化 train/validation/regression；winner
`d4b3a3ee…` 冻结后才读取 last-sixth test。`dataset_manifest.json` 写入
`heldout_materialized_after_winner_sha256` 作为顺序证据。

### 5.3 合成压力负载

合成数据单独标记为 `synthetic_multi_tenant_prefix_stress`，才允许 tenant 与 prefix hint。
生成器覆盖：

- train：3 tenants，burst size 4，burst 内间隔 30 ms；
- validation：4 tenants，burst size 4，间隔 20 ms；
- held-out moderate：4 tenants，burst size 5，间隔 15 ms；
- held-out severe：5 tenants，burst size 5，间隔 10 ms；
- 每个 regime 有反相关的 prompt/output 长度模式、多个 prefix blocks、不同 batch 压力。

官方和合成 trace 绝不混标。

## 6. D3Q 算法

D3Q 不是若干系数的线性打分，也不是给 FCFS/SJF/LIFO 改名。它在每次调度观察队列状态，
采用三个离散控制流 regime：

```text
输入：waiting, running, max_num_seqs
pressure = |waiting| + |running|
median_decode = median(waiting.requested_output_tokens)
wide_decode_dispersion = min_decode * 4 <= median_decode
has_real_prefix_signal = any(request.has_prefix_hint)

if pressure <= max_num_seqs:
    按 arrival 排序                         # 低压：避免无谓重排
else if wide_decode_dispersion and not has_real_prefix_signal:
    按 remaining_prompt 降序，再按 arrival   # 无 prefix 的混合 decode 模式保护
else:
    按 remaining_prompt + requested_output 升序
                                             # 高压：最短生命周期占用优先

在 token/sequence budget 内依次接纳；running decode 继续执行。
```

关键新机制是“**队列压力状态切换 + decode 长度离散度识别 + prefix 真实性分支 + 生命周期
占用排序**”。它避免用固定短 prompt 近似完整请求成本；低压保持 FCFS，高压时才改变策略。
源码中的 `4×` 是离散模式检测阈值，不参与连续加权求和；最终价值证据来自控制流消融，
而不是自动 novelty 分数。这里仅主张它相对本项目旧模板是有实质结构的新候选，不主张
学术上的全球首创。

关键代码：

```python
if queue_pressure <= max(1, max_num_seqs):
    _order = lambda r: r.arrival_time_s
elif mixed_decode_modes and not has_prefix_signal:
    _order = lambda r: (-r.remaining_prompt_tokens, r.arrival_time_s)
else:
    _order = lambda r: (
        r.remaining_prompt_tokens + r.num_output_tokens,
        r.arrival_time_s,
    )
```

完整、逐字节保留的生成源码见
`reports/frontier_local_evolution/sim_winner/work_variant.py`。

## 7. 搜索过程与 lineage

搜索阶段每个候选在 5 个 held-in 场景运行；搜索只用 seed 0，最终验收才用三组 paired
seeds。baseline 是每个场景在 FCFS/SJF/LJF/LIFO/current seed 中的最强者。

### 7.1 最终候选 held-in

| 场景 | split | 最强基线 | 基线 goodput | D3Q goodput | gain |
|---|---|---|---:|---:|---:|
| burstgpt_train | train | FCFS | 18.5218 | 18.5218 | 0.0000% |
| burstgpt_validation | validation | FCFS | 5.1416 | 7.8672 | +53.0120% |
| burstgpt_validation_tail | validation | LJF | 16.9050 | 16.9050 | 0.0000% |
| stress_train | train | LJF | 4.8351 | 7.2987 | +50.9529% |
| stress_validation | validation | LIFO | 2.3022 | 3.2170 | +39.7379% |

中位 held-in gain = **39.7379%**。

### 7.2 世代表

| generation | best SHA | parent | score | 说明 |
|---:|---|---|---:|---|
| 0 | `d4b3a3ee…` | root | 39.7379 | D3Q，冻结 winner |
| 1 | `f21550c8…` | `d4b3a3ee…` | 39.7379 | 子代持平，未替换 SHA |

共 9 次候选评估、2 代，终止原因 `generations_exhausted`。完整 archive 共 9 项，父代 SHA
写在 `runs/frontier_local_evolution_v2/sim_winner/evidence.json` 的
`evolution.archive`。

## 8. held-out 基线

以下是三 seed 中位 goodput（req/s）。加粗的是每场景最终对手：

| policy | BurstGPT test | stress moderate | stress severe |
|---|---:|---:|---:|
| FCFS | 22.6328 | 1.1687 | **0.6853** |
| SJF | 22.6328 | 1.3149 | 0.4927 |
| LJF | **22.6841** | 1.0048 | 0.4920 |
| LIFO | 22.6328 | **3.2117** | 0.6605 |
| current seed | 22.6328 | 1.1687 | **0.6853**（与 FCFS 并列） |

没有选择统一的弱基线；每个场景独立取五者最大值。

## 9. held-out 最终结果

所有值为 seeds `0,1,2` 的 paired 中位数。

| 场景 | 最强基线 | baseline goodput | D3Q goodput | gain | D3Q p50 TTFT ms | D3Q p99 TTFT ms | D3Q throughput req/s | 完成数 base/cand |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BurstGPT test | LJF | 22.6841 | 22.6328 | **-0.2258%** | 0.456 | 3.648 | 22.6328 | 32/32 |
| stress moderate | LIFO | 3.2117 | 3.6523 | **+13.7196%** | 240.296 | 4322.928 | 8.7655 | 60/60 |
| stress severe | FCFS | 0.6853 | 0.8387 | **+22.3897%** | 592.840 | 9040.853 | 6.2906 | 75/75 |

用于正确理解 TTFT 的最强基线侧指标：

| 场景 | baseline p50 ms | baseline p99 ms | baseline throughput req/s |
|---|---:|---:|---:|
| BurstGPT / LJF | 0.456 | 12.234 | 22.6841 |
| moderate / LIFO | 1193.312 | 6025.781 | 8.7591 |
| severe / FCFS | 4330.680 | 9231.373 | 6.4248 |

goodput 是 `throughput × (TTFT≤200ms 的完成请求比例)`，不是单独优化吞吐。severe
场景的吞吐略低于 FCFS，但满足 200ms TTFT 的请求比例更高，因此 goodput 提升。

### Marker 与饥饿风险

| 场景 | invocations（三 seed合计） | fallbacks | defers | forced | marker SHA |
|---|---:|---:|---:|---:|---|
| BurstGPT | 99 | 0 | 0 | 0 | 匹配 |
| moderate | 39279 | 0 | 0 | 0 | 匹配 |
| severe | 70488 | 0 | 0 | 0 | 匹配 |

D3Q 本身不 defer；所有请求完成，因此没有由无限 defer 引起的饥饿。bridge 对未来
defer 型候选另有八次事件强制提升的防饥饿机制与单测。

## 10. 消融

主消融 `no_decode_signal` 保留低压/高压 switch，但高压退化为 prompt-only SJF，删除
requested output 生命周期信号：

| 场景 | D3Q goodput | no-decode goodput | D3Q 相对 gain |
|---|---:|---:|---:|
| BurstGPT test | 22.6328 | 22.6328 | 0.0000% |
| moderate | 3.6523 | 1.3149 | +177.7593% |
| severe | 0.8387 | 0.4927 | +70.2364% |

中位 +70.2364%，2/3 正收益，说明主要收益来自 decode-aware 生命周期占用机制，而不是
单一常量或 seed 偶然性。

第二消融 `no_dispersion_guard` 在最终三个 held-out 场景与 D3Q 持平，说明最终 last-sixth
没有触发该分支；但它对 v1 暴露的无 prefix、decode 高离散 regression 是必要保护。该
机制不能靠最终 held-out 单独证明，因此报告不将它包装成最终收益来源。

## 11. 保留的失败尝试

第一次真实运行 `runs/frontier_local_evolution` **明确失败**：

| 场景 | 最强基线 | v1 candidate | gain |
|---|---:|---:|---:|
| 当时的 BurstGPT held-out | LJF 17.7379 | 16.8748 | **-4.8662%** |
| moderate | LIFO 3.2117 | 3.6523 | +13.7196% |
| severe | FCFS 0.6853 | 0.8387 | +22.3897% |

虽然聚合中位仍是 +13.7196%，但单场景低于 -2%，所以 v1 的 `ok=false`、验收失败，没有
降低门槛。

该失败窗口为请求 1089830–1089861，source 时间 4012044–4012045 s。v2 将这个已经看过
的窗口降级为 `validation_tail` regression，并加入 decode dispersion guard；新的最终
test 改为此前未读取的 last-sixth 请求 1248788–1248819。这样利用失败改算法，同时不把
见过的数据重新冒充 held-out。

## 12. Frontier 真实性与命令

| 项 | 值 |
|---|---|
| Frontier commit | `a4b22df8211864bf229258ecdfbe680f048f2d77` |
| bridge patch SHA256 | `a8d0262c033ad79bfab299cf7ee96ffacd9b0efb534b15ad7eaa98a6fd56ca3e` |
| runner | `/path/to/workspace/Frontier/.venv/bin/python` |
| 执行形式 | `python -m frontier.main` |
| Frontier command 记录数 | 142 |
| marker 文件数 | 128（stock FCFS 不产生 ve_policy marker） |
| 远程调用 | 0 |

代表性的最终 severe / seed 2 **实际 argv**：

```text
/path/to/workspace/Frontier/.venv/bin/python -m frontier.main
  --simulation_mode online
  --sys_arch co-location
  --cc_backend_config_type analytical
  --cluster_config_num_replicas 1
  --replica_config_model_name meta-llama/Llama-2-7b-hf
  --replica_config_attn_tensor_parallel_size 1
  --replica_config_num_pipeline_stages 1
  --replica_config_attn_data_parallel_size 1
  --replica_scheduler_config_type ve_policy
  --request_generator_config_type trace_replay
  --trace_request_generator_config_trace_file
    /path/to/workspace/vllm-evolve/runs/frontier_local_evolution_v2/raw/heldout_candidate/test/stress_test_severe/winner_d4b3a3ee8c98/trace.csv
  --trace_request_generator_config_max_tokens 4480
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps 1.0
  --random_forrest_execution_time_predictor_config_enable_dummy_mode
  --random_forrest_execution_time_predictor_config_dummy_execution_time_ms 0.001
  --metrics_config_output_dir
    /path/to/workspace/vllm-evolve/runs/frontier_local_evolution_v2/raw/heldout_candidate/test/stress_test_severe/winner_d4b3a3ee8c98/seed_2/arms/winner_d4b3a3ee8c98_2
  --metrics_config_run_id winner_d4b3a3ee8c98_2
  --metrics_config_write_metrics
  --metrics_config_store_request_metrics
  --metrics_config_store_batch_metrics
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
  --no-metrics_config_write_json_trace
  --seed 2
  --ve_policy_scheduler_config_enable_prefix_caching
  --ve_policy_scheduler_config_batch_size_cap 4
  --ve_policy_scheduler_config_max_tokens_in_batch 32768
```

实际环境还设置：

```text
VE_POLICY_PATH=/path/to/workspace/vllm-evolve/runs/frontier_local_evolution_v2/sim_winner/work_variant.py
VE_POLICY_ROOT=/path/to/workspace/vllm-evolve
VE_MARKER_DIR=<该 arm 的输出目录>
```

每一次实际调用的完整数组（没有 shell 重解释）都位于对应 arm 的
`frontier_command.json`。

### 原始指标路径

- 总证据：
  `runs/frontier_local_evolution_v2/sim_winner/evidence.json`
- 数据 manifest：
  `runs/frontier_local_evolution_v2/sim_winner/dataset_manifest.json`
- BurstGPT candidate seed 0：
  `runs/frontier_local_evolution_v2/raw/heldout_candidate/test/burstgpt_test/winner_d4b3a3ee8c98/seed_0/arms/winner_d4b3a3ee8c98_0/meta_llama_llama_2_7b_hf/online_serving/winner_d4b3a3ee8c98_0`
- moderate candidate seed 0：
  `runs/frontier_local_evolution_v2/raw/heldout_candidate/test/stress_test_moderate/winner_d4b3a3ee8c98/seed_0/arms/winner_d4b3a3ee8c98_0/meta_llama_llama_2_7b_hf/online_serving/winner_d4b3a3ee8c98_0`
- severe candidate seed 0：
  `runs/frontier_local_evolution_v2/raw/heldout_candidate/test/stress_test_severe/winner_d4b3a3ee8c98/seed_0/arms/winner_d4b3a3ee8c98_0/meta_llama_llama_2_7b_hf/online_serving/winner_d4b3a3ee8c98_0`
- 所有基线：
  `runs/frontier_local_evolution_v2/raw/heldout_baselines/test/`
- 所有消融：
  `runs/frontier_local_evolution_v2/raw/heldout_ablations/test/`

仓库内的紧凑证据副本是
`reports/frontier_local_evolution/evidence_summary.json`。

## 13. 从全新 checkout 复现

```bash
git clone https://github.com/openJiuwen-ai/agent-infer.git
cd agent-infer/tools/rsi
python3.12 -m venv .venv312
. .venv312/bin/activate
pip install -e ".[dev]"
ve init --client codex

git clone https://github.com/NetX-lab/Frontier.git ../Frontier
git -C ../Frontier checkout a4b22df8211864bf229258ecdfbe680f048f2d77
# 按 Frontier 自身 README 创建 ../Frontier/.venv 并安装依赖

export VE_FRONTIER_REPO="$(cd ../Frontier && pwd)"
export VE_FRONTIER_PYTHON="$VE_FRONTIER_REPO/.venv/bin/python"
python integrations/frontier/ensure_patch.py "$VE_FRONTIER_REPO"

git clone https://github.com/HPMLL/BurstGPT.git ../BurstGPT
git -C ../BurstGPT checkout d895a53bb7b8ec137d0d2fe203b335835a78c10a
sha256sum ../BurstGPT/data/BurstGPT_1.csv
# 期望 46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a

ve frontier-evolve \
  --burstgpt ../BurstGPT/data/BurstGPT_1.csv \
  --out runs/frontier_local_evolution_v2 \
  --seeds 0,1,2 \
  --fragment-size 32 \
  --slo-ttft-ms 200 \
  --max-num-seqs 4 \
  --generations 2 \
  --population 5 \
  --max-total-evals 10

python -m pytest -q
ruff check src tests integrations/frontier/*.py
```

macOS 可用 `shasum -a 256` 代替 `sha256sum`。

## 14. 最终验证记录

| 门禁 | 命令/方式 | 结果 |
|---|---|---|
| 全量 pytest | 带 `VE_FRONTIER_*` 与 `VE_BURSTGPT_CSV` 的 `python -m pytest -q` | **705 passed, 3 skipped** |
| ruff | `ruff check src tests integrations/frontier/*.py` | **通过** |
| frozen core + enum parity + packaging | 三个定向测试文件 | **16 passed** |
| 新增无 mock E2E | `tests/test_frontier_local_evolution_real_e2e.py`，官方 BurstGPT + 真实 Frontier | **1 passed** |
| patch 幂等 | `ensure_patch.py` 实树调用 + 临时 git repo 双调用测试 | **already_applied / 通过** |
| patch 完整性 | 在 Frontier checkout 执行 `git apply --check --reverse` | **通过** |
| winner 字节 | `shasum -a 256 work_variant.py` | **匹配 d4b3a3ee…** |
| wheel | `pip wheel . --no-deps` 并检查压缩包 | **构建成功，含 agent/hook/5 个 Codex skills** |

三个 skip 是环境门控项，不是本地 Frontier E2E；新增真实 E2E 在本次全量运行中实际执行。
`ruff check .` 还会扫描功能基线已有的 `targets/scheduling/seeds/` 策略样本 zoo，并报告
75 个历史格式问题；项目门禁因此明确 lint 产品源码、测试和 Frontier Python 集成，不把
这批未改动的历史候选格式问题伪装成本次回归。

## 15. 诚实限制与下一步

1. Frontier 是离散事件模拟器；dummy execution-time predictor、固定 Llama-2-7B 模型和
   bridge 的队列级可插拔边界不能完全代表真实 vLLM GPU 调度。
2. 三个 seed 在 trace replay 下得到相同结果，满足 paired-seed 协议，但没有覆盖模拟器
   参数不确定性；应补充 trace/config 扰动和更多工作负载。
3. 官方 BurstGPT 只有 prompt/output/arrival 类信息，不能证明真实 prefix/tenant 收益；
   prefix 结果只来自明确标记的合成场景。
4. 最终官方 test 只有 32 请求；其结果接近持平。主要收益来自合成高压场景，不能外推为
   广泛生产收益。
5. `no_dispersion_guard` 在最终 held-out 持平；guard 的证据来自 v1 失败转成的 validation
   regression，不应夸大。
6. 本目标明确禁止真实 vLLM，因此 D3Q 必须保留
   `REQUIRES_REAL_VLLM_VERIFICATION`。下一阶段应在隔离的 GPU sandbox 上做相同 trace、
   多 seed、真实 scheduler plugin 的 A/B，只有真实证据通过 frozen gate 才能 keep。
