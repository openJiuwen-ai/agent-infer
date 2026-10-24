# sr-config-tool

SR（Semantic Router）模型能力评测与路由配置生成工具包。

## 概述

本工具包提供从数据集抽样 → 双模型评测 → 成本感知校准 → 配置生成 → E2E 验证的完整流程。
所有路径和模型配置通过 `config/run_config.yaml` 统一管理，支持 SLURM/本地/SSH/Docker 执行环境。

## 文件清单

```text
sr-config-tool/
├── config/
│   └── run_config.example.yaml  # 唯一配置入口（复制为 run_config.yaml 后编辑）
├── src/
│   ├── common.py                 # 公共常量 + 工具函数 + launcher 抽象
│   ├── build_tasks.py            # 阶段 1a：分层抽样验收题集
│   ├── split_evalset.py          # 阶段 1d：扩抽+锁定+1:2 切分训练/测试集
│   ├── run_eval.py               # 阶段 1b/2a/2c：启动 vLLM → 推理 → 判分
│   ├── combine_records.py        # 阶段 2b：合并验收+新题记录并重新判分
│   ├── think_gate_eval.py        # 阶段 4b：oracle 上界 + 半拆 CV
│   ├── join_three_way.py         # 阶段 6a：离线 join（零 GPU 模拟配置效果）
│   ├── convert_eval_to_bench.py  # 阶段 6b 前置：eval→bench 测试集格式
│   ├── run_e2e.py                # 阶段 6b：E2E 终验（vLLM + router + envoy → 压测）
│   ├── sr_baseline_eval.py       # 阶段 6b 内部：bench 客户端（并发压测+统计）
│   └── pipeline.py               # 统一编排入口（子命令 + 断点恢复）
├── templates/
│   ├── gen_params_small.json     # 小模型采样参数模板
│   ├── gen_params_large.json     # 大模型采样参数模板
│   └── gen_params_large_nthink.json  # 大模型 nothink 参数模板
├── workspace/                    # 任务工作空间（自动创建）
└── requirements.txt              # pip 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 编辑配置

复制配置模板并按环境编辑（`config/run_config.yaml` 已被 gitignore，不会误提交机器相关路径）：

```bash
cp config/run_config.example.yaml config/run_config.yaml
```

编辑 `config/run_config.yaml`，修改以下关键字段：

- `paths.data_dir`：数据集根目录（含 MMLU/MMLU-Pro/CMMLU/C-Eval）
- `paths.sr_dir`：SR 代码仓库路径
- `python.eval_venv` / `python.judge_venv`：Python 解释器路径
- `launcher.type`：执行方式（local/srun/ssh/docker）
- `models.small.*` / `models.large.*`：模型路径、TP 数、GPU 数等

### 3. 运行流程

```bash
cd sr-config-tool

# 方式一：逐步执行（推荐）
python src/pipeline.py --task-id my-exp-01 build-tasks
python src/pipeline.py --task-id my-exp-01 run-eval --model small
python src/pipeline.py --task-id my-exp-01 run-eval --model large
python src/pipeline.py --task-id my-exp-01 check-differentiation
python src/pipeline.py --task-id my-exp-01 split-evalset
python src/pipeline.py --task-id my-exp-01 run-calib --model small
python src/pipeline.py --task-id my-exp-01 run-calib --model large
python src/pipeline.py --task-id my-exp-01 combine-records
python src/pipeline.py --task-id my-exp-01 run-nothink
python src/pipeline.py --task-id my-exp-01 think-gate
python src/pipeline.py --task-id my-exp-01 cost-calib
python src/pipeline.py --task-id my-exp-01 gen-config
python src/pipeline.py --task-id my-exp-01 join
python src/pipeline.py --task-id my-exp-01 convert
python src/pipeline.py --task-id my-exp-01 e2e

# 方式二：断点恢复（重发对应子命令即可；已完成的步骤按 state.json 自动跳过。
# 注意：--resume 需与子命令一起使用，单独使用只打印状态）
python src/pipeline.py --task-id my-exp-01 --resume run-eval --model small

# 方式三：dry-run（只打印命令不执行）
python src/pipeline.py --task-id my-exp-01 --dry-run build-tasks

# 查看所有历史任务
python src/pipeline.py --list-tasks
```

## 流程说明

| 阶段 | 命令 | 说明 | 需要 GPU |
| --- | --- | --- | --- |
| 1a | build-tasks | 从 4 个数据集分层抽样 380 道验收题 | 否 |
| 1b | run-eval --model small/large | 双模型各跑一遍验收题集 | 是 |
| 1c | check-differentiation | 能力区分 go/no-go（gap+饱和度+互斥率） | 否 |
| 1d | split-evalset | 扩抽+验收锁定+1:2 切分 → train/eval | 否 |
| 2a | run-calib --model small/large | think 形态跑训练集新题 | 是 |
| 2b | combine-records | 合并验收+新题→759 题全集判分 | 否 |
| 2c | run-nothink | nothink 形态跑训练集全集 | 是 |
| 4a | cost-calib | λ 成本感知校准 + token costs | 否 |
| 4b | think-gate | oracle 上界 + 半拆 CV | 否 |
| 5c | gen-config | 用 result_to_config.py 生成 SR 配置 | 否 |
| 6a | join | 离线 join（零 GPU 模拟配置效果） | 否 |
| 6b 前置 | convert | eval→bench 测试集格式转换（e2e 缺测试集时自动调用） | 否 |
| 6b | e2e | E2E 终验（vLLM + router + envoy → 压测） | 是 |

## 工作空间

每次运行创建一个任务工作空间 `workspace/{task_id}/`：

```text
workspace/my-exp-01/
├── tasks/           # accept/train/eval JSONL
├── results/         # {model}/{suffix}/{records,judged,analysis}
├── logs/            # 每步执行日志
├── config_output/   # 生成的 SR 配置 yaml
└── state.json       # 断点恢复状态
```

## 依赖的外部脚本

本工具包调用 SR 仓库的以下脚本（路径由 `run_config.yaml` 的 `paths.sr_eval_dir` 指定）：

- `multi_benchmark_eval.py` — 多数据集推理
- `mcq_judge.py` — MCQ 判分 → judged.jsonl + analysis.json
- `differentiation_check.py` — 能力区分验收
- `cost_aware_calib.py` — λ 成本感知校准
- `result_to_config.py` — SR 配置生成
- `signal_predict.py` — 信号分类器推理

## E2E 步骤说明

E2E 终验（阶段 6b）需要 SR 全栈已编译/部署：

- router 二进制（`$SR_ROOT/bin/router`）
- envoy 二进制
- candle Rust binding（`libcandle_semantic_router.so`）

用户可通过二进制或 Docker 镜像启动 SR，工具包只通过 API 调用。

## 注意事项

1. **数据集**：需要自行下载 MMLU、MMLU-Pro、CMMLU、C-Eval 到 `data_dir` 指定目录
2. **Python 环境**：推理需要 `openai` 库，判分只需 stdlib，E2E 需要 SR CLI
3. **GPU**：run-eval、run-calib、run-nothink、e2e 步骤需要 GPU
4. **SR 代码**：需要 clone SR 仓库并指定路径
