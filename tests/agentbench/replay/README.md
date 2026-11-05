# Replay CPU CI

实现依据：[RFC #52](https://github.com/openJiuwen-ai/agent-infer/issues/52)，
并复用 [PR #136](https://github.com/JiusiServe/AgentInfer/pull/136) 的 run 冒烟链路。

## 运行

仓库根目录安装 `pip install -e '.[dev]'`，使用 Python 3.10 或 3.12。
run 冒烟还需要 Node.js 20；Replay 测试只需要 Python 和回环 HTTP 服务，无需 GPU、模型或数据集下载。

```bash
# 后端 HTTP 契约；不依赖 Replay 的构造或解析代码。
python -m pytest --timeout=120 -q tests/agentbench/test_mock_server.py

# 四来源 × 并发度 1/4 × 两轮冷启动。
python -m pytest --timeout=120 -q tests/agentbench/replay/test_cold_start.py

# HTTP 依赖、故障传播和校验器反例。
python -m pytest --timeout=120 -q -m replay_smoke tests/agentbench/replay

# 精确时间和已有单元测试。
python -m pytest --timeout=120 -q -m 'not replay_smoke' tests/agentbench/replay

# 完整 CPU CI 范围。
python -m pytest --timeout=120 -q tests/agentbench tests/scheduling tests/agentcache/test_entrypoints.py
```

需要保留产物时，给 pytest 增加 `--basetemp=/tmp/replay-ci-review`。
pytest 会清理指定的 basetemp，使用专门的临时目录。
CI 沿用现有 Python 3.10/3.12 CPU 测试任务，自动收集 `tests/agentbench` 下的单元测试及 run/Replay 冒烟测试。
原始请求、响应和事件记录位于 pytest 临时目录；现有 CI 不自动上传这些产物。

## 组件边界

- `../helpers/mock_server/`：run 和 Replay 共用的独立进程服务，绑定回环随机端口。
  `run_legacy` 保留原有 `ok`、usage 7/2/9、cached 3；`replay` 由实际内容计算长度。
- `helpers/fixtures.py`：人工构造四种来源数据和预期；每轮 8 个 task、24 个正式请求。
- `helpers/entrypoint.py`：只注入 observer，调用真实 Replay CLI。
- `helpers/cold_start.py`：每轮启动新的服务和 Replay 子进程，限时等待，记录退出并回收。
- `helpers/validation.py`：核对源预期、plan、服务接收记录、客户端事件及结果。
  依赖正确性在这里和测试断言中判断，服务只控制响应。

服务不读取 Replay plan。测试驱动先校验计划，再将逻辑节点解析为 runtime request ID，
通过 `/__mock__/arm` 安装故障或 gate。请求在等待 arm **之前**被记录，提前到达不会被隐藏。

独立启动服务时，先在新目录中写入 `scenario.json`，例如：

```json
{"profile":"replay","model":"replay-ci-model","round_index":0}
```

```bash
python -m tests.agentbench.helpers.mock_server --artifact-dir /tmp/replay-mock
```

`ready.json` 给出实际地址、PID 和进程 nonce。
`POST /__mock__/shutdown` 关闭服务并排空 recorder；控制接口只绑定回环地址。
场景 schema、gate 和故障种类见 `../helpers/mock_server/protocol.py`。

## 验收覆盖

| RFC 用例 | 证据位置 | 核心断言 |
| --- | --- | --- |
| MS01–MS07 | `../test_mock_server.py`；既有 run 冒烟 | 编解码、SSE 实收内容、usage、重复捕获、gate、故障、退出、预热分类 |
| R01–R04、C01 | `test_cold_start.py` | 独立进程 nonce、稳定身份/配置/计划/fingerprint、24 个请求、对应回答均不同 |
| R05、P08 | `test_validation.py` | 正常副本通过，改参数、删/重复请求、等长历史篡改、提前发送、缺证据分别失败 |
| P01–P06、A01、A04 | `helpers/validation.py` | 实际 token 长度、当前轮历史、共享前缀、TraceLab 预热与不足量、AgentX 快照 |
| T01、T04–T07、A02 | `test_clock.py` | 同一可控时钟精确验证释放点、准备时间、lag、时间节点和 task duration |
| T02–T03 | 既有 `test_timing.py` | gap 变换、截断、跨 agent 例外、lognormal 确定性与分布 |
| D01–D07、P07、F02–F03、A03 | `test_http_dependencies.py` | gate 下不早发、标题独立、session/request 并发、失败跳过与纯时间后继继续 |
| D08 | `test_plan_validation.py` | 缺失前驱、自环和双节点环在计划校验时失败 |
| D02、F01、取消清理 | `test_clock.py`；既有 `test_executor.py` | 时间/上下文前驱分离、准备失败、取消时释放资源 |
| 校准与来源边界 | 既有 `test_prompt.py`、`test_current_turn_calibration.py`、`test_agentx.py` 等 | trim/reset、零输出、hash 作用域等已有局部覆盖 |

Unicode 码点 codec 示例：`ab → [97,98] → ab`，聊天模板额外开销为零。
因此测试能手算长度，但不验证真实模型的 tokenizer 或模板。
HTTP 契约覆盖 completions token IDs 和纯文本 chat；工具、多模态、`/v1/messages` 沿用既有局部测试。

两轮回答使用不同字符集，相同请求的输出长度保持相同。
校验器逐轮检查历史来自实际前驱回答，再比较稳定字段；完整 prompt 哈希只作诊断。
AgentX 不使用实时回答构造快照，因此其完整 prompt 跨轮必须相等。
AgentInfer 的根 prompt、共享配方片段也要求跨轮相等。

精确时间以客户端同一 monotonic 时钟计算。HTTP 测试使用事件和保护超时，
不为 CI 设置严格的毫秒发送上限，也不用服务端最后写块时间冒充客户端完成时间。
`task duration` 从获得 session 名额到任务清理结束；入场排队可由 submitted/admitted 事件计算。
这些内部测试事件不改变公开配置或 summary schema。

## 失败排查

每轮保留：

- `config.yaml`、`expected.json`、`input-manifest.json`、`replay-process.json`、`replay-exit.json`、stdout/stderr。
- `server/scenario.json`、`ready.json`、`exit.json`、`resolved-rules.json`（使用 arm 时）。
- `server/backend-requests.jsonl`、`backend-responses.jsonl`、`server-events.jsonl`。
- `client-events.jsonl`、`fingerprints.json`、`comparison.json`。
- `result/` 下真实 CLI 的 plan、execution、requests、manifest、summary 和 metrics。

请求差异以 task/source key/runtime ID 定位。丢失、重复请求按 Counter 检查；
等长历史错误按内容检查；发送过早用原始事件时间检查，不能用被截到零的 scheduler lag 掩盖。
缺少 observer/recorder 终态或子进程被强杀会失败，不视作“两轮相等”。

本套测试证明 CPU mock 条件下的负载与依赖一致性。
真实 vLLM 冷启、GPU/NPU、模型分词效果和性能延迟预算需要独立实验，不由本套 CI 证明。
