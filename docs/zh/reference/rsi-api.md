# RSI Python 与 HTTP 接口（初版）

导入 `agentinfer.rsi` 无硬件副作用。公共入口为 `python -m agentinfer.rsi`，安装包后也可用 `agentinfer-rsi`。

## 已实现目录

```text
agentinfer/rsi/
  __init__.py, __main__.py, cli.py      # 轻量入口、JSON 输出、离线运行
  controller/
    service.py                        # 10 步状态机、预算、revision、幂等命令
  feedback/
    models.py                         # FeedbackRecord、枚举、精确范围校验
    taxonomy.py                       # CUDA/Ascend 七层信号与检查地图
    probes.py                         # Probe 协议、UnavailableProbe
    engine.py                         # 冻结检查清单的证据汇总
  knowledge/
    repository.py                     # 范围检索、知识状态、历史
    seeds.py                          # 22 个命名空间的待验证种子
  storage/
    sqlite.py                         # 状态/事件/幂等/知识的事务存储
  dashboard/
    server.py                         # loopback demo HTTP API
    static/dashboard.html             # 六视图离线交互原型
examples/rsi/feedback.json             # synthetic 反馈样例
tests/rsi/                            # 标准库 CPU 行为测试
tools/rsi/build_presentation.mjs       # PPT 与工程图构建源
docs/assets/rsi/                       # PPT、PNG、SVG、Mermaid
```

## FeedbackRecord

`check_id`、`hypothesis_id`、`candidate_id`、`baseline_id` 标识验证与改动；`backend` 是 cuda/ascend，
`layer` 为七层枚举；`category` 为 correctness/system_behavior/performance；
`verdict` 为 pass/fail/inconclusive/unavailable。
`metric/value/unit` 表示测量，unavailable 的 value 必须为 null；`scope` 固定 engine_version、model_revision、workload_id，
可增加 shape、dtype、topology 等维度；`diagnostic` 标识有观测扰动；`evidence_refs` 存证据指针；
`next_test` 给出下一步；`synthetic` 标识演示。

```python
from agentinfer.rsi.feedback import evaluate_feedback, list_layers, load_records

layers = list_layers("ascend")
records = load_records("examples/rsi/feedback.json")
# scope 和 required_checks 应由外部可信调用者从冻结合同加载。
```

`evaluate_feedback(records, required_checks, scope)` 返回 status/checks/reason/next_test。
scope 还需包含 candidate_id/baseline_id/backend。缺失、重复、范围不符、无证据引用、synthetic 或 profile 性能样本返回 INCONCLUSIVE；
有效 FAIL 优先于 INCONCLUSIVE，全部要求有效 PASS 才汇总为 PASS。
重复试验需上游 validator 先按冻结统计规则聚合。当前函数不执行探针、不验证文件真实性、不重算阈值，也不能触发生产发布。

## Controller

`Controller(db_path)` 提供 create/get/list_runs/events/command。
`create(run_id, baseline=..., max_trials=3, demo=False, backend='cuda')` 默认不允许模拟验收与发布。
`command(run_id, expected_revision=..., idempotency_key=..., action=..., payload={})` 将状态、审计和幂等响应放在同一事务。

| action | 条件 / payload |
| --- | --- |
| advance | 1→2→3→4 或 5→6→7 |
| start_trial | 第 4 步，消耗 trial 配额，payload 可含 candidate_id |
| pause / resume | 暂停新工作；在途 demo 裁决、观察和恢复仍可完成 |
| update_budget | max_trials，不能抹掉已消耗配额 |
| retry | 第 7 步 FAIL/INCONCLUSIVE 且预算足够，回 3 |
| exclude | 第 5/6/7 步排除候选，回 3 |
| demo_evaluate | 第 7 步，result: PASS/FAIL/INCONCLUSIVE |
| demo_activate | 第 8 步，outcome: success/failed/unknown |
| demo_soak | 第 9 步，result: PASS/FAIL |
| demo_rollback | 第 8/9 步，healthy: boolean |
| next_round | 第 10 步且无需恢复，还要求有预算 |
| complete | 第 10 步，或第 1–4 步未启动实验时提前归档 |

demo_* 只允许 demo=True 的运行。恢复失败停止进一步工作；再确认健康恢复后才能继续。
生产运行目前不能越过独立验收与部署边界。

## KnowledgeRepository

`KnowledgeRepository(db_path)` 提供 propose/get/search/transition/seed_demo。
`search(scope=..., backend=..., component_version=..., workload=..., text='', status=None)` 精确匹配所有范围字段，默认排除
stale/retracted。
记录必须含 namespace/scope/backend/component_version/workload/title/claim；evidence 是 ref/outcome 对象列表。
transition 必须给 expected_revision、status、reason；reviewed 需来源审阅证据；validated 暂不可设置。

## HTTP

| 路径 | 用途 |
| --- | --- |
| GET /dashboard.html | HTML 原型 |
| GET /api/rsi/snapshot | 所选 demo run 与事件 |
| GET /api/rsi/taxonomy?backend=ascend | 七层地图 |
| GET /api/rsi/knowledge | 必填 scope/backend/component_version/workload 查询参数 |
| POST /api/rsi/commands | JSON: action/expected_revision/idempotency_key/payload |

Host 必须是本机地址和实际端口；若存在 Origin，必须与本机源匹配。
POST 限 JSON、64 KiB、demo run，不执行 shell。返回 409 表示无效/冲突命令，403 表示来源或运行边界拒绝。
当前无 SSE、远程认证、角色权限、真正的 artifact store 或 deployment adapter。
