# 运行 RSI 离线骨架与 Dashboard

在仓库根目录使用 Python 3.10+。以下命令仅用标准库，不加载模型、不启动 vLLM、不申请 GPU/NPU。
按照本次交付范围，不执行硬件或部署验证。

```bash
python -m agentinfer.rsi taxonomy --backend cuda
python -m agentinfer.rsi taxonomy --backend ascend
python -m agentinfer.rsi evaluate examples/rsi/feedback.json --output .rsi-demo/report.json
python -m agentinfer.rsi demo --run-dir .rsi-demo
python -m agentinfer.rsi status --run-dir .rsi-demo
python -m agentinfer.rsi serve --run-dir .rsi-demo --port 8877
```

打开 <http://127.0.0.1:8877/dashboard.html>。也可直接打开
`agentinfer/rsi/dashboard/static/dashboard.html` 查看离线交互；直接打开文件时本地 API 快照不可用。
示例评估预期为 `INCONCLUSIVE`，缺失探针和 synthetic 数据不能构成通过证据。
`demo` 初始化到第 7 步；重复执行复用已有状态，避免覆盖人工操作。

## Dashboard 使用顺序

1. **运行总览**：看当前阶段、candidate/accepted/active/last_good、预算和任务效果。
2. **自进化循环**：按 1–10 序号读闭环，查看失败、补证据、恢复和知识回流路径。
3. **组件改动**：检查示例文件差异、所属层、关联证据和知识；差异是说明性示例。
4. **稠密反馈**：按后端和七层筛选，从假设找到局部检查、证据缺口和下一项区分实验。
5. **知识与经验**：查看 system、路由、Harness 和两种推理后端的逐层知识；检查适用范围与状态。
6. **人工干预**：演练暂停、预算、补充假设、排除候选、模拟激活和恢复；审查操作记录。

HTML 按钮操作浏览器 localStorage 的模拟状态。Python CLI/HTTP 操作 SQLite demo 状态；两者尚未自动同步。
页面的 API 快照功能是只读检查入口。代码骨架的激活、观察和回滚动作均显式使用 `demo_` 前缀。
服务只绑定 IPv4 loopback；这是可信本机演示接口，没有生产认证、远程访问或真实部署能力。

## 从 CLI 干预 demo

先读取 `status` 中的 revision，然后提交命令。下例的 6 只适用于刚创建、尚未修改的 demo。
重复同一个请求和幂等键会返回首次响应；相同键提交不同内容、过期 revision 会被拒绝。

```bash
python -m agentinfer.rsi command --run-dir .rsi-demo --action pause --expected-revision 6 --idempotency-key pause-demo-1
python -m agentinfer.rsi status --run-dir .rsi-demo
```

查询精确范围的知识草稿：

```bash
python -m agentinfer.rsi knowledge --scope demo --backend cuda --component-version demo-unverified --workload demo
```

将 backend 换为 `ascend` 查询另一后端，换为 `agnostic` 查询共享契约。
所有种子都是待验证知识，不能因为被检索到就当作实验结论。

## 验证与后续接入

```bash
python -m unittest discover -s tests/rsi -p "test_*.py" -v
```

上述只验证 CPU 状态与数据合同。真实集成仍需：逐后端协议冒烟、固定输入数值对照、取消/失败恢复、
隔离配对性能测试、完整 Agent 任务、资源争抢和发布恢复验证；这些不属于本次已验证结果。

阅读[架构设计](../explanation/rsi-architecture.md)、[接口参考](../reference/rsi-api.md)与
[实施计划](../../../design/rsi/implementation-plan.md)与[后续组件目录](../../../design/rsi/component-plan.md)。
