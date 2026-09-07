# AgentInfer 文档

`docs/zh/` 是中文权威文档，`docs/en/` 是结构对称的英文版本。文档按读者意图分为 Tutorial、How-to、
Reference 和 Explanation；开发者内部设计稿位于仓根目录的 `design/`。

## 中文文档

- **Tutorial**：[快速开始](zh/tutorial/01-quick-start.md)，适合首次安装和运行 AgentInfer。
- **How-to**：[接入 vLLM](zh/how-to/integrate-vllm.md)和[运行基准测试](zh/how-to/run-benchmark.md)，
  用于完成具体任务。
- **Reference**：[Python API](zh/reference/python-api.md)、[基准命令行](zh/reference/benchmark-cli.md)、
  [基准配置](zh/reference/benchmark-config.md)和[运行产物](zh/reference/run-artifacts.md)，用于查询接口和参数。
- **Explanation**：[架构概述](zh/explanation/architecture.md)、
  [基准方法](zh/explanation/benchmark-methodology.md)和[内部设计索引](zh/explanation/design-documents.md)，
  用于理解设计原因和系统边界。

## English Documentation

- **Tutorial**: [Quick start](en/tutorial/01-quick-start.md) for first-time installation and execution.
- **How-to**: [Integrate with vLLM](en/how-to/integrate-vllm.md) and
  [run a benchmark](en/how-to/run-benchmark.md) to complete specific tasks.
- **Reference**: [Python API](en/reference/python-api.md), [benchmark CLI](en/reference/benchmark-cli.md),
  [benchmark configuration](en/reference/benchmark-config.md), and
  [run artifacts](en/reference/run-artifacts.md) for interfaces and parameters.
- **Explanation**: [Architecture](en/explanation/architecture.md),
  [benchmark methodology](en/explanation/benchmark-methodology.md), and
  [internal design index](en/explanation/design-documents.md) for design rationale and system boundaries.

## 贡献文档

新增或修改功能时，请同步更新中文权威文档和对应英文文档。分类、命名和验证要求见
[贡献指南](../CONTRIBUTING.md)。
