# InferRSI 源码快照

[`inferrsi/`](../../../inferrsi/README.md) 包含推理优化工作台
[InferMatrixLab/InferRSI](https://github.com/InferMatrixLab/InferRSI) 的完整受 Git 跟踪源码。

## 来源与目录

- 来源版本：[`76384cfb2c2af08eaa90753a18a45557ceed7ce0`][source-revision]（`main`，于 2026-10-21 同步）。
- 全部 536 个跟踪文件在 `inferrsi/` 下保留原始内容和相对路径，包括隐藏文件、源码、测试、文档、
  配方、基准测试、证据、打包技能及许可证声明。
- `inferrsi` 发行包使用 [`inferrsi/pyproject.toml`](../../../inferrsi/pyproject.toml)，
  与仓根目录的 `agentinfer` 发行包分别安装。
- `inferrsi/.github/` 保留上游工作流和 Issue 模板。GitHub 从仓根目录的 `.github/workflows/` 发现活动工作流，
  嵌套目录中的工作流作为源码参考保留。
- 主仓库的 pre-commit 钩子跳过此快照以保留上游格式；InferRSI 使用下述自身检查命令。

## 安装与本地检查

从 AgentInfer 仓根目录开始，使用 Python 3.10 或更高版本：

```bash
cd inferrsi
python -m pip install -e ".[dev]"
python -m inferrsi --help
python -m pytest tests
python -m ruff check src tests benchmarks/harness
```

CPU 测试验证工作台行为；加速器实验和性能结论需要各自的目标环境与证据。
导入的历史结果保留上游原有的验证范围。

## 文档入口

从 [InferRSI README](../../../inferrsi/README.md)、
[快速开始](../../../inferrsi/docs/quickstart.md)、
[操作指南](../../../inferrsi/docs/operations.md)及
[贡献指南](../../../inferrsi/docs/contributing.md)开始。

[source-revision]: https://github.com/InferMatrixLab/InferRSI/tree/76384cfb2c2af08eaa90753a18a45557ceed7ce0
