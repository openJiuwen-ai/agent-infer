# 验证 SWE-bench Verified 任务精度

使用官方 SWE-bench **4.1.0** 评测已有补丁，仅选择 outcome=completed 的任务。
整批运行可以包含失败任务，评测不重新调用模型。5.x 要求不同的数据集字段。

## 准备与执行

在独立环境安装固定版本，确保 Docker 可访问，并保存配置 evaluation.yaml：

```bash
uv venv --python 3.12 /path/to/swebench-venv
uv pip install --python /path/to/swebench-venv/bin/python 'swebench==4.1.0'
```

```yaml
harness_python: /path/to/swebench-venv/bin/python
expected_harness_version: "4.1.0"
max_workers: 2
timeout_seconds: 1800
overall_timeout_seconds: 7200
cleanup_timeout_seconds: 30
```

```bash
vllm bench serve --agentinfer evaluate \
  --result-path results/run-a \
  --config evaluation.yaml \
  --output-dir results/run-a/evaluations/eval-001
```

输入需要已结束的 manifest/summary、任务索引、逐任务 result/prompt 和 manifest 引用的
本地 Verified 数据集。核对任务 ID、仓库和 base commit；无成功任务时不生成成绩。
解释器相对路径以配置文件目录为基准，未知字段和不支持的版本会被拒绝。

省略 --output-dir 时，在原 run 的 evaluations/ 中创建唯一目录；已有目录不覆盖。
原始补丁、manifest 和性能 summary 不修改。首次使用会下载官方 Docker 镜像。

## 结果

每个入选任务在 task_results.jsonl 中有一条记录。correctness.json 的 counts 仅包含
total、resolved、unresolved、empty_patch、error 五项。唯一汇总精度指标为
resolved_rate = resolved / total，error 不从分母删除。
该分数表示成功完成任务子集的解决率，不代表整个 Verified 数据集。

没有补丁文件时，只有 result 与 DSH 导出状态共同证明 no_changes 才计为空补丁；
否则计为 error。失败任务即使有补丁也不入选。F2P/P2P 明细保留在 raw/ 下官方逐题报告中。
实际命令、日志、版本、镜像身份和文件哈希全部保留。

退出码：0 表示评测无错误，unresolved 是正常结果；1 表示输入、执行或评测错误；
2 表示配置错误；130 表示中断。确定任务集合后的错误仍保存汇总。

## 对比

```bash
vllm bench serve --agentinfer compare-correctness \
  --baseline results/baseline/evaluations/eval-001/correctness.json \
  --candidate results/candidate/evaluations/eval-001/correctness.json \
  --max-regressions 0
```

核对文件哈希、任务集合、题目内容、已记录的生成配置/提示词、评测环境和镜像身份。
范围不同或证据不完整时返回 insufficient_evidence，不静默取交集。
来源产物未记录的模型 checkpoint revision 无法被独立验证。

输出逐题退化和改善列表，即使总分持平也能发现退化。退出码为 0（通过）、1（超过退化预算）
或 2（证据不足）。该门禁只适用于固定任务集，不是模型总体能力的统计结论。

参考：[官方指南](https://www.swebench.com/SWE-bench/guides/evaluation/)、
[benchmark 方法](../explanation/benchmark-methodology.md)。
