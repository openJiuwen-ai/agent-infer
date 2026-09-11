# 运行完整基准验证

先运行无需模型服务的烟雾测试，验证生产 CLI、配置、数据集、工作区、运行时分派、请求代理、trace 和产物汇总：

```bash
python -m pytest tests/agentbench/test_benchmark_smoke.py -q
```

该测试使用本地测试运行时和 loopback 后端，覆盖单任务及并发任务隔离，不验证模型质量或代理进程管理。

## 运行真实场景

`tests/e2e/` 提供 JSON 驱动的 pytest 入口、服务生命周期管理和运行结果校验。
先复制并编辑适合部署的场景文件，核对 `serve_env`、`server_params`、`benchmark_params` 和 `result_root`。
内置 GLM 场景包含特定 NPU 部署路径和并行参数，应按当前主机调整。

```bash
python -m pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file /path/to/case.json \
  --task-num 1 --max-concurrency 1
```

基线和候选配置是独立文件，需分别运行。只有存在所选模型、vLLM、Agent CLI、数据集和隔离依赖时才能执行。
`plan-subagent` 以 root 启动时，可通过 `--benchmark-run-as-user USER` 指定实际 Agent 用户。
选择空闲端口和新的结果目录；环境 E2E 管理服务进程，不适用于正在承载其他任务的服务端口。

完成后使用 `vllm bench serve --agentinfer compare` 比较两个已完成目录。
服务日志和运行产物位于 JSON 指定的结果根目录；执行完成不等于 SWE-bench 补丁正确。
详细字段和入口见[E2E 测试说明](../../../tests/e2e/README.md)，比较边界见[基准方法](../explanation/benchmark-methodology.md)。
