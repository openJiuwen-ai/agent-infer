# vLLM CLI integration

AgentInfer installs a delegating `vllm` console script. Only an explicit command containing
`bench serve --agentinfer` is intercepted and routed to BenchKit. Other arguments are passed unchanged to the
pinned upstream vLLM CLI, and the process argument vector is restored after delegation.

```bash
vllm bench serve --agentinfer run --config agentinfer/agentbench/configs/swebench_vllm.yaml
```

The package must be installed in the same environment as vLLM for ordinary vLLM delegation. Benchmark modules
remain importable when vLLM is absent so lightweight unit tests and dataset preparation can run without activating
the serving integration.
