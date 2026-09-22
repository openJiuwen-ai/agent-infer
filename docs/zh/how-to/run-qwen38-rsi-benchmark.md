# Qwen3.8-27B B300 RSI benchmark 知识库

这是 PR #25 后续 4×B300 serving 目标的追加式实验记录。每一轮保留假设、
单变量修改、原始 evidence、效果和下一步；只有原始产物与独立质量门禁齐全时，
测量才具备验收资格。

## 目标与非目标

目标是在 4 张 B300 上，以 `Qwen/Qwen3.8-27B` 获得可复现的最高 E2E 输出 token
吞吐。性能负载固定为 AgentBench 的
`inferact_codex_swebenchpro` trace replay，质量门禁固定为通过 vLLM serving
的 GSM8K。

Replay 会重建请求间隔、上下文和 token 目标，但不会运行原始 Codex agent、工具或
SWE-bench 判题。因此 replay 是 serving workload，不是 agent 正确率。

系统层、runtime 层、kernel 层、证据协议和晋级门禁统一整理在
[Qwen3.8-27B B300 知识库](../../rsi/qwen38-b300-knowledge-base.md) 中，并吸收了
[vLLM kernel benchmark workflow](https://github.com/vllm-project/vllm/blob/main/.agents/skills/kernel-microbenchmark/SKILL.md)、
[vLLM Triton guidance](https://github.com/vllm-project/vllm/blob/main/.agents/skills/triton-kernel-writing/SKILL.md)、
[Z.ai dense feedback 经验](https://z.ai/blog/glm-built-its-inference-infrastructure)
以及 [NVlabs KDA workflow](https://github.com/NVlabs/kda)。

本轮冻结的证据标识如下：

* raw trace SHA256：`670f1ae8325fd70aac6ae6bdf4b03bbbd740d6ac8f2b49d8a02daaee3193fbc3`
* Unified IR bundle SHA256：`48dce67812a7138cc97e9caecd2a166430e32d047e2e6eb7d7dda94972716d0f`
* request IR SHA256：`d4d1910747438bd32673757f6d67af4544cd2cc652dc6e0fbddeb4ea35c1614f`
* GSM8K test SHA256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`，共 1,319 题

## 2026-09-22 I0：跑通真实模型链路

修改：把 PR #25 的 RSI evidence plane 接到真实 vLLM TP4 endpoint，复用带 SHA 绑定的
converted trace，打开 prefix caching 与 262,144 context，并完成一个 14 请求 smoke。

效果：14/14 请求通过 exact input accounting 并成功返回。输出吞吐 79.045 tok/s，输入吞吐
35,193.178 tok/s，请求吞吐 1.303 req/s，TTFT p50 为 81.576 ms，prefix-cache hit rate
为 0.893926。该轮 output cap 只有 64 token，因此只证明链路可执行，不作为性能冠军。

## I1：拒绝第一次完整采样

修改：采样扩大到 8 个 session、并发 8，并要求 prompt calibration residual 严格为零。

效果：该轮状态为 `failed`，不进入性能排名。8 个 task 中 3 个完成，5 个在 one-token
current-turn calibration 边界发送前失败；后端对已发送请求返回 97/97 成功，另有 154 个依赖
请求被跳过。已尝试请求的 residual 为零，但覆盖不完整，因此吞吐字段全部保留 null。

## I2：确定性边界修复与有效 TP4 样本

修改：在 suffix repair 中加入 literal punctuation boundary candidates，以 seed 228、2 个
session、并发 2 重跑；TP4、prefix caching、模型、context 与 exact accounting 保持不变。

效果：36/36 请求成功，失败数和 skip 数均为零，最大 calibration residual 为零。输出吞吐
203.280 tok/s，输入吞吐 29,167.320 tok/s，请求吞吐 0.619 req/s，TTFT p50 为 174.720 ms，
prefix-cache hit rate 为 0.942330。这是当前完成的有效对比中最高的一轮，但样本只有两个
session，仍需重复运行后才能作为最终容量结论。

## I3：两个 TP2 replica 对比

修改：把 4 张卡拆成两个 TP2 vLLM instance，通过 sticky local proxy 路由，保持相同 seed、
2-session 样本和 exact calibration。

效果：36/36 请求成功，失败数和 skip 数均为零；聚合输出吞吐为 154.174 tok/s，比 I2 低
24.16%。输入吞吐为 22,121.459 tok/s，后端 prefix-cache hit rate 为 0.928004。由于该
proxy 没有保留 streaming timing 语义，request TTFT 保持 null，不做延迟对比。当前有效对比
中仍由 TP4 作为 serving candidate。

## I4：独立 GSM8K 精度门禁

修改：把官方 GSM8K test 的 1,319 题全部发送到 TP4 endpoint，并发 16，使用
`temperature=0`、`top_p=1`、`top_k=0`、seed 42、max output 1,024；只解析最终 `####` 答案。

效果：答对 1,254/1,319，accuracy 为 0.950720（95.07%）；解析出 1,276 个 prediction，
transport error 为零，平均延迟 1.847 s，p95 为 6.558 s。该结果是独立质量门禁，与 trace
replay 性能吞吐分开记录。

本轮已完整执行 RSI taxonomy、feedback evaluate、demo state machine、真实 append-only
ledger 与 read-only dashboard。真实 dashboard 只读取 `experiments.jsonl`，HTTP POST 会被
拒绝；新增证据通过 `python -m agentinfer.rsi experiments append` 写入，每轮保留 hypothesis、
change、metrics、failure reason、next test 与 evidence hash。

## 当前结论与 50 轮效果

修正后的 sweep 为 I11–I60：10 个 TP4 serving profile、每个 5 次有效测量，每个 profile
先做一次不计入排名的 warmup；seed-228、2-task、并发 2 的 replay contract 全部冻结。
I5–I9 保留为第一次命令 harness 失败，I10 在 50 轮开始前验证了修复后的 AgentInfer
dispatcher。

当前通过质量门禁的最好 profile 是 BF16 TP4 + prefix caching +
`--max-num-batched-tokens=32768`：5 次 replay-valid 的中位输出吞吐为 210.385 tok/s，
相对重复 baseline 中位数 208.347 提升 0.98%；完整 GSM8K 为 1255/1319 = 95.1478%。
FP8 KV 的 replay 中位数略高，为 210.658 tok/s，但 GSM8K 只有 1251/1319 = 94.8446%，
低于 BF16 参考值 95.0720%，因此拒绝晋级。`batch-8192` 只有 1 次完整 replay，其余
4 次均是 27/36 覆盖，不能作为收益。

后续组合检查把 `batch-32768` 和 `async-on` 合并，5 次均 replay-valid，但中位数只有
208.134 tok/s，因此保留单变量 profile。每一轮具体的优化点与效果见
[qwen38-b300-50-round-results.md](../../rsi/qwen38-b300-50-round-results.md)；review 资产见
[dashboard](../../assets/rsi/qwen38-rsi-dashboard.png) 和
[更新后的架构图](../../assets/rsi/qwen38-b300-architecture.png)。分层知识库中已经记录
vLLM、Z.ai 与 NVlabs KDA 的流程规则和本轮晋级结论。
