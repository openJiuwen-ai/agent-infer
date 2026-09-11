# Progress-TTL 配置参考

服务端参数位于 `additional_config.agentcache.progress_ttl`，由 `build_progress_ttl_controller` 校验。
它与 AgentBench YAML 的负载配置分开；调度策略、容量和成本参数在启动 vLLM 时确定。

## 字段和默认值

| 字段 | 默认值 | 配置状态 |
| --- | --- | --- |
| `target_max_segment_rounds` | `14` | 可配置 |
| `mode` | `on` | 可配置 |
| `ttl_min_seconds` | `0.05` | 可配置 |
| `ttl_max_seconds` | `32.0` | 可配置 |
| `ttl_max_cache_miss_impact_ratio` | `1.0` | 可配置 |
| `auto_enable_utility_seconds` | `20.0` | 可配置 |
| `auto_disable_utility_seconds` | `5.0` | 可配置 |
| `ttl_prefill_model_intercept_seconds` | `0.042935` | 可配置 |
| `ttl_prefill_model_linear_seconds_per_1k_tokens` | `0.080027` | 可配置 |
| `ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared` | `0.00220962` | 可配置 |
| `ttl_decode_throughput_alpha` | `0.15` | 可配置 |
| `shared_prefix_freshness_warmup_seconds` | `100.0` | 可配置 |
| `shared_prefix_freshness_kv_turnovers` | `2.0` | 可配置 |
| `resume_capacity_ratio` | `1.0` | 可配置 |
| `resume_order` | `mru` | 可配置 |
| `resume_reclaim_acting_programs` | `True` | 可配置 |
| `pause_capacity_ratio` | `1.0` | 可配置 |
| `privileged_max_context_tokens` | `262144` | 可配置 |
| `privileged_ttl_seconds` | `5.0` | 可配置 |
| `use_fixed_input_token_growth` | `False` | 可配置 |
| `fixed_input_token_growth_per_round` | `1024` | 可配置 |
| `enable_batch_gain_admission` | `True` | 可配置 |
| `decode_step_fixed_seconds` | `0.012112` | 可配置 |
| `decode_step_seconds_per_request` | `0.0006939` | 可配置 |
| `decode_step_seconds_per_context_token` | `2.2516e-07` | 可配置 |
| `decode_buffer_tokens` | `100` | 固定实现值，不能覆盖 |
| `force_resume_timeout_scale` | `3.0` | 可配置 |
| `force_resume_timeout_min_seconds` | `30.0` | 可配置 |
| `force_resume_timeout_max_seconds` | `300.0` | 可配置 |
| `paused_program_ttl_seconds` | `1800.0` | 可配置 |

`mode` 支持 `on`、`off`、`auto`；`resume_order` 支持 `mru`、`fcfs`。
无效类型、未知字段、已移除字段以及覆盖固定字段均抛出 `ValueError`。
完整范围和字段组合约束见[配置实现](../../../agentinfer/scheduling/progress_ttl/config.py)。

## 从旧配置迁移

- 将 `ttl_prefill_seconds_per_1k_uncached_tokens` 替换为 `ttl_prefill_model_*` 的截距、线性项和二次项。
- 将 `force_resume_timeout_seconds` 替换为 `force_resume_timeout_scale`、`force_resume_timeout_min_seconds`
  和 `force_resume_timeout_max_seconds`。
- 移除 `target_min_segment_rounds`、`privileged_lookahead_rounds`、`pause_capacity_lookahead_rounds`、
  `resume_fairness_weight`、`resume_resource_penalty_weight`、`capacity_safety_margin_tokens`。
- `resume_order` 指定普通暂停 Program 的恢复顺序；`enable_batch_gain_admission` 控制批处理收益准入。

默认成本系数来自参考部署，不代表其他模型或设备上的测量值。
使用[校准工具](../../../tools/calibration/README.md)测量目标部署，再显式配置系数。
策略原理见[Progress-TTL 设计](../../../design/module/scheduling/progress-ttl-scheduling.md)。
