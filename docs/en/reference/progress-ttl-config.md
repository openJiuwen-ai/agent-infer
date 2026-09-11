# Progress-TTL configuration

Server settings live under `additional_config.agentcache.progress_ttl` and are validated by
`build_progress_ttl_controller`. They are separate from AgentBench workload YAML; scheduling policy, capacity,
and cost parameters are selected when vLLM starts.

## Fields and defaults

| Field | Default | Configuration |
| --- | --- | --- |
| `target_max_segment_rounds` | `14` | Configurable |
| `mode` | `on` | Configurable |
| `ttl_min_seconds` | `0.05` | Configurable |
| `ttl_max_seconds` | `32.0` | Configurable |
| `ttl_max_cache_miss_impact_ratio` | `1.0` | Configurable |
| `auto_enable_utility_seconds` | `20.0` | Configurable |
| `auto_disable_utility_seconds` | `5.0` | Configurable |
| `ttl_prefill_model_intercept_seconds` | `0.042935` | Configurable |
| `ttl_prefill_model_linear_seconds_per_1k_tokens` | `0.080027` | Configurable |
| `ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared` | `0.00220962` | Configurable |
| `ttl_decode_throughput_alpha` | `0.15` | Configurable |
| `shared_prefix_freshness_warmup_seconds` | `100.0` | Configurable |
| `shared_prefix_freshness_kv_turnovers` | `2.0` | Configurable |
| `resume_capacity_ratio` | `1.0` | Configurable |
| `resume_order` | `mru` | Configurable |
| `resume_reclaim_acting_programs` | `True` | Configurable |
| `pause_capacity_ratio` | `1.0` | Configurable |
| `privileged_max_context_tokens` | `262144` | Configurable |
| `privileged_ttl_seconds` | `5.0` | Configurable |
| `use_fixed_input_token_growth` | `False` | Configurable |
| `fixed_input_token_growth_per_round` | `1024` | Configurable |
| `enable_batch_gain_admission` | `True` | Configurable |
| `decode_step_fixed_seconds` | `0.012112` | Configurable |
| `decode_step_seconds_per_request` | `0.0006939` | Configurable |
| `decode_step_seconds_per_context_token` | `2.2516e-07` | Configurable |
| `decode_buffer_tokens` | `100` | Fixed; cannot be overridden |
| `force_resume_timeout_scale` | `3.0` | Configurable |
| `force_resume_timeout_min_seconds` | `30.0` | Configurable |
| `force_resume_timeout_max_seconds` | `300.0` | Configurable |
| `paused_program_ttl_seconds` | `1800.0` | Configurable |

`mode` accepts `on`, `off`, and `auto`; `resume_order` accepts `mru` and `fcfs`.
Invalid types, unknown or removed fields, and overrides of fixed fields raise `ValueError`.
See the [configuration implementation](../../../agentinfer/scheduling/progress_ttl/config.py) for ranges and
cross-field constraints.

## Migrating older configuration

- Replace `ttl_prefill_seconds_per_1k_uncached_tokens` with the intercept, linear, and quadratic `ttl_prefill_model_*` fields.
- Replace `force_resume_timeout_seconds` with `force_resume_timeout_scale`, `force_resume_timeout_min_seconds`,
  and `force_resume_timeout_max_seconds`.
- Remove `target_min_segment_rounds`, `privileged_lookahead_rounds`, `pause_capacity_lookahead_rounds`,
  `resume_fairness_weight`, `resume_resource_penalty_weight`, and `capacity_safety_margin_tokens`.
- `resume_order` controls ordinary paused Program ordering; `enable_batch_gain_admission` controls batch-gain admission.

Default cost coefficients come from a reference deployment and are not measurements for every model or device.
Use the [calibration tools](../../../tools/calibration/README.md) against the target deployment and configure its coefficients.
See the [Progress-TTL design](../../../design/module/scheduling/progress-ttl-scheduling.md) for policy rationale.
