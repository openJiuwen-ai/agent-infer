PIN_RECOMMENDATION:
- Preferred: `vllm==0.21.0`.
- Why: GitHub marks `v0.21.0` as latest stable on May 15, 2026; it is the current V1 scheduler API target. The existing `_PLUGIN_TEMPLATE` is not compatible, but pinning older versions will not preserve the assumed `SchedulerPlugin` API either.
- Fallbacks: `vllm==0.20.2`, then `vllm==0.18.2`. Treat both as fallback only if Docker/CUDA constraints block `0.21.0`.

API_SURFACE_TODAY:
- Class user must implement / subclass: `vllm.v1.core.sched.interface.SchedulerInterface`; practically subclass `vllm.v1.core.sched.scheduler.Scheduler`.
- Constructor shape: `Scheduler(vllm_config, kv_cache_config, structured_output_manager, block_size, hash_block_size=None, mm_registry=..., include_finished_set=False, log_stats=False)`.
- Method signature: `schedule(self) -> vllm.v1.core.sched.output.SchedulerOutput`.
- Return shape: `SchedulerOutput`, not `dict`. Fields include `scheduled_new_reqs`, `scheduled_cached_reqs`, `num_scheduled_tokens: dict[str, int]`, `total_num_scheduled_tokens`, `scheduled_spec_decode_tokens`, `scheduled_encoder_inputs`, `num_common_prefix_blocks`, `finished_req_ids`, `free_encoder_mm_hashes`, optional `preempted_req_ids`, connector metadata, etc.
- Input object names: V1 scheduler operates over internal `vllm.v1.request.Request`, not V0 `SequenceGroup`.
- CLI flag: `--scheduler-cls mod.custom_class`. No `--scheduler-plugin` flag found in current stable docs.
- Entry-point registration requirement: none for `--scheduler-cls`. `vllm.general_plugins` entry points exist for general process-wide plugins/patching, but they are not the custom scheduler loading mechanism.
- V0/V1 divergence: yes. vLLM docs say V0 is fully deprecated; V1 re-architects scheduler/KV/worker/sampler/API server and uses unified `{request_id: num_tokens}` scheduling internally.

DELTA_AGAINST_PLUGIN_TEMPLATE:
- Assumes `SchedulerPlugin` public API: truth is `SchedulerInterface` / `Scheduler` via `scheduler_cls`; severity `BREAKS_AT_IMPORT`.
- Assumes `EvolvedSchedulerPlugin.schedule(self, waiting, running, swapped, budget, **kwargs) -> dict`: truth is `schedule(self) -> SchedulerOutput` and state is on `self.waiting`, `self.running`, etc.; severity `BREAKS_AT_RUNTIME`.
- Assumes return dict `{"scheduled": [...], "preempted": [...], "swapped_in": [...]}`: truth is `SchedulerOutput` dataclass; severity `BREAKS_AT_RUNTIME`.
- Assumes `SequenceGroup` field access: truth is V1 `Request` with `request_id`, `prompt_token_ids`, `arrival_time`, `num_computed_tokens`, `num_tokens`, `num_prompt_tokens`, `priority`, etc.; no `seq_group.get_seqs()[0].get_prompt_token_ids()` path; severity `BREAKS_AT_RUNTIME`.
- Assumes `swapped` / `swapped_in`: V1 docs say GPU <> CPU KV cache swapping is removed; severity `BREAKS_AT_RUNTIME`.
- Assumes `--scheduler-plugin`: truth is `--scheduler-cls`; severity `BREAKS_AT_RUNTIME`.
- Assumes `plugin = EvolvedSchedulerPlugin()` module-level instance: truth is vLLM resolves/imports a class qualname and instantiates it with scheduler constructor args; severity `BREAKS_AT_IMPORT`.
- Assumes lightweight adapter can reorder request IDs externally: truth is subclass must preserve scheduler allocation, KV cache, encoder cache, prefix cache, preemption, connector metadata, and output construction; severity `SUBTLE_BEHAVIOR_DIFF` if only post-sorting output.

REQUIRED_TEMPLATE_CHANGES:
- Replace plugin class with scheduler subclass:
```diff
- class EvolvedSchedulerPlugin:
-     def schedule(self, waiting, running, swapped, budget, **kwargs) -> dict:
+ from vllm.v1.core.sched.scheduler import Scheduler
+ from vllm.v1.core.sched.output import SchedulerOutput
+
+ class EvolvedScheduler(Scheduler):
+     def schedule(self) -> SchedulerOutput:
```
- Replace module-level instance:
```diff
- plugin = EvolvedSchedulerPlugin()
+ # No module-level instance. Load with:
+ # vllm serve ... --scheduler-cls generated_scheduler:EvolvedScheduler
```
- Replace `SequenceGroup` adapter:
```diff
- def _seq_group_to_request_info(seq_group):
-     seq = seq_group.get_seqs()[0]
-     prompt_len = len(seq.get_prompt_token_ids())
+ def _request_to_request_info(request):
+     prompt_len = request.num_prompt_tokens
+     computed = request.num_computed_tokens
+     total = request.num_tokens
```
- Replace dict output with V1 output handling. Recommended minimal safe pattern: subclass `Scheduler`, reorder `self.waiting` before `super().schedule()` using evolved policy scores, then return `super().schedule()` unchanged. Do not construct `SchedulerOutput` from scratch until tests cover KV/encoder/spec/connector fields.

DOCKER_BASE_IMAGE_GUIDANCE:
- Use `vllm/vllm-openai:v0.21.0-cu129-ubuntu2404` or `vllm/vllm-openai:v0.21.0-x86_64-cu129-ubuntu2404`.
- If building yourself: use CUDA 12.9-compatible base plus `pip install vllm==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu129`, preferably in a fresh env.
- CUDA: vLLM 0.21.0 binaries are compiled with CUDA 12.9 by default; CUDA 12.8 and 13.0 variants are also published.
- GPU: NVIDIA compute capability >= 7.5. Blackwell requires CUDA >= 12.8.
- Driver: use a driver that supports CUDA 12.9; official container includes CUDA compatibility libraries for some datacenter/pro GPUs, but do not depend on that for consumer GPUs.

UNKNOWNS:
- Local PowerShell reads failed with `windows sandbox: spawn setup refresh`; local-file assertions above rely on the file/line facts in your prompt, not a fresh local read.
- I did not find an official stable `SchedulerPlugin` / `--scheduler-plugin` API. If the repo template references an unreleased PR-only API, it is not present in vLLM 0.21.0 stable.
- I verified current API from vLLM release/docs/GitHub pages, but not by importing `vllm==0.21.0` locally.

CI_GATE_CONTENT:
- Assert `vllm.__version__ == "0.21.0"`.
- Assert `SchedulerConfig().scheduler_cls is None` and `SchedulerConfig().get_scheduler_cls().__qualname__ == "Scheduler"`.
- Assert `vllm.v1.core.sched.scheduler.Scheduler.schedule` has signature `(self)`.
- Assert `Scheduler.schedule` return annotation resolves to `SchedulerOutput`.
- Assert `SchedulerOutput` has required fields: `scheduled_new_reqs`, `scheduled_cached_reqs`, `num_scheduled_tokens`, `total_num_scheduled_tokens`, `scheduled_encoder_inputs`, `finished_req_ids`, `free_encoder_mm_hashes`.
- Negative assert: importing `vllm.*SchedulerPlugin` or using `--scheduler-plugin` is unsupported/fails closed.
- Assert generated module exports a class `EvolvedScheduler` subclassing `Scheduler`, not `plugin = ...`.
- Assert adapter rejects fake `SequenceGroup` objects and accepts a minimal V1-like `Request` fixture.
