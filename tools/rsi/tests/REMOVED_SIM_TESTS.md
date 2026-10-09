# Removed sim-based tests (M3)

The DES simulator (`src/vllm_evolve/sim/`) and `openevolve/` were deleted in M3
(the project moved to real-vLLM-only evaluation). These tests exercised the
deleted simulator, the old V0 `engine/adapter.py` plugin template, or the
sim-dispatch tool. Each is listed with its rationale and real-vLLM analogue.

| Removed test | File | Why removed | Real-vLLM analogue |
|---|---|---|---|
| `test_import_simulator` | test_smoke.py | imported `sim.pd.PDSimulator` | none — sim deleted |
| `test_simulator_smoke` | test_smoke.py | ran `PDSimulator` on the seed | GPU bench smoke (`requires_gpu`) via `bench.backend` |
| `test_kv_eviction_seed_basic` | test_smoke.py | built `sim.kv_cache.KVBlock` | none — kv_eviction target → experimental (M0) |
| `test_kv_eviction_lru_order` | test_smoke.py | built `sim.kv_cache.KVBlock` | none — kv_eviction target → experimental (M0) |
| `TestScenarios.test_builtin_scenarios_count` | test_architecture.py | imported `sim.workload.SCENARIO_NAMES` | `tests/bench/test_load.py` (load regimes) |
| `TestScenarios.test_get_scenario` | test_architecture.py | imported `sim.workload.make_scenario` | `tests/bench/test_load.py` + `test_datasets.py` |
| `TestScenarios.test_get_unknown_scenario_raises` | test_architecture.py | imported `sim.workload.make_scenario` | `tests/bench/test_load.py` |
| `TestSimulateTool.*` | test_tools.py | called `tools.simulate.simulate` (sim dispatch) | `ar bench` + `tests/bench/*` |
| `TestEvolutionIntegration.test_evolution_loop_mock_1_gen` | test_tools.py | ran the in-process evolve loop on the simulator | GPU-gated live evolve loop (no off-GPU analogue) |
| `TestAdapterPlugin.*` | test_tools.py | tested the old V0 `engine.adapter` template (`EvolvedSchedulerPlugin`, `--scheduler-plugin`) | `tests/bench/test_plugin_template.py` (V1 `EvolvedScheduler`) |
| `TestErrorHandling.test_simulate_nonexistent_file` | test_tools.py | called `tools.simulate.simulate` | `ar bench` error handling |
| `TestTraceLoader.*` (3) | test_tools.py | imported `sim.trace_loader.load_trace` | `tests/bench/test_datasets.py` (BurstGPT/Azure/ShareGPT/Mooncake) |
| `TestGPUTimingModelProfile.*` (2) | test_tools.py | imported `sim.pd.GPUTimingModel` | none — real vLLM provides true hardware timing |
| `TestSimEvaluatorScenarios.*` (2) | test_tools.py | imported `sim.scenario_registry` | `tests/bench/test_load.py` (3 load regimes) |
| `test_legacy_submodule_callable_rejected_from_init[vllm_evolve.tools.simulate-simulate]` | test_phase_guard.py | parametrized over the deleted `tools.simulate` | remaining 7 legacy-tool cases still cover the guard |

Deleted source:
- `src/vllm_evolve/sim/` (entire DES simulator, ~9000 LOC)
- `src/vllm_evolve/openevolve/` (export depended on sim)
- `src/vllm_evolve/tools/simulate.py` (sim-dispatch tool)
- `src/vllm_evolve/engine/adapter.py` (broken V0 `--scheduler-plugin` template;
  replaced by `targets/scheduling/plugin_template.py`, the correct V1 template)
