# AgentInfer Tests

Mirrors [vllm-project/vllm](https://github.com/vllm-project/vllm)
test structure under `tests/agentcache/core/`.

## Test files

### `tests/agentcache/core/test_agent_scheduler.py`

Unit tests for `AgentAwareScheduler` resolution and the `EngineArgs` patch.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_subclass_of_vllm_scheduler` | `AgentAwareScheduler` extends vllm's `Scheduler` |
| `test_agent_scheduler_is_subclass_of_scheduler_interface` | `AgentAwareScheduler` satisfies `SchedulerInterface` |
| `test_get_scheduler_cls_resolves_agent_scheduler` | `SchedulerConfig.get_scheduler_cls()` resolves the class path |
| `test_get_scheduler_cls_default_is_vllm_scheduler` | Default (unpatched) resolves to vllm's `Scheduler` |
| `test_patch_sets_scheduler_cls_when_none` | `EngineArgs.__post_init__` defaults to `AgentAwareScheduler` when `scheduler_cls` is `None` |
| `test_patch_preserves_explicit_scheduler_cls` | Explicit `scheduler_cls` is not overwritten by the patch |

### `tests/agentcache/core/test_agent_scheduler_e2e.py`

In-process end-to-end test using `agentinfer.LLM`
(wraps `vllm.LLM`) with a tiny model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_configured` | `scheduler_cls` is set to `AgentAwareScheduler` in the running engine config |
| `test_generation_works` | Text generation produces output |

### `tests/agentcache/core/test_agent_scheduler_serve_e2e.py`

Subprocess end-to-end test via `vllm serve` with a real model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_configured` | Server logs contain the `AgentAwareScheduler` log |
| `test_completion` | OpenAI-compatible `/v1/completions` returns text |
| `test_chat_completion` | OpenAI-compatible `/v1/chat/completions` returns a message |

## How to run

### Prerequisites

- Python 3.10+, CUDA GPU with sufficient memory
- `vllm==0.22.1` installed

```bash
pip install -e .
```

### Run all tests

```bash
pytest tests/agentcache/ tests/agentbench/ -v
```

### Run unit tests only (no GPU needed)

```bash
pytest tests/agentcache/core/test_agent_scheduler.py -v
```

### Run in-process E2E test (requires GPU, tiny model)

```bash
pytest tests/agentcache/core/test_agent_scheduler_e2e.py -v
```

### Run serve E2E test (requires GPU, real model ~1.3 GB)

```bash
pytest tests/agentcache/core/test_agent_scheduler_serve_e2e.py -v
```
