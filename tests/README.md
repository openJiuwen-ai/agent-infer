# AgentCache Tests

Mirrors [vllm-project/vllm](https://github.com/vllm-project/vllm)
test structure under `tests/v1/core/`.

## Test files

### `tests/v1/core/test_agent_scheduler.py`

Unit tests for `AgentScheduler` resolution and the `EngineArgs` patch.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_subclass_of_vllm_scheduler` | `AgentScheduler` extends vllm's `Scheduler` |
| `test_agent_scheduler_is_subclass_of_scheduler_interface` | `AgentScheduler` satisfies `SchedulerInterface` |
| `test_get_scheduler_cls_resolves_agent_scheduler` | `SchedulerConfig.get_scheduler_cls()` resolves the class path |
| `test_get_scheduler_cls_default_is_vllm_scheduler` | Default (unpatched) resolves to vllm's `Scheduler` |
| `test_patch_sets_scheduler_cls_when_none` | `EngineArgs.__post_init__` defaults to `AgentScheduler` when `scheduler_cls` is `None` |
| `test_patch_preserves_explicit_scheduler_cls` | Explicit `scheduler_cls` is not overwritten by the patch |

### `tests/v1/core/test_agent_scheduler_e2e.py`

In-process end-to-end test using `agentcache.LLM`
(wraps `vllm.LLM`) with a tiny model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_configured` | `scheduler_cls` is set to `AgentScheduler` in the running engine config |
| `test_generation_works` | Text generation produces output |

### `tests/v1/core/test_agent_scheduler_serve_e2e.py`

Subprocess end-to-end test via `vllm-acache serve` with a real model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_configured` | Server logs contain the `AgentScheduler` warning |
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
pytest tests/v1/ -v
```

### Run unit tests only (no GPU needed)

```bash
pytest tests/v1/core/test_agent_scheduler.py -v
```

### Run in-process E2E test (requires GPU, tiny model)

```bash
pytest tests/v1/core/test_agent_scheduler_e2e.py -v
```

### Run serve E2E test (requires GPU, real model ~1.3 GB)

```bash
pytest tests/v1/core/test_agent_scheduler_serve_e2e.py -v
```
