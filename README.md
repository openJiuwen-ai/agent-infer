# AgentInfer

Efficient cache management for agent workflows, designed to plug into
LLM serving engines such as vLLM.

## Installation

AgentInfer requires Python 3.10 or later and vLLM 0.22.1.

Clone the repository and install in development mode:

```bash
git clone https://github.com/JiusiServe/AgentInfer.git
cd AgentInfer
pip install -e .
```

Once installed, the `vllm` CLI delegates standard commands to upstream vLLM while loading AgentCache:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

When `import agentinfer` is executed, vLLM `EngineArgs` are automatically
patched to use the `AgentAwareScheduler` by default, so any code or script
that imports `agentinfer` before instantiating a vLLM engine gets the
agent-aware scheduling behaviour without additional configuration.

## Custom Scheduler and Request Queue

### Architecture

AgentCache layers on top of vLLM's scheduler and request-queue primitives:

| Component            | Purpose                                          |
|----------------------|--------------------------------------------------|
| `AgentAwareQueue`    | Extends `RequestQueue` — FCFS by default         |
| `AgentAwareScheduler`| Extends vLLM `Scheduler`, uses `AgentAwareQueue` |
| `agentinfer.LLM`     | Thin wrapper around `vllm.LLM`                   |

`AgentAwareQueue` currently delegates every operation to a standard
`FCFSRequestQueue`.  The identical FCFS behaviour exists so that subclasses
can override methods (e.g. `pop_request`, `add_request`) to implement
agent-aware scheduling policies later without touching the scheduler
itself.

### Using the Wrapper

The simplest way to opt in is to use `agentinfer.LLM` instead of
`vllm.LLM`:

```python
import agentinfer

llm = agentinfer.LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

Because `import agentinfer` patches `EngineArgs`, instantiating
`agentinfer.LLM` (or `vllm.LLM` after the import) automatically selects
`AgentAwareScheduler`, which creates an `AgentAwareQueue` as its waiting
queue.

### Customising the Request Queue

Subclass `AgentAwareQueue` and override the methods you need:

```python
from agentinfer.agentcache.core.request_queue import AgentAwareQueue

class PriorityAgentQueue(AgentAwareQueue):
    def pop_request(self):
        # Insert your agent-priority logic here
        return self._q.pop_request()
```

### Customising the Scheduler

Subclass `AgentAwareScheduler` to wire in a custom queue:

```python
from agentinfer.agentcache.core.scheduler import AgentAwareScheduler

class PriorityScheduler(AgentAwareScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = PriorityAgentQueue()  # your custom queue
```

Then pass it to vLLM's `EngineArgs`:

```python
from vllm.engine.arg_utils import EngineArgs

args = EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct",
                  scheduler_cls="mymodule.PriorityScheduler")
llm = agentinfer.LLM(engine_args=args)
```

### Extension Points Summary

| Extension           | How                                              |
|---------------------|--------------------------------------------------|
| Custom queue policy | Subclass `AgentAwareQueue`                       |
| Custom scheduler    | Subclass `AgentAwareScheduler`                   |
| Opt out of patching | Set `scheduler_cls` on `EngineArgs` explicitly   |
