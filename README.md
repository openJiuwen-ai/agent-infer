# AgentCache

Efficient cache management for agent workflows, designed to plug into
LLM serving engines such as vLLM.

## Installation

AgentCache requires Python 3.10 or later and vLLM 0.22.1.

Clone the repository and install in development mode:

```bash
git clone https://github.com/JiusiServe/AgentCache.git
cd AgentCache
pip install -e .
```

Once installed, the `vllm-acache` CLI command is available.  It wraps the
standard vLLM CLI with the agent-aware scheduler already patched in:

```bash
vllm-acache serve meta-llama/Llama-3.1-8B-Instruct
```

When `import agentcache` is executed, vLLM `EngineArgs` are automatically
patched to use the `AgentAwareScheduler` by default, so any code or script
that imports `agentcache` before instantiating a vLLM engine gets the
agent-aware scheduling behaviour without additional configuration.

## Custom Scheduler and Request Queue

### Architecture

AgentCache layers on top of vLLM's scheduler and request-queue primitives:

| Component            | Purpose                                          |
|----------------------|--------------------------------------------------|
| `AgentAwareQueue`    | Extends `RequestQueue` — FCFS by default         |
| `AgentAwareScheduler`| Extends vLLM `Scheduler`, uses `AgentAwareQueue` |
| `agentcache.LLM`     | Thin wrapper around `vllm.LLM`                   |

`AgentAwareQueue` currently delegates every operation to a standard
`FCFSRequestQueue`.  The identical FCFS behaviour exists so that subclasses
can override methods (e.g. `pop_request`, `add_request`) to implement
agent-aware scheduling policies later without touching the scheduler
itself.

### Using the Wrapper

The simplest way to opt in is to use `agentcache.LLM` instead of
`vllm.LLM`:

```python
import agentcache

llm = agentcache.LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

Because `import agentcache` patches `EngineArgs`, instantiating
`agentcache.LLM` (or `vllm.LLM` after the import) automatically selects
`AgentAwareScheduler`, which creates an `AgentAwareQueue` as its waiting
queue.

### Customising the Request Queue

Subclass `AgentAwareQueue` and override the methods you need:

```python
from agentcache.core.request_queue import AgentAwareQueue

class PriorityAgentQueue(AgentAwareQueue):
    def pop_request(self):
        # Insert your agent-priority logic here
        return self._q.pop_request()
```

### Customising the Scheduler

Subclass `AgentAwareScheduler` to wire in a custom queue:

```python
from agentcache.core.scheduler import AgentAwareScheduler

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
llm = agentcache.LLM(engine_args=args)
```

### Extension Points Summary

| Extension           | How                                              |
|---------------------|--------------------------------------------------|
| Custom queue policy | Subclass `AgentAwareQueue`                       |
| Custom scheduler    | Subclass `AgentAwareScheduler`                   |
| Opt out of patching | Set `scheduler_cls` on `EngineArgs` explicitly   |
