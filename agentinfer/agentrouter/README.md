# AgentRouter native middleware patches

This directory does **not** vendor the Router source tree. It only carries the
already-merged **native middleware** delta from
[JiusiServe/AgentInfer](https://github.com/JiusiServe/AgentInfer) relative to
the upstream Router snapshot that tree was based on.

## Upstream pin

| Item | Value |
| --- | --- |
| Upstream project | [vllm-project/router](https://github.com/vllm-project/router) |
| Base revision | `d60711dc72ab8f073e33f9a3d93ee81b97274c26` |
| AgentInfer vendor commit | `5ab4ac7` (`Vendor vllm-project/router at d60711dc into agentrouter/`) |
| Feature commits (merged) | `#76`, `#95` / middleware registry, `#103` (`agent_hint_token_offsets`) |

## What the patch adds

Apply `patches/0001-agent-hint-native-middleware.patch` to a Router checkout at the
base revision above. It introduces opt-in Router-native middleware:

- `--middleware agent_hint_affinity` — copy `agent_hint.session_id` into
  `session_params.session_id` before routing
- `--middleware agent_hint_token_offsets` — rewrite message token offsets into
  KV block offsets (requires tokenizer / block-size flags)

Plus the supporting CLI/PyO3 wiring, protocol `AgentHint` fields, and
`tests/test_agent_hint_routing.rs`.

## Apply

```bash
# From a clean Router tree at the pinned revision:
git clone https://github.com/vllm-project/router.git
cd router
git checkout d60711dc72ab8f073e33f9a3d93ee81b97274c26

git apply /path/to/agent-infer/agentinfer/agentrouter/patches/0001-agent-hint-native-middleware.patch

cargo build --release
./target/release/vllm-router --help   # look for --middleware …
```

Dry-run:

```bash
git apply --check /path/to/agent-infer/agentinfer/agentrouter/patches/0001-agent-hint-native-middleware.patch
```

## Out of scope

- Vendoring or shipping the full `agentrouter/` / Router binary tree in this repo
- Unmerged WASM `--wasm-middleware` guest/host work
