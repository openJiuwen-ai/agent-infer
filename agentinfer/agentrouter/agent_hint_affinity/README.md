# `agent_hint_affinity` WASM plugin

Standalone OnRequest guest for [vLLM Router](https://github.com/vllm-project/router)
affinity adaptation. Uses the upstream HTTP-envelope WIT
(`vllm:router-middleware@0.1.0` from Router `#251`).

This crate ships **only the guest**. The Wasmtime host lives in upstream Router
(`--wasm-middleware`); AgentInfer does not vendor the Router tree.

## Contract

WIT copy: `wit/router-middleware.wit` (must stay aligned with upstream
`wit/router-middleware.wit`).

- Input: method / path / query / headers / body / request-id
- Output: `Continue` | `Modify` (body and/or headers) | `Reject(status)`

This plugin only rewrites JSON **body** fields; header mutations are unused.

## Behavior

- `Continue` when rewrite is unnecessary or unsafe (existing non-empty
  `session_params.session_id`, missing/empty `agent_hint.session_id`, or
  non-JSON body)
- `Modify` when a non-empty `agent_hint.session_id` is copied into
  `session_params.session_id`

## Build

Requires Rust with the `wasm32-wasip2` target:

```bash
./build.sh
```

Produces `agent_hint_affinity.component.wasm` in this directory and prints its
SHA-256.

Host-side unit checks for the rewrite helper (no Wasmtime):

```bash
cargo test
```

## Enable in Router

Use a Router build that includes the WASM OnRequest host (upstream `main` with
[#251](https://github.com/vllm-project/router/pull/251) or later):

```bash
./build.sh
DIGEST=$(sha256sum agent_hint_affinity.component.wasm | awk '{print $1}')

vllm-router \
  --worker-urls http://localhost:8080 \
  --policy cache_aware \
  --wasm-middleware ./agent_hint_affinity.component.wasm \
  --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions
```

Do **not** also pass `--middleware agent_hint_affinity` (legacy native path).
Native `--middleware agent_hint_token_offsets` remains a separate optional
patch; see the parent [AgentRouter README](../README.md).
