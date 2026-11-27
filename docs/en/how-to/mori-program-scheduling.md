# Serve MORI with vLLM

Use the vLLM environment that will run the API server to install this AgentInfer checkout
(`python -m pip install -e .`). The engine adapter has been exercised with vLLM 0.31.0.
Startup checks the installed scheduler, KV cache and transfer interfaces. It requires one
data-parallel rank, a single dense full-attention KV group, FCFS queues, matching hash and
allocation page sizes, and non-speculative generation. The MORI engine accepts text inputs only.

With chunked prefill, the effective step budget must be a positive multiple of the KV page size.
vLLM bounds each step by both the scheduler issue limit and the worker input capacity. Without
chunking, that effective budget must cover the maximum model length. Short prompts and the
last chunk compute their actual token counts.

## Start the service

```bash
vllm serve "$MODEL" --agentinfer --agentinfer-strategy mori \
  --kv-transfer-config '{
    "kv_connector": "MoriCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 1073741824, "kv_offload_backend": "cpu", "lazy_offload": false}
  }'
```

Size the host KV budget for the model and available RAM. The example requests 1 GiB.
The profile enables prefix caching, installs the two-way CPU connector and starts Program scheduling
in EngineCore. Async scheduling is the default. Pass `--no-async-scheduling` for synchronous execution.
Both Bridge classes handle Program admission and inherit MORI request scheduling from the engine.
The existing Progress-TTL profile is still selected by `vllm serve "$MODEL" --agentinfer`.
[Progress-TTL deployment](integrate-vllm.md)

## Send and release a Program

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg model "$MODEL" \
    '{model:$model,program_id:"session-42",messages:[{role:"user",content:"Hello"}]}')"
curl http://localhost:8000/programs/release \
  -H 'Content-Type: application/json' -d '{"program_id":"session-42"}'
```

Include the same Program ID across turns. When the service has an API key, include the native
`Authorization: Bearer …` header on both endpoints. `released: true` acknowledges release for this generation.
A later Chat using the same ID starts a new generation. `GET /programs` shows live scheduling states,
`GET /agentinfer/mori/metrics` reports effective capacities, and `GET /agentinfer/mori/cache` shows
per-generation page ownership. The Program timer defaults to five seconds and is configured through
`--additional-config '{"agentcache":{"mori":{"interval_seconds":5}}}'`.

For SGLang text-template parity use `agentcache.mori.sglang_text_chat`. Set
`agentcache.mori.sglang_unseeded_chat` to also discard per-request seeds and select joint sampling.
Override the FlashInfer filter independently with `agentcache.mori.joint_flashinfer_sampling`.
Layerwise H2D uses the connector's `mori_layerwise_loads` setting. To also select separate
prefill and decode attention backends, explicitly pass
`--worker-cls agentinfer.agentcache.mori.attention.MORIFlashAttentionPrefillWorker`.

## Program admission and release

MORI checks the server's bearer credentials for Chat and `POST /programs/release` before
sending an event to EngineCore. Both endpoints read `program_id` from the request body, then
from `extra_body`. Null and blank values fall through to the next source. Chat uses
`default` when neither source supplies an ID. Release requires an explicit ID, including
`{"program_id":"default"}` to release the default Program.

The API derives the Program ID, character count and prefix group from the original Chat JSON.
After templating and tokenization, the Bridge handles the first native ADD for that Chat as one
Program arrival. All sampling children share the admission decision and Program generation.
Waiting children enter the engine scheduler after the cache acknowledges their placement.
Concurrent Chats can arrive at the Bridge in a different order from their HTTP arrival.

Overlapping requests keep the Program in REASONING until its last request finishes.
A release that the cache cannot acknowledge returns `released: false` and can be retried.
Successful release aborts children awaiting admission. Children of an already admitted Chat
keep their original generation when a later Chat reuses the Program ID.
