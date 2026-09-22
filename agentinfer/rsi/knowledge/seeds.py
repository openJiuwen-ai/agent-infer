# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Unverified demo checklists, never empirical optimization claims."""

BASE_CLAIMS = {
    "system": "Track cross-layer contracts, ownership, and versioned evidence before proposing an optimization.",
    "semantic-router": "Check model/capability selection independently from task scheduling and worker placement.",
    "router": "Check request affinity, cancellation, and worker placement against the pinned Router revision.",
    "scheduling": "Check identity/lifecycle and admission decisions without changing vLLM physical KV ownership.",
    "harness": "Replay is not task correctness; execute real tools and independently judge the completed task.",
    "agent-cache": "Map this logical namespace to agentcache; verify prefix identity before cache reuse.",
}
ENGINE_LAYERS = {
    "api_server": "Verify streaming, reasoning fields, tool calls, cancellation, and tokenizer/template compatibility.",
    "engine": "Verify request scheduling and cache lifecycle against the selected engine revision.",
    "worker": "Check worker ownership, device memory, process cleanup, and failure recovery.",
    "model_scripts": "Compare the model implementation and quantization path to a pinned reference.",
    "parallel": "Test partitioned/unpartitioned outputs and topology-specific communication.",
    "ops.communication": "Measure dispatch, transfer, waiting, and overlap with controlled experiments.",
    "ops.compute": "Check numerical tolerances and shapes before claiming a kernel speedup.",
}
NAMESPACES = (
    set(BASE_CLAIMS)
    | {"vllm", "vllm-ascend"}
    | {f"{engine}.{layer}" for engine in ("vllm", "vllm-ascend") for layer in ENGINE_LAYERS}
)


def demo_records():
    claims = [(name, "agnostic", claim) for name, claim in BASE_CLAIMS.items()]
    for engine, backend in (("vllm", "cuda"), ("vllm-ascend", "ascend")):
        claims.append(
            (engine, backend, "Qualify the pinned model, runtime, hardware and engine combination independently.")
        )
        claims.extend((f"{engine}.{layer}", backend, claim) for layer, claim in ENGINE_LAYERS.items())
    return [
        {
            "knowledge_id": f"demo:{name}",
            "namespace": name,
            "scope": "demo",
            "backend": backend,
            "component_version": "demo-unverified",
            "workload": "demo",
            "title": f"DEMO / unverified: {name}",
            "claim": claim,
            "evidence": [],
            "demo": True,
        }
        for name, backend, claim in claims
    ]
