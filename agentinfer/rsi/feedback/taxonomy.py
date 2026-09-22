# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Logical feedback layers and collection plans, not connected hardware probes."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class Backend(str, Enum):
    CUDA = "cuda"
    ASCEND = "ascend"


class Layer(str, Enum):
    API_SERVER = "api_server"
    ENGINE = "engine"
    WORKER = "worker"
    MODEL_SCRIPTS = "model_scripts"
    PARALLEL = "parallel"
    COMMUNICATION = "ops.communication"
    COMPUTE = "ops.compute"


@dataclass(frozen=True)
class LayerSpec:
    """Describe an optimization boundary and candidate validation methods."""

    description: str
    signals: tuple[str, ...]
    validators: tuple[str, ...]
    cuda_collection: str
    ascend_collection: str

    def to_dict(self, layer: Layer, backend: Backend) -> dict[str, object]:
        return {
            "layer": layer.value,
            "backend": backend.value,
            "description": self.description,
            "signals": list(self.signals),
            "validators": list(self.validators),
            "collection_status": "planned_not_connected",
            "collection_hint": self.cuda_collection if backend is Backend.CUDA else self.ascend_collection,
        }


TAXONOMY = MappingProxyType(
    {
        Layer.API_SERVER: LayerSpec(
            "HTTP, streaming, tokenization, chat templates, and tool-call protocol boundaries.",
            ("protocol_errors", "stream_interruptions", "host_request_time", "time_to_first_token"),
            ("protocol_contract", "stream_nonstream_equivalence", "tool_roundtrip", "cancellation"),
            "Plan HTTP traces and vLLM request metrics; use Python profiling for host-side attribution.",
            "Plan HTTP traces and vLLM request metrics; MS Service Profiler can inspect host-side execution.",
        ),
        Layer.ENGINE: LayerSpec(
            "Admission, scheduler state, batching, cache policy, and request progress.",
            ("queue_time", "running_waiting_requests", "cache_usage", "prefix_hits", "starvation"),
            ("scheduler_invariants", "deterministic_request_replay", "late_callback_regression"),
            "Plan scheduler traces and vLLM metrics with version-specific field mappings.",
            "Plan MS Service Profiler Request, BatchSchedule, and KVCache domains; verify symbol versions.",
        ),
        Layer.WORKER: LayerSpec(
            "ModelRunner input preparation, graph execution, device memory, and host/device synchronization.",
            ("host_enqueue_delay", "device_idle_intervals", "graph_fallback", "peak_memory", "step_time"),
            ("eager_graph_equivalence", "dynamic_batch_shapes", "resource_lifetime", "worker_failure"),
            "Plan PyTorch Profiler or Nsight Systems traces; measure final performance without profiling.",
            "Plan torch_npu.profiler and MS Service Profiler traces; compare eager and ACL graph execution.",
        ),
        Layer.MODEL_SCRIPTS: LayerSpec(
            "Model implementation, weight loading, quantization, and attention or recurrent state semantics.",
            ("first_divergent_layer", "logprob_difference", "nonfinite_values", "state_error_by_step"),
            ("reference_model_comparison", "prefill_decode_equivalence", "long_context", "multimodal"),
            "Plan reference outputs and scoped module tensor captures at fixed shapes, dtypes, and steps.",
            "Plan msprobe statistics first, then scoped tensor dumps by module, rank, and step.",
        ),
        Layer.PARALLEL: LayerSpec(
            "TP, PP, DP, EP, and CP partition semantics and topology; not a separate sequential stack level.",
            ("rank_skew", "pipeline_bubbles", "expert_imbalance", "shard_error", "scaling_efficiency"),
            ("partitioned_reference_equivalence", "rank_group_mapping", "fixed_topology_comparison"),
            "Plan per-rank traces and distributed tests; pin supported model and parallel configurations.",
            "Plan per-rank traces, HCCL topology evidence, and supported Ascend parallel configurations.",
        ),
        Layer.COMMUNICATION: LayerSpec(
            "Collectives, dispatch/combine, KV transfer, and stream or event synchronization.",
            ("transfer_latency", "bandwidth", "enqueue_wait", "compute_transfer_overlap", "payload_errors"),
            ("collective_reference", "message_size_sweep", "concurrent_cancellation", "overlap_ablation"),
            "Plan NCCL or connector traces and compute-only, transfer-only, and combined controlled runs.",
            "Plan HCCL or connector traces and MS Service Profiler Communication events; verify topology.",
        ),
        Layer.COMPUTE: LayerSpec(
            "Compute kernels, fusion, tiling, precision, and device resource allocation.",
            ("absolute_relative_error", "error_coordinates", "kernel_latency", "bandwidth", "resource_usage"),
            ("high_precision_reference", "shape_dtype_stride_matrix", "microbenchmark", "integration_regression"),
            "Plan reference kernel tests and CUDA/Triton profiling; validate gains again in the full workload.",
            "Plan reference operator tests, torch_npu.profiler, and msprobe; require real NPU validation later.",
        ),
    }
)


def list_layers(backend: Backend | str) -> list[dict[str, object]]:
    """Return JSON-ready planning hints for one explicitly selected backend."""
    selected = Backend(backend)
    return [spec.to_dict(layer, selected) for layer, spec in TAXONOMY.items()]
