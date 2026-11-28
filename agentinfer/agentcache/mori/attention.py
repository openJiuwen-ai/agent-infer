# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Use native FlashAttention for MORI prefill and FlashInfer for decode."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend, FlashInferImpl, FlashInferMetadataBuilder
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger("vllm.agentinfer.mori.attention")


class MORIAttentionMetadataBuilder(FlashInferMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_builder = FlashAttentionMetadataBuilder(*args, **kwargs)

    def set_kernel_block_size(self, kernel_block_size):
        super().set_kernel_block_size(kernel_block_size)
        self.prefill_builder.set_kernel_block_size(kernel_block_size)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        prefill = common_attn_metadata.is_prefilling
        # Keep uniform one-token batches compatible with the captured decode
        # graph. The V2 dense runner supplies is_prefilling in host memory.
        if (
            common_attn_metadata.max_query_len > 1
            and prefill is not None
            and prefill.device.type == "cpu"
            and bool(prefill.all())
        ):
            logger.info_once("MORI uses native FlashAttention for prefill and FlashInfer for decode")
            return self.prefill_builder.build(common_prefix_len, common_attn_metadata, fast_build)
        return super().build(common_prefix_len, common_attn_metadata, fast_build)


class MORIAttentionImpl(FlashInferImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_impl = FlashAttentionImpl(*args, **kwargs)

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None
    ):
        implementation = self.prefill_impl if isinstance(attn_metadata, FlashAttentionMetadata) else super()
        return implementation.forward(
            layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )


class MORIAttentionBackend(FlashInferBackend):
    @staticmethod
    def get_builder_cls():
        return MORIAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return MORIAttentionImpl


class MORIFlashAttentionPrefillWorker(Worker):
    """Install the stage-specific backend within an opted-in vLLM worker."""

    def __init__(self, vllm_config, *args, **kwargs):
        attention = vllm_config.attention_config
        parallel = vllm_config.parallel_config
        if (
            attention.backend is not AttentionBackendEnum.FLASHINFER
            or attention.use_trtllm_attention is not False
            or vllm_config.model_config.dtype not in (torch.bfloat16, torch.float16)
            or vllm_config.cache_config.cache_dtype != "auto"
            or vllm_config.speculative_config is not None
            or not vllm_config.use_v2_model_runner
            or parallel.world_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("MORI split attention requires V2, one rank, unquantized KV and native FlashInfer")
        register_backend(AttentionBackendEnum.FLASHINFER, f"{__name__}.MORIAttentionBackend")
        super().__init__(vllm_config, *args, **kwargs)
