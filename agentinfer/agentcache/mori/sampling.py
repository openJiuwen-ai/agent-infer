# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Select the original MORI FlashInfer filter in the native GPU sampler."""

from contextlib import contextmanager
from functools import wraps

from vllm.logger import init_logger

logger = init_logger("vllm.agentinfer.mori.sampling")


@contextmanager
def mori_joint_flashinfer_sampling(use_v2_model_runner: bool):
    # vLLM creates one transfer group per worker process. Scope the override
    # to its lifetime and to the sampler module that runner actually calls.
    if use_v2_model_runner:
        from vllm.v1.worker.gpu.sample import sampler
    else:
        from vllm.v1.sample.ops import topk_topp_sampler as sampler

    native = sampler.flashinfer_sample

    @wraps(native)
    def sample(logits, k, p, *args, **kwargs):
        if k is None or p is None:
            return native(logits, k, p, *args, **kwargs)

        from flashinfer.sampling import top_k_top_p_sampling_from_logits

        logger.info_once("MORI uses native FlashInfer with the original joint top-k/top-p filter")
        return top_k_top_p_sampling_from_logits(logits, k, p, filter_apply_order="joint").view(-1)

    sampler.flashinfer_sample = sample
    try:
        yield
    finally:
        sampler.flashinfer_sample = native
