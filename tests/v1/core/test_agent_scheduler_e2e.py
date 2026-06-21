# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

import pytest

from agentcache import LLM

pytestmark = pytest.mark.cpu_test

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"
SCHEDULER_CLS_PATH = "agentcache.core.scheduler.AgentAwareScheduler"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_model_len=128,
        block_size=16,
        gpu_memory_utilization=0.3,
    )


def test_agent_scheduler_is_configured(llm):
    cfg = llm.llm_engine.vllm_config.scheduler_config
    assert cfg.scheduler_cls == SCHEDULER_CLS_PATH, f"Expected {SCHEDULER_CLS_PATH}, got {cfg.scheduler_cls}"


def test_generation_works(llm):
    outputs = llm.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1
        assert output.outputs[0].text


def test_agent_aware_queue_integration(llm):
    outputs = llm.generate([PROMPT + str(i) for i in range(5)])
    assert len(outputs) == 5
    for output in outputs:
        assert output.outputs[0].text
