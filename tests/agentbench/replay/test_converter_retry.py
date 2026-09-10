# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Exercise conversion recovery through the real synchronous tokenizer boundary."""

import json

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.converters import codex_swebenchpro
from agentinfer.agentbench.replay.runner import _prepare_replay_source


@pytest.fixture
def backend(monkeypatch):
    """Install a deterministic counting backend with controlled transport failures."""
    state = {"calls": 0, "failures": 0, "trigger": None, "failure": None, "clients": [], "delays": []}
    client_class = httpx.Client

    def respond(request):
        state["calls"] += 1
        payload = json.loads(request.content)
        prompt = payload.get("prompt", "")
        trigger = state["trigger"]
        if state["failures"] and (trigger is None or trigger in prompt):
            state["failures"] -= 1
            failure = state["failure"]
            if isinstance(failure, httpx.Response):
                return failure
            raise httpx.ReadError("injected connection reset", request=request)
        if "messages" in payload:
            prompt = (
                "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in payload["messages"])
                + "<|im_start|>assistant\n"
            )
        return httpx.Response(200, json={"count": len(prompt)})

    def client(**kwargs):
        result = client_class(transport=httpx.MockTransport(respond), **kwargs)
        state["clients"].append(result)
        return result

    monkeypatch.setattr(codex_swebenchpro.httpx, "Client", client)
    monkeypatch.setattr(codex_swebenchpro.time, "sleep", state["delays"].append)
    return state


@pytest.fixture
def config(tmp_path):
    """Create two turns so injected failures can occur after partial conversion."""
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "first"},
                    {"from": "gpt", "value": "answer"},
                    {"from": "human", "value": "second"},
                    {"from": "gpt", "value": "done"},
                ]
            }
        )
    )
    return ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_type": "inferact_codex_swebenchpro",
                "trace_path": source,
                "prompt_shape": "trace_record",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 2, "p95_seconds": 30, "p99_seconds": 90},
            }
        }
    )


@pytest.mark.parametrize("trigger", [None, "second"])
def test_conversion_recovers_without_changing_artifacts(backend, config, tmp_path, trigger):
    """A transient failure at initialization or mid-conversion preserves the entire IR."""
    _prepare_replay_source(config, tmp_path / "baseline")
    backend.update(failures=1, trigger=trigger)
    _prepare_replay_source(config, tmp_path / "recovered")
    baseline = tmp_path / "baseline" / "convert_result"
    recovered = tmp_path / "recovered" / "convert_result"
    files = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    assert files == {p.relative_to(recovered): p.read_bytes() for p in recovered.rglob("*") if p.is_file()}
    assert backend["failures"] == 0
    assert backend["delays"] == [0.1]
    assert all(c.is_closed for c in backend["clients"])


@pytest.mark.parametrize("trigger", [None, "second"])
def test_conversion_exhaustion_closes_client(backend, config, tmp_path, trigger):
    """Persistent errors escape unchanged after three attempts and close the client."""
    backend.update(failures=4, trigger=trigger)
    with pytest.raises(httpx.ReadError, match="injected connection reset"):
        _prepare_replay_source(config, tmp_path / "failed")
    assert backend["failures"] == 1
    assert backend["delays"] == [0.1, 0.2]
    assert all(c.is_closed for c in backend["clients"])


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(400), httpx.HTTPStatusError),
        (httpx.Response(500), httpx.HTTPStatusError),
        (httpx.Response(200, text="invalid json"), json.JSONDecodeError),
        (httpx.Response(200, json={}), ValueError),
    ],
)
def test_conversion_does_not_retry_invalid_responses(backend, config, tmp_path, response, error):
    """HTTP errors and invalid response bodies fail immediately and release resources."""
    backend.update(failures=1, failure=response)
    with pytest.raises(error):
        _prepare_replay_source(config, tmp_path / "invalid")
    assert backend["calls"] == 1
    assert backend["delays"] == []
    assert all(c.is_closed for c in backend["clients"])
