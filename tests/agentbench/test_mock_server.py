# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Handwritten HTTP clients verify the backend independently of Replay."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from pydantic import ValidationError

from tests.agentbench.helpers.mock_server import MockServerProcess
from tests.agentbench.helpers.mock_server.process import read_rows
from tests.agentbench.helpers.mock_server.protocol import Scenario

pytestmark = pytest.mark.cpu_test


def body(request_id="r", *, chat=False, **overrides):
    result = {
        "model": "replay-ci-model",
        "stream": True,
        "min_tokens": 2,
        "max_tokens": 2,
        "return_token_ids": True,
        "vllm_xargs": {"agentic_context": json.dumps({"session_id": "s", "request_id": request_id})},
    }
    result.update({"messages": [{"role": "user", "content": "ab"}]} if chat else {"prompt": [97, 98]})
    result.update(overrides)
    return result


def send(server, payload, *, chat=False):
    return httpx.post(
        server.base_url + ("/v1/chat/completions" if chat else "/v1/completions"),
        json=payload,
        timeout=10,
        trust_env=False,
    )


def stream_parts(response):
    response.raise_for_status()
    return [
        line[6:] if line[6:] == "[DONE]" else json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def actual_answer(parts):
    text, ids = "", []
    for part in parts:
        if isinstance(part, dict):
            for choice in part.get("choices", []):
                text += choice.get("text", choice.get("delta", {}).get("content", ""))
                ids.extend(choice.get("token_ids", []))
    return text, ids


def test_codec_stream_usage_warmup_metrics_and_duplicate_capture(tmp_path):
    answers, nonces = [], []
    for round_index in range(2):
        with MockServerProcess(
            tmp_path / str(round_index), {"profile": "replay", "round_index": round_index}
        ) as server:
            nonces.append(server.ready["nonce"])
            assert server.get("/v1/models")["data"] == [{"id": "replay-ci-model", "max_model_len": 8192}]
            assert server.post("/tokenize", {"prompt": "ab中😀"}) == {"tokens": [97, 98, 20013, 128512], "count": 4}
            assert server.post("/detokenize", {"tokens": [97, 98, 20013, 128512]}) == {"prompt": "ab中😀"}
            assert (
                server.post(
                    "/tokenize", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
                )["count"]
                == 2
            )
            for invalid in ([True], [-1], [0xD800], [0x110000]):
                assert server.client.post(server.base_url + "/detokenize", json={"tokens": invalid}).status_code == 400
            warmup = body(stream=False, min_tokens=1, max_tokens=1)
            warmup.pop("vllm_xargs")
            assert send(server, warmup).json()["usage"]["completion_tokens"] == 1
            before = server.client.get(server.base_url + "/metrics").text
            seen = []
            for chat in (False, True):
                parts = stream_parts(send(server, body(chat=chat), chat=chat))
                assert parts[-1] == "[DONE]"
                assert parts[-2]["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
                text, ids = actual_answer(parts)
                assert len(text) == len(ids) == 2 and ids == list(map(ord, text))
                assert len(parts) == 4  # Two actual chunks, usage, DONE.
                seen.append(text)
            assert seen[0] == seen[1]  # Repeated ID is recorded, not silently deduplicated.
            answers.append(seen[0])
            after = server.client.get(server.base_url + "/metrics").text
            assert 'finished_reason="length",model_name="replay-ci-model"} 1' in before
            assert 'finished_reason="length",model_name="replay-ci-model"} 3' in after
        rows = read_rows(server.root / "backend-requests.jsonl")
        formal = [r for r in rows if r["kind"] == "generation"]
        assert len(formal) == 2 and len({r["capture_id"] for r in formal}) == 2
        assert sum(r["kind"] == "unidentified_generation" for r in rows) == 1
        assert all(not r["path"].startswith("/__mock__") for r in rows)
        assert server.process.poll() == 0
    assert answers[0] != answers[1] and nonces[0] != nonces[1]


def test_armed_gate_does_not_block_receive_or_independent_response(tmp_path):
    with MockServerProcess(tmp_path / "server", {"profile": "replay", "wait_for_arm": True}) as server:
        with ThreadPoolExecutor(max_workers=2) as pool:
            held = pool.submit(send, server, body("held"))
            server.wait_event("received", request_id="held")
            assert not held.done()
            assert (
                server.client.post(
                    server.base_url + "/__mock__/arm", json={"rules": [{"request_id": "held", "typo": 1}]}
                ).status_code
                == 400
            )
            assert server.get("/__mock__/health")["armed"] is False
            server.post(
                "/__mock__/arm", {"rules": [{"request_id": "held", "gate": {"id": "gate", "phase": "before_terminal"}}]}
            )
            assert server.client.post(server.base_url + "/__mock__/arm", json={"rules": []}).status_code == 409
            server.wait_event("gate_waiting", gate_id="gate")
            independent = pool.submit(send, server, body("independent"))
            assert stream_parts(independent.result(timeout=3))[-1] == "[DONE]"
            assert not held.done()
            server.post("/__mock__/gates/gate/release")
            assert stream_parts(held.result(timeout=3))[-1] == "[DONE]"


@pytest.mark.parametrize(
    "fault", ["http_error", "omit_done", "truncate_stream", "usage_mismatch", "token_ids_missing", "token_ids_short"]
)
def test_fault_rule_hits_only_named_request(tmp_path, fault):
    with MockServerProcess(
        tmp_path / "server", {"profile": "replay", "rules": [{"request_id": "bad", "fault": fault}]}
    ) as server:
        bad = send(server, body("bad"))
        server.wait_event("rule_matched", request_id="bad", fault=fault)
        good = stream_parts(send(server, body("good")))
        assert good[-1] == "[DONE]" and len(actual_answer(good)[1]) == 2
        if fault == "http_error":
            assert bad.status_code == 500
            return
        parts = stream_parts(bad)
        text, ids = actual_answer(parts)
        if fault in {"omit_done", "truncate_stream"}:
            assert "[DONE]" not in parts
            assert len(text) == (1 if fault == "truncate_stream" else 2)
        elif fault == "usage_mismatch":
            assert parts[-2]["usage"]["prompt_tokens"] == 3
        else:
            assert len(text) == 2 and len(ids) == (0 if fault == "token_ids_missing" else 1)


def test_delay_and_shutdown_with_pending_stream(tmp_path):
    with MockServerProcess(
        tmp_path / "server",
        {
            "profile": "replay",
            "rules": [{"request_id": "delay", "delay_ms": 80}, {"request_id": "held", "gate": {"id": "g"}}],
        },
    ) as server:
        started = time.monotonic()
        assert stream_parts(send(server, body("delay")))[-1] == "[DONE]"
        assert time.monotonic() - started >= 0.08
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(send, server, body("held"))
            server.wait_event("gate_waiting", gate_id="g")
            server.post("/__mock__/shutdown")
            with pytest.raises(httpx.RemoteProtocolError):
                future.result(timeout=5)
    assert server.process.poll() == 0
    assert json.loads((server.root / "exit.json").read_text())["complete"]


@pytest.mark.parametrize(
    "config", [{}, {"profile": "wrong"}, {"profile": "replay", "delay": 0}, {"profile": "replay", "chunk_tokens": 0}]
)
def test_invalid_scenario_is_rejected(config):
    with pytest.raises(ValidationError):
        Scenario.model_validate(config)


def test_startup_failure_preserves_diagnostics_and_reaps_child(tmp_path):
    server = MockServerProcess(
        tmp_path / "bad", {"profile": "replay", "rules": [{"request_id": "duplicate"}, {"request_id": "duplicate"}]}
    )
    with pytest.raises(
        RuntimeError,
        match="mock startup failed.*",
    ):
        server.start()
    assert server.process.poll() is not None
    assert "duplicate request rule" in (server.root / "stderr.log").read_text()
    assert json.loads((server.root / "exit.json").read_text())["complete"] is False


def test_recorder_write_error_cannot_be_reported_as_complete(tmp_path):
    from tests.agentbench.helpers.mock_server.recorder import Recorder

    recorder = Recorder(tmp_path, "write-error")
    recorder.submit("missing-directory/rows.jsonl", {"event": "lost"})
    with pytest.raises(RuntimeError, match="recorder failed: FileNotFoundError"):
        recorder.close()
