"""R4: the box-side served-model quality probe (runs ON the GPU host over SSH). The serve + HTTP
boundary is mocked, so the measurement LOGIC is tested offline. Honesty property: any serve/HTTP
failure or missing field -> None (the driver treats None as NOT MEASURED -> no false certify).
"""
from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace

import vllm_evolve.engine.quality_probe as qp
from vllm_evolve.engine.quality_runner import calibration_set

_PROMPTS = [c["prompt"] for c in calibration_set()]
_EXPECTED = {c["prompt"]: c["expected"] for c in calibration_set()}


@contextlib.contextmanager
def _fake_serve(*a, **k):
    yield SimpleNamespace(port=8270)


def _fake_post(url, payload):
    prompt = payload["prompt"]
    cont = _EXPECTED.get(prompt, " x")
    return {"choices": [{"text": prompt + cont,
                         "logprobs": {"token_logprobs": [None, -0.5, -0.5]}}]}


def test_probe_returns_three_signals_and_continuations(monkeypatch):
    monkeypatch.setattr(qp, "serve_vllm", _fake_serve)
    monkeypatch.setattr(qp, "_http_post", _fake_post)
    out = qp.probe({"model": "facebook/opt-125m"}, _PROMPTS)
    assert out is not None
    assert out["perplexity"] > 0
    assert out["task_em"] == 1.0
    assert len(out["continuations"]) == len(_PROMPTS)


def test_probe_candidate_engine_config_reaches_serve(monkeypatch):
    # a representation-changing win (quantization / kv_cache_dtype / artifact / TP / batching) must
    # be SERVED as that representation on the box when its quality is measured.
    calls: list = []

    @contextlib.contextmanager
    def _capture_serve(model, plugin_dir, **kw):
        calls.append({"model": model, **kw})
        yield SimpleNamespace(port=8270)

    monkeypatch.setattr(qp, "serve_vllm", _capture_serve)
    monkeypatch.setattr(qp, "_http_post", _fake_post)
    cand = {"model": "facebook/opt-125m", "quantization": "fp8", "kv_cache_dtype": "fp8",
            "max_num_seqs": 512, "tensor_parallel_size": 2, "max_num_batched_tokens": 4096,
            "model_artifact_kind": "fp8-ckpt", "quantized_model_id": "org/opt-125m-fp8",
            "async_scheduling": True}
    assert qp.probe(cand, _PROMPTS) is not None
    c = calls[-1]
    assert c["model"] == "org/opt-125m-fp8"
    assert c["max_num_seqs"] == 512 and c["tensor_parallel_size"] == 2
    ea = c["extra_args"]
    assert ea[ea.index("--quantization") + 1] == "fp8"
    assert ea[ea.index("--kv-cache-dtype") + 1] == "fp8"
    assert ea[ea.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--async-scheduling" in ea


def test_quality_bench_config_preserves_frozen_remote_environment():
    config = qp.bench_config_for({
        "model": "m",
        "gpus": "0,1",
        "tensor_parallel_size": 2,
        "remote": "box",
        "remote_repo": "/frozen/repo",
        "conda_env": "/frozen/env",
        "conda_sh": "/frozen/conda.sh",
        "remote_workspace": "/frozen/workspace",
        "local_artifact_root": "/frozen/artifacts",
        "hf_endpoint": "https://frozen.example",
    })

    assert config.runner.remote_repo == "/frozen/repo"
    assert config.runner.conda_env == "/frozen/env"
    assert config.runner.conda_sh == "/frozen/conda.sh"
    assert config.runner.remote_workspace == "/frozen/workspace"
    assert config.runner.local_artifact_root == "/frozen/artifacts"
    assert config.runner.hf_endpoint == "https://frozen.example"


def test_probe_serve_failure_is_none(monkeypatch):
    class _FailingServe(contextlib.AbstractContextManager):
        def __enter__(self):
            raise RuntimeError("no GPU on this host")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(qp, "serve_vllm", lambda *a, **k: _FailingServe())
    assert qp.probe({"model": "m"}, _PROMPTS) is None


def test_probe_missing_logprobs_is_none(monkeypatch):
    monkeypatch.setattr(qp, "serve_vllm", _fake_serve)
    monkeypatch.setattr(qp, "_http_post",
                        lambda url, payload: {"choices": [{"text": "x", "logprobs": None}]})
    assert qp.probe({"model": "m"}, _PROMPTS) is None


def test_probe_no_model_is_none():
    assert qp.probe({}, _PROMPTS) is None


def test_main_reads_stdin_and_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(qp, "serve_vllm", _fake_serve)
    monkeypatch.setattr(qp, "_http_post", _fake_post)
    monkeypatch.setattr("sys.stdin",
                        io.StringIO(json.dumps({"config": {"model": "m"}, "prompts": _PROMPTS})))
    assert qp.main() == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["task_em"] == 1.0 and "continuations" in out


def test_main_bad_input_prints_null(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert qp.main() == 0
    assert capsys.readouterr().out.strip() == "null"
