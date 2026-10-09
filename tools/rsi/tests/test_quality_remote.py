"""R4: the remote quality DRIVER. It must run the served-model probe ON the GPU box over SSH (the
same host the bench uses), NOT on the local no-GPU driver. The SSH boundary is mocked, so the driver
logic is tested offline. Honesty property: ssh/serve failure, non-zero exit, or null probe output ->
None (NOT MEASURED -> the accept gate hard-fails -> no false certify).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import vllm_evolve.engine.quality_remote as qr
from vllm_evolve.engine.quality_runner import (
    calibration_set,
    is_quality_certified,
    measured_quality_verdict,
)

_PROMPTS = [c["prompt"] for c in calibration_set()]


def _probe_ok(payload_json: str):
    p = json.loads(payload_json)
    body = {"perplexity": 10.0, "task_em": 0.8, "continuations": [" x"] * len(p["prompts"])}
    return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")


def test_driver_runs_probe_on_the_remote_host(monkeypatch):
    # the probe MUST be dispatched over ssh to bc.runner.remote (not served locally); the winner
    # config travels to the box in the payload.
    seen: dict = {}

    def fake_ssh(remote, remote_cmd, payload_json):
        seen["remote"] = remote
        seen["cmd"] = remote_cmd
        seen["payload"] = json.loads(payload_json)
        return _probe_ok(payload_json)

    monkeypatch.setattr(qr, "_ssh_run", fake_ssh)
    qmf = qr.make_remote_quality_measure_fn({"model": "facebook/opt-125m", "remote": "gpu-box"})
    m = qmf({"model": "facebook/opt-125m", "max_num_seqs": 512})(_PROMPTS)
    assert m is not None and m["perplexity"] == 10.0
    assert m["output_agreement"] == 1.0                      # baseline measured against itself
    assert seen["remote"] == "gpu-box"
    assert "python -m vllm_evolve.engine.quality_probe" in seen["cmd"]
    assert seen["payload"]["config"]["max_num_seqs"] == 512  # winner config sent to the box


def test_driver_certifies_quality_through_the_runner(monkeypatch):
    monkeypatch.setattr(qr, "_ssh_run", lambda remote, cmd, payload: _probe_ok(payload))
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    cand = {"model": "m", "remote": "gpu-box", "max_num_seqs": 512}
    verdict = measured_quality_verdict(qmf({"model": "m", "remote": "gpu-box"}), qmf(cand))
    assert is_quality_certified(verdict) is True


def test_ssh_failure_is_not_measured(monkeypatch):
    def _boom(remote, cmd, payload):
        raise RuntimeError("box unreachable")

    monkeypatch.setattr(qr, "_ssh_run", _boom)
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    assert qmf({"model": "m", "remote": "gpu-box"})(_PROMPTS) is None


def test_nonzero_exit_is_not_measured(monkeypatch):
    monkeypatch.setattr(qr, "_ssh_run",
                        lambda remote, cmd, payload: SimpleNamespace(returncode=1, stdout="",
                                                                     stderr="serve crashed"))
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    assert qmf({"model": "m", "remote": "gpu-box"})(_PROMPTS) is None


def test_null_probe_output_is_not_measured(monkeypatch):
    # the box probe prints "null" when it could not measure (e.g. serve failed there)
    monkeypatch.setattr(qr, "_ssh_run",
                        lambda remote, cmd, payload: SimpleNamespace(returncode=0, stdout="null\n",
                                                                     stderr=""))
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    assert qmf({"model": "m", "remote": "gpu-box"})(_PROMPTS) is None


def test_baseline_resets_each_round_for_shifted_bottleneck(monkeypatch):
    # measured_quality_verdict measures baseline then candidate, once each per round. Across a
    # shifted-bottleneck multi-round run the SAME closure is reused; each round's candidate must
    # agree vs THAT round's baseline, not the first baseline ever. Round 1's outputs ("b") differ
    # from round 0's ("a"); the old "first baseline ever" code would score the round-1 candidate vs
    # "a" (agreement 0), wrongly. The fix re-establishes the baseline each round.
    seq = [["a", "a"], ["a", "a"],          # round 0: baseline, candidate
           ["b", "b"], ["b", "b"]]          # round 1 (adopted, shifted): baseline, candidate
    i = {"n": 0}

    def fake_ssh(remote, cmd, payload):
        conts = seq[i["n"]]
        i["n"] += 1
        body = {"perplexity": 10.0, "task_em": 0.8, "continuations": conts}
        return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(qr, "_ssh_run", fake_ssh)
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    prompts = ["p1", "p2"]
    qmf({"model": "m", "remote": "gpu-box"})(prompts)                       # r0 baseline
    c0 = qmf({"model": "m", "remote": "gpu-box", "k": 1})(prompts)          # r0 candidate
    qmf({"model": "m", "remote": "gpu-box", "k": 1})(prompts)               # r1 baseline (" b")
    c1 = qmf({"model": "m", "remote": "gpu-box", "k": 2})(prompts)          # r1 candidate (" b")
    assert c0["output_agreement"] == 1.0           # candidate agrees with round-0 baseline
    assert c1["output_agreement"] == 1.0           # agrees with round-1 baseline, NOT round-0


def test_ssh_timeout_cleans_up_remote_and_is_not_measured(monkeypatch):
    # on an ssh TIMEOUT the box-side `vllm serve` keeps running; we must kill it (free GPU/port)
    # before swallowing the timeout, mirroring run_remote_bench (Codex review P2).
    import subprocess

    import vllm_evolve.bench.dispatch as dispatch
    cleaned: dict = {}

    def _timeout(remote, cmd, payload):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

    monkeypatch.setattr(qr, "_ssh_run", _timeout)
    monkeypatch.setattr(dispatch, "_remote_cleanup",
                        lambda remote, port: cleaned.update(remote=remote, port=port))
    qmf = qr.make_remote_quality_measure_fn({"model": "m", "remote": "gpu-box"})
    assert qmf({"model": "m", "remote": "gpu-box"})(_PROMPTS) is None
    assert cleaned == {"remote": "gpu-box", "port": 8270}      # killed the box-side serve by port


def test_agreement_no_tokens_is_not_perfect_agreement():
    # all-empty continuations (a degenerate / echo-only serve generated NOTHING) must NOT read as
    # 1.0 agreement and certify quality — no evidence is a FAIL, not a skip (self-review).
    assert qr._agreement(["", ""], ["", ""]) == 0.0
    assert qr._agreement(["a b", "c"], ["a b", "c"]) == 1.0


def test_no_model_is_not_measured():
    assert qr.make_remote_quality_measure_fn({})({}) is None
