from pathlib import Path

from vllm_evolve.bench import dispatch


def test_pull_optional_retries_transient_scp(monkeypatch, tmp_path: Path) -> None:
    attempts = []

    def flaky_scp(src: str, dst: str) -> None:
        attempts.append((src, dst))
        if len(attempts) < 3:
            raise RuntimeError("transient remote read failure")

    monkeypatch.setattr(dispatch, "_scp", flaky_scp)

    destination = tmp_path / "raw_requests.jsonl"
    assert dispatch._pull_optional("gpu-box", "/run/raw_requests.jsonl", destination)
    assert attempts == [
        ("gpu-box:/run/raw_requests.jsonl", str(destination)),
        ("gpu-box:/run/raw_requests.jsonl", str(destination)),
        ("gpu-box:/run/raw_requests.jsonl", str(destination)),
    ]


def test_pull_optional_remains_bounded_for_missing_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    attempts = 0

    def missing_scp(src: str, dst: str) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("missing")

    monkeypatch.setattr(dispatch, "_scp", missing_scp)

    assert not dispatch._pull_optional(
        "gpu-box", "/run/policy.py", tmp_path / "policy.py"
    )
    assert attempts == 3
