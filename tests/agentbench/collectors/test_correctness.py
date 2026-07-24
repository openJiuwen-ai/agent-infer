import json

from agentinfer.agentbench.benchkit.collectors.correctness import load_correctness_artifact


def test_load_correctness_artifact_returns_raw_object(tmp_path) -> None:
    path = tmp_path / "correctness.json"
    path.write_text(json.dumps({"resolved": True, "score": 1}), encoding="utf-8")

    capture = load_correctness_artifact(path)

    assert capture.available is True
    assert capture.path == path
    assert capture.metadata == {"resolved": True, "score": 1}


def test_load_correctness_artifact_reports_missing_directory_and_invalid_input(tmp_path) -> None:
    missing = load_correctness_artifact(tmp_path / "missing.json")
    assert missing.available is False
    assert missing.reason == "artifact missing"

    directory = load_correctness_artifact(tmp_path)
    assert directory.available is False
    assert directory.reason == "artifact missing"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    capture = load_correctness_artifact(invalid)
    assert capture.available is False
    assert capture.reason == "artifact must contain a JSON object"

    invalid_encoding = tmp_path / "invalid-encoding.json"
    invalid_encoding.write_bytes(b"\xff")
    capture = load_correctness_artifact(invalid_encoding)
    assert capture.available is False
    assert capture.path == invalid_encoding
    assert capture.reason is not None
