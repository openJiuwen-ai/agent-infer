import platform
import sys

from agentcache.benchmarks.benchkit.collectors.environment import collect_environment


def test_collect_environment_returns_reproducibility_metadata(monkeypatch) -> None:
    monkeypatch.setattr(platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(platform, "node", lambda: "test-host")

    capture = collect_environment()

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {
        "platform": "test-platform",
        "hostname": "test-host",
        "python_version": sys.version,
    }
