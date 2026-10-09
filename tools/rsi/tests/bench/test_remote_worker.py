from __future__ import annotations

import json
import sys

import pytest

import vllm_evolve.bench.remote_worker as worker


def _snapshot(memory_used=4, apps=None):
    return {
        "captured_at_unix_s": 1.0,
        "nvidia_smi_exit": 0,
        "compute_apps_exit": 0,
        "gpus": [
            {
                "index": "1",
                "uuid": "GPU-one",
                "name": "L20X",
                "driver_version": "570",
                "memory_total_mib": 143771,
                "memory_used_mib": memory_used,
                "utilization_gpu_pct": 0,
                "compute_mode": "Default",
            }
        ],
        "compute_apps": apps or [],
        "raw_gpu_csv": "",
        "raw_compute_apps_csv": "",
    }


def _run(monkeypatch, tmp_path, command):
    monkeypatch.setattr(worker, "gpu_snapshot", lambda: _snapshot())
    monkeypatch.setattr(worker, "_cleanup_vllm_port", lambda port: None)
    return worker.run_guarded(
        command,
        run_dir=tmp_path / "runs" / "r1",
        workspace=tmp_path,
        gpus="1",
        tensor_parallel_size=1,
        port=8260,
        timeout_s=5,
        candidate_id="candidate-a",
        git_sha="abc123",
        config_sha256="cfg123",
    )


def test_success_always_writes_status_and_gpu_evidence(monkeypatch, tmp_path):
    status = _run(monkeypatch, tmp_path, [sys.executable, "-c", "print('ok')"])
    run_dir = tmp_path / "runs" / "r1"
    assert status["ok"] is True and status["state"] == "succeeded"
    assert json.loads((run_dir / "status.json").read_text())["candidate_id"] == "candidate-a"
    assert json.loads((run_dir / "manifest.json").read_text())["gpus"] == ["1"]
    assert (run_dir / "gpu_before.json").exists()
    assert (run_dir / "gpu_after.json").exists()
    summary = json.loads((run_dir / "gpu_summary.json").read_text())
    assert summary["sample_count"] >= 1
    assert summary["contaminated"] is False
    assert "ok" in (run_dir / "native.stdout.log").read_text()


def test_busy_gpu_refuses_without_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "gpu_snapshot", lambda: _snapshot(memory_used=4096))
    monkeypatch.setattr(worker, "_cleanup_vllm_port", lambda port: None)
    status = worker.run_guarded(
        [sys.executable, "-c", "raise SystemExit(99)"],
        run_dir=tmp_path / "runs" / "busy",
        workspace=tmp_path,
        gpus="1",
        tensor_parallel_size=1,
        port=8260,
        timeout_s=5,
        candidate_id="c",
        git_sha="g",
        config_sha256="s",
    )
    assert status["state"] == "resource_busy"
    assert status["exit_code"] == worker.BUSY_EXIT
    assert not (tmp_path / "runs" / "busy" / "native.stdout.log").exists()


def test_failure_is_structured_and_cleans_port(monkeypatch, tmp_path):
    cleaned = []
    monkeypatch.setattr(worker, "gpu_snapshot", lambda: _snapshot())
    monkeypatch.setattr(worker, "_cleanup_vllm_port", cleaned.append)
    status = worker.run_guarded(
        [sys.executable, "-c", "raise SystemExit(7)"],
        run_dir=tmp_path / "runs" / "failed",
        workspace=tmp_path,
        gpus="1",
        tensor_parallel_size=1,
        port=8261,
        timeout_s=5,
        candidate_id="c",
        git_sha="g",
        config_sha256="s",
    )
    assert status["state"] == "failed" and status["exit_code"] == 7
    assert cleaned == [8261]
    assert json.loads((tmp_path / "runs" / "failed" / "status.json").read_text())["ok"] is False


def test_existing_run_directory_is_never_overwritten(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "same"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep"
    sentinel.write_text("original")
    monkeypatch.setattr(worker, "gpu_snapshot", lambda: _snapshot())
    monkeypatch.setattr(worker, "_cleanup_vllm_port", lambda port: None)
    with pytest.raises(FileExistsError):
        worker.run_guarded(
            [sys.executable, "-c", "print('no')"],
            run_dir=run_dir,
            workspace=tmp_path,
            gpus="1",
            tensor_parallel_size=1,
            port=8262,
            timeout_s=5,
            candidate_id="c",
            git_sha="g",
            config_sha256="s",
        )
    assert sentinel.read_text() == "original"


def test_foreign_gpu_process_during_run_invalidates_measurement(
    monkeypatch, tmp_path
):
    app = {
        "gpu_uuid": "GPU-one",
        "pid": 987654,
        "process_name": "other-user-vllm",
        "used_memory_mib": 4096,
    }
    snapshots = iter([
        _snapshot(),
        _snapshot(memory_used=4096, apps=[app]),
        _snapshot(memory_used=4096, apps=[app]),
    ])
    last = _snapshot(memory_used=4096, apps=[app])

    def snapshot():
        return next(snapshots, last)

    monkeypatch.setattr(worker, "gpu_snapshot", snapshot)
    monkeypatch.setattr(worker, "_is_descendant_process", lambda pid, root: False)
    monkeypatch.setattr(worker, "_cleanup_vllm_port", lambda port: None)
    status = worker.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        run_dir=tmp_path / "runs" / "contaminated",
        workspace=tmp_path,
        gpus="1",
        tensor_parallel_size=1,
        port=8263,
        timeout_s=5,
        candidate_id="c",
        git_sha="g",
        config_sha256="s",
    )
    assert status["ok"] is False
    assert status["state"] == "resource_contaminated"
    assert status["exit_code"] == worker.CONTAMINATED_EXIT
    assert status["gpu_summary"]["contaminated"] is True
    assert status["gpu_summary"]["foreign_compute_apps"][0]["pid"] == 987654


def test_owned_gpu_pid_stays_owned_after_teardown_reparenting(monkeypatch):
    ancestry = iter([True, False])
    monkeypatch.setattr(
        worker,
        "_is_descendant_process",
        lambda pid, root: next(ancestry),
    )
    owned = set()

    assert worker._owned_by_run(1234, 99, owned) is True
    # The second call must use the sticky positive identity; ancestry is no
    # longer consulted after vLLM reparents the EngineCore during shutdown.
    assert worker._owned_by_run(1234, 99, owned) is True
    assert owned == {1234}
