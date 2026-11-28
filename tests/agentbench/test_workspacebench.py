# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunResult
from agentinfer.agentbench.benchkit import cli
from agentinfer.agentbench.benchkit.config import load_config
from agentinfer.agentbench.benchkit.dataset import Task
from agentinfer.agentbench.benchkit.workspacebench import (
    FILES_REVISION,
    REVISION,
    TASK_REVISION,
    evaluation,
    prepare,
    runner,
    worker,
)
from agentinfer.agentbench.benchkit.workspacebench.trace import jiuwen_trace, zcode_trace
from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter
from agentinfer.agentbench.request_proxy.server import RequestProxyServer
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")
CONFIG = Path(__file__).resolve().parents[2] / "agentinfer/agentbench/configs/workspacebench_jiuwenswarm.yaml"


def test_full_pair_index_and_selection(tmp_path):
    root = tmp_path / "en"
    standard = root / "filesys/kaifa_standard"
    standard.mkdir(parents=True)
    for i in range(1, 389):
        td = root / "tasks" / str(i)
        td.mkdir(parents=True)
        (td / "metadata.json").write_text(
            json.dumps({"absolute_id": i, "task": "merge files", "language": "en", "file_system": "dev"})
        )
    rows = prepare._index(root, "en", {"dev": "kaifa"})
    cfg = load_config(CONFIG)
    cfg.dataset.index_path = tmp_path / "instances.jsonl"
    cfg.dataset.selection_path = tmp_path / "list"
    cfg.dataset.index_path.write_text("\n".join(json.dumps(row) for row in rows))
    cfg.dataset.selection_path.write_text("wsb-en-0002\nwsb-en-0001")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "workspacebench",
                "task_revision": TASK_REVISION,
                "workspace_revision": FILES_REVISION,
                "evaluation_revision": REVISION,
            }
        )
    )
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/jiuwenswarm-runtime.json").write_text('{"installation":{"image_id":"test"}}')
    tasks = runner.select_tasks(cfg, tmp_path)
    assert [task.instance_id for task in tasks] == ["wsb-en-0002", "wsb-en-0001"]
    assert tasks[0].standard_path == standard
    assert json.loads((tmp_path / "workspacebench.json").read_text())["tasks"][0]["absolute_id"] == 2
    cfg.dataset.selection_path.write_text("wsb-en-0999")
    with pytest.raises(ValueError, match="unknown"):
        runner.select_tasks(cfg, tmp_path)


def test_jiuwen_trace_preserves_nested_ids_and_failed_tools(tmp_path):
    frames = [
        (
            "chat.tool_call",
            {"execution_id": who, "tool_call": {"name": "read", "arguments": '{"path":"a"}', "tool_call_id": "1"}},
        )
        for who in ("root", "child")
    ]
    frames += [
        ("chat.tool_update", {"execution_id": "root", "tool_call_id": "1", "arguments": {"path": "src/a"}}),
        ("chat.tool_result", {"execution_id": "root", "tool_call_id": "1", "result": "contents", "success": True}),
        ("chat.tool_result", {"execution_id": "child", "tool_call_id": "1", "result": "missing", "success": False}),
        ("chat.delta", {"content": "done"}),
    ]
    raw = tmp_path / "jiuwenswarm.jsonl"
    raw.write_text("\n".join(json.dumps({"event": event, "payload": payload}) for event, payload in frames))
    trace = jiuwen_trace(raw)
    first, second, text = trace["executionTrace"]
    assert first["input"] == {"path": "src/a"} and first["output"]["result"] == "contents"
    assert second["input"] == {"path": "a"} and second["output"]["success"] is False
    assert text["content"] == trace["lastText"] == "done"


def test_zcode_trace_retains_attempts(tmp_path):
    record = {"sessionId": "s", "toolCallId": "c", "toolName": "Read", "toolInput": {"path": "/a"}}
    (tmp_path / "zcode-hooks.jsonl").write_text(
        "\n".join(
            json.dumps({"event": {**record, "hookEventName": k, "toolResponse": {"status": "failed"}}})
            for k in ["PreToolUse", "PostToolUseFailure"]
        )
    )
    (event,) = zcode_trace(tmp_path)["executionTrace"]
    assert event["tool"] == "Read" and event["output"]["event"] == "PostToolUseFailure"


def test_native_bridge_uses_official_prompt(monkeypatch, tmp_path):
    import sys

    from agentinfer.agentbench.agents.jiuwenswarm import runner as jiuwen

    def official(**kwargs):
        assert kwargs["eval_while_running"] is False and kwargs["isolated_workdir"]
        result = kwargs["run_fn"](
            prompt="生成月报", work_dir=str(tmp_path), sandbox_dir=str(tmp_path), timeout_s=15, api_provider={}
        )
        assert result["status"] == "ok"
        return {"case": {"status": "passed"}, "summary": {"passed": 1}}

    monkeypatch.setitem(sys.modules, "agent_runner", SimpleNamespace(_run_one_case=official))
    execute = AsyncMock(
        return_value=AgentRunResult("jiuwenswarm", "agent.code.normal", outcome=AgentRunOutcome.COMPLETED)
    )
    monkeypatch.setattr(jiuwen, "run_jiuwenswarm", execute)
    opts = load_config(CONFIG).agent.model_dump(mode="json")
    worker.run_case(
        {
            "agent": opts,
            "task": {"id": "1", "task": "test", "file_system": "dev"},
            "model": "test",
            "proxy": "http://test",
            "session_id": "s",
            "timeout": 15,
            "prompt_timeout": 5,
        },
        tmp_path,
        tmp_path,
    )
    assert execute.await_args.kwargs["workspace_prompt"] == "生成月报"
    assert execute.await_args.args[0].session_id == "s"


def test_host_execution_copies_shared_role_directory(tmp_path):
    config = load_config(CONFIG)
    config.dataset.workspacebench_execution = "host"
    standard = tmp_path / "standard"
    standard.mkdir()
    (standard / "input.txt").write_text("original")

    # Exercise the host worker command without a system Agent installation.
    evaluation_dir = tmp_path / "evaluation"
    src = evaluation_dir / "src"
    src.mkdir(parents=True)
    (src / "agent_runner.py").write_text(
        """import json
import shutil
from pathlib import Path

def _run_one_case(meta, runs_root, standard_work_dir_map, isolated_workdir, **kwargs):
    assert isolated_workdir
    assert "rubrics" not in meta
    case = Path(runs_root) / meta["id"]
    workdir = case / "workdir"
    shutil.copytree(standard_work_dir_map[meta["file_system"]], workdir)
    source = workdir / "input.txt"
    (case / "output").mkdir()
    (case / "output" / "report.txt").write_text(source.read_text())
    source.write_text(meta["id"])
    (case / "agent.json").write_text(json.dumps({"runnerStatus": "ok"}))
    return {"summary": {"passed": 1}, "case": {"status": "passed"}}
"""
    )
    (tmp_path / "workspacebench.json").write_text(json.dumps({"evaluation_checkout": str(evaluation_dir)}))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "jiuwenswarm-runtime.json").write_text(json.dumps({"executable": "jiuwenswarm"}))

    for number in (1, 2):
        instance_id = f"wsb-en-{number:04d}"
        metadata = tmp_path / f"{number}.json"
        metadata.write_text(
            json.dumps(
                {
                    "absolute_id": number,
                    "task": "report",
                    "file_system": "dev",
                    "language": "en",
                    "output_files": ["report.txt"],
                    "rubrics": ["judge only"],
                }
            )
        )
        output = tmp_path / "tasks" / instance_id
        output.mkdir(parents=True)
        task = Task(instance_id, "", "", "report", metadata_path=metadata, standard_path=standard)
        result = asyncio.run(runner.run_case(config, task, "http://localhost", output, instance_id))
        case = output / "official" / str(number)
        assert result.outcome == AgentRunOutcome.COMPLETED
        assert (case / "output" / "report.txt").read_text() == "original"
        assert not (case / "workdir").exists()
        assert json.loads((output / "metadata.json").read_text())["rubrics"] == ["judge only"]
    assert (standard / "input.txt").read_text() == "original"


def test_host_collector_rejects_links_outside_workdir(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "report.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="outside the task directory"):
        worker._validate_workspace_links(workdir)


@pytest.mark.parametrize("endpoint", ["/v1/messages", "/v1/chat/completions"])
def test_official_sdk_request_attribution(tmp_path, endpoint):
    async def go():
        session = "43c618db-da45-4866-b564-c00d06a504bb"
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        server = RequestProxyServer("http://backend", "run", writer, 10, "token", endpoint)
        await server.client.aclose()

        def upstream(request):
            assert request.url.path == endpoint
            return httpx.Response(200, json={"usage": {}})

        server.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        await writer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://local") as client:
            res = await client.post("/sessions/" + session + endpoint, json={"model": "test", "messages": []})
        await server.close()
        assert res.status_code == 200
        assert json.loads((tmp_path / "requests.jsonl").read_text())["session_id"] == session

    asyncio.run(go())


@pytest.mark.parametrize("host", [False, True])
def test_evaluate_dispatch_regrade_and_coverage(tmp_path, monkeypatch, host):
    rows = [{"instance_id": f"wsb-{lang}-0001", "absolute_id": 1, "language": lang} for lang in ("en", "cn")]
    (tmp_path / "manifest.json").write_text('{"config":{"dataset":{"name":"workspacebench"}}}')
    manifest = {"tasks": rows, "source": {}}
    if not host:
        manifest["image_id"] = "digest"
    (tmp_path / "workspacebench.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(evaluation, "image_id", lambda: "host-judge")
    for key, val in {"JUDGE_API_KEY": "test", "JUDGE_BASE_URL": "http://test", "JUDGE_MODEL": "my/model"}.items():
        monkeypatch.setenv(key, val)

    def grade(run, row, directory, config, image):
        assert image == ("host-judge" if host else "digest")
        if row["language"] == "cn":
            raise ValueError("incomplete rubric")
        return 2, 3

    monkeypatch.setattr(evaluation, "_judge_one", grade)
    for i in (1, 2):
        dest = tmp_path / f"eval{i}"
        assert cli.main(["evaluate", "--result-path", str(tmp_path), "--output-dir", str(dest)]) == 1
        out = json.loads((dest / "correctness.json").read_text())
        assert out["languages"]["en"]["rubric_pass_rate"] == 2 / 3
        assert out["languages"]["cn"]["coverage"] == 0 and out["languages"]["cn"]["rubric_pass_rate"] is None
        assert not out["complete"] and len(out["tasks"]) == 2


def test_judge_result_requires_all_rubrics():
    rows = [{"index": i, "rubric": t, "passed": i == 0} for i, t in enumerate(["a", "b"])]
    assert evaluation.score({"judge": {"error": None}, "rubrics": rows}, ["a", "b"]) == 1
    for changes in ({"rubrics": [rows[0], rows[0]]}, {"rubrics": rows[:1]}, {"judge": {"error": "service failed"}}):
        with pytest.raises(ValueError):
            evaluation.score({"judge": {"model": "a"}, "rubrics": rows, **changes}, ["a", "b"])


def test_prepare_retries_and_keeps_full_role_snapshots(tmp_path, monkeypatch):
    import subprocess

    # Use the official helper signatures with a small filesystem fixture.
    upstream = tmp_path / "upstream"
    evaluation_dir = upstream / "evaluation"
    (evaluation_dir / "scripts").mkdir(parents=True)
    (evaluation_dir / "src").mkdir()
    (evaluation_dir / "scripts/build_run_config.py").write_text("ROLE_DIRS={'开发人员':'kaifa'}")
    (evaluation_dir / "src/filesys_utils.py").write_text(
        "from shutil import copytree\ndef make_filesys(raw, standard, role, tasks):\n    copytree(raw, standard)\n"
    )
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(
        prepare,
        "REVISION",
        subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
    )
    attempts = []

    def assets(evaluation, root, language):
        attempts.append(language)
        if attempts == ["en", "cn"]:
            raise OSError("download interrupted")
        if (root / "tasks").exists():
            return
        raw = root / "filesys/kaifa_raw"
        raw.mkdir(parents=True)
        (raw / "noise.md").write_text("role filesystem")
        for i in range(1, 389):
            case = root / "tasks" / str(i)
            case.mkdir(parents=True)
            (case / "metadata.json").write_text(
                json.dumps({"absolute_id": i, "language": language, "file_system": "开发人员", "task": "x"})
            )

    monkeypatch.setattr(prepare, "_assets", assets)
    out = tmp_path / "data"
    with pytest.raises(OSError, match="interrupted"):
        prepare.prepare(out, upstream)
    assert not (out / "manifest.json").exists()
    prepared = prepare.prepare(out, upstream)
    assert prepared.rows == 776
    assert prepared.selection_path.read_text().splitlines() == [
        "wsb-en-0002",
        "wsb-cn-0002",
        "wsb-en-0001",
        "wsb-cn-0001",
    ]
    for lang in ("en", "cn"):
        assert (out / lang / "filesys/kaifa_standard/noise.md").read_text() == "role filesystem"
    with pytest.raises(FileExistsError):
        prepare.prepare(out, upstream)
