# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Worker entry points for a single official case and an independent judge."""

import asyncio
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from ...agents.contracts import AgentRunOutcome, AgentRunRequest, TerminationReason
from ..dataset import Task
from .trace import jiuwen_trace, zcode_trace

OFFICIAL = Path("/opt/workspacebench/evaluation")


def probe(options: dict) -> dict:
    from ...agents.registry import get_runtime

    kind = options["type"]
    if kind in ("jiuwenswarm", "zcode"):
        report = asyncio.run(get_runtime(kind).preflight(Path(options["executable"])))
        if kind == "jiuwenswarm":
            subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "--unshare-user", "--unshare-pid", "--proc", "/proc", "true"],
                check=True,
            )
            versions = {pkg: version(pkg) for pkg in ("workswarm", "openjiuwen")}
            if versions != {"workswarm": "0.2.8", "openjiuwen": "0.1.20"}:
                raise ValueError(f"Unexpected Jiuwen versions: {versions}")
            return versions
        return report.versions
    if kind == "dsh":
        return {"deepseek-harness-sdk": version("deepseek-harness-sdk")}
    package = OFFICIAL / "node_modules/@anthropic-ai/claude-agent-sdk/package.json"
    return {"claude-agent-sdk": json.loads(package.read_text())["version"]}


def run_case(payload: dict, destination: Path, standard: Path, *, host: bool = False) -> None:
    import agent_runner

    from ...agents.jiuwenswarm.runner import run_jiuwenswarm
    from ...agents.zcode.runner import run_zcode

    options = payload["agent"]
    kind = options["type"]
    meta = payload["task"]
    session = payload["session_id"]
    role = meta["file_system"]

    def run_native(*, prompt, work_dir, sandbox_dir, timeout_s, api_provider):
        raw = Path(sandbox_dir) / "raw"
        request = AgentRunRequest(
            agent_type=kind,
            profile_name=options["profile"],
            executable=Path(options["executable"]),
            task=Task(meta["id"], "", "", meta["task"]),
            model=payload["model"],
            api_base_url=payload["proxy"],
            workspace=Path(work_dir),
            artifact_dir=raw,
            session_id=session,
            timeout_seconds=int(timeout_s),
            patch_flush_seconds=0,
            prompt_delivery_timeout_seconds=payload["prompt_timeout"],
            tmux_startup_seconds=0,
            terminal_capture_interval_seconds=1,
            context_window=options["context_window"],
            max_tokens=options["max_tokens"],
            max_subagents=options["max_subagents"],
            max_concurrent_subagents=options["max_concurrent_subagents"],
        )
        execute = run_jiuwenswarm if kind == "jiuwenswarm" else run_zcode
        if host:
            _validate_workspace_links(request.workspace)
        result = asyncio.run(execute(request, workspace_prompt=prompt))
        if host:
            _validate_workspace_links(request.workspace)
        trace = jiuwen_trace(raw / "jiuwenswarm.jsonl", session) if kind == "jiuwenswarm" else zcode_trace(raw)
        status = "ok" if result.outcome == AgentRunOutcome.COMPLETED else "error"
        if result.termination_reason == TerminationReason.TIMEOUT:
            status = "timeout"
        return {"status": status, "errorMessage": result.termination_reason, "trace": trace, "paths": []}

    official_agents = {"claude": "claudecode", "dsh": "deepseekharness"}
    run = agent_runner._load_agent_run(official_agents[kind]) if kind in official_agents else run_native
    scoped_url = f"{payload['proxy']}/sessions/{session}"
    provider = {
        "model": payload["model"],
        "model_name": payload["model"],
        "apiKey": "agentbench",
        "provider_type": "anthropic" if kind == "claude" else "deepseek",
        "baseUrl": scoped_url if kind == "claude" else scoped_url + "/v1",
        "__deepseek_harness_runtime__": {"max_tokens": options["max_tokens"]},
    }
    report = agent_runner._run_one_case(
        idx=0,
        meta=meta,
        runs_root=str(destination),
        run_fn=run,
        prompt_head="",
        prompt_tail="",
        task_target_output_dir="model_output",
        timeout_sec=payload["timeout"],
        api_provider=provider,
        eval_while_running=False,
        eval_yaml="",
        work_dir_map={role: str(standard)},
        standard_work_dir_map={role: str(standard)},
        agent_name=kind,
        model_name=payload["model"],
        isolated_workdir=True,
        task_workdir_cleanup="always",
    )
    (destination / "agent_runner_report.json").write_text(
        json.dumps({"summary": report["summary"], "cases": [report["case"]]}, ensure_ascii=False)
    )


def _validate_workspace_links(workspace: Path) -> None:
    # The official collector runs with host access and follows file symlinks.
    root = workspace.resolve()
    if workspace.is_symlink():
        raise ValueError(f"Workspace must be a directory: {workspace}")
    for path in workspace.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"Workspace link points outside the task directory: {path}")


def judge() -> None:
    import agent_as_a_judge

    # The upstream Claude SDK can write its provider config under raw/. Only
    # export the final rubric and graph; keep transient judge files in /tmp.
    meta = json.loads(Path("/evidence/metadata.json").read_text())
    case = Path("/tmp/judge") / str(int(meta["absolute_id"]))
    case.mkdir(parents=True)
    for name in ("metadata.json", "agent.json", "output"):
        (case / name).symlink_to(Path("/evidence") / name)
    result = agent_as_a_judge.evaluate_task(str(case), eval_yaml_path="/evaluation.yaml")
    if not result.get("success") or not result.get("rubricsPath"):
        raise RuntimeError(result.get("error") or "official judge produced no rubric file")
    for key in ("rubricsPath", "depGraphPath"):
        if path := result.get(key):
            shutil.copyfile(path, Path("/scores") / Path(path).name)


def main() -> None:
    action = sys.argv[1]
    if action == "run-host":
        evaluation, destination, standard = (Path(arg) for arg in sys.argv[2:])
        sys.path.insert(0, str(evaluation / "src"))
        run_case(json.load(sys.stdin), destination, standard, host=True)
        return
    sys.path.insert(0, str(OFFICIAL / "src"))
    if action == "judge":
        judge()
    elif action == "probe":
        print(json.dumps(probe(json.load(sys.stdin))))
    elif action == "run":
        run_case(json.load(sys.stdin), Path("/cases"), Path("/standard"))
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
