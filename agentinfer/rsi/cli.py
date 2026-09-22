# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""CPU-only entry point for offline feedback, demo state and evidence."""

import argparse
import json
import sys
from pathlib import Path


def _write(value, output=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")


def _controller(run_dir):
    from agentinfer.rsi.controller import Controller

    return Controller(Path(run_dir) / "state.sqlite")


def _demo(run_dir, run_id):
    from agentinfer.rsi.knowledge import KnowledgeRepository

    controller = _controller(run_dir)
    existing = {run["run_id"] for run in controller.list_runs()}
    if run_id in existing:
        run = controller.get(run_id)
        if not run["demo"]:
            raise ValueError("Refusing to reuse a non-demo run")
    else:
        run = controller.create(run_id, baseline="v0.8.2", max_trials=3, demo=True, backend="cuda")
        actions = [("advance", {})] * 3 + [("start_trial", {"candidate_id": "v0.8.3-rc24"})] + [("advance", {})] * 2
        for index, (action, payload) in enumerate(actions):
            run = controller.command(
                run_id,
                expected_revision=run["revision"],
                idempotency_key=f"demo-bootstrap-{index}",
                action=action,
                payload=payload,
            )
    knowledge = KnowledgeRepository(Path(run_dir) / "state.sqlite").seed_demo()
    return {"synthetic": True, "run": run, "seed_records": len(knowledge), "production_connected": False}


def parser():
    result = argparse.ArgumentParser(description="Offline RSI scaffold. No model or deployment is started.")
    commands = result.add_subparsers(dest="command", required=True)
    taxonomy = commands.add_parser("taxonomy", help="Describe logical layers and backend-specific probe plans")
    taxonomy.add_argument("--backend", choices=("cuda", "ascend"), default="cuda")
    evaluate = commands.add_parser("evaluate", help="Evaluate frozen checks from a feedback JSON manifest")
    evaluate.add_argument("input", type=Path)
    evaluate.add_argument("--output", type=Path)
    experiments = commands.add_parser("experiments", help="Append or list reported experiment evidence")
    experiment_commands = experiments.add_subparsers(dest="experiment_command", required=True)
    append = experiment_commands.add_parser("append", help="Append one record from JSON; never execute it")
    append.add_argument("--input", type=Path, required=True)
    listing = experiment_commands.add_parser("list", help="Read experiments.jsonl without creating state")
    for item in (append, listing):
        item.add_argument("--run-dir", type=Path, required=True)
    for name in ("demo", "status", "command", "knowledge", "serve"):
        item = commands.add_parser(name)
        item.add_argument("--run-dir", type=Path, default=Path(".rsi-demo"))
        if name != "knowledge":
            item.add_argument("--run-id", default="demo-r024")
        if name == "command":
            item.add_argument("--action", required=True)
            item.add_argument("--expected-revision", type=int, required=True)
            item.add_argument("--idempotency-key", required=True)
            item.add_argument("--payload", default="{}", help="JSON object; never interpreted as shell code")
        if name == "knowledge":
            item.add_argument("--scope", required=True)
            item.add_argument("--backend", required=True)
            item.add_argument("--component-version", required=True)
            item.add_argument("--workload", required=True)
            item.add_argument("--text", default="")
        if name == "serve":
            item.add_argument("--port", type=int, default=8877)
            item.add_argument(
                "--real-experiments", action="store_true", help="Serve only the read-only experiment ledger"
            )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "taxonomy":
            from agentinfer.rsi.feedback import list_layers

            _write(list_layers(args.backend))
        elif args.command == "evaluate":
            from agentinfer.rsi.feedback import evaluate_feedback, load_records

            manifest = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("CLI evaluation requires a manifest with scope and required_checks")
            records = load_records(args.input)
            report = evaluate_feedback(records, manifest["required_checks"], manifest["scope"])
            report["synthetic"] = manifest.get("synthetic", False) or any(record.synthetic for record in records)
            _write(report, args.output)
        elif args.command == "demo":
            _write(_demo(args.run_dir, args.run_id))
        elif args.command == "experiments":
            from agentinfer.rsi.experiments import append_experiment, list_experiments, load_experiment_json

            if args.experiment_command == "append":
                record = load_experiment_json(args.input.read_text(encoding="utf-8"))
                _write(append_experiment(args.run_dir, record))
            else:
                _write(list_experiments(args.run_dir))
        elif args.command == "status":
            _write(_controller(args.run_dir).get(args.run_id))
        elif args.command == "command":
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise ValueError("--payload must be a JSON object")
            _write(
                _controller(args.run_dir).command(
                    args.run_id,
                    expected_revision=args.expected_revision,
                    idempotency_key=args.idempotency_key,
                    action=args.action,
                    payload=payload,
                )
            )
        elif args.command == "knowledge":
            from agentinfer.rsi.knowledge import KnowledgeRepository

            _write(
                KnowledgeRepository(args.run_dir / "state.sqlite").search(
                    scope=args.scope,
                    backend=args.backend,
                    component_version=args.component_version,
                    workload=args.workload,
                    text=args.text,
                )
            )
        elif args.command == "serve":
            from agentinfer.rsi.dashboard.server import serve

            serve(args.run_dir, run_id=args.run_id, port=args.port, real_experiments=args.real_experiments)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"RSI: {error}", file=sys.stderr)
        return 2
    return 0
