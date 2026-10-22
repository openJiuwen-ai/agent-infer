# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Independent wire, history, timing and artifact checks for tiny source fixtures."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tests.agentbench.helpers.mock_server.process import read_rows


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition, field, expected, actual):
    if not condition:
        raise AssertionError(json.dumps({"field": field, "expected": expected, "actual": actual}, ensure_ascii=False))


def equal(field, actual, expected):
    require(actual == expected, field, expected, actual)


def check_counts(expected_ids, records):
    actual = Counter(row["identity"].get("request_id") for row in records)
    equal("request_counts", actual, Counter(expected_ids))


def response_for(records, capture_id):
    chunks = [row["payload"] for row in records if row["capture_id"] == capture_id]
    text, ids, usage = "", [], None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for choice in chunk.get("choices", []):
            text += choice.get("text", choice.get("delta", {}).get("content", ""))
            ids.extend(choice.get("token_ids", []))
        if "usage" in chunk:
            usage = chunk["usage"]
    return {"text": text, "ids": ids, "usage": usage, "done": "[DONE]" in chunks}


def validate_plan(expected, plan):
    """Reject a plan inconsistent with the fixture's independent length/edge tables."""
    equal("tasks", len(plan["tasks"]), expected["task_num"])
    equal("seed", plan["sample_seed"], 9)
    equal("unique_sessions", len({task["runtime_session_id"] for task in plan["tasks"]}), expected["task_num"])
    ids = [node["runtime_request_id"] for task in plan["tasks"] for node in task["requests"]]
    equal("unique_request_ids", len(set(ids)), expected["task_num"] * 3)
    equal("execution_ready", plan["execution_ready"], True)
    equal(
        "sample_order",
        [task["source_session_id"] for task in plan["tasks"]],
        expected["sample_order"][: expected["task_num"]],
    )
    counts = Counter(task["source_session_id"] for task in plan["tasks"])
    if expected["task_num"] == 8:
        equal("sampled_session_counts", sorted(counts.values()), [4, 4])
    for position, task in enumerate(plan["tasks"]):
        equal("task_index", task["task_index"], position)
        nodes = task["requests"]
        equal(f"task[{position}].inputs", [node["planned_input_tokens"] for node in nodes], expected["inputs"])
        equal(f"task[{position}].outputs", [node["planned_output_tokens"] for node in nodes], expected["outputs"])
        for i, node in enumerate(nodes):
            label = f"task[{position}].{node['source_key']}"
            previous = nodes[i - 1]["source_key"] if i else None
            context = previous if expected["source"] != "agentX" else None
            interval = 0.05 if i else 0
            if expected.get("title"):
                previous = nodes[0]["source_key"] if i == 2 else None
                context = previous
                interval = [0, 0.2, 0.1][i]
            if expected.get("overlap"):
                previous = None if i < 2 else nodes[0]["source_key"]
                interval = [0, 0.02, 0.02][i]
            equal(label + ".send_after", node["send_after"], previous)
            equal(label + ".context_after", node["context_after"], context)
            equal(label + ".role", node["actor_role"], "lead")
            equal(label + ".node_type", node["node_type"], "request")
            if expected["source"] != "inferact_codex_swebenchpro":
                require(
                    abs(node["effective_interval_seconds"] - interval) < 1e-6,
                    label + ".interval",
                    interval,
                    node["effective_interval_seconds"],
                )


def validate_round(expected: dict, root: Path):
    """Check a completed round and persist a pass/fail report before returning evidence."""
    try:
        return _validate_round(expected, root)
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        (root / "comparison.json").write_text(
            json.dumps(
                {"passed": False, "round": str(root), "source": expected["source"], "error": str(exc)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise


def _validate_round(expected, root):
    def read(name):
        return json.loads((root / name).read_text())

    plan = read("result/replay-plan.json")
    validate_plan(expected, plan)
    equal("server_profile", read("server/ready.json")["profile"], "replay")
    equal("server_complete", read("server/exit.json")["complete"], True)
    equal("replay_exit", read("replay-exit.json"), {"returncode": 0, "forced": False})
    equal("manifest", read("result/manifest.json")["status"], "completed")
    rows = read_rows(root / "server/backend-requests.jsonl")
    responses = read_rows(root / "server/backend-responses.jsonl")
    events = read_rows(root / "client-events.jsonl")
    require(bool(events), "client_events", "nonempty", events)
    equal("observer_closed", events[-1]["event"], "observer_closed")
    equal("event_sequence", [e["sequence"] for e in events], list(range(1, len(events) + 1)))
    server_events = read_rows(root / "server/server-events.jsonl")
    require(bool(server_events), "server_events", "nonempty", server_events)
    equal("recorder_closed", server_events[-1]["event"], "recorder_closed")
    equal("server_event_sequence", [e["sequence"] for e in server_events], list(range(1, len(server_events) + 1)))
    requests = [row for row in rows if row["kind"] == "generation"]
    unknown = [row for row in rows if row["kind"] == "unidentified_generation"]
    equal("warmup_count", len(unknown), int(expected["source"] == "tracelab"))
    warmup = []
    if unknown:
        body = unknown[0]["body"]
        equal("warmup_length", len(body["prompt"]), 21)
        equal("warmup_output", body["max_tokens"], 1)
        equal("warmup_stream", body.get("stream", False), False)
        warmup = body["prompt"]
    expected_ids = [n["runtime_request_id"] for t in plan["tasks"] for n in t["requests"]]
    check_counts(expected_ids, requests)
    facts = read_rows(root / "result/requests.jsonl")
    equal("trace_counts", Counter(f["request_id"] for f in facts), Counter(expected_ids))
    execution = read("result/replay-execution.json")
    equal(
        "execution_counts",
        Counter(n["runtime_request_id"] for t in execution["tasks"] for n in t["nodes"]),
        Counter(expected_ids),
    )
    equal("trace_pending", execution["trace_writer"]["pending"], 0)
    for name in ("transport_entered", "http_dispatch", "node_finished"):
        equal(
            "events." + name,
            Counter(e.get("runtime_request_id") for e in events if e["event"] == name),
            Counter(expected_ids),
        )
    by_id = {r["identity"]["request_id"]: r for r in requests}
    facts_by_id = {fact["request_id"]: fact for fact in facts}
    times = defaultdict(dict)
    for event in events:
        if event.get("runtime_request_id"):
            times[event["runtime_request_id"]][event["event"]] = event
    bodies, replies, hashes = {}, {}, {}
    root_prompts = []
    for task in plan["tasks"]:
        task_events = [event for event in events if event.get("runtime_session_id") == task["runtime_session_id"]]
        task_times = {
            event["event"]: event["t_monotonic"] for event in task_events if event.get("runtime_request_id") is None
        }
        require(
            task_times["task_submitted"]
            <= task_times["task_admitted"]
            <= task_times["graph_started"]
            <= task_times["task_finished"],
            "task_event_order",
            "submitted/admitted/graph/finished",
            task_times,
        )
        result_task = next(t for t in execution["tasks"] if t["runtime_session_id"] == task["runtime_session_id"])
        require(
            abs(result_task["duration_seconds"] - (task_times["task_finished"] - task_times["task_admitted"])) < 1e-6,
            "duration",
            "admitted to cleanup",
            result_task["duration_seconds"],
        )
        nodes = {node["source_key"]: node for node in task["requests"]}
        previous_body = previous_answer = None
        task_root = None
        for index, node in enumerate(task["requests"]):
            rid = node["runtime_request_id"]
            label = f"task[{task['task_index']}].{node['source_key']}"
            row = by_id[rid]
            body = row["body"]
            answer = response_for(responses, row["capture_id"])
            equal(label + ".session", row["identity"]["session_id"], task["runtime_session_id"])
            equal(label + ".actor", row["identity"]["agent_id"], node["actor_id"])
            equal(label + ".model", body["model"], "replay-ci-model")
            equal(
                label + ".endpoint",
                row["path"],
                "/v1/chat/completions" if expected["source"] == "inferact_codex_swebenchpro" else "/v1/completions",
            )
            for field in ("min_tokens", "max_tokens"):
                equal(label + "." + field, body[field], expected["outputs"][index])
            equal(label + ".seed", body["seed"], node["backend_sampling_seed"])
            equal(label + ".stream", body["stream"], True)
            equal(label + ".ignore_eos", body["ignore_eos"], True)
            prompt = body.get("prompt")
            length = len(prompt) if prompt is not None else sum(len(m["content"]) for m in body["messages"])
            equal(label + ".input_tokens", length, expected["inputs"][index])
            fact = facts_by_id[rid]
            for field, value in {
                "session_id": task["runtime_session_id"],
                "actor_id": node["actor_id"],
                "actor_role": node["actor_role"],
                "status": "success",
                "status_code": 200,
                "input_tokens": expected["inputs"][index],
                "output_tokens": expected["outputs"][index],
            }.items():
                equal(label + ".trace." + field, fact[field], value)
            equal(label + ".output_text_length", len(answer["text"]), expected["outputs"][index])
            equal(
                label + ".usage",
                answer["usage"],
                {
                    "prompt_tokens": length,
                    "completion_tokens": len(answer["text"]),
                    "total_tokens": length + len(answer["text"]),
                },
            )
            equal(label + ".done", answer["done"], True)
            if body.get("return_token_ids"):
                equal(label + ".output_ids", answer["ids"], list(map(ord, answer["text"])))
            if expected["source"] == "inferact_codex_swebenchpro":
                messages = body["messages"]
                equal(label + ".roles", [m["role"] for m in messages], ["user", "assistant"] * index + ["user"])
                equal(label + ".user", messages[-1]["content"], ["hello", "abcdefghij", "klmnopqrst"][index])
                if previous_body:
                    equal(label + ".history", messages[:-2], previous_body["messages"])
                    equal(label + ".answer_history", messages[-2]["content"], previous_answer["text"])
            elif expected["source"] == "agentX":
                if previous_body:
                    equal(label + ".snapshot_prefix", prompt[: len(previous_body["prompt"])], previous_body["prompt"])
            elif expected["source"] == "tracelab":
                history = previous_body["prompt"] + previous_answer["ids"] if previous_body else warmup
                count = [20, 34, 48][index]
                equal(label + ".history", prompt[:count], history[:count])
            elif node["context_after"]:
                predecessor = nodes[node["context_after"]]["runtime_request_id"]
                history = bodies[predecessor]["prompt"] + replies[predecessor]["ids"]
                equal(label + ".history", prompt[: len(history)], history)
            if index == 0:
                task_root = prompt
                if prompt is not None:
                    root_prompts.append(prompt)
            ev = times[rid]
            for name in (
                "prompt_prepare_started",
                "prompt_prepare_finished",
                "release_scheduled",
                "transport_entered",
                "http_dispatch",
                "node_finished",
            ):
                require(name in ev, label + ".events", name, list(ev))
            equal(label + ".status", ev["node_finished"]["status"], "success")
            start = ev["transport_entered"]["t_monotonic"]
            base = (
                times[nodes[node["send_after"]]["runtime_request_id"]]["node_finished"]["t_monotonic"]
                if node["send_after"]
                else task_times["graph_started"]
            )
            release = base + node["effective_interval_seconds"]
            require(
                abs(ev["release_scheduled"]["release_clock"] - release) < 1e-6,
                label + ".release",
                release,
                ev["release_scheduled"]["release_clock"],
            )
            require(start + 1e-6 >= release, label + ".early_send", release, start)
            require(
                start + 1e-6 >= ev["prompt_prepare_finished"]["t_monotonic"],
                label + ".unprepared_send",
                "after preparation",
                start,
            )
            require(
                ev["http_dispatch"]["t_monotonic"] >= start, label + ".dispatch", "after transport", ev["http_dispatch"]
            )
            if node["context_after"]:
                context_end = times[nodes[node["context_after"]]["runtime_request_id"]]["node_finished"]["t_monotonic"]
                require(
                    ev["prompt_prepare_started"]["t_monotonic"] >= context_end,
                    label + ".context_wait",
                    context_end,
                    ev["prompt_prepare_started"],
                )
            result_node = next(n for n in result_task["nodes"] if n["runtime_request_id"] == rid)
            equal(label + ".result_status", result_node["status"], "success")
            require(
                abs(result_node["scheduler_lag_seconds"] - (start - release)) < 1e-6,
                label + ".lag",
                start - release,
                result_node["scheduler_lag_seconds"],
            )
            if expected["source"] == "tracelab":
                equal(
                    label + ".shortfall", result_node["prompt_calibration"]["prefix_shortfall_tokens"], [0, 0, 4][index]
                )
            bodies[rid], replies[rid] = body, answer
            hashes[rid] = {"body": fingerprint(body), "prompt": fingerprint(body.get("prompt", body.get("messages")))}
            previous_body, previous_answer = body, answer
        if expected["source"] == "agentinfer" and task_root:
            for other in root_prompts[:-1]:
                for lo, hi in [(0, 5), (9, 21), (60, 64)]:
                    equal("shared_prefix", task_root[lo:hi], other[lo:hi])
                require(task_root[5:9] != other[5:9], "private_prefix", "distinct sessions", task_root[5:9])
    active = peak = 0
    config = yaml.safe_load((root / "config.yaml").read_text())
    for event in events:
        if event["event"] == "task_admitted":
            active += 1
            peak = max(peak, active)
        if event["event"] == "task_finished":
            active -= 1
        require(
            0 <= active <= config["experiment"]["max_concurrency"],
            "session_concurrency",
            config["experiment"]["max_concurrency"],
            active,
        )
    equal("active_at_exit", active, 0)
    summary = read("result/summary.json")
    total = expected["task_num"] * 3
    for field, value in {
        "requests": total,
        "successful_requests": total,
        "input_tokens": expected["task_num"] * sum(expected["inputs"]),
        "output_tokens": expected["task_num"] * sum(expected["outputs"]),
    }.items():
        equal("summary." + field, summary["requests"][field], value)
    equal("summary.completed", summary["tasks"]["completed"], expected["task_num"])
    for name in ("failed_requests", "pre_send_failed_requests", "dependency_skipped_requests", "failed_tasks"):
        equal("execution." + name, execution["summary"][name], 0)
    equal(
        "metrics.inputs", summary["vllm"]["counters"]["vllm:prompt_tokens_total"], summary["requests"]["input_tokens"]
    )
    equal("metrics.successes", summary["vllm"]["request_finished_by_reason"]["length"], total)
    (root / "fingerprints.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (root / "comparison.json").write_text(
        json.dumps({"passed": True, "requests": total, "peak_sessions": peak}), encoding="utf-8"
    )
    return {"plan": plan, "bodies": bodies, "replies": replies, "root": root}


def compare_rounds(expected, first, second):
    """Compare stable fields while requiring different answers and new process nonces."""
    equal("workload_fingerprint", first["plan"]["workload_fingerprint"], second["plan"]["workload_fingerprint"])
    equal("plans", first["plan"]["tasks"], second["plan"]["tasks"])
    equal("source_sha256", first["plan"]["source_sha256"], second["plan"]["source_sha256"])
    equal(
        "raw_source", *(json.loads((result["root"] / "input-manifest.json").read_text()) for result in (first, second))
    )
    configs = []
    for result in (first, second):
        cfg = yaml.safe_load((result["root"] / "config.yaml").read_text())
        cfg["backend"].pop("base_url")
        cfg["experiment"].pop("result_dir")
        configs.append(cfg)
    equal("config", *configs)
    planned = {n["runtime_request_id"]: (t, n) for t in first["plan"]["tasks"] for n in t["requests"]}
    for rid, a in first["bodies"].items():
        b = second["bodies"][rid]
        stable_a = {key: value for key, value in a.items() if key not in {"prompt", "messages"}}
        stable_b = {key: value for key, value in b.items() if key not in {"prompt", "messages"}}
        equal(rid + ".parameters", stable_a, stable_b)
        require(
            first["replies"][rid]["text"] != second["replies"][rid]["text"],
            rid + ".answer_variation",
            "different answers",
            first["replies"][rid]["text"],
        )
        if expected["source"] == "agentX":
            equal(rid + ".snapshot", a["prompt"], b["prompt"])
        elif expected["source"] == "agentinfer":
            task, node = planned[rid]
            if node["context_after"] is None:
                equal(rid + ".independent_prompt", a["prompt"], b["prompt"])
            else:
                parent = next(n for n in task["requests"] if n["source_key"] == node["context_after"])
                boundary = parent["planned_input_tokens"] + parent["planned_output_tokens"]
                equal(rid + ".current_recipe", a["prompt"][boundary:], b["prompt"][boundary:])
    for path in ("server/ready.json", "replay-process.json"):
        a, b = (json.loads((r["root"] / path).read_text()) for r in (first, second))
        require(a["nonce"] != b["nonce"], "cold_start." + path, "new process nonce", a["nonce"])
