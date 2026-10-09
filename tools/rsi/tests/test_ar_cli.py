"""CLI tests for the `ar` round-lifecycle verbs (M4). No GPU needed."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from vllm_evolve.cli.main import build_parser
from vllm_evolve.cli.main import main as ar_main
from vllm_evolve.tools import phase_guard
from vllm_evolve.tools.phase_guard import Phase

# Legal transition chain from INIT (matches phase_guard._TRANSITIONS).
_CHAIN = [
    Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
    Phase.VERIFY_PASSED, Phase.BENCHMARK, Phase.KEEP_OR_DISCARD,
    Phase.COMMIT_OR_ROLLBACK,
]


def _advance_to(target: Phase) -> None:
    phase_guard.reset_state()  # -> INIT
    for ph in _CHAIN:
        phase_guard.transition(ph)
        if ph == target:
            return


@pytest.fixture(autouse=True)
def _reset_phase():
    phase_guard.reset_state()
    yield
    phase_guard.reset_state()


def _eval_json(tmp, name, vals, metric="goodput_req_s", runner_kind="strong_baseline"):
    from vllm_evolve.bench.config import build_bench_config
    bc = build_bench_config(runner_kind=runner_kind, model="facebook/opt-125m").to_dict()
    p = tmp / name
    p.write_text(
        json.dumps({
            # LOCK D: only a clean real run (source + outcome_class) may be compared/kept
            "source": "real_vllm", "outcome_class": "eval_result",
            "primary_metric": metric,
            "raw_per_seed_metrics": [{"primary_value": v} for v in vals],
            "bench_config": bc,   # same-caliber provenance for the A/B gate
        }),
        encoding="utf-8",
    )
    return str(p)


def _formal_acceptance(policy_source: str, tmp_path) -> dict:
    from vllm_evolve.bench.eval_result import (
        acceptance_evidence_sha256,
        build_acceptance_artifact_manifest,
    )
    from vllm_evolve.engine.real_evolve_suite import evaluate_suite

    scenarios = (
        "burstgpt_saturated",
        "burstgpt_high_pressure",
        "burstgpt_severe_pressure",
    )
    workload = {
        "required": True,
        "valid": True,
        "verdict": "valid_saturated_real_vllm",
    }
    policy_sha = hashlib.sha256(policy_source.encode()).hexdigest()
    control_sha = "c" * 64

    def result(value: float, *, policy: str | None = None) -> dict:
        row = {
            "source": "real_vllm",
            "outcome_class": "eval_result",
            "primary_metric": "goodput_req_s",
            "aggregate_metrics": {"median": value, "cv": 0.0},
            "raw_per_seed_metrics": [
                {
                    "seed": seed,
                    "primary_value": value,
                    "metrics": {
                        "num_requests": 512,
                        "num_completed": 512,
                        "num_failed": 0,
                    },
                }
                for seed in (0, 1, 2)
            ],
            "workload_validity": workload,
        }
        if policy:
            row.update({
                "policy_sha256": policy,
                "marker_verified": True,
                "effective": True,
                "plugin_provenance": {
                    "fallback": False,
                    "deferred_request_actions": 3,
                },
            })
        return row

    raw_pairs = {
        scenario: {
            "baseline": result(100.0),
            "candidate": result(105.0, policy=policy_sha),
        }
        for scenario in scenarios
    }
    raw_controls = {
        scenario: result(100.0, policy=control_sha) for scenario in scenarios
    }
    artifact_root = tmp_path / "suite_artifacts"
    quality_dir = artifact_root / "quality_evidence"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "measurement.json").write_text(
        json.dumps({"baseline_perplexity": 10.0, "candidate_perplexity": 10.0}),
        encoding="utf-8",
    )

    def write_eval(name: str, value: dict) -> str:
        path = artifact_root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    pair_records = {}
    control_records = {}
    eval_paths = []
    for scenario in scenarios:
        baseline_path = write_eval(f"{scenario}_baseline", raw_pairs[scenario]["baseline"])
        candidate_path = write_eval(f"{scenario}_candidate", raw_pairs[scenario]["candidate"])
        control_path = write_eval(f"{scenario}_control", raw_controls[scenario])
        eval_paths.extend((baseline_path, candidate_path, control_path))
        pair_records[scenario] = {
            "baseline": {
                "eval_result": raw_pairs[scenario]["baseline"],
                "local_eval_path": baseline_path,
            },
            "candidate": {
                "eval_result": raw_pairs[scenario]["candidate"],
                "local_eval_path": candidate_path,
            },
        }
        control_records[scenario] = {
            "eval_result": raw_controls[scenario],
            "local_eval_path": control_path,
        }
    acceptance = evaluate_suite(
        raw_pairs,
        quality_ok=True,
        controls=raw_controls,
        require_mechanism_control=True,
    )
    suite = {
        "source": "real_vllm",
        "outcome_class": "real_acceptance_suite",
        "policy_sha256": policy_sha,
        "control_sha256": control_sha,
        "required_action_counters": ["deferred_request_actions"],
        "pairs": pair_records,
        "controls": control_records,
        "quality": {
            "ok": True,
            "reasons": [],
            "measured": {
                "baseline": {
                    "perplexity": 10.0,
                    "task_em": 0.8,
                    "output_agreement": 1.0,
                },
                "candidate": {
                    "perplexity": 10.0,
                    "task_em": 0.8,
                    "output_agreement": 1.0,
                },
            },
            "evidence_dirs": [str(quality_dir)],
        },
        "acceptance": acceptance,
    }
    suite["artifact_manifest"] = build_acceptance_artifact_manifest(
        artifact_root,
        eval_result_paths=eval_paths,
        quality_evidence_dirs=[str(quality_dir)],
    )
    suite["evidence_sha256"] = acceptance_evidence_sha256(suite)
    return suite


# ── parser / help ─────────────────────────────────────────────

def test_help_lists_all_verbs(capsys):
    with pytest.raises(SystemExit):
        ar_main(["--help"])
    out = capsys.readouterr().out
    for verb in ["phase", "verify", "bench", "context", "design", "compare", "keep", "discard"]:
        assert verb in out


def test_build_parser_registers_new_verbs():
    parser = build_parser()
    # the subparser action holds the verb choices
    choices = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    for verb in ["context", "design", "compare", "keep", "discard"]:
        assert verb in choices


# ── phase gating ──────────────────────────────────────────────

def test_context_rejected_from_init(capsys):
    rc = ar_main(["context", "scheduling"])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["outcome_class"] == "phase_violation"


def test_compare_rejected_outside_keep_phase(tmp_path, capsys):
    _advance_to(Phase.DESIGN)  # wrong phase for compare
    base = _eval_json(tmp_path, "b.json", [10.0, 10.0, 10.0])
    cand = _eval_json(tmp_path, "c.json", [12.0, 12.0, 12.0])
    rc = ar_main(["compare", base, cand])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "phase_violation"


# ── context ───────────────────────────────────────────────────

def test_context_ok_in_read_context(tmp_path, capsys):
    _advance_to(Phase.READ_CONTEXT)
    rc = ar_main(["context", "scheduling", "--db", str(tmp_path / "t.db")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert "schedule_batch" in out["evolvable_functions"]
    assert "def schedule_batch" in out["seed"]


# ── design ────────────────────────────────────────────────────

def test_design_records_note(tmp_path, capsys):
    _advance_to(Phase.DESIGN)
    out_path = tmp_path / "design.md"
    rc = ar_main(["design", "--note", "try priority by wait time", "scheduling",
                  "--out", str(out_path)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert "try priority by wait time" in out_path.read_text(encoding="utf-8")


# ── compare (3-state verdict) ─────────────────────────────────

def test_compare_better(tmp_path, capsys):
    _advance_to(Phase.KEEP_OR_DISCARD)
    base = _eval_json(tmp_path, "b.json", [10.0, 10.2, 9.8, 10.1, 9.9])
    cand = _eval_json(tmp_path, "c.json", [14.0, 14.2, 13.8, 14.1, 13.9])
    rc = ar_main(["compare", base, cand, "--seed", "1"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "better"


def test_compare_inconclusive_when_overlapping(tmp_path, capsys):
    _advance_to(Phase.KEEP_OR_DISCARD)
    base = _eval_json(tmp_path, "b.json", [10.0, 10.5, 9.5, 10.2, 9.8])
    cand = _eval_json(tmp_path, "c.json", [10.1, 10.4, 9.6, 10.1, 9.9])
    rc = ar_main(["compare", base, cand, "--seed", "3"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "inconclusive"


def test_compare_rejects_missing_caliber_provenance(tmp_path, capsys):
    # Codex R2: eval_results without bench_config provenance -> unverifiable, not a pass
    _advance_to(Phase.KEEP_OR_DISCARD)
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    for f in (b, c):
        f.write_text(json.dumps({"source": "real_vllm", "outcome_class": "eval_result",
                     "primary_metric": "goodput_req_s",
                     "raw_per_seed_metrics": [{"primary_value": 10.0}]}), encoding="utf-8")
    rc = ar_main(["compare", str(b), str(c)])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "same_caliber_unverifiable"


def test_compare_rejects_caliber_mismatch(tmp_path, capsys):
    # candidate ran a DIFFERENT model -> not the same caliber -> rejected
    from vllm_evolve.bench.config import build_bench_config
    _advance_to(Phase.KEEP_OR_DISCARD)
    base = _eval_json(tmp_path, "b.json", [10.0, 10.0, 10.0])
    cand = tmp_path / "c.json"
    bc = build_bench_config(runner_kind="candidate", model="other/model-7b").to_dict()
    cand.write_text(json.dumps({"source": "real_vllm", "outcome_class": "eval_result",
                    "primary_metric": "goodput_req_s",
                    "raw_per_seed_metrics": [{"primary_value": 12.0}],
                    "bench_config": bc}), encoding="utf-8")
    rc = ar_main(["compare", base, str(cand)])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["outcome_class"] == "same_caliber_mismatch" and out["diffs"]


# ── keep (store + git commit) ─────────────────────────────────

def test_keep_archives_and_commits(tmp_path, monkeypatch, capsys):
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    monkeypatch.chdir(tmp_path)
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)

    policy = tmp_path / "work.py"
    policy_source = "def schedule_batch():\n    return None\n"
    policy.write_text(policy_source, encoding="utf-8")
    suite = _formal_acceptance(policy_source, tmp_path)
    ev = Path(suite["pairs"]["burstgpt_saturated"]["candidate"]["local_eval_path"])
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(suite), encoding="utf-8")

    rc = ar_main(["keep", str(policy), "scheduling", "--eval-result", str(ev),
                  "--acceptance-evidence", str(acceptance),
                  "--archive-root", "archive_policies", "--db", str(tmp_path / "t.db")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kept"] is True
    assert out["committed"]  # a commit sha was produced
    arch = tmp_path / "archive_policies" / "scheduling" / out["run_id"]
    assert (arch / "policy.py").exists()
    assert (arch / "eval_result.json").exists()
    assert (arch / "acceptance_evidence.json").exists()


def test_keep_rejects_real_eval_without_formal_acceptance(tmp_path, capsys):
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    ev = tmp_path / "eval.json"
    ev.write_text(
        json.dumps({"source": "real_vllm", "outcome_class": "eval_result"}),
        encoding="utf-8",
    )
    rc = ar_main([
        "keep", str(policy), "scheduling", "--eval-result", str(ev),
        "--archive-root", str(tmp_path / "arch"), "--db", str(tmp_path / "t.db"),
    ])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == (
        "real_acceptance_evidence_required"
    )


def test_adoption_recomputes_raw_suite_and_rejects_forged_summary(tmp_path):
    from vllm_evolve.bench.eval_result import (
        acceptance_evidence_sha256,
        real_adoption_block,
    )

    source = "def schedule_batch():\n    return None\n"
    policy_sha = hashlib.sha256(source.encode()).hexdigest()
    suite = _formal_acceptance(source, tmp_path)
    for pair in suite["pairs"].values():
        pair["candidate"]["eval_result"].pop("raw_per_seed_metrics")
    suite["evidence_sha256"] = acceptance_evidence_sha256(suite)
    block = real_adoption_block(
        suite["pairs"]["burstgpt_saturated"]["candidate"]["eval_result"],
        suite,
        policy_sha256=policy_sha,
        eval_result_path=suite["pairs"]["burstgpt_saturated"]["candidate"][
            "local_eval_path"
        ],
    )
    assert block is not None
    assert block["outcome_class"] == "real_acceptance_evidence_blocked"


def test_keep_rejected_outside_commit_phase(tmp_path, capsys):
    _advance_to(Phase.KEEP_OR_DISCARD)  # compare phase, not commit
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    rc = ar_main(["keep", str(policy)])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "phase_violation"


# ── keep real-eval gate (P1d) ─────────────────────────────────

def test_keep_rejects_missing_eval_result(tmp_path, capsys):
    # default keep REQUIRES a clean real eval_result; absent -> rejected (not archived)
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    rc = ar_main(["keep", str(policy), "scheduling",
                  "--archive-root", str(tmp_path / "arch"), "--db", str(tmp_path / "t.db")])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "eval_result_required"


def test_keep_rejects_unreadable_eval_result(tmp_path, capsys):
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    rc = ar_main(["keep", str(policy), "scheduling", "--eval-result", str(bad),
                  "--archive-root", str(tmp_path / "arch"), "--db", str(tmp_path / "t.db")])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "eval_result_unreadable"


def test_keep_rejects_local_smoke_eval(tmp_path, capsys):
    # a synthetic local_smoke result can never be kept (LOCK D / real_source_block)
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    ev = tmp_path / "smoke.json"
    ev.write_text(json.dumps({"source": "local_smoke", "outcome_class": "eval_result"}),
                  encoding="utf-8")
    rc = ar_main(["keep", str(policy), "scheduling", "--eval-result", str(ev),
                  "--archive-root", str(tmp_path / "arch"), "--db", str(tmp_path / "t.db")])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["outcome_class"] == "non_real_source_blocked"


def test_keep_manual_archives_without_gain_credit(tmp_path, monkeypatch, capsys):
    # --manual: archive-only escape hatch; succeeds with NO eval, but never a gain
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    monkeypatch.chdir(tmp_path)
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    policy = tmp_path / "work.py"
    policy.write_text("def schedule_batch():\n    return None\n", encoding="utf-8")
    rc = ar_main(["keep", str(policy), "scheduling", "--manual",
                  "--archive-root", "archive_policies", "--db", str(tmp_path / "t.db")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kept"] is True and out["manual"] is True
    assert out["archive_only"] is True and out["gain_credit"] is False
    arch = tmp_path / "archive_policies" / "scheduling" / out["run_id"]
    assert (arch / "policy.py").exists()
    assert not (arch / "eval_result.json").exists()   # manual writes no eval_result


# ── discard ───────────────────────────────────────────────────

def test_discard(capsys):
    _advance_to(Phase.COMMIT_OR_ROLLBACK)
    rc = ar_main(["discard", "work.py", "--reason", "regressed latency"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["discarded"] is True
    assert out["reason"] == "regressed latency"


def test_bench_remote_success_emits_eval_result(monkeypatch, tmp_path, capsys):
    # Codex review P1: a successful REMOTE ve bench must emit the eval_result + artifact paths
    # (RemoteBench.to_dict) so the next ve compare/keep has the JSON — not just provenance.
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench
    monkeypatch.delenv("VE_BENCH_BACKEND", raising=False)
    phase_guard.reset_state()
    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY, Phase.VERIFY_PASSED):
        phase_guard.transition(ph)
    monkeypatch.setattr(d, "run_remote_bench_config", lambda c: RemoteBench(
        {"source": "real_vllm", "primary_metric": "goodput_req_s",
         "raw_per_seed_metrics": [{"primary_value": 7.0}]},
        "serve log", str(tmp_path / "e.json"), str(tmp_path / "l.log"),
        remote_cmd="ssh box native"))
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    rc = ar_main(["bench", str(pol), "scheduling", "--runner", "strong_baseline"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["result"]["eval_result"]["source"] == "real_vllm"
    assert out["result"]["local_eval_path"] == str(tmp_path / "e.json")
    assert out["result"]["remote_cmd"] == "ssh box native"
    phase_guard.reset_state()


def test_calibrate_reads_real_profile_signals(monkeypatch, capsys):
    # Codex review P1: calibrate must read throughput + GPU saturation from the PROFILE, not the
    # absent top-level eval_result fields -> sweep points carry real signals (not zeros), and a
    # saturated GPU yields client_saturated=False.
    import vllm_evolve.engine.profile as prof_mod
    from vllm_evolve.core.schemas import Profile
    phase_guard.reset_state()

    def fake_collect(cfg):
        return Profile(config=cfg, metrics={"tok_s": 1234.0},
                       gpu={"sm_util_max": 97.0, "mem_used_mb": 39000.0, "mem_total_mb": 40000.0},
                       vllm={"kv_util": 0.8, "preempt": 0}, saturated=True,
                       eval_result={"source": "real_vllm", "rendered_serve_args": ["--x"],
                                    "remote_cmd": "ssh box"}, remote_cmd="ssh box")

    monkeypatch.setattr(prof_mod, "collect_profile", fake_collect)
    ar_main(["calibrate"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    ok_pts = [p for p in out["evidence"]["points"] if p["status"] == "ok"]
    assert ok_pts, out
    sig = ok_pts[0]["signals"]
    assert sig["tok_s"] == 1234.0 and sig["sm_util_max"] == 97.0
    assert sig["client_saturated"] is False     # saturated GPU -> client was NOT the limit
    phase_guard.reset_state()
