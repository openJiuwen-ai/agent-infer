"""AC6: the supported surface is truthful — exactly ONE target (scheduling) is wired to the real
backend, the supported target's config carries no simulator evaluator, and the round verbs refuse
any other target instead of silently running scheduling (zero GPU).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.tools import phase_guard  # noqa: E402
from vllm_evolve.tools.phase_guard import Phase  # noqa: E402

_TARGETS_DIR = REPO_ROOT / "config" / "targets"


@pytest.fixture(autouse=True)
def _reset_phase():
    phase_guard.reset_state()
    yield
    phase_guard.reset_state()


def _yaml(p: Path) -> dict:
    import yaml
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return doc.get("spec", doc)


def test_exactly_one_supported_target_is_scheduling():
    supported = [p.stem for p in _TARGETS_DIR.glob("*.yaml")
                 if _yaml(p).get("maturity") == "supported"]
    assert supported == ["scheduling"], supported


def test_no_kept_yaml_references_the_deleted_simulator():
    # the deleted DES simulator must not linger in ANY kept target config (Codex R0 AC6)
    offenders = [p.name for p in _TARGETS_DIR.glob("*.yaml")
                 if _yaml(p).get("evaluator") == "simulator"]
    assert offenders == [], offenders


def test_ar_bench_rejects_non_scheduling_target(capsys):
    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
               Phase.VERIFY_PASSED):
        phase_guard.transition(ph)
    seed = str(REPO_ROOT / "targets" / "scheduling" / "seed.py")
    rc = cli_main.main(["bench", seed, "kv_eviction", "--runner", "strong_baseline",
                      "--backend", "local_smoke"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "unsupported_target"
    assert out["target"] == "kv_eviction"


def test_ar_context_rejects_non_scheduling_target(capsys):
    phase_guard.transition(Phase.READ_CONTEXT)
    rc = cli_main.main(["context", "spec_decode"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "unsupported_target"


def test_target_registry_is_central_source_of_truth():
    # P4 step 1: the single-target rule lives in targets/registry (pure), reused everywhere
    from vllm_evolve.targets import registry
    assert registry.is_supported("scheduling") is True
    assert registry.rejection_payload("scheduling") is None
    assert registry.rejection_payload("") is None            # default/unset -> not rejected here
    bad = registry.rejection_payload("kv_eviction")
    assert bad and bad["outcome_class"] == "unsupported_target" and bad["target"] == "kv_eviction"


def test_design_keep_discard_reject_non_scheduling(tmp_path, capsys):
    # P4 step 1: design/keep/discard previously lacked the target check; now central + enforced
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")

    phase_guard.transition(Phase.READ_CONTEXT)
    phase_guard.transition(Phase.DESIGN)
    rc = cli_main.main(["design", "--note", "x", "kv_eviction", "--out", str(tmp_path / "d.md")])
    assert rc == 2 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "outcome_class"] == "unsupported_target"

    for ph in (Phase.GENERATE, Phase.VERIFY, Phase.VERIFY_PASSED, Phase.BENCHMARK,
               Phase.KEEP_OR_DISCARD, Phase.COMMIT_OR_ROLLBACK):
        phase_guard.transition(ph)
    rc = cli_main.main(["keep", str(pol), "kv_eviction", "--archive-root", str(tmp_path / "a")])
    assert rc == 2 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "outcome_class"] == "unsupported_target"
    rc = cli_main.main(["discard", str(pol), "spec_decode", "--reason", "x",
                      "--archive-root", str(tmp_path / "a")])
    assert rc == 2 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
        "outcome_class"] == "unsupported_target"


def test_ar_discard_writes_audit_row(tmp_path, capsys):
    # M6: `ve discard` appends an auditable row (policy hash + reason + timestamp + target/run).
    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
               Phase.VERIFY_PASSED, Phase.BENCHMARK, Phase.KEEP_OR_DISCARD,
               Phase.COMMIT_OR_ROLLBACK):
        phase_guard.transition(ph)
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    arch = tmp_path / "arch"
    rc = cli_main.main(["discard", str(pol), "scheduling", "--reason", "no gain",
                      "--archive-root", str(arch)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["discarded"] is True
    audit = arch / "scheduling" / "discarded.jsonl"
    assert audit.exists()
    row = json.loads(audit.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["reason"] == "no gain" and row["policy_sha256"] and row["timestamp"]
    assert row["target"] == "scheduling"


def test_no_kept_config_references_deleted_skills():
    # M6: root skills/ was deleted; no kept config may point at a missing skills/*.md path.
    import re
    offenders = []
    for y in (REPO_ROOT / "config").rglob("*.yaml"):
        for ref in re.findall(r"skills/[\w./-]+\.md", y.read_text(encoding="utf-8")):
            if not (REPO_ROOT / ref).exists():
                offenders.append(f"{y.relative_to(REPO_ROOT)}: {ref}")
    assert offenders == [], offenders


def test_no_kept_source_config_has_simulator_or_strategy_residue():
    # M6 (Codex R3): kept Python config modules must not re-introduce the deleted simulator default,
    # the deleted strategy-skill namespace, or a missing skills/*.md ref.
    import re
    cfg_modules = [REPO_ROOT / "src" / "vllm_evolve" / "config.py"]
    offenders = []
    for f in cfg_modules:
        text = f.read_text(encoding="utf-8")
        if re.search(r"""evaluator\s*:\s*str\s*=\s*["']simulator["']""", text):
            offenders.append(f"{f.name}: evaluator default 'simulator'")
        if re.search(r"""strategy\s*[:=].*["']diff_mutation["']""", text):
            offenders.append(f"{f.name}: stale strategy 'diff_mutation'")
        for ref in re.findall(r"skills/[\w./-]+\.md", text):
            if not (REPO_ROOT / ref).exists():
                offenders.append(f"{f.name}: missing skills ref {ref}")
    # the deleted duplicate target-spec module must not come back
    dup = REPO_ROOT / "src" / "vllm_evolve" / "targets_spec.py"
    assert not dup.exists(), "stale duplicate targets_spec.py is back"
    assert offenders == [], offenders
