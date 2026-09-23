"""``ar`` — short CLI for the vllm-evolve harness.

Verbs: the phase-locked round (``ve phase`` / ``context`` / ``design`` / ``verify`` /
``bench`` / ``compare`` / ``keep`` / ``discard``) over REAL vLLM, plus the autopt
layer (``goal`` / ``profile`` [+ ``--cuda``] / ``diagnose`` / ``targets`` / ``optimize`` /
``calibrate`` / ``verify-gain`` / ``autopt``) and ``ve init``. ``ve bench`` is a real
BenchConfig-backed runner (vanilla / strong_baseline / candidate); when the GPU box is
unreachable it reports ``box_gated_blocked`` with provenance — never a fabricated pass.
Every verb routes through ``phase_guard.check_or_fail`` before non-trivial work, regardless
of how it is invoked (CLI, Python import, or scaffold).

Output is always a single JSON object on stdout. Errors are returned via
non-zero exit codes:

* ``0`` — success
* ``2`` — phase violation or safety rejection (rejected input, not an error)
* ``1`` — unexpected runtime error
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from vllm_evolve.targets import registry as _target_registry
from vllm_evolve.tools import phase_guard
from vllm_evolve.tools.phase_guard import Phase, PhaseError

# The round-lifecycle verb->phase mapping (context/design/compare/keep/discard) now lives in the
# canonical table in phase_guard._VERB_PHASES (M4) — no import-time registration here, so the hook
# no longer needs to import this module to know those phases.


# Repo root resolved from the source location; the skeleton-parity check
# loads ``targets/<target>/skeleton.py`` to compare arg names.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # cli/main.py -> cli -> vllm_evolve -> src -> repo


# Pattern matched by ``verify_tool.verify_code``: nested for-loops over the
# same iterable, skipping list/dict/set comprehensions (those have ``[`` or
# ``{`` ahead of the ``for`` token, so they sit on the same indented line).
_NESTED_LOOP_PATTERN = re.compile(
    r"^[ \t]+for\s+\w+\s+in\s+(\w+).*\n.*?^[ \t]+for\s+\w+\s+in\s+\1",
    re.MULTILINE,
)


# Heuristic for fabricated request IDs. Authoritative detection requires
# running the scheduler and comparing outputs against inputs (a runtime
# check that lands in ``ve bench``); ve verify only flags string literals
# that look obviously fake. Tokens chosen to be conservative -- they catch
# the obvious test fixture without crying wolf on legitimate code.
_FABRICATED_ID_PATTERN = re.compile(
    r"""['"](?:FAKE|fake|dummy|sample|placeholder|stub|test_req)_?\w*['"]""",
)


# --- verify helpers --------------------------------------------------------


def _load_target_spec(target_name: str):
    """Best-effort target spec load. Returns ``None`` if the target is
    missing or the YAML is malformed; downstream callers treat ``None``
    as "skip the spec-dependent check"."""

    try:
        from vllm_evolve.config import load_target_spec

        return load_target_spec(target_name)
    except Exception:
        return None


def _load_expected_functions(target_name: str) -> list[str]:
    spec = _load_target_spec(target_name)
    if spec is None:
        return []
    try:
        return [fn["name"] for fn in spec.evolvable_functions]
    except Exception:
        return []


def _parse_or_none(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _check_return_annotations(
    code: str, expected_functions: Iterable[str]
) -> list[str]:
    """Reject if any expected evolvable function has no return annotation."""

    tree = _parse_or_none(code)
    if tree is None:
        return []
    expected = set(expected_functions)
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            if node.returns is None:
                issues.append(
                    f"L1: function {node.name!r} is missing the return "
                    "type annotation required by the target skeleton"
                )
    return issues


def _skeleton_signatures(target_name: str) -> dict[str, list[str]]:
    """Parse ``targets/<target>/skeleton.py`` and return
    ``{function_name: [arg_name, ...]}`` for each evolvable function the
    skeleton defines."""

    spec = _load_target_spec(target_name)
    if spec is None:
        return {}
    skel_relative = getattr(spec, "skeleton_path", None)
    if not skel_relative:
        return {}
    skel_path = _REPO_ROOT / skel_relative
    if not skel_path.exists():
        return {}
    try:
        skel_source = skel_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    tree = _parse_or_none(skel_source)
    if tree is None:
        return {}
    expected = set(_load_expected_functions(target_name))
    sigs: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            sigs[node.name] = [a.arg for a in node.args.args]
    return sigs


def _check_signature_parity(code: str, target: str) -> list[str]:
    """Reject if a policy defines an evolvable function with arg names
    that diverge from the skeleton's. Covers the AC-5 signature-mismatch
    category at the parity level (the trust.safety.check_signatures
    function only covers the name-missing case)."""

    skel_sigs = _skeleton_signatures(target)
    if not skel_sigs:
        return []
    tree = _parse_or_none(code)
    if tree is None:
        return []
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in skel_sigs:
            actual = [a.arg for a in node.args.args]
            expected = skel_sigs[node.name]
            if actual != expected:
                issues.append(
                    "L2: signature of "
                    f"{node.name!r} does not match skeleton "
                    f"(expected args {expected}, got {actual})"
                )
    return issues


def _check_fabricated_ids(code: str) -> list[str]:
    """Heuristic L2 check for fabricated request IDs.

    A match indicates the policy hardcodes a request-id-shaped string
    literal (e.g. ``"FAKE_REQ_001"``) that almost certainly cannot
    correspond to a real input request_id. The authoritative runtime
    check happens in ``ve bench``; this heuristic catches the obvious
    static case so the bench is not wasted on a known-bad policy.
    """

    if _FABRICATED_ID_PATTERN.search(code):
        return [
            "L2: heuristic match for fabricated request_id literal -- "
            "request_ids must come from input objects. Authoritative "
            "check lands in ve bench at runtime; this static heuristic "
            "catches the obvious 'FAKE_*' / 'dummy_*' shape"
        ]
    return []


def _verify_code(code: str, target: str) -> list[str]:
    """Run L1 (safety + return-annotation) + L2 (signature parity, nested
    loop, fabricated-id heuristic). Returns the list of issue strings.
    Empty list means the policy passed every static check."""

    from vllm_evolve.trust.safety import (
        check_reserved_class_redefinitions,
        check_safety,
        check_signatures,
    )

    issues: list[str] = []

    safety = check_safety(code)
    if not safety.ok:
        issues.append(f"L1: {safety.reason}")

    expected = _load_expected_functions(target)
    if expected:
        sig = check_signatures(code, expected)
        if not sig.ok:
            issues.append(f"L2: {sig.reason}")
        issues.extend(_check_return_annotations(code, expected))

    issues.extend(_check_signature_parity(code, target))

    if target == "scheduling":
        bridge_types = check_reserved_class_redefinitions(
            code,
            frozenset({"RequestInfo", "ScheduleDecision"}),
        )
        if not bridge_types.ok:
            issues.append(f"L2: {bridge_types.reason}")

    for var in _NESTED_LOOP_PATTERN.findall(code):
        issues.append(f"L2: Potential O(n^2) nested loop over '{var}'")

    issues.extend(_check_fabricated_ids(code))

    return issues


# --- verbs -----------------------------------------------------------------


def cmd_phase(args: argparse.Namespace) -> int:
    phase_guard.check_or_fail("phase")  # administrative; returns current

    if args.action == "status":
        current = phase_guard.read_phase()
        print(json.dumps({"ok": True, "phase": current.value}))
        return 0

    if args.action == "init":
        phase_guard.reset_state()
        current = phase_guard.read_phase()
        print(json.dumps({"ok": True, "action": "init", "phase": current.value}))
        return 0

    if args.action == "set":
        try:
            target = Phase(args.target)
        except ValueError:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "outcome_class": "unknown_phase",
                        "given": args.target,
                        "known": sorted(p.value for p in Phase),
                    }
                )
            )
            return 2
        try:
            phase_guard.transition(target)
        except PhaseError as exc:
            print(json.dumps(exc.to_dict()))
            return 2
        print(json.dumps({"ok": True, "action": "set", "phase": target.value}))
        return 0

    raise NotImplementedError(args.action)


SUPPORTED_TARGET = _target_registry.DEFAULT_TARGET


def _reject_unsupported_target(target: str) -> int | None:
    """Refuse a round verb on any non-supported target — central check in
    ``targets/registry.py`` (CLAUDE.md: only ``scheduling`` is wired to the real backend)."""
    payload = _target_registry.rejection_payload(target)
    if payload is not None:
        print(json.dumps(payload))
        return 2
    return None


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("verify")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc

    code_path = Path(args.policy)
    if not code_path.exists():
        print(
            json.dumps(
                {
                    "ok": False,
                    "outcome_class": "policy_not_found",
                    "policy_path": str(code_path),
                }
            )
        )
        return 2
    code = code_path.read_text(encoding="utf-8")

    issues = _verify_code(code, args.target)
    if issues:
        print(
            json.dumps(
                {
                    "ok": False,
                    "outcome_class": "safety_rejection",
                    "issues": issues,
                    "policy_path": str(code_path),
                    "target": args.target,
                }
            )
        )
        return 2

    phase_guard.mark_verify_passed()
    print(
        json.dumps(
            {
                "ok": True,
                "outcome_class": "verify_pass",
                "policy_path": str(code_path),
                "target": args.target,
            }
        )
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Real, BenchConfig-backed bench runner (phase-gated by VERIFY_PASSED).

    Builds a single ``BenchConfig`` (the execution source of truth) for the chosen
    runner kind (``vanilla`` / ``strong_baseline`` / ``candidate``), renders the full
    serve command (all wired levers), runs the anti-forgery check on candidate policy
    source, then invokes the BenchConfig-consuming backend. When the box is
    unreachable the run is reported as ``box_gated_blocked`` with full provenance +
    the attempted command — NEVER a fabricated pass.
    """
    from vllm_evolve.bench.config import CANDIDATE, build_bench_config

    try:
        phase_guard.check_or_fail("bench")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc

    runner_kind = getattr(args, "runner", None) or CANDIDATE
    scheduler_cls = "generated_scheduler.EvolvedScheduler" if runner_kind == CANDIDATE else None
    eng = {k: getattr(args, k, None) for k in (
        "quantization", "kv_cache_dtype", "quantized_model_id", "model_artifact_kind",
        "max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization",
        "max_model_len", "tensor_parallel_size", "enable_prefix_caching",
        "enable_chunked_prefill", "enforce_eager")}
    workload = {k: getattr(args, k, None) for k in (
        "trace_path", "concurrency", "n_requests", "load_mode")}
    config = build_bench_config(
        runner_kind=runner_kind, policy_path=args.policy,
        model=getattr(args, "model", None) or "facebook/opt-125m", scheduler_cls=scheduler_cls,
        profile=getattr(args, "profile", None) or "throughput",
        gpus=getattr(args, "gpus", None) or "0",
        port=getattr(args, "port", None) or 8200,
        max_seeds=getattr(args, "max_seeds", None),
        # None when --remote was not given -> RunnerConfig honors VE_REMOTE (Codex review P2)
        remote=getattr(args, "remote", None),
        **{k: v for k, v in {**eng, **workload}.items() if v is not None})
    prov = config.provenance()

    # Anti-fabrication (Codex HIGH#3): a candidate policy must not forge provenance markers.
    if runner_kind == CANDIDATE and args.policy and Path(args.policy).exists():
        from vllm_evolve.bench.runtime import contains_marker_forgery
        if contains_marker_forgery(Path(args.policy).read_text(encoding="utf-8")):
            print(json.dumps({"ok": False, "outcome_class": "policy_marker_forgery",
                              "runner_kind": runner_kind, "provenance": prov}))
            return 2

    # Backend seam: default 'remote' is the real SSH/vLLM path; 'local_smoke' (synthetic in-process)
    # and 'frontier_sim' (out-of-process Frontier simulator) are both NON-REAL plumbing backends
    # (quarantined — can never be a real gain/keep/AC6).
    from vllm_evolve.bench.local_smoke import run_local_smoke, selected_backend
    backend = selected_backend(args)
    try:
        if backend == "local_smoke":
            result = run_local_smoke(config)
        elif backend == "frontier_sim":
            from vllm_evolve.bench.frontier_sim import run_frontier_sim
            result = run_frontier_sim(config)
        else:
            from vllm_evolve.bench.dispatch import run_remote_bench_config
            result = run_remote_bench_config(config)
    except Exception as exc:  # noqa: BLE001 - honest remote/bench failure surface
        print(json.dumps({
            "ok": False, "outcome_class": "box_gated_blocked", "runner_kind": runner_kind,
            "backend": backend, "error": str(exc)[-300:], "model_served": prov["model_served"],
            "attempted_serve_args": prov["rendered_serve_args"], "provenance": prov}))
        return 1
    # Remote ``gpus=auto`` is bound in-place by the dispatch boundary.  Emit the
    # resolved config rather than the pre-discovery request in success output.
    prov = config.provenance()
    out = {"ok": True, "runner_kind": runner_kind, "backend": backend, "provenance": prov}
    if backend in ("local_smoke", "frontier_sim"):
        out["banner"] = result.banner
    if hasattr(result, "to_dict"):
        out["result"] = result.to_dict()
    print(json.dumps(out))
    return 0


# --- round-lifecycle verbs (M4) -------------------------------------------


def _first_evolvable_fn(target: str) -> str:
    """First evolvable function name from config/targets/<target>.yaml."""
    import yaml

    cfg = _REPO_ROOT / "config" / "targets" / f"{target}.yaml"
    if not cfg.exists():
        return "unknown"
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        spec = data.get("spec", data)
        fns = spec.get("evolvable_functions", [])
        if fns and isinstance(fns[0], dict):
            return fns[0].get("name", "unknown")
    except Exception:
        pass
    return "unknown"


def _git_commit(path: str, message: str) -> str | None:
    """Best-effort ``git add <path> && git commit``. Returns short SHA or None."""
    import subprocess

    try:
        subprocess.run(["git", "add", path], check=True, capture_output=True)
        r = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return sha.stdout.strip()[:12] if sha.returncode == 0 else "committed"
    except Exception:
        return None


def cmd_context(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("context")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc

    target = args.target
    tdir = _REPO_ROOT / "targets" / target
    if not tdir.is_dir():
        print(json.dumps({"ok": False, "outcome_class": "target_not_found", "target": target}))
        return 2

    def _read(p: Path) -> str:
        return p.read_text(encoding="utf-8") if p.exists() else ""

    fns: list[str] = []
    cfg = _REPO_ROOT / "config" / "targets" / f"{target}.yaml"
    if cfg.exists():
        import yaml
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            spec = data.get("spec", data)
            fns = [f["name"] for f in spec.get("evolvable_functions", [])
                   if isinstance(f, dict) and "name" in f]
        except Exception:
            fns = []

    best: list = []
    try:
        from vllm_evolve.store.db import Store
        store = Store(args.db) if args.db else Store()
        best = store.best_policies(target, n=3)
        store.close()
    except Exception:
        best = []

    runtime_contract = None
    if target == "scheduling":
        from vllm_evolve.engine.scheduling_contract import load_scheduling_author_contract

        runtime_contract = load_scheduling_author_contract(backend="frontier")

    print(json.dumps({
        "ok": True, "phase": Phase.READ_CONTEXT.value, "target": target,
        "evolvable_functions": fns,
        "skeleton": (
            runtime_contract["rendered"]
            if runtime_contract is not None
            else _read(tdir / "skeleton.py")
        ),
        "static_skeleton": _read(tdir / "skeleton.py"),
        "runtime_contract": runtime_contract,
        "seed": _read(tdir / "seed.py"),
        "prompt_hints": _read(tdir / "prompt_hints.md"),
        "best": best,
    }, default=str))
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("design")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"\n## Design note ({args.target})\n\n{args.note}\n")
    print(json.dumps({
        "ok": True, "phase": Phase.DESIGN.value,
        "design_path": str(out), "note": args.note, "target": args.target,
    }))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("compare")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2

    def _load(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    try:
        base = _load(args.baseline)
        cand = _load(args.candidate)
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "outcome_class": "eval_result_unreadable",
                          "error": str(exc)}))
        return 2

    # LOCK D: a comparison may consume ONLY real-vLLM results. A synthetic local_smoke artifact
    # (or any non-real source) is HARD-REFUSED here — it can never become a gain.
    from vllm_evolve.bench.eval_result import real_source_block
    for ev in (base, cand):
        blk = real_source_block(ev)
        if blk is not None:
            print(json.dumps(blk))
            return 2

    # AC1: enforce SAME CALIBER before comparing (Codex R2). A candidate may only be
    # compared against a baseline that ran the same口径; missing BenchConfig provenance
    # is NOT a pass — it is same_caliber_unverifiable.
    from vllm_evolve.bench.config import BenchConfig, same_caliber
    bc_b, bc_c = base.get("bench_config"), cand.get("bench_config")
    if not bc_b or not bc_c:
        print(json.dumps({"ok": False, "outcome_class": "same_caliber_unverifiable",
                          "hint": "both eval_results must carry bench_config provenance "
                                  "(re-run via ve bench so the A/B caliber is enforceable)"}))
        return 2
    cal_ok, cal_diffs = same_caliber(BenchConfig.from_dict(bc_b), BenchConfig.from_dict(bc_c))
    if not cal_ok:
        print(json.dumps({"ok": False, "outcome_class": "same_caliber_mismatch",
                          "diffs": cal_diffs}))
        return 2

    def _vals(ev: dict) -> list[float]:
        return [r["primary_value"] for r in ev.get("raw_per_seed_metrics", [])
                if isinstance(r, dict) and r.get("primary_value") is not None]

    bvals, cvals = _vals(base), _vals(cand)
    if not bvals or not cvals:
        print(json.dumps({"ok": False, "outcome_class": "invalid_metrics",
                          "hint": "both eval_results need raw_per_seed_metrics[].primary_value"}))
        return 2

    metric = cand.get("primary_metric") or base.get("primary_metric") or "goodput_req_s"
    higher_is_better = not metric.endswith("_ms")  # latency metrics are lower-better

    from vllm_evolve.bench.compare import compare_metric

    cmp = compare_metric(
        metric, bvals, cvals, higher_is_better=higher_is_better,
        epsilon_pct=args.epsilon, cv_threshold=args.cv_threshold, seed=args.seed,
    )
    out = cmp.to_dict()
    out.update({"ok": True, "phase": Phase.KEEP_OR_DISCARD.value})
    print(json.dumps(out))
    return 0


def cmd_keep(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("keep")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(json.dumps({"ok": False, "outcome_class": "policy_not_found",
                          "policy_path": str(policy_path)}))
        return 2
    source = policy_path.read_text(encoding="utf-8")
    import hashlib

    # A kept WINNER must carry a clean real eval_result. `--manual` is the only
    # escape hatch: an archive-only keep that never counts as a gain / AC6.
    manual = getattr(args, "manual", False)
    ev: dict = {}
    acceptance_evidence: dict = {}
    if not manual:
        if not args.eval_result:
            print(json.dumps({"ok": False, "outcome_class": "eval_result_required",
                              "hint": "keep needs --eval-result with a real_vllm result; "
                                      "use --manual for an archive-only keep (never a gain)"}))
            return 2
        try:
            with open(args.eval_result, encoding="utf-8") as f:
                ev = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(json.dumps({"ok": False, "outcome_class": "eval_result_unreadable",
                              "eval_result": args.eval_result}))
            return 2
        # LOCK D: never archive a non-real eval_result as a kept winner (a synthetic
        # local_smoke result can never be "kept").
        from vllm_evolve.bench.eval_result import real_source_block
        blk = real_source_block(ev)
        if blk is not None:
            print(json.dumps(blk))
            return 2
        if not args.acceptance_evidence:
            print(json.dumps({
                "ok": False,
                "outcome_class": "real_acceptance_evidence_required",
                "hint": (
                    "keep requires --acceptance-evidence from the formal three-scenario "
                    "real-vLLM suite; use --manual for archive-only"
                ),
            }))
            return 2
        try:
            acceptance_evidence = json.loads(
                Path(args.acceptance_evidence).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            print(json.dumps({
                "ok": False,
                "outcome_class": "real_acceptance_evidence_unreadable",
                "acceptance_evidence": args.acceptance_evidence,
            }))
            return 2

        from vllm_evolve.bench.eval_result import real_adoption_block

        adoption_block = real_adoption_block(
            ev,
            acceptance_evidence,
            policy_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            eval_result_path=args.eval_result,
        )
        if adoption_block is not None:
            print(json.dumps(adoption_block))
            return 2

    run_id = args.run_id or hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    archive_root = Path(args.archive_root)
    run_dir = archive_root / args.target / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "policy.py").write_text(source, encoding="utf-8")
    if ev:
        (run_dir / "eval_result.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
        (run_dir / "acceptance_evidence.json").write_text(
            json.dumps(acceptance_evidence, indent=2), encoding="utf-8"
        )

    policy_id = None
    try:
        from vllm_evolve.store.db import Store
        store = Store(args.db) if args.db else Store()
        policy_id = store.put_policy(source, _first_evolvable_fn(args.target), args.target)
        if ev:
            store.put_eval("policy", policy_id, ev.get("profile", "?"), "real_vllm",
                           metrics=ev.get("aggregate_metrics"), wall_time_s=ev.get("wall_time_s"))
        store.close()
    except Exception:
        policy_id = None

    committed = _git_commit(str(archive_root), f"keep {args.target} {run_id}")
    print(json.dumps({
        "ok": True, "phase": Phase.COMMIT_OR_ROLLBACK.value, "kept": True,
        "manual": manual, "archive_only": manual, "gain_credit": (not manual),
        "policy_id": policy_id, "run_id": run_id,
        "archive_dir": str(run_dir), "committed": committed,
    }))
    return 0


def cmd_discard(args: argparse.Namespace) -> int:
    try:
        phase_guard.check_or_fail("discard")
    except PhaseError as exc:
        print(json.dumps(exc.to_dict()))
        return 2
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc
    import datetime
    import hashlib

    policy_path = Path(args.policy)
    source = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    policy_sha = hashlib.sha256(source.encode("utf-8")).hexdigest() if source else ""
    target = getattr(args, "target", None) or "scheduling"
    run_id = getattr(args, "run_id", None) or (policy_sha[:12] if policy_sha else "unknown")
    archive_root = Path(getattr(args, "archive_root", None) or "archive_policies")
    target_dir = archive_root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    audit = target_dir / "discarded.jsonl"
    row = {
        "discarded": True, "policy_path": str(policy_path), "policy_sha256": policy_sha,
        "reason": args.reason, "target": target, "run_id": run_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(audit, "a", encoding="utf-8") as f:    # append-only audit trail (never overwrites)
        f.write(json.dumps(row) + "\n")
    print(json.dumps({
        "ok": True, "phase": Phase.COMMIT_OR_ROLLBACK.value, "discarded": True,
        "policy": args.policy, "reason": args.reason, "run_id": run_id, "audit": str(audit),
    }))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Install / uninstall / check Claude Code and/or Codex assets.

    Administrative (no phase gate): this is project setup, not a round step.
    """
    from vllm_evolve.install import codex_installer, installer

    client = getattr(args, "client", "claude")
    installers = {
        "claude": installer,
        "codex": codex_installer,
    }

    def _one(name: str):
        selected = installers[name]
        if args.uninstall:
            return selected.uninstall(args.dir)
        if args.check:
            return selected.check(args.dir)
        result = selected.install(args.dir, force=args.force)
        return {
            "status": result.status,
            "target": result.target,
            "created_files": result.created_files,
            "hook_path": result.hook_path,
        }

    if client == "both":
        print(json.dumps({"ok": True, "client": "both",
                          "results": {name: _one(name) for name in ("claude", "codex")}}))
    else:
        print(json.dumps({"ok": True, "client": client, **_one(client)}))
    return 0


# --- parser ---------------------------------------------------------------


def cmd_frontier_evolve(args: argparse.Namespace) -> int:
    """Run the local-only real-Frontier generational evolution workflow."""
    phase_guard.register_verb("frontier-evolve", None)
    phase_guard.check_or_fail("frontier-evolve")
    from vllm_evolve.engine.local_frontier_evolve import run_local_frontier_evolution

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    author_fn = None
    if getattr(args, "author_command", None):
        from vllm_evolve.engine.authoring import command_source_author

        author_fn = command_source_author(args.author_command, author_kind="codex")
    from vllm_evolve.store.db import Store

    store = Store(args.db) if getattr(args, "db", None) else Store()
    try:
        result = run_local_frontier_evolution(
            burstgpt_path=args.burstgpt,
            out_dir=args.out,
            seeds=seeds,
            fragment_size=args.fragment_size,
            slo_ttft_ms=args.slo_ttft_ms,
            max_num_seqs=args.max_num_seqs,
            generations=args.generations,
            population=args.population,
            max_total_evals=args.max_total_evals,
            author_fn=author_fn,
            research_snapshot=getattr(args, "research_snapshot", None),
            research_mode=getattr(args, "research_mode", "auto"),
            research_refresh=bool(getattr(args, "research_refresh", False)),
            store=store,
            goal=args.goal,
            resume=bool(args.resume),
        )
    except Exception as exc:  # noqa: BLE001 - honest local simulator failure surface
        print(json.dumps({
            "ok": False,
            "outcome_class": "frontier_local_evolution_failed",
            "source": "frontier_sim",
            "remote_executed": False,
            "error": str(exc)[-1000:],
        }))
        return 1
    finally:
        store.close()
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def cmd_real_evolve(args: argparse.Namespace) -> int:
    """Run the fail-closed real-vLLM-only generational controller."""
    phase_guard.register_verb("real-evolve", Phase.DESIGN)
    phase_guard.check_or_fail("real-evolve")
    from vllm_evolve.bench.config import BenchConfig
    from vllm_evolve.engine.authoring import command_source_author
    from vllm_evolve.engine.real_evolution import run_real_evolution
    from vllm_evolve.knowledge.compiler import load_research_context

    try:
        bench_config = BenchConfig.from_dict(
            json.loads(Path(args.bench_config).read_text(encoding="utf-8"))
        )
        requested_gpus = getattr(args, "gpus", "auto")
        if requested_gpus != "config":
            bench_config.environment.gpus = requested_gpus
            bench_config.environment.gpu_selection_mode = "fixed"
            bench_config.environment.gpu_selection_evidence = {}
        research = load_research_context(args.research_snapshot)
        author_fn = command_source_author(
            args.author_command,
            timeout_s=args.author_timeout_s,
            author_kind="codex",
        )
        result = run_real_evolution(
            author_fn=author_fn,
            baseline_config=bench_config,
            research=research,
            out_dir=args.out,
            generations=args.generations,
            population=args.population,
            max_total_evals=args.max_total_evals,
            repair_limit=args.repair_limit,
        )
    except Exception as exc:  # noqa: BLE001 - preserve an honest blocked/error surface
        print(
            json.dumps(
                {
                    "ok": False,
                    "source": "real_vllm",
                    "outcome_class": "real_evolution_failed",
                    "error": f"{type(exc).__name__}: {exc}"[-2000:],
                }
            )
        )
        return 1
    ok = result.get("status") == "real_winner_proposal"
    print(json.dumps({"ok": ok, **result}))
    return 0 if ok else 2


def _json_object_arg(value: str | None, *, default: dict | None = None) -> dict:
    if value is None:
        return dict(default or {})
    try:
        text = value
        if not value.lstrip().startswith(("{", "[")):
            path = Path(value)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object/path {value!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def _json_list_arg(value: str | None) -> list:
    if value is None:
        return []
    try:
        text = value
        if not value.lstrip().startswith(("{", "[")):
            path = Path(value)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON list/path {value!r}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
    return parsed


def cmd_research(args: argparse.Namespace) -> int:
    """Research is an internal/admin step, not a fourth optimization mode."""
    phase_guard.register_verb("research", None)
    phase_guard.check_or_fail("research")
    from vllm_evolve.knowledge import (
        ResearchCompiler,
        gather,
        live_required_compiler,
        load_research_context,
        offline_compiler,
    )

    try:
        if args.action == "real-scheduler":
            from vllm_evolve.engine.real_research import (
                build_real_scheduler_research,
            )

            result = build_real_scheduler_research(
                out_dir=args.out,
                installed_vllm_commit=args.installed_vllm_commit,
                environment=_json_object_arg(args.environment),
                refresh=args.refresh,
            )
            context = result.context
            print(
                json.dumps(
                    {
                        "ok": True,
                        "outcome_class": context.outcome_class,
                        "snapshot_id": context.snapshot_id,
                        "snapshot_hash": context.snapshot_hash,
                        "artifact_dir": result.artifact_dir,
                        "source_count": len(context.sources),
                        "mechanism_count": len(context.mechanism_cards),
                        "mechanism_ids": [
                            card.mechanism_id for card in context.mechanism_cards
                        ],
                        "upstream_main_ref": context.upstream_capability_matrix.get(
                            "upstream_main_ref"
                        ),
                        "citation_check": context.validate_citations(),
                    }
                )
            )
            return 0

        if args.action == "build":
            from vllm_evolve.intent.spec import parse_goal

            spec = parse_goal(args.goal).to_dict()
            diagnosis = _json_object_arg(
                args.diagnosis,
                default={"bottleneck": args.target, "status": "research_requested"},
            )
            environment = _json_object_arg(args.environment)
            if (
                args.target == "scheduling"
                and environment.get("source") == "real_vllm"
                and args.mode == "live-required"
            ):
                from vllm_evolve.engine.real_research import (
                    build_real_scheduler_research,
                )

                installed_commit = str(
                    environment.get("installed_vllm_commit") or ""
                )
                if not installed_commit:
                    raise ValueError(
                        "real-vLLM scheduling research requires "
                        "environment.installed_vllm_commit"
                    )
                result = build_real_scheduler_research(
                    out_dir=args.out,
                    installed_vllm_commit=installed_commit,
                    environment=environment,
                    spec=spec,
                    diagnosis=diagnosis,
                    refresh=args.refresh,
                )
            else:
                try:
                    internal = gather(
                        args.target,
                        db=args.db,
                        n=args.internal_limit,
                        regime=environment.get("profile"),
                        metric=spec.get("metric"),
                    )
                except Exception as exc:  # noqa: BLE001 - research can proceed without DB
                    internal = {
                        "lessons": [],
                        "hypotheses": [],
                        "knowledge_status": (
                            f"unavailable: {type(exc).__name__}: {exc}"
                        ),
                    }
                compiler = (
                    offline_compiler()
                    if args.mode == "offline"
                    else live_required_compiler()
                    if args.mode == "live-required"
                    else ResearchCompiler()
                )
                result = compiler.compile(
                    target=args.target,
                    spec=spec,
                    diagnosis=diagnosis,
                    environment=environment,
                    out_dir=args.out,
                    internal_evidence=internal,
                    refresh=args.refresh,
                )
            context = result.context
            print(json.dumps({
                "ok": True,
                "outcome_class": context.outcome_class,
                "snapshot_id": context.snapshot_id,
                "snapshot_hash": context.snapshot_hash,
                "artifact_dir": result.artifact_dir,
                "reused_frozen_snapshot": result.reused_frozen_snapshot,
                "source_count": len(context.sources),
                "mechanism_count": len(context.mechanism_cards),
                "provider_status": context.provider_status,
                "citation_check": context.validate_citations(),
            }))
            return 0

        if args.action == "verify":
            context = load_research_context(args.snapshot)
            check = context.validate_citations()
            requested = [
                item.strip() for item in (args.citations or "").split(",") if item.strip()
            ]
            known = {f"source:{source.source_id}" for source in context.sources}
            fabricated = sorted(set(requested) - known)
            payload = {
                "ok": check["ok"] and not fabricated,
                "snapshot_id": context.snapshot_id,
                "snapshot_hash": context.snapshot_hash,
                "citation_check": check,
                "requested_citations": requested,
                "fabricated_citations": fabricated,
            }
            print(json.dumps(payload))
            return 0 if payload["ok"] else 2

        if args.action == "export-author":
            from vllm_evolve.engine.authoring import export_author_bundle
            from vllm_evolve.engine.evolve_schemas import AuthorContext

            research = load_research_context(args.snapshot)
            skeleton_path = Path(args.skeleton) if args.skeleton else (
                _REPO_ROOT / "targets" / research.target / "skeleton.py"
            )
            context = AuthorContext(
                spec=_json_object_arg(args.spec, default=research.goal),
                diagnosis=_json_object_arg(args.diagnosis, default=research.diagnosis),
                skeleton=skeleton_path.read_text(encoding="utf-8"),
                parents=_json_list_arg(args.parents),
                peers=_json_list_arg(args.peers),
                lessons=_json_list_arg(args.lessons) or research.internal_lessons,
                research_context=research.to_dict(),
                last_errors=_json_list_arg(args.last_errors),
                budget=_json_object_arg(
                    args.budget,
                    default={"evals_remaining": 1, "repair_remaining": 0},
                ),
                generation=args.generation,
                author_kind=args.author_kind,
            )
            bundle = export_author_bundle(context, args.out)
            print(json.dumps({"ok": True, **bundle}))
            return 0

        if args.action == "import-candidate":
            from vllm_evolve.engine.authoring import import_author_submission
            from vllm_evolve.engine.evolve_schemas import AuthorContext

            payload = json.loads(Path(args.author_context).read_text(encoding="utf-8"))
            payload.pop("doctrine", None)
            context = AuthorContext(**payload)
            submission = import_author_submission(
                source_path=args.source,
                manifest_path=args.manifest,
                context=context,
            )
            manifest = (
                submission.manifest.to_dict()
                if hasattr(submission.manifest, "to_dict")
                else submission.manifest
            )
            out = Path(args.out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"source": submission.source, "manifest": manifest}, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({"ok": True, "response_path": str(out), "manifest": manifest}))
            return 0
    except Exception as exc:  # noqa: BLE001 - one honest JSON failure surface
        print(json.dumps({
            "ok": False,
            "outcome_class": "research_step_failed",
            "action": args.action,
            "error": str(exc)[-2000:],
        }))
        return 2
    print(json.dumps({"ok": False, "outcome_class": "unknown_research_action"}))
    return 2


def cmd_diagnose(args: argparse.Namespace) -> int:
    """L1: localize the bottleneck from a Profile JSON (pure, no GPU)."""
    phase_guard.register_verb("diagnose", None)
    phase_guard.check_or_fail("diagnose")
    from vllm_evolve.core.schemas import Profile
    from vllm_evolve.engine.diagnose import diagnose
    p = Path(args.profile)
    if not p.is_file():
        print(json.dumps({"ok": False, "outcome_class": "profile_not_found",
                          "path": str(p)}))
        return 2
    try:
        prof = Profile.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "outcome_class": "profile_unreadable",
                          "error": str(exc)[:200]}))
        return 2
    d = diagnose(prof)
    print(json.dumps({"ok": True, **d.to_dict()}))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Read-only store views + the cited-fact contract (ve-research support). No GPU, no mutation
    (SELECT only); usable in any phase. The orchestrator runs this to gather the knowledge base and
    to VERIFY a research summary's citations before trusting it."""
    phase_guard.register_verb("inspect", None)
    phase_guard.check_or_fail("inspect")
    from vllm_evolve.ops.inspect import inspect_top, verify_citations
    db = getattr(args, "db", None)
    if db:
        # explicit path -> fresh read-only handle. Refuse a non-existent path: a read verb must
        # NEVER materialize a stray empty store (a typo'd --db would otherwise rubber-stamp).
        if not Path(db).exists():
            print(json.dumps({"ok": False, "outcome_class": "store_not_found", "db": db}))
            return 2
        from vllm_evolve.store.db import Store
        store = Store(db)
    else:
        from vllm_evolve.tools.store_tool import get_store
        store = get_store()
    cited = getattr(args, "verify_citations", None)
    if cited is not None:
        ids = [s.strip() for s in cited.split(",") if s.strip()]
        print(json.dumps({"ok": True, **verify_citations(store, ids)}))
        return 0
    if getattr(args, "top", None):
        print(json.dumps({"ok": True, **inspect_top(store, args.top, args.n)}))
        return 0
    print(json.dumps({"ok": False, "outcome_class": "inspect_needs_a_selector",
                      "hint": "pass --top or --verify-citations"}))
    return 2


def cmd_targets(args: argparse.Namespace) -> int:
    """L2: rank candidate interventions for a Diagnosis JSON (pure, no GPU)."""
    phase_guard.register_verb("targets", None)
    phase_guard.check_or_fail("targets")
    from vllm_evolve.core.schemas import Diagnosis
    from vllm_evolve.engine.targets import select_targets
    p = Path(args.diagnosis)
    if not p.is_file():
        print(json.dumps({"ok": False, "outcome_class": "diagnosis_not_found",
                          "path": str(p)}))
        return 2
    try:
        diag = Diagnosis.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "outcome_class": "diagnosis_unreadable",
                          "error": str(exc)[:200]}))
        return 2
    plan = select_targets(diag)
    print(json.dumps({"ok": True, **plan.to_dict()}))
    return 0


def cmd_goal(args: argparse.Namespace) -> int:
    """L0: parse a natural-language goal into a Spec (pure, no GPU)."""
    phase_guard.register_verb("goal", None)
    phase_guard.check_or_fail("goal")
    from vllm_evolve.intent.spec import goal_notes, parse_goal
    text = " ".join(args.text).strip()
    if not text:
        print(json.dumps({"ok": False, "outcome_class": "empty_goal"}))
        return 2
    spec = parse_goal(text)
    print(json.dumps({"ok": True, **spec.to_dict(), "notes": goal_notes(text)}))
    return 0


def _make_research_fn(args: argparse.Namespace, *, purpose: str, backend: str):
    """Build the profile→diagnosis→research callback used by autopt/run.

    The closure freezes one snapshot per round/target and keeps external research
    in the proposal layer.  It never calls a benchmark or adoption gate.
    """
    from vllm_evolve.knowledge import (
        ResearchCompiler,
        gather,
        live_required_compiler,
        load_research_context,
        offline_compiler,
    )

    explicit_snapshot = getattr(args, "research_snapshot", None)
    if explicit_snapshot:
        frozen = load_research_context(explicit_snapshot)

        def reuse_snapshot(**_kwargs):
            return frozen

        return reuse_snapshot

    mode = getattr(args, "research_mode", "auto")
    compiler = (
        offline_compiler()
        if mode == "offline"
        else live_required_compiler()
        if mode == "live-required"
        else ResearchCompiler()
    )
    root = Path(
        getattr(args, "research_out", None)
        or f"runs/auto_research/{purpose}"
    ).resolve()
    refresh = bool(getattr(args, "research_refresh", False))
    db = getattr(args, "research_db", None)

    def research_fn(*, spec, diagnosis, base_config, target, round_index):
        target_name = "scheduling" if target == "schedule_batch" else target
        try:
            internal = gather(
                target_name,
                db=db,
                n=5,
                regime=base_config.get("profile"),
                metric=spec.metric,
            )
        except Exception as exc:  # noqa: BLE001 - external expert brief still compiles
            internal = {
                "lessons": [],
                "hypotheses": [],
                "knowledge_status": f"unavailable: {type(exc).__name__}: {exc}",
            }
        environment = {
            **{
                key: value for key, value in base_config.items()
                if key in {
                    "model",
                    "gpus",
                    "profile",
                    "workload_spec",
                    "max_num_seqs",
                    "concurrency",
                    "n_requests",
                    "remote",
                }
            },
            "backend": backend,
            "vllm_version": base_config.get("vllm_version", "runtime-detected"),
        }
        result = compiler.compile(
            target=target_name,
            spec=spec.to_dict(),
            diagnosis=diagnosis,
            environment=environment,
            out_dir=root / f"round_{round_index}_{target_name}",
            internal_evidence=internal,
            refresh=refresh,
        )
        return result.context

    return research_fn


def cmd_autopt(args: argparse.Namespace) -> int:
    """Top: parse a goal and run the L1->L4 optimization loop on real vLLM."""
    phase_guard.register_verb("autopt", None)
    phase_guard.check_or_fail("autopt")
    from vllm_evolve.bench.local_smoke import selected_backend
    from vllm_evolve.engine.profile import (
        collect_profile,
        collect_profile_frontier,
        collect_profile_local_smoke,
    )
    from vllm_evolve.intent.spec import parse_goal
    text = " ".join(args.text).strip()
    if not text:
        print(json.dumps({"ok": False, "outcome_class": "empty_goal"}))
        return 2
    spec = parse_goal(text)
    base = {
        k: getattr(args, k) for k in (
            "model", "gpus", "gpu_memory_utilization", "max_num_seqs", "concurrency",
            "max_model_len", "n_requests", "max_seeds", "profile", "remote", "policy")
        if getattr(args, k, None) is not None
    }
    if getattr(args, "workload_spec", None):
        try:
            base["workload_spec"] = json.loads(args.workload_spec)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "outcome_class": "bad_workload_spec_json"}))
            return 2
    # the goal's metric drives what the bench reports (e.g. goodput needs the SLO path)
    base["primary_metric"] = spec.metric
    # PIN the model so the remote quality hook serves the SAME model the bench serves when --model
    # is omitted, same as `ve run` (Codex review P2).
    base["model"] = _run_default_model(base.get("model"))
    # Backend seam: local_smoke runs the WHOLE L1->L4 loop off-box (synthetic, -> DoD-B only).
    backend = selected_backend(args)
    eval_fn = {"local_smoke": collect_profile_local_smoke,
               "frontier_sim": collect_profile_frontier}.get(backend, collect_profile)
    evolve_fn = None
    author_fn = None
    evolve_params = None
    research_fn = None
    store = None
    lessons: list = []
    if args.evolve:   # enable evolving schedule_batch when scheduling is the bottleneck
        # The generational engine is backend-agnostic. Research/authoring proposes;
        # the selected eval_fn and existing gate remain the only measurement/judgment.
        from vllm_evolve.engine.evolve_target import template_author_fn
        from vllm_evolve.store.db import Store

        if getattr(args, "author_command", None):
            from vllm_evolve.engine.authoring import command_source_author

            author_fn = command_source_author(args.author_command, author_kind="codex")
        else:
            author_fn = template_author_fn
        store = Store()
        lessons = store.get_lessons(metric=spec.metric, limit=5)
        evolve_params = {
            "generations": args.generations, "population": args.population,
            "repair_limit": args.repair_limit, "lessons": lessons,
        }
        research_fn = _make_research_fn(
            args, purpose="autopt", backend=backend
        )
    from vllm_evolve.modes import autopt as autopt_mode
    # A real-vLLM (remote) candidate that clears bootstrap+holdout still needs MEASURED quality to
    # adopt; wire the same backend-selected hook `ve run` uses so the direct autopt entry is not a
    # dead path for real gains (off-box -> None -> NOT MEASURED -> DoD-B).
    res = autopt_mode.run(spec, eval_fn=eval_fn, evolve_fn=evolve_fn,
                          author_fn=author_fn, evolve_params=evolve_params, store=store,
                          research_fn=research_fn,
                          quality_measure_fn=_run_quality_measure_fn(backend, base),
                          base_config=base,
                          max_rounds=args.max_rounds, max_evals=args.max_evals,
                          run_holdout=not args.no_holdout,
                          accept_threshold_pct=getattr(args, "accept_threshold_pct", None),
                          backend=backend)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **res}))
    return 0


# ── ve run: the natural-language front door (declare 4 fields, system self-configures) ──

_RUN_NO_GPU_HW = {"", "none", "cpu", "no-gpu", "nogpu", "no_gpu"}
_RUN_TUNE_POLICY = "targets/scheduling/seed.py"   # default policy the tune path optimizes around


def _run_resolve_backend(hw: str, override: str | None) -> str:
    """Hardware -> backend. An explicit --backend wins; else none/cpu/empty -> the off-box
    simulator (frontier_sim, DoD-B), any real GPU spec -> remote (real vLLM)."""
    if override:
        return override
    return "frontier_sim" if (hw or "").strip().lower() in _RUN_NO_GPU_HW else "remote"


def _run_resolve_profile(metric: str) -> str:
    """Metric phrasing -> load profile (latency-tight / trace-replay / saturating throughput)."""
    m = (metric or "").lower()
    if "ttft" in m or "latency" in m:
        return "latency"
    if "trace" in m or "replay" in m:
        return "replay"
    return "throughput"


def _run_default_model(model: str | None) -> str:
    """The model to serve when --model is omitted — the SAME default build_bench_config uses, so the
    bench (collect_profile) and the remote quality probe serve the same model. Without this the
    quality hook gets no model -> None for every measurement -> remote DoD-A is unreachable."""
    from vllm_evolve.bench.config import EngineConfig
    return model or EngineConfig.model


def _run_dod_stamp(backend: str, result: dict) -> tuple[str, str]:
    """DoD level + honesty note, decided by the BACKEND (the quarantine boundary): any off-box
    backend is DoD-B and never a real gain — EVEN IF the underlying result says 'adopted' — and
    never touches the adoption gate. Only a real-vLLM (remote) adoption is DoD-A."""
    if backend in ("local_smoke", "frontier_sim"):
        return ("DoD-B",
                "off-box backend: scores rank search candidates only; a real-vLLM gain is "
                "box-gated (--hw <gpu> --backend remote). No adoption gate was touched.")
    if result.get("outcome") == "adopted" and result.get("adopted"):
        return ("DoD-A",
                "real vLLM: candidate adopted via the deterministic bootstrap+holdout accept gate.")
    return ("DoD-B", "real vLLM run with no adopted candidate (no provable gain).")


def _run_quality_measure_fn(backend: str, base: dict):
    """Backend-selected measured-quality hook for the adoption gate. remote -> the REAL box-gated
    served-model measure, so a real-vLLM win can clear quality and reach DoD-A. Off-box
    (frontier_sim/local_smoke) -> None: quality is NOT measured, nothing can adopt, run stays
    DoD-B. This only SUPPLIES the gate's quality input; the gate itself is unchanged."""
    if backend == "remote":
        from vllm_evolve.engine.quality_remote import make_remote_quality_measure_fn
        return make_remote_quality_measure_fn(base)
    return None


def _run_invoke_mode(method: str, backend: str, spec, eval_fn, base: dict,
                     args: argparse.Namespace) -> dict:
    """Call the underlying mode.run DIRECTLY (no subprocess) so the gate/provenance are reused."""
    from vllm_evolve.modes import autopt as autopt_mode
    from vllm_evolve.modes import tune as tune_mode
    # Real-GPU wins go through the EXISTING accept gate, which certifies quality only when a
    # measured-quality hook is supplied; remote gets the real box-gated measure, off-box gets None
    # (NOT MEASURED -> cannot adopt -> DoD-B). The gate is unchanged; we only feed its input.
    quality_measure_fn = _run_quality_measure_fn(backend, base)
    if method == "tune":
        return tune_mode.run(_RUN_TUNE_POLICY, spec, eval_fn=eval_fn,
                             quality_measure_fn=quality_measure_fn, base_config=base,
                             backend=backend)
    # evolve: ALWAYS drive the real generational engine (template author + Archive + repair) on
    # EVERY backend — the engine is backend-agnostic; eval_fn does the measuring (collect_profile
    # = real vLLM on remote, the simulator on frontier_sim, synthetic on local_smoke). Adoption
    # stays inside the existing bootstrap/holdout + LOCK-D gate (an off-box profile can never be
    # adopted, on metadata alone) and _run_dod_stamp keeps off-box runs DoD-B. The legacy no-op
    # default_evolve_fn is intentionally NOT used here: method=evolve must actually evolve.
    from vllm_evolve.engine.evolve_target import template_author_fn
    from vllm_evolve.store.db import Store
    store = Store()
    if getattr(args, "author_command", None):
        from vllm_evolve.engine.authoring import command_source_author

        author_fn = command_source_author(args.author_command, author_kind="codex")
    else:
        author_fn = template_author_fn
    evolve_params = {"generations": getattr(args, "generations", 2),
                     "population": getattr(args, "population", 3), "repair_limit": 1,
                     "lessons": store.get_lessons(metric=spec.metric, limit=5)}
    research_fn = _make_research_fn(args, purpose="run", backend=backend)
    return autopt_mode.run(spec, eval_fn=eval_fn, evolve_fn=None, author_fn=author_fn,
                           evolve_params=evolve_params, store=store,
                           research_fn=research_fn,
                           quality_measure_fn=quality_measure_fn, base_config=base,
                           backend=backend)


def cmd_run(args: argparse.Namespace) -> int:
    """ve run — the natural-language front door. Declare {model, hardware, metric, method} and the
    harness self-configures backend + load profile + evolve|tune path, runs the chosen mode, and
    emits ONE DoD-stamped verdict. Thin orchestration over autopt/tune mode.run: it never bypasses
    the adoption gate or real_source_block, and off-box backends stay DoD-B."""
    phase_guard.register_verb("run", None)          # admin verb, phase-independent (like autopt)
    phase_guard.check_or_fail("run")
    from vllm_evolve.engine.profile import (
        collect_profile,
        collect_profile_frontier,
        collect_profile_local_smoke,
    )
    from vllm_evolve.intent.spec import parse_goal
    metric = args.metric
    if isinstance(metric, list):
        metric = " ".join(metric)
    metric = (metric or "").strip()
    if not metric:
        print(json.dumps({"ok": False, "outcome_class": "empty_metric"}))
        return 2
    method = (args.method or "evolve").strip().lower()
    if method not in ("evolve", "tune"):
        print(json.dumps({"ok": False, "outcome_class": "unknown_method", "method": method}))
        return 2
    hw = (args.hw or "").strip()
    backend = _run_resolve_backend(hw, getattr(args, "backend", None))
    profile = _run_resolve_profile(metric)
    spec = parse_goal(metric)
    eval_fn = {"local_smoke": collect_profile_local_smoke,
               "frontier_sim": collect_profile_frontier}.get(backend, collect_profile)
    model = _run_default_model(args.model)        # pin so the bench + quality probe serve one model
    base: dict = {"primary_metric": spec.metric, "profile": profile, "model": model}
    # A load operating point: optional, but adoption (DoD-A) needs it — the accept gate's holdout
    # re-measures at a DIFFERENT wired operating point, so without one the holdout is skipped and a
    # win can never clear the gate (same as `ve autopt`). The 4 fields stay the headline.
    for _k in ("concurrency", "n_requests", "remote"):
        if getattr(args, _k, None) is not None:
            base[_k] = getattr(args, _k)
    declaration = {"model": model, "hardware": hw or "none", "metric": metric, "method": method}
    resolved = {"backend": backend, "profile": profile, "goal": spec.to_dict()}

    def _emit(verdict: dict) -> None:
        if args.out:
            Path(args.out).write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(json.dumps(verdict))

    # The front door promises ONE JSON verdict. A box-gated failure (the remote GPU host is
    # unreachable, serve startup fails, ...) must be reported as structured JSON, never a traceback
    # — same contract cmd_bench/cmd_profile keep. Failure means NOT MEASURED -> DoD-B, never DoD-A.
    try:
        result = _run_invoke_mode(method, backend, spec, eval_fn, base, args)
    except Exception as e:        # noqa: BLE001 - keep the single-JSON contract for box-gated failures
        _emit({"ok": False, "outcome_class": "run_failed", "declaration": declaration,
               "resolved": resolved, "error": repr(e), "dod_level": "DoD-B",
               "honesty_note": "the run failed before producing a measurement (e.g. the remote GPU "
                               "box was unreachable or serve failed); nothing measured/adopted."})
        return 1        # box-gated/runtime failure (exit 1), NOT a rejected-input (exit 2)
    dod_level, honesty = _run_dod_stamp(backend, result)
    _emit({"ok": True, "declaration": declaration, "resolved": resolved, "result": result,
           "dod_level": dod_level, "honesty_note": honesty})
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """tune mode: optimize a GIVEN policy's serving config (composite; gate-preserving)."""
    phase_guard.register_verb("tune", None)
    phase_guard.check_or_fail("tune")
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc
    policy = Path(args.policy)
    if not policy.exists():
        print(json.dumps({"ok": False, "outcome_class": "policy_not_found",
                          "policy_path": str(policy)}))
        return 2
    from vllm_evolve.bench.local_smoke import selected_backend
    from vllm_evolve.engine.profile import (
        collect_profile,
        collect_profile_frontier,
        collect_profile_local_smoke,
    )
    from vllm_evolve.intent.spec import parse_goal
    from vllm_evolve.modes import tune as tune_mode
    spec = parse_goal(" ".join(args.goal).strip() if args.goal else "maximize goodput")
    base = {
        k: getattr(args, k) for k in (
            "model", "gpus", "concurrency", "n_requests", "max_seeds", "profile", "remote")
        if getattr(args, k, None) is not None
    }
    backend = selected_backend(args)
    eval_fn = {"local_smoke": collect_profile_local_smoke,
               "frontier_sim": collect_profile_frontier}.get(backend, collect_profile)
    res = tune_mode.run(policy, spec, eval_fn=eval_fn, base_config=base,
                        max_rounds=args.max_rounds, max_evals=args.max_evals,
                        run_holdout=not args.no_holdout,
                        accept_threshold_pct=getattr(args, "accept_threshold_pct", None),
                        backend=backend)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **res}))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """runs ls / gc: browse and garbage-collect the runs/ working area (admin; no phase)."""
    from vllm_evolve.artifacts import layout
    runs_root = getattr(args, "runs_root", None) or "runs"
    if args.action == "ls":
        rows = layout.list_runs(runs_root, target=getattr(args, "target", None))
        print(json.dumps({"ok": True, "runs": rows, "count": len(rows)}))
        return 0
    if args.action == "gc":
        removed = layout.gc_runs(
            runs_root, keep_last=args.keep_last,
            older_than_days=getattr(args, "older_than_days", None),
            dry_run=args.dry_run)
        print(json.dumps({"ok": True, "removed": [r["run_id"] for r in removed],
                          "removed_count": len(removed), "dry_run": args.dry_run}))
        return 0
    print(json.dumps({"ok": False, "outcome_class": "unknown_runs_action"}))
    return 2


def _load_experiment_spec_doc(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_experiment_spec(doc: dict):
    """A spec JSON -> ExperimentSpec. Workload is a trace (template+params or inline rows);
    metrics are MetricExpr-shaped; arms are {name, policy_path?, knobs?}."""
    from vllm_evolve.engine.experiment import Arm, ExperimentSpec
    from vllm_evolve.engine.workload_synth import WorkloadSpec
    w = doc.get("workload", {})
    workload = WorkloadSpec(template=w.get("template"), params=w.get("params", {}),
                            inline_rows=w.get("inline_rows", []))
    arms = [Arm(name=a["name"], policy_path=a.get("policy_path"), knobs=a.get("knobs", {}))
            for a in doc.get("arms", [])]
    return ExperimentSpec(workload=workload, arms=arms, metrics=doc.get("metrics", []),
                          seeds=doc.get("seeds", [0]), knobs=doc.get("knobs", {}),
                          budget=int(doc.get("budget", 24)), knob_ranges=doc.get("knob_ranges", {}))


def cmd_experiment(args: argparse.Namespace) -> int:
    """ve experiment register / run / list — open-world hypothesis research (admin; no phase).

    ``run`` performs the REAL chain: register the prediction (prediction-first) -> run the R1
    experiment -> R5 adjudicate -> append R4 ledger execution+adjudication events. Ledger rows are
    PROPOSAL-layer; this verb never touches the adoption gate."""
    phase_guard.register_verb("experiment", None)          # admin verb, phase-independent
    phase_guard.check_or_fail("experiment")
    from vllm_evolve.store.db import Store
    store = Store(args.db) if getattr(args, "db", None) else Store()
    try:
        if args.action == "list":
            views = store.list_hypotheses(verdict=getattr(args, "verdict", None),
                                          limit=getattr(args, "limit", 50))
            print(json.dumps({"ok": True, "hypotheses": views, "count": len(views)}))
            return 0
        if args.action == "register":
            doc = _load_experiment_spec_doc(args.spec)
            eid = store.register_prediction(
                hypothesis_id=doc["hypothesis_id"], run_id=doc.get("run_id", ""),
                statement=doc.get("statement", ""), prediction=doc["prediction"],
                source=doc.get("source", "frontier_sim"))
            print(json.dumps({"ok": True, "registered": True,
                              "hypothesis_id": doc["hypothesis_id"], "prediction_event_id": eid}))
            return 0
        if args.action == "run":
            from vllm_evolve.engine import experiment_chain
            doc = _load_experiment_spec_doc(args.spec)
            res = experiment_chain.run_hypothesis(
                store, hypothesis_id=doc["hypothesis_id"], statement=doc.get("statement", ""),
                prediction=doc["prediction"], spec=_build_experiment_spec(doc),
                eval_fn=experiment_chain.frontier_catalog_eval,
                base_out_dir=getattr(args, "out", None) or "runs/experiments",
                run_id=doc.get("run_id", ""))
            print(json.dumps({"ok": True, **res}))
            return 0
        print(json.dumps({"ok": False, "outcome_class": "unknown_experiment_action"}))
        return 2
    finally:
        store.close()


def cmd_port(args: argparse.Namespace) -> int:
    """port mode: version x hardware regression/adaptation matrix -> port_report.json."""
    phase_guard.register_verb("port", None)
    phase_guard.check_or_fail("port")
    rc = _reject_unsupported_target(getattr(args, "target", SUPPORTED_TARGET))
    if rc is not None:
        return rc
    policy = Path(args.policy)
    if not policy.exists():
        print(json.dumps({"ok": False, "outcome_class": "policy_not_found",
                          "policy_path": str(policy)}))
        return 2
    from vllm_evolve.bench.local_smoke import selected_backend
    from vllm_evolve.engine.profile import (
        collect_profile,
        collect_profile_frontier,
        collect_profile_local_smoke,
    )
    from vllm_evolve.modes import port as port_mode
    versions = [v.strip() for v in (args.versions or "").split(",") if v.strip()]
    hardware = [h.strip() for h in (args.hardware or "").split(",") if h.strip()]
    backend = selected_backend(args)
    eval_fn = {"local_smoke": collect_profile_local_smoke,
               "frontier_sim": collect_profile_frontier}.get(backend, collect_profile)
    report = port_mode.run(policy, versions=versions, hardware=hardware, eval_fn=eval_fn,
                           target=args.target, backend=backend)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **report}))
    return 0


def cmd_verify_gain(args: argparse.Namespace) -> int:
    """L4: bootstrap + holdout + bottleneck-shift verdict for a candidate (no GPU)."""
    phase_guard.register_verb("verify-gain", None)
    phase_guard.check_or_fail("verify-gain")
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.core.verify import extract_values, verify_gain

    def _load(path):
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    base, cand = _load(args.baseline), _load(args.candidate)
    if base is None or cand is None:
        print(json.dumps({"ok": False, "outcome_class": "input_not_found_or_unreadable"}))
        return 2
    # LOCK D: the adoption boundary may consume ONLY real-vLLM results. A synthetic local_smoke
    # input is HARD-REFUSED before any gain math — it can never be verified into a gain.
    from vllm_evolve.bench.eval_result import real_source_block
    for ev in (base, cand):
        blk = real_source_block(ev)
        if blk is not None:
            blk["adopt"] = False
            print(json.dumps(blk))
            return 2
    # AC1: enforce SAME CALIBER at the adoption boundary too (Codex R3), not just ve compare.
    # Missing BenchConfig provenance is a hard fail — never a silent adopt.
    from vllm_evolve.bench.config import BenchConfig, same_caliber
    bc_b, bc_c = base.get("bench_config"), cand.get("bench_config")
    if not bc_b or not bc_c:
        print(json.dumps({"ok": False, "outcome_class": "same_caliber_unverifiable",
                          "adopt": False, "hint": "both inputs must carry bench_config "
                          "provenance (run via ve bench / run_remote_bench_config)"}))
        return 2
    # The knob(s) under test may legitimately differ (a config-optimization A/B); the user declares
    # them with --exempt-knob, mirroring the autopt loop's exempt=searched_knobs. Any OTHER drift
    # still trips same_caliber_mismatch (Codex review P2).
    exempt = set(getattr(args, "exempt_knob", None) or ())
    cal_ok, cal_diffs = same_caliber(BenchConfig.from_dict(bc_b), BenchConfig.from_dict(bc_c),
                                     exempt=exempt)
    if not cal_ok:
        print(json.dumps({"ok": False, "outcome_class": "same_caliber_mismatch",
                          "adopt": False, "diffs": cal_diffs}))
        return 2
    bvals = extract_values(base, args.metric)
    cvals = extract_values(cand, args.metric)
    # honest holdout needs BOTH baseline and candidate at the SAME different config
    h_base_vals = h_cand_vals = None
    if args.holdout_baseline and args.holdout_candidate:
        hb, hc = _load(args.holdout_baseline), _load(args.holdout_candidate)
        if hb is None or hc is None:
            print(json.dumps({"ok": False, "outcome_class": "input_not_found_or_unreadable",
                              "adopt": False, "which": "holdout"}))
            return 2
        # The holdout A/B must clear the SAME real-source + same-caliber boundary as the primary —
        # else a synthetic/local_smoke or mismatched holdout could satisfy holdout_ok and enable
        # adoption of an otherwise-real primary (Codex review P1).
        for ev in (hb, hc):
            blk = real_source_block(ev)
            if blk is not None:
                blk["adopt"] = False
                blk["which"] = "holdout"
                print(json.dumps(blk))
                return 2
        hbc_b, hbc_c = hb.get("bench_config"), hc.get("bench_config")
        if not hbc_b or not hbc_c:
            print(json.dumps({"ok": False, "outcome_class": "same_caliber_unverifiable",
                              "adopt": False, "which": "holdout"}))
            return 2
        h_ok, h_diffs = same_caliber(BenchConfig.from_dict(hbc_b), BenchConfig.from_dict(hbc_c),
                                     exempt=exempt)   # holdout A/B differs by the same knob
        if not h_ok:
            print(json.dumps({"ok": False, "outcome_class": "same_caliber_mismatch",
                              "adopt": False, "which": "holdout", "diffs": h_diffs}))
            return 2
        h_base_vals = extract_values(hb, args.metric)
        h_cand_vals = extract_values(hc, args.metric)
    spec = Spec(metric=args.metric, direction=args.direction)
    v = verify_gain(
        bvals, cvals, spec,
        # anti-fabrication: absent marker_verified == NOT verified (default False)
        candidate_verified=bool(cand.get("marker_verified", False)),
        holdout_baseline_values=h_base_vals or None,
        holdout_candidate_values=h_cand_vals or None, holdout_floor=args.holdout_floor,
        baseline_bottleneck=base.get("bottleneck"),
        candidate_bottleneck=cand.get("bottleneck"),
    )
    # Strict strong-baseline gate BINDS adoption (Codex HIGH): hard X% on point + CI-low,
    # frozen holdout, explicit effective (from probe/marker) + quality_ok (None if the
    # quality runner did not certify -> reject). 'better' alone cannot adopt.
    from vllm_evolve.core.accept import _DEFAULT_ACCEPT_THRESHOLD_PCT, accept_vs_strong_baseline
    # the accept-gate threshold is a CLI/config decision, never the (possibly agent-produced) Spec.
    threshold = getattr(args, "accept_threshold_pct", None) or _DEFAULT_ACCEPT_THRESHOLD_PCT
    strict = accept_vs_strong_baseline(
        bvals, cvals, threshold_pct=threshold, higher_is_better=(args.direction != "min"),
        # require EXPLICIT effective proof (invoked + reordered + no fallback); do NOT fall back to
        # marker_verified, which only proves the plugin LOADED, not that it changed admission (P1).
        candidate_effective=bool(cand.get("effective", False)),
        quality_ok=cand.get("quality_ok"),
        holdout_baseline=h_base_vals or None, holdout_candidate=h_cand_vals or None)
    print(json.dumps({"ok": True, "verify_gain": v.to_dict(), "strict_accept": strict.to_dict(),
                      "adopt": bool(v.verdict == "better" and strict.accepted)}))
    return 0


def _coerce_scalar(s: str):
    """Parse a search-space token as int, then float, else leave as string."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            continue
    return s


def cmd_optimize(args: argparse.Namespace) -> int:
    """L3: search a config knob's space on real vLLM -> best Candidate."""
    phase_guard.register_verb("optimize", None)
    phase_guard.check_or_fail("optimize")
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine.optimize import optimize
    from vllm_evolve.engine.profile import WIRED_KNOBS, collect_profile
    knob = args.target.split(":", 1)[1] if ":" in args.target else args.target
    if knob not in WIRED_KNOBS:
        # honesty: searching a knob the bench doesn't apply would run the SAME
        # config every trial and report a fake "verified winner".
        print(json.dumps({"ok": False, "outcome_class": "knob_not_wired", "knob": knob,
                          "supported": sorted(WIRED_KNOBS),
                          "detail": "this knob is not yet threaded through the real bench; "
                                    "searching it would not actually change the run"}))
        return 2
    space = [_coerce_scalar(x.strip()) for x in args.space.split(",") if x.strip()]
    if not space:
        print(json.dumps({"ok": False, "outcome_class": "empty_search_space"}))
        return 2
    spec = Spec(metric=args.metric, direction=args.direction)
    base = {
        k: getattr(args, k) for k in (
            "model", "gpus", "gpu_memory_utilization", "max_num_seqs", "concurrency",
            "max_model_len", "n_requests", "max_seeds", "profile", "remote", "policy",
            "port")
        if getattr(args, k, None) is not None
    }
    cand = optimize({knob: space}, base, spec, collect_profile,
                    strategy=args.strategy, max_evals=args.max_evals)
    out = {"ok": True, **cand.to_dict()}
    if args.out:
        Path(args.out).write_text(json.dumps(cand.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(out))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """L1: run a real bench while sampling GPU -> Profile (+ optional diagnose)."""
    phase_guard.register_verb("profile", None)
    phase_guard.check_or_fail("profile")
    from vllm_evolve.engine.profile import collect_profile
    config = {
        k: getattr(args, k) for k in (
            "model", "gpus", "gpu_memory_utilization", "max_num_seqs", "concurrency",
            "max_model_len", "n_requests", "max_seeds", "profile", "remote", "policy",
            "port")
        if getattr(args, k, None) is not None
    }
    try:
        prof = collect_profile(config)
    except Exception as exc:  # noqa: BLE001 - honest remote/bench failure
        print(json.dumps({"ok": False, "outcome_class": "profile_failed",
                          "error": str(exc)[-300:]}))
        return 1
    out = {"ok": True, **prof.to_dict()}
    if args.diagnose:
        from vllm_evolve.engine.diagnose import diagnose
        out["diagnosis"] = diagnose(prof).to_dict()
    if getattr(args, "cuda", None):
        # AC4: a separate SAME-CALIBER profiled diagnostic window. The collector reconstructs the
        # unprofiled run's BenchConfig and runs the SAME model + levers + remote under a profiler
        # wrapper (box-gated). No provenance -> degraded_no_provenance (never a generic profile).
        # The live run degrades honestly with the attempted command preserved — never fabricated.
        import os
        import tempfile

        from vllm_evolve.bench.cuda_profile import collect_cuda_profile, make_cuda_collector
        run_dir = (os.path.dirname(os.path.abspath(args.out)) if args.out
                   else tempfile.mkdtemp(prefix="ve_cuda_"))
        if not prof.bench_config:
            out["cuda_profile"] = {
                "profiler": args.cuda, "status": "degraded_no_provenance",
                "error": "the unprofiled profile carries no bench_config; cannot run a "
                         "same-caliber CUDA window (Codex R7)"}
        else:
            from vllm_evolve.bench.config import BenchConfig
            from vllm_evolve.engine.cuda_parse import parse_torch_trace
            bc = BenchConfig.from_dict(prof.bench_config)
            try:
                collect_fn, attempted = make_cuda_collector(args.cuda, bench_config=bc,
                                                            run_dir=run_dir)
            except Exception as exc:  # noqa: BLE001 - candidate policy unreadable -> degrade
                out["cuda_profile"] = {
                    "profiler": args.cuda, "status": "degraded_staging",
                    "error": f"could not stage candidate policy: {str(exc)[-200:]}"}
            else:
                cuda = collect_cuda_profile(
                    args.cuda, collect_fn=collect_fn, attempted_cmd=attempted,
                    parse_fn=parse_torch_trace if args.cuda == "torch" else None,
                    remote=bc.runner.remote)
                out["cuda_profile"] = cuda.to_dict()
                if cuda.breakdown:
                    out["cuda_breakdown"] = cuda.breakdown
    if args.out:
        # persist the FULL output (incl. the cuda payload), not just the base Profile (Codex R6)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """AC2: drive a real saturation sweep to lock the max-throughput single-card口径.

    Sweeps a rising closed-loop load against the strong baseline, measures each point,
    and locks the highest-throughput saturated + plateaued operating point. Box
    unreachable -> every point fails -> honest box_gated_blocked (never fabricated).
    """
    phase_guard.register_verb("calibrate", None)
    phase_guard.check_or_fail("calibrate")
    from vllm_evolve.bench.config import STRONG_BASELINE
    from vllm_evolve.engine.calibrate import run_calibration_sweep
    from vllm_evolve.engine.profile import collect_profile

    base = {
        "model": getattr(args, "model", None) or "facebook/opt-125m",
        "gpus": getattr(args, "gpus", None) or "0",
        # None when --remote not given -> build_bench_config/RunnerConfig honors VE_REMOTE (P2)
        "remote": getattr(args, "remote", None)}

    def eval_fn(cfg: dict) -> dict:
        # Profile the strong-baseline at this sweep point: collect_profile runs the bench AND
        # samples the GPU, so the saturation signals (sm_util / mem / kv) actually exist — the raw
        # run_remote_bench eval_result alone has none of them, and nests throughput under
        # aggregate_metrics (Codex review P1). Box unreachable -> raises -> failed point.
        prof = collect_profile({**cfg, "runner_kind": STRONG_BASELINE})
        er = prof.eval_result or {}
        agg = er.get("aggregate_metrics") or {}
        tok_s = prof.metrics.get("tok_s") or agg.get("median") or 0.0
        gpu, vllm = prof.gpu, prof.vllm
        # client_saturated: the CLIENT was the limit iff the GPU was NOT driven near saturation
        # (the server had spare capacity). prof.saturated = sustained-high SM util / near-full mem.
        client_saturated = (not prof.saturated) if prof.saturated is not None else None
        return {
            "tok_s": float(tok_s),    # already coalesced to a number above
            "sm_util_max": float(gpu.get("sm_util_max") or 0.0),
            "mem_used_mb": float(gpu.get("mem_used_mb") or 0.0),
            "mem_total_mb": float(gpu.get("mem_total_mb") or 1.0),
            "kv_util": vllm.get("kv_util"), "preempt": vllm.get("preempt"),
            "client_saturated": client_saturated,
            # provenance for per-point evidence + freezing the strong baseline (Codex R3)
            "_eval_result": er, "_rendered_serve_args": er.get("rendered_serve_args"),
            "_remote_cmd": er.get("remote_cmd") or prof.remote_cmd}

    result = run_calibration_sweep(base, eval_fn)
    payload = {
        "ok": result.saturated,
        "outcome_class": ("calibrated" if result.saturated
                          else ("box_gated_blocked" if result.evidence.get("sweep_failures")
                                else "not_saturated")),
        "locked_config": result.locked_config, "locked_tok_s": result.locked_tok_s,
        "reasons": result.reasons, "evidence": result.evidence}
    if result.saturated and getattr(args, "out", None):
        # FROZEN strong-baseline artifact (Codex R3): the binding A/B opponent for every
        # candidate. Carries the locked config + its full eval_result (with bench_config
        # provenance) + rendered command + selection evidence.
        locked_er = result.evidence.get("locked_eval_result") or {}
        frozen = {
            "artifact": "frozen_strong_baseline",
            "locked_config": result.locked_config, "locked_tok_s": result.locked_tok_s,
            "bench_config": locked_er.get("bench_config"),
            "eval_result": locked_er, "rendered_serve_args": locked_er.get("rendered_serve_args"),
            "remote_cmd": locked_er.get("remote_cmd"), "evidence": result.evidence}
        Path(args.out).write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        payload["frozen_strong_baseline_path"] = args.out
    print(json.dumps(payload))
    return 0 if result.saturated else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ve",
        description="vllm-evolve harness CLI: phase-locked real-vLLM rounds, the autopt layer, "
        "and quarantined local real-Frontier generational search.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_phase = sub.add_parser("phase", help="Inspect or mutate the phase machine state.")
    p_phase_sub = p_phase.add_subparsers(dest="action", required=True)
    p_phase_sub.add_parser("status", help="Print current phase as JSON.")
    p_phase_sub.add_parser("init", help="Reset phase state to INIT.")
    p_phase_set = p_phase_sub.add_parser("set", help="Transition to a specific phase.")
    p_phase_set.add_argument("target", help="Target phase name.")
    p_phase.set_defaults(handler=cmd_phase)

    p_verify = sub.add_parser("verify", help="L1+L2 verify a policy file.")
    p_verify.add_argument("policy", help="Path to policy .py file.")
    p_verify.add_argument(
        "target",
        nargs="?",
        default="scheduling",
        help="Target name (default: scheduling).",
    )
    p_verify.set_defaults(handler=cmd_verify)

    p_bench = sub.add_parser(
        "bench",
        help="Run a real BenchConfig-backed bench (vanilla/strong_baseline/candidate); "
        "box-unreachable reports box_gated_blocked with provenance.",
    )
    p_bench.add_argument("policy", help="Path to policy .py file (candidate runner).")
    p_bench.add_argument("target", nargs="?", default="scheduling",
                         help="Target name (default: scheduling).")
    p_bench.add_argument("--runner", choices=["vanilla", "strong_baseline", "candidate"],
                         default="candidate", help="Runner kind (default: candidate).")
    p_bench.add_argument("--backend", choices=["remote", "local_smoke", "frontier_sim"],
                         help="remote=real SSH/vLLM (default); local_smoke=synthetic off-box "
                              "plumbing (NEVER a real result). Also via VE_BENCH_BACKEND.")
    p_bench.add_argument("--profile", default="throughput", help="Bench profile/regime.")
    p_bench.add_argument("--model", default=None, help="Base model id.")
    p_bench.add_argument(
        "--gpus",
        default=None,
        help="CUDA device IDs, or auto for idle same-caliber discovery (default: 0).",
    )
    p_bench.add_argument("--remote", default=None, help="Remote box alias (default: gpu-host).")
    p_bench.add_argument("--port", type=int, default=8200, help="Run-unique vLLM port.")
    p_bench.add_argument("--trace-path", dest="trace_path", default=None,
                         help="Materialized exact-token real workload JSON.")
    p_bench.add_argument("--load-mode", choices=["closed", "open"], default=None)
    p_bench.add_argument("--concurrency", type=int, default=None)
    p_bench.add_argument("--n-requests", dest="n_requests", type=int, default=None)
    p_bench.add_argument("--max-seeds", dest="max_seeds", type=int, default=None)
    # wired levers (rendered through BenchConfig.to_serve_args)
    p_bench.add_argument("--quantization", default=None, help="fp8 | awq | gptq")
    p_bench.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default=None)
    p_bench.add_argument("--quantized-model-id", dest="quantized_model_id", default=None)
    p_bench.add_argument("--model-artifact-kind", dest="model_artifact_kind", default=None)
    p_bench.add_argument("--max-num-batched-tokens", dest="max_num_batched_tokens",
                         type=int, default=None)
    p_bench.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=None)
    p_bench.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization",
                         type=float, default=None)
    p_bench.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    p_bench.add_argument("--tensor-parallel-size", dest="tensor_parallel_size",
                         type=int, default=None)
    p_bench.add_argument("--enable-prefix-caching", dest="enable_prefix_caching",
                         action="store_true", default=None)
    p_bench.add_argument("--disable-prefix-caching", dest="enable_prefix_caching",
                         action="store_false", default=None)
    p_bench.add_argument("--enable-chunked-prefill", dest="enable_chunked_prefill",
                         action="store_true", default=None)
    p_bench.add_argument("--disable-chunked-prefill", dest="enable_chunked_prefill",
                         action="store_false", default=None)
    p_bench.add_argument("--enforce-eager", dest="enforce_eager",
                         action="store_true", default=None)
    p_bench.set_defaults(handler=cmd_bench)

    p_context = sub.add_parser("context", help="Emit target context (READ_CONTEXT).")
    p_context.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_context.add_argument("--db", default=None, help="Store DB path (default vllm_evolve.db).")
    p_context.set_defaults(handler=cmd_context)

    p_design = sub.add_parser("design", help="Record a design hypothesis (DESIGN).")
    p_design.add_argument("--note", required=True, help="Design hypothesis text.")
    p_design.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_design.add_argument("--out", default=".ve/design.md", help="Design notes path.")
    p_design.set_defaults(handler=cmd_design)

    p_compare = sub.add_parser("compare", help="Compare two eval_result.json (KEEP_OR_DISCARD).")
    p_compare.add_argument("baseline", help="Baseline eval_result.json path.")
    p_compare.add_argument("candidate", help="Candidate eval_result.json path.")
    p_compare.add_argument("--epsilon", type=float, default=2.0, help="Tie dead-band percent.")
    p_compare.add_argument("--cv-threshold", type=float, default=None, dest="cv_threshold",
                           help="CV above which verdict is high_variance_inconclusive.")
    p_compare.add_argument("--seed", type=int, default=0, help="Bootstrap seed (deterministic).")
    p_compare.set_defaults(handler=cmd_compare)

    p_keep = sub.add_parser("keep", help="Archive + store a kept policy; git-commit.")
    p_keep.add_argument("policy", help="Path to the kept policy .py file.")
    p_keep.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_keep.add_argument("--eval-result", default=None, dest="eval_result",
                        help="eval_result.json to archive.")
    p_keep.add_argument(
        "--acceptance-evidence",
        default=None,
        dest="acceptance_evidence",
        help=(
            "Formal real-vLLM three-scenario suite result.json. Required for non-manual keep "
            "and hash-bound to the kept policy."
        ),
    )
    p_keep.add_argument("--archive-root", default="archive_policies", dest="archive_root",
                        help="Archive root dir.")
    p_keep.add_argument("--run-id", default=None, dest="run_id",
                        help="Override run id (default: policy hash).")
    p_keep.add_argument("--db", default=None, help="Store DB path.")
    p_keep.add_argument("--manual", action="store_true",
                        help="Archive-only manual keep (no real eval_result; never counts as "
                             "a gain / AC6). Without it, keep REQUIRES a clean real eval_result.")
    p_keep.set_defaults(handler=cmd_keep)

    p_discard = sub.add_parser("discard", help="Record a discarded policy (audited; no commit).")
    p_discard.add_argument("policy", help="Path to the discarded policy .py file.")
    p_discard.add_argument("--reason", default="", help="Why it was discarded.")
    p_discard.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_discard.add_argument("--archive-root", default="archive_policies", dest="archive_root")
    p_discard.add_argument("--run-id", default=None, dest="run_id")
    p_discard.set_defaults(handler=cmd_discard)

    p_init = sub.add_parser("init", help="Install Claude Code and/or Codex assets.")
    p_init.add_argument("--dir", default=".", help="Target project dir (default: cwd).")
    p_init.add_argument("--client", choices=["claude", "codex", "both"], default="claude",
                        help="Integration to install (default: claude).")
    p_init.add_argument("--uninstall", action="store_true", help="Remove a prior install.")
    p_init.add_argument("--check", action="store_true", help="Report install status.")
    p_init.add_argument("--force", action="store_true", help="Reinstall (restore then re-install).")
    p_init.set_defaults(handler=cmd_init)

    p_frontier_evolve = sub.add_parser(
        "frontier-evolve",
        help="Local-only real Frontier evolution over BurstGPT fragments + stress traces.",
    )
    p_frontier_evolve.add_argument(
        "--burstgpt", required=True, help="Official BurstGPT CSV path."
    )
    p_frontier_evolve.add_argument(
        "--out", default="runs/frontier_local_evolution",
        help="Artifact root (default: runs/frontier_local_evolution).",
    )
    p_frontier_evolve.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the frozen request fingerprint and artifacts match.",
    )
    p_frontier_evolve.add_argument("--seeds", default="0,1,2")
    p_frontier_evolve.add_argument(
        "--goal",
        default="maximize goodput while preserving completion and TTFT SLO",
        help="Natural-language optimization goal; normalized and frozen into the run request.",
    )
    p_frontier_evolve.add_argument("--fragment-size", type=int, default=64,
                                   dest="fragment_size")
    p_frontier_evolve.add_argument("--slo-ttft-ms", type=float, default=200.0,
                                   dest="slo_ttft_ms")
    p_frontier_evolve.add_argument("--max-num-seqs", type=int, default=4,
                                   dest="max_num_seqs")
    p_frontier_evolve.add_argument("--generations", type=int, default=3)
    p_frontier_evolve.add_argument("--population", type=int, default=8)
    p_frontier_evolve.add_argument("--max-total-evals", type=int, default=24,
                                   dest="max_total_evals")
    p_frontier_evolve.add_argument(
        "--research-mode",
        choices=["live-required", "auto", "offline"],
        default="auto",
        dest="research_mode",
        help="auto=live primary-source refresh with curated fallback; offline=curated only.",
    )
    p_frontier_evolve.add_argument(
        "--research-snapshot", default=None, dest="research_snapshot",
        help="Reuse a frozen research_snapshot.json instead of rebuilding research.",
    )
    p_frontier_evolve.add_argument(
        "--research-refresh", action="store_true", dest="research_refresh",
        help="Refresh research even when the request fingerprint matches an existing snapshot.",
    )
    p_frontier_evolve.add_argument(
        "--author-command", default=None, dest="author_command",
        help="External Codex-compatible command: reads AuthorContext JSON on stdin and prints "
             "{source,manifest} JSON. Omit for the deterministic template fallback.",
    )
    p_frontier_evolve.add_argument(
        "--db", default=None,
        help="Knowledge DB for immutable evolution lessons (default: vllm_evolve.db).",
    )
    p_frontier_evolve.set_defaults(handler=cmd_frontier_evolve)

    p_real_evolve = sub.add_parser(
        "real-evolve",
        help="Run generational evolution using only valid saturated real-vLLM fitness.",
    )
    p_real_evolve.add_argument(
        "--bench-config",
        required=True,
        dest="bench_config",
        help="Frozen formal strong-baseline BenchConfig JSON.",
    )
    p_real_evolve.add_argument(
        "--research-snapshot",
        required=True,
        dest="research_snapshot",
        help="Frozen live-required research_snapshot.json with live-gap cards.",
    )
    p_real_evolve.add_argument(
        "--gpus",
        default="auto",
        help=(
            "GPU selection: auto (default) discovers an idle same-caliber set "
            "and freezes it for the round; use config to honor the frozen "
            "BenchConfig value, or pass explicit CUDA device IDs."
        ),
    )
    p_real_evolve.add_argument(
        "--author-command",
        required=True,
        dest="author_command",
        help="Local Codex-compatible command reading AuthorContext JSON on stdin.",
    )
    p_real_evolve.add_argument(
        "--author-timeout-s",
        type=float,
        default=600.0,
        dest="author_timeout_s",
    )
    p_real_evolve.add_argument("--out", required=True)
    p_real_evolve.add_argument("--generations", type=int, default=2)
    p_real_evolve.add_argument("--population", type=int, default=4)
    p_real_evolve.add_argument(
        "--max-total-evals",
        type=int,
        default=8,
        dest="max_total_evals",
    )
    p_real_evolve.add_argument(
        "--repair-limit",
        type=int,
        default=1,
        dest="repair_limit",
    )
    p_real_evolve.set_defaults(handler=cmd_real_evolve)

    p_research = sub.add_parser(
        "research",
        help="Internal Auto Research Expert step (admin utility, not an optimization mode).",
    )
    p_research_sub = p_research.add_subparsers(dest="action", required=True)
    p_research_build = p_research_sub.add_parser(
        "build", help="Compile and freeze a target-aware expert brief."
    )
    p_research_build.add_argument("--target", default="scheduling")
    p_research_build.add_argument("--goal", required=True)
    p_research_build.add_argument(
        "--diagnosis", default=None, help="Diagnosis JSON object or file path."
    )
    p_research_build.add_argument(
        "--environment", default=None, help="Environment JSON object or file path."
    )
    p_research_build.add_argument("--out", required=True)
    p_research_build.add_argument(
        "--mode",
        choices=["live-required", "auto", "offline"],
        default="auto",
    )
    p_research_build.add_argument("--refresh", action="store_true")
    p_research_build.add_argument("--db", default=None)
    p_research_build.add_argument("--internal-limit", type=int, default=5,
                                  dest="internal_limit")

    p_research_real = p_research_sub.add_parser(
        "real-scheduler",
        help="Build live-pinned structural scheduler gap cards for real evolution.",
    )
    p_research_real.add_argument("--out", required=True)
    p_research_real.add_argument(
        "--installed-vllm-commit",
        required=True,
        dest="installed_vllm_commit",
    )
    p_research_real.add_argument(
        "--environment",
        default=None,
        help="Additional frozen environment JSON object or file path.",
    )
    p_research_real.add_argument("--refresh", action="store_true")

    p_research_verify = p_research_sub.add_parser(
        "verify", help="Verify snapshot hash and source citations."
    )
    p_research_verify.add_argument("snapshot")
    p_research_verify.add_argument("--citations", default=None)

    p_research_export = p_research_sub.add_parser(
        "export-author", help="Export the exact Codex AuthorContext prompt bundle."
    )
    p_research_export.add_argument("snapshot")
    p_research_export.add_argument("--out", required=True)
    p_research_export.add_argument("--spec", default=None)
    p_research_export.add_argument("--diagnosis", default=None)
    p_research_export.add_argument("--skeleton", default=None)
    p_research_export.add_argument("--parents", default=None)
    p_research_export.add_argument("--peers", default=None)
    p_research_export.add_argument("--lessons", default=None)
    p_research_export.add_argument("--last-errors", default=None, dest="last_errors")
    p_research_export.add_argument("--budget", default=None)
    p_research_export.add_argument("--generation", type=int, default=0)
    p_research_export.add_argument("--author-kind", default="codex", dest="author_kind")

    p_research_import = p_research_sub.add_parser(
        "import-candidate", help="Validate/import Codex source + CandidateManifest."
    )
    p_research_import.add_argument("--author-context", required=True, dest="author_context")
    p_research_import.add_argument("--source", required=True)
    p_research_import.add_argument("--manifest", required=True)
    p_research_import.add_argument("--out", required=True)
    p_research.set_defaults(handler=cmd_research)


    # --- autopt L0-L4 verbs (merged from vllm-evolve) ---
    # --- autopt L1: profile + diagnose ---
    p_diag = sub.add_parser("diagnose", help="L1: localize the bottleneck from a Profile JSON.")
    p_diag.add_argument("profile", help="Path to a Profile JSON (from `ve profile`).")
    p_diag.set_defaults(handler=cmd_diagnose)

    p_insp = sub.add_parser("inspect", help="Read-only store views + cited-fact check.")
    p_insp.add_argument("--top", default=None, help="Top-N policies for a target.")
    p_insp.add_argument("--n", type=int, default=5, help="N for --top.")
    p_insp.add_argument("--verify-citations", default=None, dest="verify_citations",
                        help="Comma-separated policy_ids to verify exist (fabricated flagged).")
    p_insp.add_argument("--db", default=None, help="Store path (default vllm_evolve.db).")
    p_insp.set_defaults(handler=cmd_inspect)

    p_tgt = sub.add_parser("targets", help="L2: ranked interventions for a Diagnosis JSON.")
    p_tgt.add_argument("diagnosis", help="Path to a Diagnosis JSON (from `ve diagnose`).")
    p_tgt.set_defaults(handler=cmd_targets)

    p_opt = sub.add_parser("optimize", help="L3: search a config knob on real vLLM.")
    p_opt.add_argument("--target", required=True, help="config:<knob>, e.g. config:max_num_seqs.")
    p_opt.add_argument("--space", required=True, help="Comma-separated values to try.")
    p_opt.add_argument("--metric", default="goodput_req_s", help="Objective metric.")
    p_opt.add_argument("--direction", default="max", choices=["max", "min"])
    p_opt.add_argument("--strategy", default="grid", choices=["grid", "coordinate"])
    p_opt.add_argument("--max-evals", type=int, default=8, dest="max_evals")
    p_opt.add_argument("--model", default=None)
    p_opt.add_argument("--gpus", default="0")
    p_opt.add_argument("--gpu-memory-utilization", type=float, default=None,
                       dest="gpu_memory_utilization")
    p_opt.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs")
    p_opt.add_argument("--concurrency", type=int, default=None)
    p_opt.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    p_opt.add_argument("--n-requests", type=int, default=None, dest="n_requests")
    p_opt.add_argument("--max-seeds", type=int, default=None, dest="max_seeds")
    p_opt.add_argument("--profile", default="throughput")
    p_opt.add_argument("--policy", default=None)
    p_opt.add_argument("--remote", default=None)
    p_opt.add_argument("--port", type=int, default=8260)
    p_opt.add_argument("--out", default=None, help="Write the Candidate JSON here.")
    p_opt.set_defaults(handler=cmd_optimize)

    p_goal = sub.add_parser("goal", help="L0: parse a natural-language goal into a Spec.")
    p_goal.add_argument("text", nargs="+", help="The goal, e.g. 'cut P99 TTFT to 200ms'.")
    p_goal.set_defaults(handler=cmd_goal)

    p_auto = sub.add_parser("autopt", help="Top: parse a goal, run L1->L4 on real vLLM.")
    p_auto.add_argument("text", nargs="+", help="Natural-language goal.")
    p_auto.add_argument("--max-rounds", type=int, default=3, dest="max_rounds")
    p_auto.add_argument("--max-evals", type=int, default=8, dest="max_evals")
    p_auto.add_argument("--no-holdout", action="store_true", dest="no_holdout")
    p_auto.add_argument("--accept-threshold-pct", type=float, default=None,
                        dest="accept_threshold_pct",
                        help="Hard X%% accept gate vs strong baseline (config decision).")
    p_auto.add_argument("--backend", choices=["remote", "local_smoke", "frontier_sim"],
                        help="remote=real vLLM (default); local_smoke=synthetic off-box loop "
                             "(can only conclude DoD-B). Also via VE_BENCH_BACKEND.")
    p_auto.add_argument("--evolve", action="store_true",
                        help="Allow evolving schedule_batch when scheduling is the "
                             "bottleneck (box-gated; GPU evolution).")
    p_auto.add_argument("--model", default=None)
    p_auto.add_argument("--gpus", default="0")
    p_auto.add_argument("--gpu-memory-utilization", type=float, default=None,
                        dest="gpu_memory_utilization")
    p_auto.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs")
    p_auto.add_argument("--concurrency", type=int, default=None)
    p_auto.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    p_auto.add_argument("--n-requests", type=int, default=None, dest="n_requests")
    p_auto.add_argument("--max-seeds", type=int, default=None, dest="max_seeds")
    p_auto.add_argument("--profile", default="throughput")
    p_auto.add_argument("--policy", default=None)
    p_auto.add_argument("--remote", default=None)
    p_auto.add_argument("--out", default=None, help="Write the full trajectory JSON here.")
    p_auto.add_argument("--workload-spec", default=None, dest="workload_spec",
                        help='JSON workload spec, e.g. {"length_dist":"uniform",...}.')
    p_auto.add_argument("--generations", type=int, default=2,
                        help="Evolution v2: generations per code-target round (engine caps "
                             "total evals at 24 regardless).")
    p_auto.add_argument("--population", type=int, default=3,
                        help="Evolution v2: children per generation.")
    p_auto.add_argument("--repair-limit", type=int, default=1, dest="repair_limit",
                        help="Evolution v2: verify-failure repair attempts per child.")
    p_auto.add_argument(
        "--research-mode",
        choices=["live-required", "auto", "offline"],
        default="auto",
        dest="research_mode",
    )
    p_auto.add_argument("--research-snapshot", default=None, dest="research_snapshot")
    p_auto.add_argument("--research-out", default=None, dest="research_out")
    p_auto.add_argument("--research-db", default=None, dest="research_db")
    p_auto.add_argument("--research-refresh", action="store_true", dest="research_refresh")
    p_auto.add_argument(
        "--author-command", default=None, dest="author_command",
        help="Codex-compatible stdin/stdout JSON author command; no template fallback when set.",
    )
    p_auto.set_defaults(handler=cmd_autopt)

    p_tune = sub.add_parser("tune", help="Tune a GIVEN policy's serving config (composite mode).")
    p_tune.add_argument("policy", help="Path to the policy .py to tune.")
    p_tune.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_tune.add_argument("--goal", nargs="*", default=None, help="Natural-language objective.")
    p_tune.add_argument("--backend", choices=["remote", "local_smoke", "frontier_sim"],
                        help="remote=real vLLM (default); local_smoke=synthetic plumbing (DoD-B).")
    p_tune.add_argument("--max-rounds", type=int, default=2, dest="max_rounds")
    p_tune.add_argument("--max-evals", type=int, default=8, dest="max_evals")
    p_tune.add_argument("--no-holdout", action="store_true", dest="no_holdout")
    p_tune.add_argument("--accept-threshold-pct", type=float, default=None,
                        dest="accept_threshold_pct")
    p_tune.add_argument("--model", default=None)
    p_tune.add_argument("--gpus", default="0")
    p_tune.add_argument("--concurrency", type=int, default=None)
    p_tune.add_argument("--n-requests", type=int, default=None, dest="n_requests")
    p_tune.add_argument("--max-seeds", type=int, default=None, dest="max_seeds")
    p_tune.add_argument("--profile", default="throughput")
    p_tune.add_argument("--remote", default=None)
    p_tune.add_argument("--out", default=None, help="Write the full result JSON here.")
    p_tune.set_defaults(handler=cmd_tune)

    p_port = sub.add_parser("port", help="Adapt/regression-check a policy across vLLM "
                                         "versions x hardware (composite mode).")
    p_port.add_argument("policy", help="Path to the policy .py to port.")
    p_port.add_argument("target", nargs="?", default="scheduling", help="Target name.")
    p_port.add_argument("--versions", default="", help="Comma-separated vLLM versions.")
    p_port.add_argument("--hardware", default="", help="Comma-separated hardware ids.")
    p_port.add_argument("--backend", choices=["remote", "local_smoke", "frontier_sim"],
                        help="remote=real vLLM per cell (@requires_gpu); local_smoke=plumbing.")
    p_port.add_argument("--out", default=None, help="Write port_report.json here.")
    p_port.set_defaults(handler=cmd_port)

    p_runs = sub.add_parser("runs", help="Browse / garbage-collect the runs/ working area.")
    p_runs_sub = p_runs.add_subparsers(dest="action", required=True)
    p_runs_ls = p_runs_sub.add_parser("ls", help="List run bundles (newest first).")
    p_runs_ls.add_argument("--target", default=None, help="Filter by target.")
    p_runs_ls.add_argument("--runs-root", default=None, dest="runs_root")
    p_runs_gc = p_runs_sub.add_parser("gc", help="Remove old runs (kept winners are exempt).")
    p_runs_gc.add_argument("--keep-last", type=int, default=20, dest="keep_last")
    p_runs_gc.add_argument("--older-than-days", type=float, default=None, dest="older_than_days")
    p_runs_gc.add_argument("--runs-root", default=None, dest="runs_root")
    p_runs_gc.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_runs.set_defaults(handler=cmd_runs)

    p_run = sub.add_parser(
        "run",
        help="Front door: declare {model, hw, metric, method} -> the harness self-configures "
             "backend+profile+evolve|tune and emits ONE DoD-stamped verdict.")
    p_run.add_argument("--model", default=None, help="Model id/path to serve.")
    p_run.add_argument("--hw", "--hardware", dest="hw", default="none",
                       help="Your hardware: a GPU spec (e.g. 1xA100) -> real vLLM (remote); "
                            "none/cpu -> off-box simulator (frontier_sim, DoD-B).")
    p_run.add_argument("--metric", nargs="+", required=True,
                       help="What to optimize, in plain words: 'maximize goodput', "
                            "'minimize p99 TTFT', 'maximize throughput' ...")
    p_run.add_argument("--method", choices=["evolve", "tune"], default="evolve",
                       help="evolve=evolutionary search of schedule_batch; tune=tune config knobs.")
    p_run.add_argument("--backend", choices=["remote", "local_smoke", "frontier_sim"], default=None,
                       help="Override the hardware->backend mapping (e.g. local_smoke for an "
                            "offline smoke). off-box backends stay DoD-B.")
    p_run.add_argument("--concurrency", type=int, default=None,
                       help="Optional load operating point; adoption (DoD-A) needs one so the "
                            "accept gate's holdout can re-measure at a different point.")
    p_run.add_argument("--n-requests", type=int, default=None, dest="n_requests",
                       help="Optional request count at the operating point (enables the holdout).")
    p_run.add_argument("--remote", default=None,
                       help="remote backend: ssh host/alias of the GPU box (default VE_REMOTE); "
                            "the bench AND the quality probe run there.")
    p_run.add_argument("--generations", type=int, default=2,
                       help="evolve+frontier_sim: generations (engine caps total evals at 24).")
    p_run.add_argument("--population", type=int, default=3,
                       help="evolve+frontier_sim: children per generation.")
    p_run.add_argument(
        "--research-mode",
        choices=["live-required", "auto", "offline"],
        default="auto",
        dest="research_mode",
    )
    p_run.add_argument("--research-snapshot", default=None, dest="research_snapshot")
    p_run.add_argument("--research-out", default=None, dest="research_out")
    p_run.add_argument("--research-db", default=None, dest="research_db")
    p_run.add_argument("--research-refresh", action="store_true", dest="research_refresh")
    p_run.add_argument("--author-command", default=None, dest="author_command")
    p_run.add_argument("--out", default=None, help="Write the full verdict JSON here.")
    p_run.set_defaults(handler=cmd_run)

    p_exp = sub.add_parser("experiment",
                           help="Open-world hypothesis research: register / run / list (admin).")
    p_exp_sub = p_exp.add_subparsers(dest="action", required=True)
    p_exp_reg = p_exp_sub.add_parser("register", help="Register a hypothesis prediction (first).")
    p_exp_reg.add_argument("spec", help="Experiment spec JSON (hypothesis_id/prediction/...).")
    p_exp_reg.add_argument("--db", default=None)
    p_exp_run = p_exp_sub.add_parser("run", help="Register -> run -> adjudicate -> ledger.")
    p_exp_run.add_argument("spec", help="Experiment spec JSON.")
    p_exp_run.add_argument("--db", default=None)
    p_exp_run.add_argument("--out", default=None, help="Base dir for experiment artifacts.")
    p_exp_list = p_exp_sub.add_parser("list", help="List hypotheses (optionally by verdict).")
    p_exp_list.add_argument("--db", default=None)
    p_exp_list.add_argument("--verdict", default=None,
                            choices=["supported", "falsified", "inconclusive", "unadjudicated"])
    p_exp_list.add_argument("--limit", type=int, default=50)
    p_exp.set_defaults(handler=cmd_experiment)

    p_vg = sub.add_parser("verify-gain", help="L4: bootstrap+holdout verdict for a candidate.")
    p_vg.add_argument("baseline", help="JSON with per-seed values (eval_result/Candidate/values).")
    p_vg.add_argument("candidate", help="JSON with per-seed values for the candidate.")
    p_vg.add_argument("--metric", default="goodput_req_s", help="Metric to compare.")
    p_vg.add_argument("--direction", default="max", choices=["max", "min"])
    p_vg.add_argument("--holdout-baseline", default=None, dest="holdout_baseline",
                      help="JSON: baseline per-seed values at a different config.")
    p_vg.add_argument("--holdout-candidate", default=None, dest="holdout_candidate",
                      help="JSON: candidate per-seed values at the SAME different config.")
    p_vg.add_argument("--holdout-floor", type=float, default=0.95, dest="holdout_floor")
    p_vg.add_argument("--accept-threshold-pct", type=float, default=None,
                      dest="accept_threshold_pct",
                      help="Hard X%% gate vs strong baseline (point + CI-low). Default: Spec 20%%.")
    p_vg.add_argument("--exempt-knob", action="append", default=None, dest="exempt_knob",
                      help="Engine/workload knob UNDER TEST that may differ between baseline "
                           "and candidate (e.g. max_num_seqs for a config A/B). Repeatable. "
                           "Other drift still trips same_caliber_mismatch.")
    p_vg.set_defaults(handler=cmd_verify_gain)

    p_prof = sub.add_parser("profile", help="L1: bench + sample GPU -> Profile (real vLLM).")
    p_prof.add_argument("--model", default=None, help="Model id/path.")
    p_prof.add_argument("--gpus", default="0", help="Card(s); profile samples the first.")
    p_prof.add_argument("--gpu-memory-utilization", type=float, default=None,
                        dest="gpu_memory_utilization")
    p_prof.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs")
    p_prof.add_argument("--concurrency", type=int, default=None)
    p_prof.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    p_prof.add_argument("--n-requests", type=int, default=None, dest="n_requests")
    p_prof.add_argument("--max-seeds", type=int, default=None, dest="max_seeds")
    p_prof.add_argument("--profile", default="throughput", help="Bench profile.")
    p_prof.add_argument("--policy", default=None, help="Policy .py (default seed).")
    p_prof.add_argument("--remote", default=None, help="ssh host/alias of the GPU box.")
    p_prof.add_argument("--port", type=int, default=8260)
    p_prof.add_argument("--out", default=None, help="Write the Profile JSON here.")
    p_prof.add_argument("--diagnose", action="store_true", help="Also run diagnose.")
    p_prof.add_argument("--cuda", choices=["nsys", "torch", "ncu"], default=None,
                        help="AC4: also run a profiled CUDA diagnostic window (box-gated; "
                        "degraded with capability evidence when unavailable).")
    p_prof.set_defaults(handler=cmd_profile)

    p_cal = sub.add_parser("calibrate", help="AC2: sweep load to lock the saturated "
                           "max-throughput operating point (box-gated real runs).")
    p_cal.add_argument("--model", default=None, help="Base model id.")
    p_cal.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES (default: 0).")
    p_cal.add_argument("--remote", default=None, help="Remote box alias.")
    p_cal.add_argument("--out", default=None, help="Persist the locked saturated config JSON.")
    p_cal.set_defaults(handler=cmd_calibrate)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _entrypoint() -> None:
    """Console-script entry point referenced by ``pyproject.toml``."""
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entrypoint()
