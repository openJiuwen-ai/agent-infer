"""Frozen issue #10 gates; missing or inconsistent evidence is INCONCLUSIVE.

Usage: python validate.py manifest.json
Manifest: corpus, baseline/candidate {sha, benchmark, quality: [files]},
controls (matching the plan), and checks {L0, L1, L2}: paths to result JSON
with a verdict. Paths are relative to the manifest. This validates recorded
evidence; it does not authenticate hardware or replace review of launch logs.
"""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def close(actual, expected):
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)


def load_runtime(root, spec, plan, corpus):
    require(len(spec["sha"]) == 40 and int(spec["sha"], 16) >= 0, "Missing source pin")
    directory = root / spec["benchmark"]
    rows = json.loads((directory / "summary.json").read_text())
    expected = {(c, r) for c in plan["concurrency"] for r in range(plan["repetitions"])}
    require(len(rows) == len(expected), "Incomplete or extra performance repetitions")
    results, request_ids = {}, {}
    for summary in rows:
        key = summary["concurrency"], summary["repeat"]
        require(key in expected and key not in results, "Unexpected or duplicate repetition")
        label = f"c{key[0]}-r{key[1]}"
        records = [json.loads(s) for s in (directory / f"{label}.jsonl").read_text().splitlines()]
        require(len(records) == len(corpus), f"Incomplete performance records: {label}")
        require({r["sample_id"] for r in records} == corpus.keys(), f"Wrong samples: {label}")
        require(all(r["success"] and r["http_status"] == 200 for r in records), f"Failed: {label}")
        require(summary["valid"] and summary["failed"] == 0, f"Invalid run: {label}")
        require(summary["requests"] == summary["completed"] == len(corpus), label)
        elapsed = summary["makespan_s"]
        require(math.isfinite(elapsed) and elapsed > 0, "Invalid makespan")
        for r in records:
            frames = r["audio_samples"] / 2048
            require(0 < frames <= plan["controls"]["sampling"]["max_frames"], "Invalid length")
            require(close(r["acoustic_frames"], frames), "Incorrect frame accounting")
            require(close(r["audio_duration_s"], r["audio_samples"] / 44100), "Incorrect duration")
            require(close(r["rtf"], r["last_audio_s"] / r["audio_duration_s"]), "Incorrect RTF")
            require(0 < r["ttfa_s"] <= r["last_audio_s"] <= r["completion_s"], "Invalid timing")
            require(len(r["audio_sha256"]) == 64, "Missing PCM fingerprint")
        recomputed = {
            "throughput_frames_s": sum(r["audio_samples"] for r in records) / 2048 / elapsed,
            "ttfa_p50_ms": percentile([r["ttfa_s"] for r in records], 50) * 1000,
            "ttfa_p95_ms": percentile([r["ttfa_s"] for r in records], 95) * 1000,
            "rtf_mean": statistics.mean(r["rtf"] for r in records),
        }
        require(all(close(summary[k], v) for k, v in recomputed.items()), "Summary mismatch")
        results[key] = recomputed
        request_ids[label] = {r["request_id"]: r["sample_id"] for r in records}
        require(len(request_ids[label]) == len(corpus), "Duplicate request IDs")
    quality, asr = {}, None
    for filename in spec["quality"]:
        doc = json.loads((root / filename).read_text())
        require(asr is None or asr == doc["asr"], "ASR settings differ")
        asr = doc["asr"]
        for label, group in doc["groups"].items():
            require(label not in quality and label in request_ids, "Unexpected quality group")
            records = group["records"]
            require(group["complete"] and len(records) == len(corpus), "Incomplete quality")
            require(
                {r["request_id"]: r["sample_id"] for r in records} == request_ids[label],
                "Quality/performance samples differ",
            )
            for r in records:
                require(
                    r["is_success"] and r["target_text"] == corpus[r["sample_id"]],
                    "Invalid quality sample",
                )
                require(math.isfinite(r["speaker_similarity"]), "Missing SIM")
                require(
                    all(
                        isinstance(r[k], int) and r[k] >= 0
                        for k in ("substitutions", "deletions", "insertions", "hits")
                    ),
                    "Invalid WER",
                )
            errors = sum(
                r[k] for r in records for k in ("substitutions", "deletions", "insertions")
            )
            words = sum(r[k] for r in records for k in ("substitutions", "deletions", "hits"))
            require(words > 0 and close(group["wer"]["wer_corpus"], errors / words), "WER mismatch")
            sim = sum(r["speaker_similarity"] for r in records)
            require(close(group["similarity"]["mean"], sim / len(records)), "SIM mismatch")
            quality[label] = errors, words, sim, len(records)
    require(quality.keys() == request_ids.keys(), "Missing quality repetitions")
    return results, quality, asr


def validate(manifest_path):
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads((Path(__file__).parent.parent / "plan/plan.json").read_text())
    require(manifest["controls"] == plan["controls"], "Run controls differ from frozen plan")
    data = (root / manifest["corpus"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == plan["controls"]["corpus_sha256"], "Wrong corpus")
    samples = [json.loads(s) for s in data.splitlines()]
    corpus = {s["id"]: s["text"] for s in samples}
    require(len(corpus) == len(samples) == plan["requests_per_repetition"], "Invalid corpus size")
    require(manifest["baseline"]["sha"] == plan["baseline_sha"], "Wrong baseline revision")
    checks = []
    for level in ("L0", "L1", "L2"):
        evidence = json.loads((root / manifest["checks"][level]).read_text())
        require(evidence["verdict"] in ("PASS", "FAIL"), f"Incomplete {level}")
        require(evidence["candidate_sha"] == manifest["candidate"]["sha"], f"Unmatched {level}")
        checks.append(evidence["verdict"] == "PASS")
    baseline, bq, ba = load_runtime(root, manifest["baseline"], plan, corpus)
    candidate, cq, ca = load_runtime(root, manifest["candidate"], plan, corpus)
    require(ba == ca, "Baseline/candidate ASR differs")
    output = []
    for c in plan["concurrency"]:
        b = statistics.median(
            baseline[c, r]["throughput_frames_s"] for r in range(plan["repetitions"])
        )
        t = statistics.median(
            candidate[c, r]["throughput_frames_s"] for r in range(plan["repetitions"])
        )
        require(close(b * 1.5, plan["target_frames_s"][str(c)]), "Baseline target mismatch")

        def aggregate(q, concurrency=c):
            sums = [
                sum(q[f"c{concurrency}-r{r}"][i] for r in range(plan["repetitions"]))
                for i in range(4)
            ]
            return sums[0] / sums[1] * 100, sums[2] / sums[3]

        bw, bs = aggregate(bq)
        cw, cs = aggregate(cq)
        passed = (
            t >= b * 1.5
            and cw <= bw + plan["quality"]["wer_max_increase_pp"]
            and cs >= bs - plan["quality"]["speaker_similarity_max_decrease"]
        )
        checks.append(passed)
        output.append(
            dict(
                concurrency=c,
                throughput_ratio=t / b,
                wer_delta_pp=cw - bw,
                sim_delta=cs - bs,
                passed=passed,
            )
        )
    return {"verdict": "PASS" if all(checks) else "FAIL", "concurrency": output}


if __name__ == "__main__":
    try:
        result = validate(Path(sys.argv[1]).resolve())
    except (KeyError, ValueError, OSError, TypeError, IndexError, ZeroDivisionError) as error:
        result = {"verdict": "INCONCLUSIVE", "reason": str(error)}
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["verdict"] == "PASS" else 1)
