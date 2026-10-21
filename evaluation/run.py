"""Run from the project directory: python -m evaluation.run --mode simulated."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

from config import BASE_DIR, OUTPUT_DIR
from evaluation.metrics import relevance_metrics, jaccard
from evaluation.protocol import validate_protocol


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(qrels_path=None):
    files = list(BASE_DIR.glob("*.py")) + list((BASE_DIR / "evaluation").glob("*.py"))
    files += list((BASE_DIR / "scripts").glob("*.py"))
    files += list((BASE_DIR / "tests").glob("*.py"))
    files += list((BASE_DIR / "templates").glob("*.html")) + [BASE_DIR / "api_prompt.txt"]
    files += list(OUTPUT_DIR.glob("*")) + list((BASE_DIR / "requirements").glob("*.lock"))
    files += [BASE_DIR / "requirements.txt", BASE_DIR / "evaluation/queries.json",
              BASE_DIR / "evaluation/qrels.json", BASE_DIR / "requirements/model-revisions.json"]
    if qrels_path is not None: files.append(Path(qrels_path).resolve())
    from catalog import default_manager
    try:
        snapshot = default_manager().current()
        catalog = dict(version=snapshot.version, sha256=snapshot.manifest["sha256"], recipe=snapshot.manifest["recipe"])
    except Exception as exc:
        catalog = dict(status="unavailable", reason=str(exc))
    return {
        "catalog": catalog,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "sha256": {str(p.relative_to(BASE_DIR)) if p.is_relative_to(BASE_DIR) else str(p): digest(p)
                   for p in sorted(set(files)) if p.is_file()},
        "models": {"lyrics": "nomic-ai/nomic-embed-text-v1.5", "audio": "laion/clap-htsat-unfused",
                   "revisions": __import__('config').MODEL_REVISIONS,
                   "historical_catalog_revisions_verified": False},
    }


def run(mode="simulated", repeats=3, refinement=False, baseline_id="block5-post-P20", qrels_path=None):
    import app
    import os
    fixture = json.loads((BASE_DIR / "evaluation/queries.json").read_text(encoding="utf-8"))
    qrels = json.loads(Path(qrels_path or BASE_DIR / "evaluation/qrels.json").read_text(encoding="utf-8"))
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "refinement": refinement, "repeats": repeats,
              "baseline_id": baseline_id,
              "evidence": "simulated_control" if mode == "simulated" else "real_models",
              "manifest": manifest(qrels_path), "runs": [],
              "limitations": ["Quality metrics require applicable and sufficiently complete human labels.",
                              "Simulated timings do not measure model latency or musical quality.",
                              "First run is process-first, not guaranteed disk/cache cold.",
                              "No price or external API call-count instrumentation: monetary cost is unknown."]}
    app.engine.load_indices()
    if not app.engine.ready:
        report.update(status="blocked", blocked_reason="Catalog validation failed")
        return report
    snapshot = app.engine.pin_snapshot()
    if report["manifest"]["catalog"].get("version") != snapshot.version:
        report.update(status="blocked", blocked_reason="Catalog changed while preparing evaluation")
        return report
    if qrels.get("catalog") is not None and qrels["catalog"] != dict(
            version=snapshot.version, sha256=snapshot.manifest["sha256"]):
        report.update(status="blocked", blocked_reason="Human judgments belong to a different catalog generation")
        return report
    if mode == "real" and not os.getenv("GROQ_API_KEY"):
        report.update(status="blocked", blocked_reason="GROQ_API_KEY is not configured")
        return report
    ids = sorted(int(v) for v in snapshot.df_meta["song_id"])
    validate_protocol(fixture, qrels, ids)
    from judge import ExclusionPool
    for case in fixture["queries"]:
        for repeat in range(repeats):
            counters = {"planner_invocations": 0, "judge_evaluations": 0,
                        "refiner_invocations": 0, "lyrics_worker_calls": 0, "audio_worker_calls": 0}
            with ExitStack() as stack:
                stack.enter_context(patch.object(app, "REFINEMENT_ENABLED", refinement))
                original_router, original_judge = app._import_router, app._import_judge
                original_refiner = app._import_refiner
                original_lyrics, original_audio = app._import_semantic_search, app._import_clap_search

                def planner(inputs):
                    counters["planner_invocations"] += 1
                    if mode == "real":
                        return original_router().invoke(inputs)
                    return {k: case[k] for k in ("lyric_positive", "lyric_negative", "clap_positive", "clap_negative")}

                def evaluate(**kw):
                    counters["judge_evaluations"] += 1
                    if mode == "real":
                        return original_judge()[0](**kw)
                    status = "rejected" if counters["judge_evaluations"] == 1 else "approved"
                    return {"status": status, "approved": status == "approved",
                            "reason": "SIMULATED control verdict; not a musical judgment"}

                def refine(**kw):
                    counters["refiner_invocations"] += 1
                    if mode == "real" and refinement:
                        return original_refiner()(**kw)
                    plan = dict(kw["plan"])
                    modalities = []
                    if mode == "simulated" and refinement:
                        key = "lyric_positive" if plan["lyric_positive"] else "clap_positive"
                        plan[key] = [text + " simulated variant" for text in plan[key]]
                        modalities = ["lyrics" if key.startswith("lyric") else "audio"]
                    return {"plan": plan, "modalities": modalities,
                            "reason": "SIMULATED rewrite" if refinement else "Refinement disabled for baseline"}

                def worker(modality, original, **kw):
                    counters[modality + "_worker_calls"] += 1
                    if mode == "real":
                        return original()(**kw)
                    key = json.dumps([modality, kw["positive"], kw["negative"]], sort_keys=True)
                    ranking = sorted(ids, key=lambda sid: hashlib.sha256(f"{key}:{sid}".encode()).digest())
                    return [(sid, 1 / (i + 1)) for i, sid in enumerate(ranking[:kw["top_k"]])]

                stack.enter_context(patch.object(app, "_import_router", return_value=type("Planner", (), {"invoke": staticmethod(planner)})()))
                stack.enter_context(patch.object(app, "_import_judge", return_value=(evaluate, ExclusionPool)))
                stack.enter_context(patch.object(app, "_import_refiner", return_value=refine))
                stack.enter_context(patch.object(app, "_import_semantic_search", return_value=lambda **kw: worker("lyrics", original_lyrics, **kw)))
                stack.enter_context(patch.object(app, "_import_clap_search", return_value=lambda **kw: worker("audio", original_audio, **kw)))
                started = time.perf_counter()
                try:
                    result = app.get_top_k_results(case["query"], search_engine=snapshot)
                    error = None
                except Exception as exc:
                    result, error = {"results": [], "judge_status": "error"}, str(exc)
                elapsed = time.perf_counter() - started
            ranked_ids = [song["song_id"] for song in result["results"]]
            excluded = {entry["song_id"] for entry in result.get("exclusions", [])}
            report["runs"].append({"query_id": case["id"], "category": case["category"],
                "repeat": repeat, "process_first_search": len(report["runs"]) == 0,
                "elapsed_seconds": elapsed, "counts": counters, "result": result, "error": error,
                "external_api_calls": 0 if mode == "simulated" else None, "monetary_cost": None,
                "duplicates": len(ranked_ids) - len(set(ranked_ids)),
                "historical_reconsiderations": len(set(ranked_ids) & excluded),
                "excluded_reappearances": sum(sid in excluded and not any(
                    row["song_id"] == sid and row.get("plan_version") == result.get("plan_version")
                    and (row["status"] == "approved" or result.get("judge_status") == "retrieved")
                    for row in result.get("judge_log", [])) for sid in ranked_ids),
                "metrics": relevance_metrics(ranked_ids, qrels["judgments"].get(case["id"]) if mode == "real" else None)})
    stability = {}
    for case in fixture["queries"]:
        rows = [r for r in report["runs"] if r["query_id"] == case["id"]]
        values = [jaccard([s["song_id"] for s in a["result"]["results"]],
                          [s["song_id"] for s in b["result"]["results"]])
                  for a, b in zip(rows, rows[1:])]
        observed = [v for v in values if v is not None]
        stability[case["id"]] = statistics.mean(observed) if observed else None
    report["summary"] = {"runs": len(report["runs"]),
        "status_counts": {status: sum(r["result"]["judge_status"] == status for r in report["runs"])
                          for status in sorted({r["result"]["judge_status"] for r in report["runs"]})},
        "error_runs": sum(r["result"]["judge_status"] == "error" for r in report["runs"]),
        "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in report["runs"]),
        "stability_jaccard": stability,
        "musical_quality": "not evaluated" if mode == "simulated" or not qrels["judgments"] else "see per-query metrics"}
    report["status"] = "completed" if not report["summary"]["error_runs"] else "completed_with_errors"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulated", "real"], default="simulated")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refinement", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-id", default="block5-post-P20")
    parser.add_argument("--qrels", type=Path, help="Human-reviewed qrels; does not replace the original file")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report = run(args.mode, args.repeats, args.refinement, args.baseline_id, args.qrels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
