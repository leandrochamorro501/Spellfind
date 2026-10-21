"""Paired comparisons; simulated control cannot authorize a quality claim."""
import argparse
import json
from pathlib import Path
import statistics


def compare(baseline, refined):
    if baseline.get("mode") != refined.get("mode"):
        raise ValueError("Comparison modes differ")
    if baseline.get("refinement") is not False or refined.get("refinement") is not True:
        raise ValueError("Need baseline without refinement and treatment with refinement")
    if baseline["manifest"]["sha256"] != refined["manifest"]["sha256"]:
        raise ValueError("Source/data manifests differ; recapture both with the same sources")
    if baseline["manifest"].get("catalog") != refined["manifest"].get("catalog"):
        raise ValueError("Catalog generations differ")
    result = dict(mode=baseline["mode"], baseline_id=baseline["baseline_id"],
                  policy="disabled_by_default_experimental_opt_in",
                  decision="pending_real_quality_and_human_calibration", musical_quality_delta=None)
    if baseline.get("status") == "blocked" or refined.get("status") == "blocked":
        return dict(result, status="blocked", paired_runs=0)
    left = {(row["query_id"], row["repeat"]): row for row in baseline["runs"]}
    right = {(row["query_id"], row["repeat"]): row for row in refined["runs"]}
    if left.keys() != right.keys() or not left:
        raise ValueError("Runs are not paired")
    pairs = []
    for key, a in left.items():
        b = right[key]
        pairs.append(dict(query_id=key[0], repeat=key[1],
            latency_delta_seconds=b["elapsed_seconds"] - a["elapsed_seconds"],
            count_deltas={name: b["counts"][name] - value for name, value in a["counts"].items()},
            baseline_status=a["result"]["judge_status"], refined_status=b["result"]["judge_status"]))
    deltas = [row["latency_delta_seconds"] for row in pairs]
    result.update(status="completed", paired_runs=len(pairs), pairs=pairs,
        mean_latency_delta_seconds=statistics.mean(deltas),
        stdev_latency_delta_seconds=statistics.stdev(deltas) if len(deltas) > 1 else None,
        count_delta_totals={name: sum(row["count_deltas"][name] for row in pairs) for name in pairs[0]["count_deltas"]},
        limitations=["Synthetic verdicts/rankings do not measure relevance or provider cost.",
                      "Latency includes order/cache effects; use separate processes and more repetitions for real timing.",
                      "Call counters describe service boundaries, not billable provider requests."])
    # Expose real quality only when all paired measurements have human-derived labels.
    quality = [(a["metrics"]["ndcg_at_k"], right[key]["metrics"]["ndcg_at_k"]) for key, a in left.items()]
    if baseline["mode"] == "real" and all(a is not None and b is not None for a, b in quality):
        result["musical_quality_delta"] = statistics.mean(b - a for a, b in quality)
        result["decision"] = "manual_review_required_quality_cost_and_negative_violations"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                     json.loads(args.refined.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}))


if __name__ == "__main__":
    main()
