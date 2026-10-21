"""Metrics that abstain when human judgments are incomplete."""
import math


def relevance_metrics(ids, judgments=None, k=5):
    if not judgments:
        return {"ndcg_at_k": None, "recall_at_k": None, "negative_violation_rate": None,
                "unjudged_results": len(ids[:k])}
    grades = judgments.get("grades", {})
    top = ids[:k]
    unjudged = sum(str(sid) not in grades for sid in top)
    ndcg = recall = violation = None
    if top and not unjudged:
        gains = [int(grades[str(sid)]) for sid in top]
        ideal = sorted((int(v) for v in grades.values()), reverse=True)[:k]
        dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        # Only complete catalog labels justify the ideal ranking and recall denominator.
        if judgments.get("catalog_complete"):
            ndcg = dcg / idcg if idcg else None
            relevant = {int(sid) for sid, grade in grades.items() if int(grade) > 0}
            recall = len(set(top) & relevant) / len(relevant) if relevant else None
        negatives = judgments.get("negative_violations", {})
        if all(str(sid) in negatives for sid in top):
            violation = sum(bool(negatives[str(sid)]) for sid in top) / len(top)
    return {"ndcg_at_k": ndcg, "recall_at_k": recall,
            "negative_violation_rate": violation, "unjudged_results": unjudged}


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else None
