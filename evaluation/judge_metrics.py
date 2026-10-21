"""Judge calibration requires explicit human decisions, never simulated labels."""


def calibration_metrics(rows):
    valid = {"approved", "rejected", "abstained"}
    if any(row.get("human_status") not in valid or row.get("model_status") not in valid | {"error"}
           for row in rows):
        raise ValueError("Calibration rows require human_status and model_status")
    approved = [row for row in rows if row["human_status"] == "approved"]
    rejected = [row for row in rows if row["human_status"] == "rejected"]
    return dict(annotated_pairs=len(rows),
        false_approval_rate=sum(row["model_status"] == "approved" for row in rejected) / len(rejected) if rejected else None,
        false_rejection_rate=sum(row["model_status"] == "rejected" for row in approved) / len(approved) if approved else None,
        abstention_rate=sum(row["model_status"] == "abstained" for row in rows) / len(rows) if rows else None,
        error_rate=sum(row["model_status"] == "error" for row in rows) / len(rows) if rows else None)
