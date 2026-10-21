"""Prepare catalog-bound human review material without inventing relevance labels."""
import argparse
import json
from pathlib import Path

from config import BASE_DIR
from catalog import default_manager
from evaluation.protocol import validate_protocol


def review_packet(snapshot, fixture):
    return dict(schema_version=1, status="pending_human_annotation", reviewed_by=None,
        catalog=dict(version=snapshot.version, sha256=snapshot.manifest["sha256"]),
        instructions=["Assign human relevance grades 0-3 and negative violations true/false.",
            "Leave unknown songs absent; do not convert unknown into grade zero.",
            "Audio relevance and violations require listening; lyrics alone cannot establish sound.",
            "Set catalog_complete only after grading every song for that query.",
            "Use status human_reviewed and reviewed_by only after an actual human review."],
        queries=fixture["queries"],
        songs=[snapshot.get_song_by_id(sid) for sid in sorted(snapshot.song_id_map.values())],
        judgments={case["id"]: dict(catalog_complete=False, grades={}, negative_violations={})
                   for case in fixture["queries"]})


def reviewed_qrels(packet, snapshot, fixture):
    if packet.get("catalog") != dict(version=snapshot.version, sha256=snapshot.manifest["sha256"]):
        raise ValueError("Human review belongs to a different catalog generation")
    reviewer = packet.get("reviewed_by")
    if packet.get("status") != "human_reviewed" or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Actual human review and reviewer attribution are required")
    if packet.get("queries") != fixture["queries"]:
        raise ValueError("Human review belongs to different query definitions")
    result = dict(schema_version=1, status="human_reviewed", reviewed_by=reviewer.strip(),
        catalog=packet["catalog"], judgments=packet.get("judgments"))
    validate_protocol(fixture, result, snapshot.song_id_map.values())
    if not any(entry.get("grades") for entry in result["judgments"].values()):
        raise ValueError("No human relevance grades were supplied")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((BASE_DIR / "evaluation/queries.json").read_text(encoding="utf-8"))
    snapshot = default_manager().current()
    if args.action == "export":
        result = review_packet(snapshot, fixture)
    else:
        if args.input is None: parser.error("import requires --input")
        result = reviewed_qrels(json.loads(args.input.read_text(encoding="utf-8")), snapshot, fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing reviews unless the user deliberately selects a new output.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=result["status"], catalog_version=snapshot.version, output=str(args.output))))


if __name__ == "__main__": main()
