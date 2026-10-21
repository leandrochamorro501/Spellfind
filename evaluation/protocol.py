"""Validate fixtures and human annotations before reporting quality metrics."""
from search_contract import normalize_plan


def validate_protocol(fixture, qrels, song_ids):
    cases = fixture.get("queries", [])
    ids = [case["id"] for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Evaluation query IDs must be nonempty and unique")
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("Invalid evaluation query")
        normalize_plan(case)
    judgments = qrels.get("judgments")
    if not isinstance(judgments, dict) or not set(judgments).issubset(ids):
        raise ValueError("Unknown query in human judgments")
    known = {str(sid) for sid in song_ids}
    for entry in judgments.values():
        if not isinstance(entry, dict) or type(entry.get("catalog_complete", False)) is not bool:
            raise ValueError("Invalid human judgment schema")
        grades = entry.get("grades", {})
        negatives = entry.get("negative_violations", {})
        if not isinstance(grades, dict) or not isinstance(negatives, dict):
            raise ValueError("Human grades and violations must be mappings")
        if not set(grades).issubset(known) or not set(negatives).issubset(known):
            raise ValueError("Unknown song ID in human judgments")
        if any(type(grade) is not int or not 0 <= grade <= 3 for grade in grades.values()):
            raise ValueError("Human grades must be integers 0-3")
        if any(type(value) is not bool for value in negatives.values()):
            raise ValueError("Negative violations must be booleans")
        if entry.get("catalog_complete") and set(grades) != known:
            raise ValueError("Complete judgments must cover the full catalog")
