"""Offline residual-risk probes, not a musical-quality evaluation."""
from contextlib import redirect_stdout
import io
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intent import prepare_plan
import judge


def run():
    query = "Una canción sobre volver a casa sin romance"
    raw = dict(lyric_positive=["returning home"], lyric_negative=["violence"],
        clap_positive=[], clap_negative=[], constraints=[dict(modality="lyrics",
        polarity="negative", text="violence", origin="user_explicit", source_quote="sin romance")])
    # A verbatim quote is deliberately paired with the wrong translated concept.
    try:
        plan = prepare_plan(raw, query)
        translation_error = None
    except ValueError as exc:
        plan, translation_error = None, str(exc)
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="APPROVED", tool_calls=[], invalid_tool_calls=[])
    with patch.object(judge, "_get_judge_llm", return_value=llm), redirect_stdout(io.StringIO()):
        verdict = judge.evaluate_song("A song about returning home", "Unrelated words",
                                     ["returning home"], [], 1, "Control")
    qrels = json.loads((ROOT / "evaluation/qrels.json").read_text(encoding="utf-8"))
    return dict(evidence="offline_control_no_provider_calls", findings={
        "E13_P07_P08": dict(confirmed=plan is not None and plan["lyric_negative"] == ["violence"],
            input_query=query, wrong_translation="violence", accepted_plan=plan,
            rejection_reason=translation_error,
            meaning="Quote provenance does not verify semantic equivalence"),
        "P10_legacy": dict(confirmed=verdict["status"] == "approved" and not verdict["evidence"],
            verdict=verdict, meaning="Legacy compatibility can approve without verified evidence"),
        "P04_P10_P12": dict(human_annotation_status=qrels["status"],
            annotated_queries=len(qrels["judgments"]), real_quality_verified=False)},
        provider_calls=0, network_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/pending_fixes/post-fix-audit.json')
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: row.get("confirmed", row.get("human_annotation_status"))
                      for key, row in report["findings"].items()}))
