"""Block III behavior, with real graph/HTTP and controlled model boundaries."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import judge
from console import safe_print
from intent import prepare_plan, plan_fingerprint
from retrieval import RetrievalError
from workflow import build_workflow, initial_state
from evaluation.judge_metrics import calibration_metrics
from evaluation.compare import compare


def plan():
    return dict(lyric_positive=["returning home", "family reunion"], lyric_negative=[],
                clap_positive=[], clap_negative=[])


def response(raw):
    return SimpleNamespace(content=json.dumps(raw) if isinstance(raw, dict) else raw,
                           tool_calls=[], invalid_tool_calls=[])


class Block3Tests(unittest.TestCase):
    def graph(self, **options):
        defaults = dict(planner=lambda _: plan(),
            search=lambda **_: ([(1, .9), (2, .8)], [], [(1, .9), (2, .8)]),
            judge=lambda **_: dict(status="approved", reason="match"),
            refiner=lambda **kw: dict(plan=kw["plan"], modalities=[], reason="keep"),
            lookup=lambda sid: dict(title=str(sid), lyrics="Returning home to my family"),
            infer_weights=lambda *_: (1., 0.))
        defaults.update(options)
        return build_workflow(**defaults)

    def run_graph(self, **options):
        return self.graph(**options).invoke(initial_state("Una letra sobre volver a casa", "block3"),
                                           config={"recursion_limit": 160})

    def test_equivalent_reorder_does_not_release_rejections(self):
        def reorder(**kw):
            p = deepcopy(kw["plan"]); p["lyric_positive"].reverse(); p["constraints"].reverse()
            return dict(plan=p, modalities=["lyrics"], reason="order only")
        state = self.run_graph(judge=lambda **_: dict(status="rejected", reason="wrong"), refiner=reorder)
        self.assertEqual([r["song_id"] for r in state["judge_log"]], [1, 2])
        self.assertEqual(state["plan_version"], 1)
        self.assertEqual(state["status"], "no_candidates")

    def test_real_descriptor_change_changes_fingerprint(self):
        first = prepare_plan(plan(), "about home")
        second = deepcopy(first); second["lyric_positive"] = ["leaving home"]
        self.assertNotEqual(plan_fingerprint(first), plan_fingerprint(second))

    def test_both_planning_routes_reject_invalid_metadata(self):
        for status in ("ready", "needs_clarification"):
            for bad in (float("nan"), float("inf"), object()):
                with self.subTest(status=status, bad=type(bad).__name__), self.assertRaises((ValueError, TypeError)):
                    prepare_plan(dict(plan(), planning_status=status, confidence=bad), "about home")

    def test_invalid_clarification_metadata_recovers_to_json_state(self):
        state = self.run_graph(planner=lambda _: dict(plan(), planning_status="needs_clarification", confidence=float("nan")))
        json.dumps(state, allow_nan=False)
        self.assertEqual(state["status"], "clarification_needed")

    def test_console_cp1252_does_not_raise(self):
        data = io.BytesIO()
        stream = io.TextIOWrapper(data, encoding="cp1252")
        safe_print("Ready ✅", file=stream, flush=True)
        self.assertIn(b"Ready", data.getvalue())
        self.assertIn(b"\\u2705", data.getvalue())

    def retrieve(self, lyrics, audio):
        with patch.object(app, "_import_semantic_search", return_value=lyrics), \
             patch.object(app, "_import_clap_search", return_value=audio):
            return app._execute_search(["home"], ["war"], ["piano"], [], .6, .4)

    def test_partial_audio_failure_preserves_lyrics_and_weights(self):
        batch = self.retrieve(Mock(return_value=[(1, .9)]), Mock(side_effect=RuntimeError("audio offline")))
        self.assertEqual(batch[2][0][0], 1)
        self.assertEqual(batch.details["effective_weights"], {"lyrics": 1., "audio": 0.})
        self.assertEqual(batch.details["branches"]["audio"]["status"], "failed")

    def test_both_failed_modalities_are_not_empty_success(self):
        with self.assertRaises(RetrievalError) as caught:
            self.retrieve(Mock(side_effect=RuntimeError("lyrics offline")), Mock(side_effect=RuntimeError("audio offline")))
        self.assertTrue(all(b["status"] == "failed" for b in caught.exception.details["branches"].values()))

    def test_refiner_failure_continues_without_repeating_failed_refiner(self):
        evaluator = Mock(side_effect=[dict(status="rejected", reason="wrong"), dict(status="approved", reason="match")])
        refiner = Mock(side_effect=RuntimeError("offline"))
        state = self.run_graph(judge=evaluator, refiner=refiner)
        self.assertEqual(state["selected_song"], 2)
        self.assertEqual(state["degradations"][0]["stage"], "refine")
        refiner.assert_called_once()

    def test_all_failed_new_search_restores_matching_plan_and_ranking(self):
        def rewrite(**kw):
            p = deepcopy(kw["plan"]); p["lyric_positive"] = ["new hypothesis"]
            return dict(plan=p, modalities=["lyrics"], reason="change")
        search = Mock(side_effect=[([(1, .9), (2, .8)], [], [(1, .9), (2, .8)]),
                                   RuntimeError("retrieval unavailable"),
                                   ([(1, .9), (2, .8)], [], [(1, .9), (2, .8)])])
        evaluator = Mock(side_effect=[dict(status="rejected", reason="wrong"), dict(status="approved", reason="match")])
        state = self.run_graph(search=search, judge=evaluator, refiner=rewrite)
        self.assertEqual(state["selected_song"], 2)
        self.assertEqual(state["plan"], state["initial_plan"])
        self.assertEqual(state["plan_version"], 1)
        self.assertEqual(state["degradations"][0]["stage"], "search")

    def test_metadata_skips_do_not_consume_judge_budget(self):
        ids = [(i, 1 / i) for i in range(1, 6)]
        state = self.run_graph(search=lambda **_: (ids, [], ids),
            lookup=lambda sid: None if sid < 5 else dict(title="5", lyrics="Returning home"))
        self.assertEqual(state["selected_song"], 5)
        self.assertEqual(state["counters"]["judge_evaluations"], 1)
        self.assertEqual(state["counters"]["metadata_skips"], 4)

    def test_candidate_budget_bounds_metadata_scans(self):
        state = self.run_graph(lookup=lambda _: None, max_candidates=1)
        self.assertEqual(state["status"], "limit_reached")
        self.assertEqual(state["budget_reason"], "candidate_budget")
        self.assertEqual(state["counters"]["judge_evaluations"], 0)

    def test_retrieval_expansion_finds_next_candidate_once(self):
        def search(**kw):
            rows = [(sid, 1 / sid) for sid in range(1, kw["top_k"] + 1)]
            return rows, [], rows
        state = self.run_graph(search=search, initial_top_k=1, max_expansions=1,
            lookup=lambda sid: None if sid == 1 else dict(title="2", lyrics="Returning home"))
        self.assertEqual(state["selected_song"], 2)
        self.assertEqual(state["counters"]["expansions"], 1)
        self.assertEqual(state["requested_top_k"], 2)

    def test_expansion_budget_does_not_claim_full_catalog_exhaustion(self):
        def search(**kw):
            rows = [(sid, 1 / sid) for sid in range(1, kw["top_k"] + 1)]
            return rows, [], rows
        state = self.run_graph(search=search, lookup=lambda _: None, initial_top_k=1, max_expansions=1)
        self.assertEqual(state["counters"]["candidate_visits"], 2)
        self.assertEqual(state["budget_reason"], "retrieved_window_exhausted")

    def evaluate(self, answers, lyrics="Returning home to my family"):
        llm = Mock(); llm.invoke.side_effect = answers
        with patch.object(judge, "_get_judge_llm", return_value=llm):
            verdict = judge.evaluate_song("about home", lyrics, ["home"], [], 1, "title")
        return verdict, llm

    def test_judge_sees_and_cites_ending_beyond_original_cutoff(self):
        lyrics = "x" * 4000 + " Returning home"
        verdict, llm = self.evaluate([response(dict(status="approved", reason="home", evidence=["Returning home"]))], lyrics)
        self.assertEqual(verdict["status"], "approved")
        self.assertIn("Returning home", llm.invoke.call_args.args[0][1].content)
        self.assertEqual(verdict["model_calls"], 1)

    def test_invalid_quote_retried_once_then_valid(self):
        verdict, llm = self.evaluate([
            response(dict(status="approved", reason="home", evidence=["invented quote"])),
            response(dict(status="approved", reason="home", evidence=["Returning home"]))])
        self.assertEqual(verdict["status"], "approved")
        self.assertEqual(llm.invoke.call_count, 2)

    def test_malformed_output_stops_after_two_attempts(self):
        verdict, llm = self.evaluate([response("maybe"), response("maybe again")])
        self.assertEqual(verdict["status"], "error")
        self.assertEqual(verdict["model_calls"], 2)
        self.assertEqual(llm.invoke.call_count, 2)

    def test_provider_failure_is_not_retried(self):
        verdict, llm = self.evaluate([RuntimeError("offline")])
        self.assertEqual(verdict["status"], "error")
        self.assertEqual(verdict["model_calls"], 1)
        self.assertEqual(llm.invoke.call_count, 1)

    def test_incomplete_coverage_cannot_approve(self):
        lyrics = "x" * 15000 + "Returning home"
        verdict, _ = self.evaluate([response(dict(status="approved", reason="home", evidence=["Returning home"]))], lyrics)
        self.assertEqual(verdict["status"], "abstained")
        self.assertFalse(verdict["coverage"]["complete"])

    def test_abstention_does_not_trigger_refinement_or_fake_rejection(self):
        evaluator = Mock(side_effect=[dict(status="abstained", reason="uncertain"), dict(status="approved", reason="match")])
        refiner = Mock(side_effect=AssertionError("must not refine uncertainty"))
        state = self.run_graph(judge=evaluator, refiner=refiner)
        self.assertEqual(state["selected_song"], 2)
        self.assertEqual(state["exclusions"][0]["decision"], "abstained")
        refiner.assert_not_called()

    def test_disabled_refinement_uses_next_candidate_without_refiner(self):
        evaluator = Mock(side_effect=[dict(status="rejected", reason="wrong"), dict(status="approved", reason="match")])
        refiner = Mock(side_effect=AssertionError("disabled"))
        state = self.run_graph(judge=evaluator, refiner=refiner, enable_refinement=False)
        self.assertEqual(state["selected_song"], 2)
        refiner.assert_not_called()

    def test_http_partial_retrieval_preserves_requested_audio_without_approving_it(self):
        engine = SimpleNamespace(ready=True, error=None, get_song_by_id=lambda sid:
            dict(song_id=sid, title="Home", artist="artist", lyrics="Returning home"))
        raw = dict(plan(), clap_positive=["soft piano"])
        client = app.create_app({"TESTING": True}, search_engine=engine, initialize=False).test_client()
        with patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=lambda _: raw)), \
             patch.object(app, "_import_judge", return_value=(lambda **_: dict(status="approved", reason="home"), None)), \
             patch.object(app, "_import_semantic_search", return_value=lambda **_: [(1, .9)]), \
             patch.object(app, "_import_clap_search", side_effect=RuntimeError("CLAP offline")):
            result = client.post("/search", json={"query": "Una letra sobre casa con piano"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["judge_status"], "partially_validated")
        self.assertEqual(result.json["requested_modalities"], ["lyrics", "audio"])
        self.assertEqual(result.json["audio_weight"], 0)
        self.assertEqual(result.json["results"][0]["validation"]["audio"]["status"], "not_evaluated")
        self.assertTrue(result.json["warning"])
        self.assertFalse(result.json["catalog_exhausted"])

    def test_calibration_does_not_invent_human_labels(self):
        empty = calibration_metrics([])
        self.assertIsNone(empty["false_approval_rate"])
        measured = calibration_metrics([
            dict(human_status="rejected", model_status="approved"),
            dict(human_status="approved", model_status="abstained")])
        self.assertEqual(measured["false_approval_rate"], 1)
        self.assertEqual(measured["abstention_rate"], .5)

    def test_comparison_rejects_changed_manifest(self):
        left = dict(mode="simulated", refinement=False, manifest={"sha256": {"file": "a"}})
        right = dict(mode="simulated", refinement=True, manifest={"sha256": {"file": "b"}})
        with self.assertRaises(ValueError):
            compare(left, right)

    def test_simulated_comparison_cannot_claim_quality_improvement(self):
        row = dict(query_id="q", repeat=0, elapsed_seconds=1, counts={"refiner_invocations": 0},
                   result={"judge_status": "approved"}, metrics={"ndcg_at_k": None})
        left = dict(mode="simulated", refinement=False, manifest={"sha256": {}}, baseline_id="test", runs=[row])
        right = deepcopy(left); right["refinement"] = True
        right["runs"][0]["counts"]["refiner_invocations"] = 1
        result = compare(left, right)
        self.assertIsNone(result["musical_quality_delta"])
        self.assertEqual(result["count_delta_totals"]["refiner_invocations"], 1)
        self.assertIn("pending_real", result["decision"])


if __name__ == "__main__":
    unittest.main()
