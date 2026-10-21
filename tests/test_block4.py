"""Product controls, exact preselection, honest evidence and profiling isolation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from profiling import worker_profile, add, phase
from result_evidence import lyric_excerpt
from scoring import hybrid_scores
from search_options import validate_weights, validate_filters, matches_metadata, catalog_year


def raw_plan():
    return dict(lyric_positive=["returning home"], lyric_negative=[], clap_positive=["soft piano"], clap_negative=[])


class FixtureEngine:
    ready, error = True, None

    def __init__(self):
        self.rows = [dict(song_id=sid, artist="Target" if sid == 25 else "Other", tag="rock" if sid % 2 else "pop",
                          year=str(2000 + sid), title=f"Song {sid}", lyrics="Opening words. " * 30 + "Returning home to my family.")
                     for sid in range(1, 31)]

    def metadata_rows(self):
        return [{key: row[key] for key in ("song_id", "artist", "tag", "year")} for row in self.rows]

    def get_song_by_id(self, sid):
        return next((dict(row) for row in self.rows if row["song_id"] == sid), None)


class Block4Tests(unittest.TestCase):
    def setUp(self):
        self.engine = FixtureEngine()
        self.client = app.create_app({"TESTING": True}, search_engine=self.engine, initialize=False).test_client()

    def search(self, payload=None, plan=None, judge=None):
        self.worker = Mock(side_effect=lambda **kw: [(sid, 1 / sid) for sid in range(1, min(30, kw["top_k"]) + 1)])
        self.evaluator = judge or Mock(return_value=dict(status="approved", reason="CONTROL verdict", format="structured",
                                                         evidence=["Returning home"], coverage={"complete": True}))
        with patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=lambda _: deepcopy(plan or raw_plan()))), \
             patch.object(app, "_import_semantic_search", return_value=self.worker), \
             patch.object(app, "_import_clap_search", return_value=self.worker), \
             patch.object(app, "_import_judge", return_value=(self.evaluator, None)), \
             patch.object(app, "REFINEMENT_ENABLED", False):
            return self.client.post("/search", json=payload or {"query": "Una letra sobre volver con piano"})

    def test_fusion_is_bounded_additive_and_not_a_probability(self):
        ranked, parts = hybrid_scores([(1, 100), (2, .1)], [(2, -100), (1, -1000)], .7, .3)
        for sid, score in ranked:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
            self.assertAlmostEqual(score, sum(p["contribution"] for p in parts[str(sid)].values()))
        scaled, _ = hybrid_scores([(1, 1e6), (2, 100)], [(2, 8), (1, 1)], .7, .3)
        self.assertEqual(ranked, scaled)  # Worker score magnitudes are not comparable spaces.

    def test_weight_change_changes_order_and_zero_disables_contribution(self):
        lyrics, audio = [(1, 10), (2, 5)], [(2, 9), (1, 3)]
        self.assertEqual(hybrid_scores(lyrics, audio, .9, .1)[0][0][0], 1)
        self.assertEqual(hybrid_scores(lyrics, audio, .1, .9)[0][0][0], 2)
        ranked, parts = hybrid_scores([(1, 1)], [(2, 1)], 1, 0)
        self.assertEqual([sid for sid, _ in ranked], [1])
        self.assertEqual(parts["1"]["audio"]["contribution"], 0)

    def test_fusion_ties_are_deterministic(self):
        ranked, _ = hybrid_scores([(2, 1), (1, .9)], [(1, 1), (2, .9)], .5, .5)
        self.assertEqual([sid for sid, _ in ranked], [1, 2])

    def test_invalid_weights_fail_before_any_provider_import(self):
        bad = [True, [], {}, {"lyrics": .5}, {"lyrics": True, "audio": 0},
               {"lyrics": float("nan"), "audio": .4}, {"lyrics": float("inf"), "audio": 0},
               {"lyrics": -.1, "audio": 1.1}, {"lyrics": 0, "audio": 0},
               {"lyrics": .6, "audio": .6}, {"lyrics": "0.6", "audio": .4}]
        with patch.object(app, "_import_router", side_effect=AssertionError("must not plan")):
            for weights in bad:
                with self.subTest(weights=weights):
                    response = self.client.post("/search", json=dict(query="q", weights=weights))
                    self.assertEqual(response.status_code, 400)

    def test_explicit_weights_are_exposed_and_zero_branch_is_not_called(self):
        response = self.search(dict(query="Una letra sobre volver con piano", weights={"lyrics": 1, "audio": 0}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.worker.call_count, 1)
        self.assertEqual(response.json["retrieval"]["branches"]["audio"]["status"], "zero_weight")
        self.assertEqual(response.json["requested_weights"], {"lyrics": 1., "audio": 0.})
        self.assertEqual(response.json["judge_status"], "partially_validated")
        self.assertFalse(response.json["score_is_probability"])
        self.assertNotIn("score_pct", response.json["results"][0])

    def test_weights_do_not_invent_an_absent_modality(self):
        plan = raw_plan(); plan["clap_positive"] = []
        response = self.search(dict(query="Una letra sobre casa", weights={"lyrics": 0, "audio": 1}), plan=plan)
        self.assertEqual(response.json["judge_status"], "no_candidates")
        self.assertIn("peso positivo", response.json["warning"])
        self.worker.assert_not_called(); self.evaluator.assert_not_called()

    def test_zero_weight_keeps_user_constraints_for_judge(self):
        plan = raw_plan(); plan["lyric_negative"] = ["romance"]
        plan["constraints"] = [dict(modality="lyrics", polarity="negative", text="romance",
                                    origin="user_explicit", source_quote="sin romance")]
        response = self.search(dict(query="Una letra sobre casa sin romance con piano", weights={"lyrics": 0, "audio": 1}), plan=plan)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.evaluator.call_args.kwargs["lyric_negative"], ["romance"])
        self.assertEqual(response.json["plan"]["lyric_negative"], ["romance"])

    def test_metadata_filters_use_exact_names_inclusive_years_and_intersection(self):
        row = dict(artist="Beyoncé", tag="Rock", year="2020.0")
        self.assertTrue(matches_metadata(row, dict(artist=" BEYONCE\u0301 ", genre="rock", year_min=2020, year_max=2020)))
        self.assertFalse(matches_metadata(row, dict(artist="Beyonce")))
        self.assertFalse(matches_metadata(row, dict(artist="Bey")))
        self.assertFalse(matches_metadata(row, dict(genre="pop", year_min=2020)))
        self.assertFalse(matches_metadata(dict(row, year=""), dict(year_max=2030)))
        self.assertIsNone(catalog_year("2020.5"))

    def test_bad_filter_types_and_ranges_are_400(self):
        for filters in ([], {"artist": ""}, {"artist": 4}, {"genre": None}, {"year_min": True},
                        {"year_max": 2020.5}, {"year_min": "2020"}, {"year_min": 2021, "year_max": 2020},
                        {"year_min": 0}, {"unknown": "x"}):
            with self.subTest(filters=filters):
                self.assertEqual(self.client.post("/search", json=dict(query="q", filters=filters)).status_code, 400)

    def test_filter_can_find_a_candidate_outside_the_unfiltered_top_ten(self):
        response = self.search(dict(query="Una letra sobre casa con piano",
            filters=dict(artist=" target ", genre="rock", year_min=2025, year_max=2025)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row["song_id"] for row in response.json["results"]], [25])
        self.assertEqual(self.evaluator.call_args.kwargs["song_id"], 25)
        self.assertTrue(all(call.kwargs["top_k"] == 30 for call in self.worker.call_args_list))
        self.assertEqual(response.json["filter_summary"]["matched_catalog_songs"], 1)
        self.assertEqual(response.json["retrieval"]["filter_scope"], "full_catalog_then_exact_filter")

    def test_no_metadata_matches_short_circuits_all_model_calls(self):
        with patch.object(app, "_import_router", side_effect=AssertionError("must not plan")), \
             patch.object(app, "_import_judge", side_effect=AssertionError("must not initialize judge")):
            response = self.client.post("/search", json=dict(query="q", filters={"artist": "Missing artist"}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["termination_reason"], "no_filter_matches")
        self.assertEqual(response.json["results"], [])
        self.assertIsNone(response.json["plan"])

    def test_catalog_options_are_metadata_only_and_do_not_need_models(self):
        response = self.client.get("/catalog/options")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["artists"], ["Other", "Target"])
        self.assertEqual(response.json["genres"], ["pop", "rock"])
        self.assertEqual((response.json["year_min"], response.json["year_max"]), (2001, 2030))
        self.assertNotIn("lyrics", response.json)
        self.engine.ready = False
        self.assertEqual(self.client.get("/catalog/options").status_code, 503)

    def test_catalog_without_optional_year_genre_remains_filterable(self):
        import pandas as pd
        engine = app.SearchEngine()
        engine.df_meta = pd.DataFrame([dict(song_id=1, artist="Artist")])
        row = engine.metadata_rows()[0]
        self.assertEqual(row["tag"], "")
        self.assertTrue(matches_metadata(row, {"artist": "Artist"}))
        self.assertFalse(matches_metadata(row, {"year_min": 2000}))

    def test_verified_quote_is_shown_only_on_the_judged_candidate(self):
        response = self.search()
        first, second = response.json["results"][:2]
        self.assertEqual(first["lyric_excerpt"]["source"], "judge_quote")
        self.assertEqual(first["lyric_excerpt"]["text"], "Returning home")
        self.assertEqual(first["selection_reason"], "validated_recommendation")
        self.assertEqual(second["validation_status"], "not_evaluated")
        self.assertNotEqual(second["lyric_excerpt"]["source"], "judge_quote")
        self.assertNotIn("lyrics", first)

    def test_unverified_quote_never_appears_as_judge_evidence_excerpt(self):
        excerpt = lyric_excerpt("Returning home", {"lyrics": {"format": "structured", "evidence": ["invented"]}}, ["home"], True)
        self.assertEqual(excerpt["source"], "lexical_excerpt")
        self.assertIsNone(lyric_excerpt("Returning home", {}, ["home"], False))

    def test_snippet_indices_match_original_unicode_text_and_stay_short(self):
        lyrics = "Árbol y canción. " * 100 + "Returning home to my family"
        excerpt = lyric_excerpt(lyrics, {}, ["returning home"], True)
        self.assertEqual(excerpt["text"], lyrics[excerpt["start"]:excerpt["end"]])
        self.assertLessEqual(len(excerpt["text"]), 320)
        self.assertIn("Returning home", excerpt["text"])

    def test_api_exposes_additive_scores_and_real_stage_timings(self):
        result = self.search().json
        for row in result["results"]:
            self.assertAlmostEqual(row["score"], sum(p["contribution"] for p in row["score_components"].values()))
        self.assertGreaterEqual(result["timings"]["total_ms"], result["timings"]["workflow_ms"])
        self.assertTrue(all(row["elapsed_ms"] >= 0 for row in result["node_timings"]))
        json.dumps(result, allow_nan=False)

    def test_request_cache_is_observable_without_recounting_worker_times(self):
        cache = {}
        worker = Mock(return_value=[(1, 1.)])
        with patch.object(app, "_import_semantic_search", return_value=worker):
            first = app._execute_search(["home"], [], [], [], 1., 0., cache=cache)
            second = app._execute_search(["home"], [], [], [], 1., 0., cache=cache)
        self.assertFalse(first.details["branches"]["lyrics"]["request_cache_hit"])
        self.assertTrue(second.details["branches"]["lyrics"]["request_cache_hit"])
        self.assertIsNone(second.details["branches"]["lyrics"]["worker_profile"])
        worker.assert_called_once()

    def test_concurrent_profiles_do_not_share_counters(self):
        def task(count):
            with worker_profile() as measured:
                for _ in range(count):
                    add("calls")
                with phase("work_ms"):
                    pass
            add("outside", 100)
            return measured
        with ThreadPoolExecutor(2) as pool:
            profiles = list(pool.map(task, [2, 7]))
        self.assertEqual([p["calls"] for p in profiles], [2, 7])
        self.assertTrue(all("outside" not in p for p in profiles))


if __name__ == "__main__":
    unittest.main()
