"""Module boundary and evidence claims, including the music compatibility path."""
import unittest
from unittest.mock import Mock

import app
from media_contracts import Capability, condition_record, evidence_record, verdict_record
from music_module import MusicModule, registry
from workflow import build_workflow, initial_state


def music(**overrides):
    values = dict(planner=lambda _: dict(lyric_positive=["rain"], lyric_negative=[],
                    clap_positive=[], clap_negative=[]),
                  search=lambda **_: ([(1, .9)], [], [(1, .9)]),
                  judge=lambda **_: dict(status="approved", reason="seen", evidence=["rain"],
                                       format="structured", coverage={"complete": True}, model_calls=1),
                  refiner=lambda **kw: dict(plan=kw["plan"], modalities=[], reason="none"),
                  lookup=lambda _: dict(song_id=1, title="Rain", artist="A", lyrics="rain"),
                  infer_weights=lambda *_: (1., 0.))
    values.update(overrides)
    return MusicModule(**values)


class MediaModuleTests(unittest.TestCase):
    def test_contract_rejects_unsupported_proof_and_false_provenance(self):
        with self.assertRaises(ValueError):
            condition_record(dict(aspect="audio", text="piano", polarity="positive",
                                  origin="user_explicit", source_quote=""))
        with self.assertRaises(ValueError):
            verdict_record("c1", "meets", "Sounds right")
        self.assertEqual(verdict_record("c1", "unverified", "No audio evidence")["status"], "unverified")

    def test_one_lyric_condition_can_use_complete_verified_quote(self):
        module = music()
        plan = dict(lyric_positive=["rain"], lyric_negative=[], clap_positive=[], clap_negative=[],
                    constraints=[dict(id="c1", modality="lyrics", polarity="positive", text="rain",
                                      origin="user_explicit", source_quote="rain")])
        verdicts = module.condition_verdicts(plan, dict(lyrics=dict(status="approved", format="structured",
            evidence=["rain"], coverage={"complete": True})), catalog_version="v1")
        self.assertEqual(verdicts[0]["status"], "meets")
        self.assertEqual(verdicts[0]["evidence"][0]["catalog_version"], "v1")
        plan["constraints"].append(dict(id="c2", modality="lyrics", polarity="negative", text="war",
                                        origin="user_explicit", source_quote="no war"))
        self.assertEqual([row["status"] for row in module.condition_verdicts(plan,
            dict(lyrics=dict(status="approved", format="structured", evidence=["rain"],
                             coverage={"complete": True})))], ["unverified", "unverified"])

    def test_audio_retrieval_is_not_claimed_as_acoustic_proof(self):
        module = music()
        plan = dict(constraints=[dict(id="c", modality="audio", polarity="positive", text="piano",
                                      origin="user_explicit", source_quote="piano")])
        self.assertEqual(module.condition_verdicts(plan, dict(audio=dict(status="approved")))[0]["status"],
                         "unverified")
        self.assertFalse(next(cap for cap in module.capabilities if cap.aspect == "audio").verifiable)

    def test_workflow_accepts_module_and_keeps_music_selection(self):
        graph = build_workflow(module=music(), enable_refinement=False)
        state = graph.invoke(initial_state("rain", "test"), config={"recursion_limit": 100})
        self.assertEqual(state["selected_song"], 1)
        self.assertEqual(state["status"], "approved")

    def test_registry_and_public_capabilities_only_advertise_music(self):
        self.assertEqual(list(registry(music())), ["music"])
        client = app.create_app(config={"TESTING": True}, initialize=False).test_client()
        response = client.get("/media/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row["media_type"] for row in response.json["media"]], ["music"])
        with self.assertRaisesRegex(ValueError, "aún no está disponible"):
            app.get_top_k_results("rain", search_engine=Mock(ready=True), media_type="film")

    def test_workflow_core_accepts_a_non_music_module_contract(self):
        class ExampleFilm:
            media_type = "film"
            def plan(self, query): return dict(planning_status="ready", subject=query)
            def prepare_plan(self, raw, query): return raw
            def fallback_plan(self, query): return dict(planning_status="needs_clarification")
            def infer_weights(self, plan): return (1., 0.)
            def retrieve_plan(self, plan, weights, cache, top_k):
                return ([(7, .8)], [], [(7, .8)])
            def requested_aspects(self, plan): return dict(primary=True, secondary=False)
            def lookup(self, item_id): return dict(title="Rain film", item_id=item_id)
            def evaluate_candidate(self, **kwargs):
                return dict(verdict=dict(status="approved", reason="Reviewed synopsis", model_calls=0),
                    validation=dict(synopsis=dict(status="approved")), judge_called=True,
                    metadata_skip=False, scope="plan")
            def refine(self, **kwargs): raise AssertionError("No refinement expected")
            def validate_refinement(self, current, result): return result
            def plan_fingerprint(self, plan): return plan["subject"]
        graph = build_workflow(module=ExampleFilm(), enable_refinement=False)
        state = graph.invoke(initial_state("rain", "film-control"), config={"recursion_limit": 100})
        self.assertEqual(state["selected_item"], 7)
        self.assertEqual(state["status"], "approved")


if __name__ == "__main__":
    unittest.main()
