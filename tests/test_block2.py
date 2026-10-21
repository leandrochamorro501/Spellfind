"""Block II: decision ownership, modality honesty, provenance and safe fallback."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from intent import prepare_plan, constraint, audio_fallback
from search_contract import normalize_plan, build_fallback_plan
from refiner import validate_refinement
from workflow import build_workflow, initial_state
import app
import judge


def plan(lyrics=True,audio=False):
    return {"lyric_positive":["returning home"] if lyrics else [],"lyric_negative":[],
            "clap_positive":["soft piano"] if audio else [],"clap_negative":[]}


class Block2Tests(unittest.TestCase):
    def run_graph(self,raw=None,query="Una letra sobre volver a casa",lyrics="some lyrics",
                  judgments=None,refine=None,planner=None):
        self.judge=Mock(side_effect=judgments or [{"status":"approved","reason":"match"}])
        self.search=Mock(return_value=([(1,.9),(2,.8)],[(1,.9),(2,.8)],[(1,.9),(2,.8)]))
        self.refine=Mock(side_effect=refine or (lambda **kw:{"plan":kw['plan'],"modalities":[],"reason":"keep"}))
        self.graph=build_workflow(planner=planner or (lambda _:raw or plan()),search=self.search,
            judge=self.judge,refiner=self.refine,lookup=lambda sid:{"song_id":sid,"title":str(sid),"lyrics":lyrics},
            infer_weights=app._infer_weights)
        return self.graph.invoke(initial_state(query,"test"),config={"recursion_limit":20})

    def test_state_is_json_and_has_no_pool_or_cache(self):
        state=self.run_graph(judgments=[{"status":"rejected","reason":"wrong"},{"status":"approved","reason":"match"}])
        json.loads(json.dumps(state,allow_nan=False))
        self.assertNotIn('pool',state);self.assertNotIn('cache',state)
        self.assertEqual(len(state['exclusions']),1)
        self.assertEqual(state['exclusions'][0]['song_id'],1)
        self.assertEqual(state['selected_song'],2)

    def test_judge_returns_decision_without_touching_a_pool(self):
        pool=Mock();pool.exclude.side_effect=AssertionError('judge must not write')
        response=types.SimpleNamespace(content=json.dumps(dict(status='rejected', reason='wrong topic', evidence=['lyrics'])),
            invalid_tool_calls=[], tool_calls=[])
        llm=Mock();llm.invoke.return_value=response
        with patch.object(judge,'_get_judge_llm',return_value=llm):
            verdict=judge.evaluate_song('q','lyrics',['q'],[],1,'title',pool=pool)
        self.assertEqual(verdict['status'],'rejected');pool.exclude.assert_not_called()

    def test_requests_do_not_share_decisions_even_on_same_graph(self):
        graph=build_workflow(planner=lambda _:plan(),search=lambda **kw: ([(1,.9),(2,.8)],[],[(1,.9),(2,.8)]),
            judge=lambda **kw:{'status':'rejected' if kw['request_id']=='a' and kw['song_id']==1 else 'approved','reason':'decision'},
            refiner=lambda **kw:{'plan':kw['plan'],'modalities':[],'reason':'keep'},
            lookup=lambda sid:{'title':str(sid),'lyrics':'words'},infer_weights=app._infer_weights)
        with ThreadPoolExecutor(2) as executor:
            states=list(executor.map(lambda rid:graph.invoke(initial_state('about home',rid)),['a','b']))
        self.assertEqual([e['song_id'] for e in states[0]['exclusions']],[1])
        self.assertEqual(states[1]['exclusions'],[])

    def test_audio_only_instrumental_needs_no_lyrics_or_judge(self):
        state=self.run_graph(raw=plan(False,True),query='Instrumental sin voces',lyrics='')
        self.assertEqual(state['status'],'retrieved')
        self.assertEqual(state['validation']['audio']['status'],'not_evaluated')
        self.assertEqual(state['validation']['lyrics']['status'],'not_requested')
        self.assertEqual(state['exclusions'],[])
        self.judge.assert_not_called();self.refine.assert_not_called()

    def test_mixed_request_never_claims_full_validation(self):
        state=self.run_graph(raw=plan(True,True))
        self.assertEqual(state['status'],'partially_validated')
        self.assertEqual(state['validation']['lyrics']['status'],'approved')
        self.assertEqual(state['validation']['audio']['status'],'not_evaluated')

    def test_lyrics_only_approval(self):
        state=self.run_graph()
        self.assertEqual(state['status'],'approved')
        self.assertEqual(state['validation']['audio']['status'],'not_requested')

    def test_negative_only_lyric_constraint_still_requires_lyric_evaluation(self):
        raw=plan(False,True);raw['lyric_negative']=['romance']
        raw['constraints']=[constraint('lyrics','negative','romance','user_explicit','sin romance')]
        state=self.run_graph(raw=raw,query='Piano y letra sin romance')
        self.assertEqual(state['status'],'partially_validated')
        self.judge.assert_called_once()
        self.assertEqual(self.judge.call_args.kwargs['lyric_positive'],[])
        self.assertEqual(self.judge.call_args.kwargs['lyric_negative'],['romance'])

    def test_missing_lyric_evidence_is_not_a_judge_rejection(self):
        state=self.run_graph(lyrics='')
        self.judge.assert_not_called()
        self.assertEqual(state['judge_log'][0]['status'],'abstained')
        self.assertFalse(state['judge_log'][0]['judge_called'])

    def test_simple_audio_intent_corrects_gratuitous_lyric_branch(self):
        p=prepare_plan(plan(True,True),'Piano lento sin voces')
        self.assertEqual(p['lyric_positive'],[])
        self.assertEqual(p['clap_negative'],['vocals'])
        self.assertTrue(all(c['origin']=='user_explicit' for c in p['constraints']))

    def explicit_plan(self):
        raw=plan();raw['lyric_negative']=['romance']
        raw['constraints']=[constraint('lyrics','negative','romance','user_explicit','sin romance')]
        return prepare_plan(raw,'Una letra triste sin romance')

    def test_explicit_restrictions_cannot_be_removed_or_relabelled(self):
        current=self.explicit_plan();new=deepcopy(current);new['lyric_negative']=[];new['constraints']=[]
        with self.assertRaises(ValueError):
            validate_refinement(current,{'plan':new,'modalities':['lyrics'],'reason':'remove'})

    def test_model_invented_negative_can_be_corrected(self):
        raw=plan();raw['lyric_negative']=['romance']
        current=prepare_plan(raw,'Una letra sobre volver')
        new=deepcopy(current);new['lyric_negative']=[]
        revised=validate_refinement(current,{'plan':new,'modalities':['lyrics'],'reason':'unsupported hypothesis'})
        self.assertEqual(revised['plan']['lyric_negative'],[])
        self.assertFalse(any(c['text']=='romance' for c in revised['plan']['constraints']))

    def test_quoted_user_evidence_is_validated(self):
        raw=plan();raw['lyric_negative']=['romance']
        for quote in ['invented text','romance']:
            raw['constraints']=[constraint('lyrics','negative','romance','user_explicit',quote)]
            with self.subTest(quote=quote),self.assertRaises(ValueError):
                prepare_plan(raw,'Una letra triste sin romance')

    def test_activate_omitted_audio_requires_intent_evidence(self):
        current=prepare_plan(plan(),'Una letra sobre volver con piano')
        new=deepcopy(current);new['clap_positive']=['piano']
        proposal={'plan':new,'modalities':['audio'],'reason':'repair omitted sound'}
        with self.assertRaises(ValueError):validate_refinement(current,proposal)
        proposal['modality_evidence']=[{'modality':'audio','source_quote':'piano','diagnosis':'intent_correction'}]
        self.assertEqual(validate_refinement(current,proposal)['plan']['clap_positive'],['piano'])

    def test_can_remove_hypothesized_lyrics_for_instrumental_intent(self):
        current=prepare_plan(plan(True,True),'Instrumental relajante para estudiar')
        new=deepcopy(current);new['lyric_positive']=[]
        result=validate_refinement(current,{'plan':new,'modalities':['lyrics'],'reason':'sound only',
            'modality_evidence':[{'modality':'lyrics','source_quote':'Instrumental','diagnosis':'intent_correction'}]})
        self.assertEqual(result['plan']['lyric_positive'],[])

    def test_lyric_feedback_alone_cannot_rewrite_audio(self):
        current=prepare_plan(plan(True,True),'Una letra triste con piano')
        new=deepcopy(current);new['clap_positive']=['loud rock']
        with self.assertRaises(ValueError):
            validate_refinement(current,{'plan':new,'modalities':['audio'],'reason':'lyric was sad'})

    def test_versioned_rejections_need_reevaluation(self):
        def revise(**kw):
            new=deepcopy(kw['plan']);new['lyric_positive']=['I return to my hometown']
            return {'plan':new,'modalities':['lyrics'],'reason':'preserve exact home image'}
        state=self.run_graph(judgments=[{'status':'rejected','reason':'not matching old descriptors'},
                                       {'status':'approved','reason':'matches updated descriptors'}],refine=revise)
        self.assertEqual(state['plan_version'],2)
        self.assertEqual(state['selected_song'],1)
        self.assertEqual(state['exclusions'][0]['plan_version'],1)
        self.assertEqual(state['judge_log'][1]['plan_version'],2)

    def test_bounded_bilingual_fallback_preserves_audio_negatives(self):
        for query in ['Música tranquila con guitarra sin sintetizadores','quiet guitar without synthesizers']:
            p=build_fallback_plan(query)
            self.assertEqual(p['planning_status'],'ready')
            self.assertEqual(p['lyric_positive'],[])
            self.assertIn('synthesizers',p['clap_negative'])
            self.assertTrue(all(c['source_quote'] for c in p['constraints']))

    def test_unsupported_or_ambiguous_fallback_abstains(self):
        for query in ['triste pero sin amor romántico','not only piano but drums','sin piano, guitarra',
                      'instrumental without singing vocals','piano without drums guitar','piano sin piano','vuelve a mi ciudad']:
            p=build_fallback_plan(query)
            self.assertEqual(p['planning_status'],'needs_clarification')
            self.assertEqual(p['lyric_positive'],[]);self.assertEqual(p['clap_positive'],[])

    def test_fallback_reuses_only_a_valid_plan_for_same_query(self):
        current=self.explicit_plan();query=current['intent_query']
        recovered=build_fallback_plan(query,current)
        self.assertEqual(recovered['degradation'],'reused_valid_plan')
        self.assertEqual(recovered['lyric_negative'],['romance'])
        self.assertEqual(build_fallback_plan('otra consulta',current)['planning_status'],'needs_clarification')

    def test_router_failure_never_searches_an_uncertain_plan(self):
        state=self.run_graph(query='Triste pero sin amor romántico',planner=Mock(side_effect=RuntimeError('offline')))
        self.assertEqual(state['status'],'clarification_needed')
        self.search.assert_not_called();self.judge.assert_not_called()

    def test_router_cannot_silently_drop_an_unquoted_negation(self):
        state=self.run_graph(query='Una letra triste sin romance',raw=plan())
        self.assertEqual(state['status'],'clarification_needed')
        self.search.assert_not_called()

    def test_known_fallback_remains_useful_without_router(self):
        state=self.run_graph(query='Piano lento sin voces',lyrics='',planner=Mock(side_effect=RuntimeError('offline')))
        self.assertEqual(state['status'],'retrieved')
        self.assertIn('vocals',self.search.call_args.kwargs['clap_negative'])
        self.judge.assert_not_called()

    def test_http_exposes_audio_scope_and_safe_clarification(self):
        engine=types.SimpleNamespace(ready=True,error=None,get_song_by_id=lambda sid:{
            'song_id':sid,'title':'Instrumental','artist':'artist','lyrics':''})
        client=app.create_app({'TESTING':True},search_engine=engine,initialize=False).test_client()
        audio=Mock(return_value=[(1,.9)])
        evaluator=Mock(side_effect=AssertionError('audio must not invoke lyric judge'))
        with patch.object(app,'_import_router',side_effect=RuntimeError('offline')), \
             patch.object(app,'_import_judge',return_value=(evaluator,judge.ExclusionPool)), \
             patch.object(app,'_import_clap_search',return_value=audio):
            result=client.post('/search',json={'query':'Piano instrumental sin voces'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json['judge_status'],'retrieved')
            self.assertEqual(result.json['judge_iterations'],0)
            self.assertEqual(result.json['validation_scope'],'none')
            self.assertEqual(result.json['results'][0]['validation']['audio']['status'],'not_evaluated')
            audio.reset_mock()
            unknown=client.post('/search',json={'query':'Letra triste sin amor romántico'})
            self.assertEqual(unknown.json['judge_status'],'clarification_needed')
            self.assertEqual(unknown.json['results'],[])
            audio.assert_not_called();evaluator.assert_not_called()

    def test_real_router_template_formats_contract_without_phantom_variables(self):
        import router
        from langchain_core.runnables import RunnableLambda
        from langchain_core.messages import AIMessage
        captured=[]
        def respond(value):
            captured.append(value.to_messages()[0].content)
            return AIMessage(content=json.dumps(plan()))
        with patch.dict(os.environ,{'GROQ_API_KEY':'test-local-only'}), \
             patch.object(router,'_chain_cache',None), \
             patch.object(router,'_download_prompt',return_value='Local system prompt.'), \
             patch.object(router,'ChatGroq',return_value=RunnableLambda(respond)):
            result=router.get_router_chain().invoke({'query':'Una letra sobre volver'})
        self.assertEqual(result['planning_status'],'ready')
        self.assertFalse(result.get('_fallback'))
        self.assertIn('"modality"',captured[0])


if __name__=='__main__':unittest.main(verbosity=2)
