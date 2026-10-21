"""New workflow regressions using the real LangGraph runtime."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from unittest.mock import Mock, patch

from test_search_regressions import app, judge, Message, ROOT, module, fixture_plan
from intent import prepare_plan, constraint
from refiner import validate_refinement
import resources


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.plan = fixture_plan('original request')
        self.query='Una letra sobre volver con piano'
        self.lyrics = Mock(return_value=[(1, .9), (2, .8), (3, .7), (4, .6)])
        self.audio = Mock(return_value=[(1, .9), (2, .8), (3, .7), (4, .6)])
        self.refine = Mock(side_effect=lambda **kw: {
            'plan':kw['plan'], 'modalities':[], 'reason':'No justified rewrite'})

    def run_flow(self, statuses=('rejected', 'approved'), lookup=None):
        verdicts = iter(statuses)
        self.judge = Mock(side_effect=lambda **kw: {
            'status':next(verdicts), 'reason':'feedback', 'approved':False})
        # Approval is driven by explicit status, never a truthy flag or exhaustion.
        engine = types.SimpleNamespace(ready=True, get_song_by_id=lookup or (lambda sid: {
            'song_id':sid, 'title':str(sid), 'lyrics':'lyrics', 'artist':'artist'}))
        chain = types.SimpleNamespace(invoke=lambda _:self.plan)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(app, 'REFINEMENT_ENABLED', True))
            for name, value in [
                ('engine',engine), ('_import_router',lambda:chain),
                ('_import_judge',lambda:(self.judge,judge.ExclusionPool)),
                ('_import_refiner',lambda:self.refine),
                ('_import_semantic_search',lambda:self.lyrics),
                ('_import_clap_search',lambda:self.audio),
            ]:
                stack.enter_context(patch.object(app,name,value))
            return app.get_top_k_results(self.query)

    def rewrite(self, modality):
        def change(**kw):
            plan=copy.deepcopy(kw['plan'])
            prefix='lyric' if modality=='lyrics' else 'clap'
            plan[prefix+'_positive']=['new description']
            return {'plan':plan,'modalities':[modality],'reason':'Specific feedback',
                    'modality_evidence':[{'modality':'audio','source_quote':'piano','diagnosis':'descriptor_mismatch'}] if modality=='audio' else []}
        self.refine.side_effect=change

    def test_noop_refinement_and_final_response_reuse_rankings(self):
        result=self.run_flow(('rejected','rejected','approved'))
        self.assertEqual(self.lyrics.call_count,1)
        self.assertEqual(self.audio.call_count,1)
        self.assertEqual(result['judge_iterations'],3)
        self.assertEqual(result['termination_reason'],'partially_validated')
        self.assertEqual([x['song_id'] for x in result['results']],[3,4])
        self.assertEqual(len(result['refinement_log']),2)

    def test_lyrics_rewrite_preserves_audio_and_exclusions(self):
        self.rewrite('lyrics')
        self.lyrics.side_effect=[[(1,.9),(2,.8)],[(1,1),(3,.95),(2,.8)]]
        result=self.run_flow()
        self.assertEqual(self.lyrics.call_count,2)
        self.assertEqual(self.audio.call_count,1)
        self.assertEqual(result['judge_log'][0]['plan_version'],1)
        self.assertEqual(result['judge_log'][1]['plan_version'],2)
        self.assertEqual(result['results'][0]['song_id'],1)  # explicitly reevaluated under the new plan
        self.assertEqual(result['plan']['clap_positive'],self.plan['clap_positive'])
        self.assertEqual(result['refinement_log'][0]['modalities'],['lyrics'])

    def test_audio_rewrite_preserves_lyrics(self):
        self.rewrite('audio')
        result=self.run_flow()
        self.assertEqual(self.audio.call_count,2)
        self.assertEqual(self.lyrics.call_count,1)
        self.assertEqual(result['plan']['lyric_positive'],self.plan['lyric_positive'])
        self.assertEqual(result['validation_scope'],'lyrics')

    def test_both_modalities_can_change(self):
        def change(**kw):
            p=copy.deepcopy(kw['plan'])
            p['lyric_positive']=['new lyric']; p['clap_positive']=['new sound']
            return {'plan':p,'modalities':['lyrics','audio'],'reason':'Both descriptors need changes',
                    'modality_evidence':[{'modality':'audio','source_quote':'piano','diagnosis':'descriptor_mismatch'}]}
        self.refine.side_effect=change
        result=self.run_flow()
        self.assertEqual(self.audio.call_count,2)
        self.assertEqual(self.lyrics.call_count,2)
        self.assertEqual(result['judge_status'],'partially_validated')

    def test_limit_never_approves_or_refines_after_last_round(self):
        result=self.run_flow(('rejected',)*3)
        self.assertEqual(result['judge_status'],'limit_reached')
        self.assertEqual(self.refine.call_count,2)
        self.assertTrue(result['warning'])
        self.assertTrue(all(r['validation_status']=='not_evaluated' for r in result['results']))

    def test_empty_retrieval_never_calls_judge_or_refiner(self):
        self.lyrics.return_value=[]; self.audio.return_value=[]
        result=self.run_flow(())
        self.assertEqual(result['judge_status'],'no_candidates')
        self.judge.assert_not_called(); self.refine.assert_not_called()
        self.assertEqual(result['results'],[])

    def test_exhaustion_of_cached_candidates_is_not_approval(self):
        self.lyrics.return_value=[(1,.9)]; self.audio.return_value=[]
        result=self.run_flow(('rejected',))
        self.assertEqual(result['judge_status'],'no_candidates')
        self.assertEqual(self.judge.call_count,1)
        self.assertEqual(result['results'],[])

    def test_failed_refiner_continues_and_keeps_rejections(self):
        self.refine.side_effect=RuntimeError('offline')
        result=self.run_flow(('rejected','approved'))
        self.assertEqual(result['judge_status'],'partially_validated')
        self.assertEqual(result['degradations'][0]['stage'],'refine')
        self.assertNotIn(1,[r['song_id'] for r in result['results']])
        self.assertEqual(self.judge.call_count,2)

    def test_failed_lyric_retrieval_keeps_audio_and_reports_degradation(self):
        self.rewrite('lyrics')
        self.lyrics.side_effect=[[(1,.9),(2,.8)],RuntimeError('FAISS failure')]
        result=self.run_flow(('rejected','approved'))
        self.assertEqual(result['judge_status'],'partially_validated')
        self.assertEqual(result['retrieval']['branches']['lyrics']['status'],'failed')
        self.assertEqual(result['lyric_weight'],0)
        self.assertEqual(result['audio_weight'],1)
        self.assertEqual(self.judge.call_count,2)

    def test_missing_lyrics_skip_llm_and_still_terminate(self):
        result=self.run_flow((),lookup=lambda sid:None)
        self.assertEqual(result['judge_status'],'no_candidates')
        self.assertEqual(len(result['exclusions']),4)
        self.assertEqual(result['counters']['judge_evaluations'],0)
        self.judge.assert_not_called(); self.refine.assert_not_called()

    def test_unselected_modality_change_is_discarded(self):
        def bad(**kw):
            p=copy.deepcopy(kw['plan']);p['clap_positive']=['new sound']
            return {'plan':p,'modalities':['lyrics'],'reason':'lyrics feedback'}
        self.refine.side_effect=bad
        result=self.run_flow(('rejected','approved'))
        self.assertEqual(result['judge_status'],'partially_validated')
        self.assertEqual(result['degradations'][0]['stage'],'refine')
        self.assertEqual(result['plan'],result['initial_plan'])


class RefinementContractTests(unittest.TestCase):
    def test_cannot_remove_negatives_or_activate_absent_modality(self):
        p=fixture_plan('q');p['lyric_negative']=['romance']
        p['constraints']=[constraint('lyrics','negative','romance','user_explicit','sin romance')]
        p['intent_query']='q sin romance'
        proposed=copy.deepcopy(p);proposed['lyric_negative']=[]
        with self.assertRaises(ValueError):
            validate_refinement(p,{'plan':proposed,'modalities':['lyrics'],'reason':'x'})
        p['clap_positive']=[]
        proposed=copy.deepcopy(p);proposed['clap_positive']=['piano']
        with self.assertRaises(ValueError):
            validate_refinement(p,{'plan':proposed,'modalities':['audio'],'reason':'x'})

    def test_malformed_response_and_noop(self):
        p=fixture_plan('q')
        for bad in [None,{}, {'modalities':['unknown'],'reason':'x','plan':p},
                    {'modalities':[],'reason':'','plan':p}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError): validate_refinement(p,bad)
        result=validate_refinement(p,{'plan':p,'modalities':['lyrics'],'reason':'keep'})
        self.assertEqual(result['modalities'],[])

    def test_real_refiner_parses_json_and_preserves_original_request(self):
        import refiner
        p=prepare_plan(fixture_plan('query'),'piano please');new=copy.deepcopy(p);new['clap_positive']=['soft piano']
        model=Mock();model.invoke.return_value=types.SimpleNamespace(content=json.dumps(
            {'plan':new,'modalities':['audio'],'reason':'Use the requested instrument',
             'modality_evidence':[{'modality':'audio','source_quote':'piano','diagnosis':'descriptor_mismatch'}]}))
        with patch.object(refiner,'_get_refiner_llm',return_value=model):
            result=refiner.refine_plan('piano please',p,{'reason':'wrong topic'},{'lyrics':'words'})
        payload=json.loads(model.invoke.call_args.args[0][1].content)
        self.assertEqual(payload['query'],'piano please')
        self.assertEqual(payload['audio_evidence'],'not available')
        self.assertEqual(result['modalities'],['audio'])


class ResourceTests(unittest.TestCase):
    def setUp(self):
        resources._catalog.cache_clear();resources._index.cache_clear()
        resources._song_id_map.cache_clear();resources._metadata.cache_clear()
        self.addCleanup(resources._catalog.cache_clear)
        self.addCleanup(resources._index.cache_clear)
        self.addCleanup(resources._song_id_map.cache_clear)
        self.addCleanup(resources._metadata.cache_clear)

    def test_process_resources_are_loaded_once_even_concurrently(self):
        import faiss
        import numpy as np
        def valid_index(path):
            dimension = 256 if 'lyrics' in path else 512
            index = faiss.IndexFlatIP(dimension)
            vector = np.ones((1, dimension), dtype=np.float32)
            faiss.normalize_L2(vector)
            index.add(vector)
            return index
        reader=Mock(side_effect=valid_index)
        with patch.dict(sys.modules,{'faiss':module('faiss',read_index=reader)}):
            with ThreadPoolExecutor(4) as executor:
                results=list(executor.map(lambda _:resources.get_index('lyrics'),range(8)))
            audio=resources.get_index('audio')
        self.assertTrue(all(x is results[0] for x in results))
        self.assertIsNot(audio,results[0])
        self.assertEqual(reader.call_count,2)

    def test_worker_model_initialization_is_shared_across_threads(self):
        worker=self.load_worker('semantic_search')
        model=object(); index=object(); mapping={'0':1}
        factory=Mock(return_value=model)
        with patch.object(worker,'SentenceTransformer',factory), \
             patch.object(worker,'get_index',return_value=index) as read_index, \
             patch.object(worker,'get_song_id_map',return_value=mapping):
            with ThreadPoolExecutor(4) as executor:
                results=list(executor.map(lambda _:worker._load_resources(),range(8)))
        self.assertTrue(all(r is results[0] for r in results))
        self.assertIs(results[0]['index'],index)
        self.assertNotIn('df',results[0])
        factory.assert_called_once();read_index.assert_called_once()

    def test_catalog_map_and_lookup_reuse_and_metadata_is_copied(self):
        import pandas as pd
        with patch.object(pd,'read_csv',wraps=pd.read_csv) as read:
            catalog=resources.get_catalog()
            self.assertIs(catalog,resources.get_catalog())
            sid=int(catalog.iloc[0]['song_id'])
            meta=resources.get_song(sid);meta['lyrics']='modified'
            self.assertNotEqual(resources.get_song(sid)['lyrics'],'modified')
            self.assertEqual(read.call_count,1)
        self.assertIs(resources.get_song_id_map(),resources.get_song_id_map())

    def load_worker(self,name):
        spec=importlib.util.spec_from_file_location('tested_worker_'+name,ROOT/(name+'.py'))
        worker=importlib.util.module_from_spec(spec)
        deps={'sentence_transformers':module('sentence_transformers',SentenceTransformer=object),
              'torch':module('torch',cuda=types.SimpleNamespace(is_available=lambda:False)),
              'transformers':module('transformers',ClapModel=object,ClapProcessor=object)}
        with patch.dict(sys.modules,deps):spec.loader.exec_module(worker)
        return worker

    def test_lyrics_embeddings_are_reused_and_never_mutated(self):
        import numpy as np
        worker=self.load_worker('semantic_search')
        class Embedder:
            def __init__(self):self.calls=[]
            def encode(self,text,**kwargs):
                self.calls.append(text)
                return np.arange(1,769,dtype=np.float32) if 'negative' not in text else np.ones(768,dtype=np.float32)
        model=Embedder()
        baseline=worker._encode_text('positive',model)
        first=worker._build_query_vector('positive',['negative'],model)
        second=worker._build_query_vector('positive',['negative'],model)
        np.testing.assert_allclose(first,second)
        np.testing.assert_array_equal(baseline,worker._encode_text('positive',model))
        self.assertEqual(len(model.calls),2)
        self.assertEqual(first.shape,(256,))
        self.assertAlmostEqual(float(np.linalg.norm(first)),1,places=5)

    def test_audio_embeddings_are_reused_and_return_independent_copies(self):
        import numpy as np
        worker=self.load_worker('clap_search')
        encoder=Mock(side_effect=lambda text,*args:np.arange(1,513,dtype=np.float32)
                     if text=='positive' else np.ones(512,dtype=np.float32))
        with patch.object(worker,'_encode_cached',lru_cache(maxsize=512)(encoder)):
            model,processor=object(),object()
            baseline=worker._encode_text('positive',model,processor)
            first=worker._build_query_vector('positive',['negative'],model,processor)
            second=worker._build_query_vector('positive',['negative'],model,processor)
            np.testing.assert_allclose(first,second)
            np.testing.assert_array_equal(baseline,worker._encode_text('positive',model,processor))
            self.assertEqual(encoder.call_count,2)


if __name__=='__main__':unittest.main(verbosity=2)
