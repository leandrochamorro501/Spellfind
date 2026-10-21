"""Regression tests; external libraries/providers are replaced at import boundaries."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_contract import normalize_plan, build_fallback_plan

def fixture_plan(query):
    return normalize_plan(dict(lyric_positive=[query], lyric_negative=[], clap_positive=[query], clap_negative=[]))



def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result

class Message:
    def __init__(self, content):
        self.content = content

class Flask:
    def __init__(self, *args): pass
    def route(self, *args, **kwargs): return lambda fn: fn

stubs = {
    'langchain_groq': module('langchain_groq', ChatGroq=object),
    'langchain_core.tools': module('langchain_core.tools', tool=lambda fn: fn),
    'langchain_core.messages': module('langchain_core.messages', HumanMessage=Message, SystemMessage=Message),
    'langchain_core.prompts': module('langchain_core.prompts', ChatPromptTemplate=object),
    'langchain_core.output_parsers': module('langchain_core.output_parsers', JsonOutputParser=object),
}
def load(name):
    spec = importlib.util.spec_from_file_location('tested_' + name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(result)
    return result

judge, app, router = load('judge'), load('app'), load('router')

def response(text='', calls=None, invalid=None):
    return types.SimpleNamespace(content=text, tool_calls=calls or [], invalid_tool_calls=invalid or [])

def call(sid, reason='Wrong topic', name='exclude_song'):
    return {'name':name, 'args':{'song_id':sid, 'reason':reason}}


def decision(status, evidence='some lyrics'):
    return response(json.dumps(dict(status=status, reason='Control decision', evidence=[evidence])))

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def evaluate(self, answer, sid=1, pool=None):
        pool = pool or judge.ExclusionPool(10)
        llm = types.SimpleNamespace(invoke=lambda messages: answer)
        with patch.object(judge, '_get_judge_llm', return_value=llm):
            result = judge.evaluate_song('query','lyrics',['positive'],[],sid,'title',pool)
        return result, pool

    def test_plan_strings_legacy_null_and_duplicates(self):
        p = normalize_plan(dict(lyric_positive={'text':' lyric ', 'weight':0.7},
                                lyric_negative=None, clap_positive=[' audio ','audio',''], clap_negative=''))
        self.assertEqual(p['lyric_positive'], ['lyric'])
        self.assertEqual(p['clap_positive'], ['audio'])
        self.assertEqual(p['lyric_negative'], [])
        self.assertEqual(router._validate_plan(p), p)

    def test_invalid_plans(self):
        bad = [None, {}, dict(fixture_plan('query'), clap_positive=[3]),
               dict(fixture_plan('query'), lyric_positive={'text':4,'weight':0.7}),
               dict(fixture_plan('query'), lyric_positive={'text':'x','weight':2}),
               dict(fixture_plan('query'), lyric_positive=[], clap_positive=[])]
        for p in bad:
            with self.subTest(p=p), self.assertRaises(ValueError): normalize_plan(p)

    def test_router_fallback_contract(self):
        p = router._build_fallback_plan('piano lento sin voces')
        self.assertEqual(p['clap_positive'], ['piano', 'slow'])
        self.assertEqual(p['clap_negative'], ['vocals'])
        self.assertEqual(normalize_plan(p), p)

    def test_only_explicit_approval(self):
        for text in ['','maybe','APPROVED but uncertain', 'NOT APPROVED']:
            with self.subTest(text=text):
                verdict,pool=self.evaluate(response(text))
                self.assertEqual(verdict['status'],'error')
                self.assertFalse(verdict['approved'])
                self.assertFalse(pool.excluded_ids)
        self.assertEqual(self.evaluate(response(' APPROVED '))[0]['status'], 'abstained')
        self.assertTrue(self.evaluate(decision('approved', 'lyrics'))[0]['approved'])

    def test_rejection_text_and_tool(self):
        for answer in [response('REJECTED: wrong topic'),response(calls=[call('1')])]:
            verdict,pool=self.evaluate(answer)
            self.assertEqual(verdict['status'],'abstained')
            self.assertEqual(pool.excluded_ids,set())

    def test_bad_tools_do_not_mutate_pool(self):
        for answer in [response(calls=[call('2')]),response(calls=[call('1', '')]),
                       response(calls=[call('1', name='unknown')]),
                       response(calls=[call('1'),call('2')]),response(invalid=[{}]),
                       response(calls=[call(1.5)]),response(calls=[call(True)])]:
            verdict,pool=self.evaluate(answer)
            self.assertEqual(verdict['status'],'abstained')
            self.assertFalse(pool.excluded_ids)

    def test_provider_and_initialization_errors(self):
        for llm in [None, types.SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(RuntimeError('offline')))]:
            kwargs={'side_effect':RuntimeError('missing key')} if llm is None else {'return_value':llm}
            with patch.object(judge,'_get_judge_llm',**kwargs):
                result=judge.evaluate_song('q','l',[],[],1,'t',judge.ExclusionPool(2))
                self.assertEqual(result['status'],'error')

    def test_interleaved_requests_are_isolated(self):
        barrier=Barrier(2)
        def invoke(messages):
            sid=int(messages[1].content.split('SONG ID: ')[1].splitlines()[0])
            barrier.wait(timeout=5)
            return decision('rejected', 'l')
        pools=[judge.ExclusionPool(10),judge.ExclusionPool(10)]
        with patch.object(judge,'_get_judge_llm',return_value=types.SimpleNamespace(invoke=invoke)):
            with ThreadPoolExecutor(2) as executor:
                futures=[executor.submit(judge.evaluate_song,'q','l',[],[],sid,'t',pool)
                         for sid,pool in zip([1,2],pools)]
                self.assertTrue(all(f.result()['status']=='rejected' for f in futures))
        self.assertEqual(pools[0].excluded_ids,set())
        self.assertEqual(pools[1].excluded_ids,set())
        self.assertFalse(judge.ExclusionPool(10).excluded_ids)

    def run_search(self, answers, broken_router=False):
        fake_engine=types.SimpleNamespace(ready=True, get_song_by_id=lambda sid:
            {'song_id':sid,'title':str(sid),'artist':'artist','lyrics':'some lyrics'})
        chain=types.SimpleNamespace(invoke=lambda _: fixture_plan('query'))
        llm=types.SimpleNamespace(invoke=lambda _: next(answers))
        def worker(**kwargs):
            self.assertIsInstance(kwargs['positive'],list)
            self.assertTrue(all(isinstance(t,str) for t in kwargs['positive']))
            return [(1,0.9),(2,0.8),(3,0.7),(4,0.6)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(app,'engine',fake_engine))
            stack.enter_context(patch.object(app,'_import_router',**({'side_effect':RuntimeError('offline')} if broken_router else {'return_value':chain})))
            stack.enter_context(patch.object(app,'_import_semantic_search',return_value=worker))
            stack.enter_context(patch.object(app,'_import_clap_search',return_value=worker))
            stack.enter_context(patch.object(app,'_import_judge',return_value=(judge.evaluate_song,judge.ExclusionPool)))
            stack.enter_context(patch.object(app,'_import_refiner',return_value=lambda **kw:
                {'plan':kw['plan'], 'modalities':[], 'reason':'Keep existing descriptors'}))
            stack.enter_context(patch.object(judge,'_get_judge_llm',return_value=llm))
            return app.get_top_k_results('query')

    def test_integrated_rejections_never_return(self):
        result=self.run_search(iter([decision('rejected'),decision('rejected'),decision('approved')]))
        self.assertEqual([r['song_id'] for r in result['results']],[3,4])
        self.assertEqual(result['results'][0]['validation_status'],'partially_validated')
        self.assertEqual(len(result['exclusions']),2)
        self.assertEqual(result['judge_status'],'partially_validated')

    def test_integrated_error_stops_evaluation(self):
        result=self.run_search(iter([response('ambiguous')]))
        self.assertEqual(result['judge_iterations'],1)
        self.assertEqual(result['judge_status'],'error')
        self.assertTrue(result['warning'])
        self.assertEqual(result['results'][0]['validation_status'],'error')
        self.assertTrue(all(r['validation_status']=='not_evaluated' for r in result['results'][1:]))

    def test_limit_does_not_refill_with_rejected_songs(self):
        result=self.run_search(iter([decision('rejected')]*3))
        self.assertEqual([r['song_id'] for r in result['results']],[4])
        self.assertEqual(result['judge_status'],'limit_reached')

    def test_unknown_fallback_requires_clarification(self):
        result=self.run_search(iter([response('APPROVED')]),broken_router=True)
        self.assertTrue(result['plan']['_fallback'])
        self.assertEqual(result['judge_status'],'clarification_needed')
        self.assertEqual(result['results'],[])
        self.assertEqual(result['judge_iterations'],0)

    def test_new_search_has_fresh_exclusions_and_request_id(self):
        first=self.run_search(iter([decision('rejected'),decision('approved')]))
        second=self.run_search(iter([decision('approved')]))
        self.assertNotEqual(first['request_id'],second['request_id'])
        self.assertEqual(second['exclusions'],[])
        self.assertEqual(second['results'][0]['song_id'],1)

if __name__=='__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main(verbosity=2)
