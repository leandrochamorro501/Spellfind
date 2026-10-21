"""Block I acceptance tests with real Flask, NumPy, pandas and FAISS."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import faiss
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import app
from vector_validation import unit_vector, validate_index, validate_alignment
from evaluation.metrics import relevance_metrics, jaccard
from evaluation.protocol import validate_protocol


def index(dimension, count=2):
    result = faiss.IndexFlatIP(dimension)
    values = np.zeros((count, dimension), dtype=np.float32)
    values[:, 0] = 1
    result.add(values)
    return result


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.engine = types.SimpleNamespace(ready=True, error=None,
            df_meta=[1, 2], idx_lyrics=types.SimpleNamespace(ntotal=2),
            idx_audio=types.SimpleNamespace(ntotal=2), load_indices=Mock())
        self.application = app.create_app({"TESTING":True,"MAX_QUERY_LENGTH":20,"MAX_CONTENT_LENGTH":1024},
                                          search_engine=self.engine, initialize=False)
        self.client = self.application.test_client()

    def test_real_template_and_health(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertIn(b'SPELLFIND', self.client.get('/').data)
        self.assertEqual(self.client.get('/health').json['songs'],2)

    def test_malformed_and_wrong_type_bodies_are_400(self):
        for raw in ['null','[]','true','3','"text"','{', '{}',
                    '{"query":null}', '{"query":3}', '{"query":[]}', '{"query":"  "}']:
            with self.subTest(raw=raw):
                result=self.client.post('/search',data=raw,content_type='application/json')
                self.assertEqual(result.status_code,400)
                self.assertIn('error',result.json)
        self.assertEqual(self.client.post('/search',data='text').status_code,400)

    def test_length_limit_and_payload_limit(self):
        self.assertEqual(self.client.post('/search',json={'query':'x'*21}).status_code,400)
        self.assertEqual(self.client.post('/search',json={'query':'x'*20000}).status_code,413)

    def test_valid_query_and_degraded_error_codes(self):
        with patch.object(app,'get_top_k_results',return_value={'judge_status':'approved','results':[]}) as search:
            self.assertEqual(self.client.post('/search',json={'query':' query '}).status_code,200)
            search.assert_called_once_with('query',k=5,search_engine=self.engine)
        with patch.object(app,'get_top_k_results',return_value={'judge_status':'error','results':[]}):
            self.assertEqual(self.client.post('/search',json={'query':'query'}).status_code,503)

    def test_unavailable_engine_and_internal_error(self):
        self.engine.ready=False
        self.assertEqual(self.client.get('/health').status_code,503)
        self.assertEqual(self.client.post('/search',json={'query':'query'}).status_code,503)
        self.engine.ready=True
        with patch.object(app,'get_top_k_results',side_effect=RuntimeError('internal detail')):
            with self.assertLogs(self.application.logger,level='ERROR'):
                result=self.client.post('/search',json={'query':'query'})
        self.assertEqual(result.status_code,500)
        self.assertNotIn('internal detail',result.get_data(as_text=True))

    def test_factory_initializes_engine_and_defaults_to_no_debug(self):
        self.assertFalse(self.application.debug)
        app.create_app({'TESTING':True},search_engine=self.engine)
        self.engine.load_indices.assert_called_once()

    def test_actual_catalog_initializes_without_model_weights(self):
        application=app.create_app({'TESTING':True},search_engine=app.SearchEngine())
        health=application.test_client().get('/health')
        self.assertEqual(health.status_code,200)
        self.assertEqual(health.json['songs'],258)


class NumericalTests(unittest.TestCase):
    def test_normalization_and_nonmutation(self):
        value=np.array([3.,4.])
        result=unit_vector(value,2)
        np.testing.assert_allclose(result,[.6,.8])
        np.testing.assert_array_equal(value,[3,4])
        self.assertEqual(result.dtype,np.float32)

    def test_invalid_query_vectors(self):
        for value in [[],[0,0],[np.nan,1],[np.inf,1],[[1,2]],[1,2,3],[1e-30,1e-30]]:
            with self.subTest(value=value),self.assertRaises(ValueError): unit_vector(value,2)

    def test_index_dimension_metric_empty_and_vectors(self):
        validate_index(index(256),'lyrics')
        bad=[faiss.IndexFlatIP(256), index(255),faiss.IndexFlatL2(256)]
        for value in bad:
            with self.assertRaises(ValueError):validate_index(value,'lyrics')
        for vector in [np.zeros(256),np.full(256,np.nan),np.ones(256)]:
            value=faiss.IndexFlatIP(256);value.add(np.array([vector],dtype=np.float32))
            with self.assertRaises(ValueError):validate_index(value,'lyrics')

    def test_map_and_catalog_correspondence(self):
        frame=pd.DataFrame({'song_id':[10,20]})
        validate_alignment(frame,{'0':10,'1':20},index(256),index(512))
        for mapping in [{'0':20,'1':10},{'0':10},{'0':True,'1':20},{'0':10,'1':20,'2':30}]:
            with self.assertRaises(ValueError):
                validate_alignment(frame,mapping,index(256),index(512))
        with self.assertRaises(ValueError):
            validate_alignment(frame,{'0':10,'1':20},index(256,1),index(512))

    def test_degenerate_model_query_never_reaches_faiss(self):
        import semantic_search
        model=Mock();model.encode.return_value=np.ones(768,dtype=np.float32)
        with self.assertRaises(ValueError):
            semantic_search._build_query_vector('same',['same'],model)


class EvaluationTests(unittest.TestCase):
    def test_false_complete_annotations_are_rejected(self):
        fixture=json.loads((BASE/'evaluation/queries.json').read_text(encoding='utf-8'))
        invalid={'judgments':{'lyric_grief':{'catalog_complete':True,'grades':{'1':3}}}}
        with self.assertRaises(ValueError):validate_protocol(fixture,invalid,[1,2])
        invalid['judgments']['lyric_grief']={'grades':{'1':True}}
        with self.assertRaises(ValueError):validate_protocol(fixture,invalid,[1,2])

    def test_missing_labels_are_unknown(self):
        result=relevance_metrics([1,2])
        self.assertIsNone(result['ndcg_at_k'])
        self.assertIsNone(result['negative_violation_rate'])
        self.assertEqual(result['unjudged_results'],2)

    def test_metrics_require_complete_relevance_labels(self):
        labels={'catalog_complete':False,'grades':{'1':3,'2':0},
                'negative_violations':{'1':False,'2':True}}
        result=relevance_metrics([1,2],labels)
        self.assertIsNone(result['recall_at_k'])
        self.assertEqual(result['negative_violation_rate'],.5)
        labels['catalog_complete']=True
        self.assertEqual(relevance_metrics([1,2],labels)['ndcg_at_k'],1)
        self.assertEqual(relevance_metrics([1,2],labels)['recall_at_k'],1)

    def test_stability_and_fixed_fixture(self):
        self.assertEqual(jaccard([1,2],[2,3]),1/3)
        self.assertIsNone(jaccard([],[]))
        fixture=json.loads((BASE/'evaluation/queries.json').read_text(encoding='utf-8'))
        self.assertEqual(len(fixture['queries']),12)
        self.assertEqual(len({q['id'] for q in fixture['queries']}),12)
        self.assertEqual(len({q['category'] for q in fixture['queries']}),6)


if __name__=='__main__':unittest.main(verbosity=2)
