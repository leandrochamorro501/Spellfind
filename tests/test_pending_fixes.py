"""Residual-risk regressions; no model or provider calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import judge
from intent import prepare_plan, quoted_constraint, constraint, fallback
from workflow import build_workflow, initial_state
from evaluation.annotations import review_packet, reviewed_qrels


class PendingFixTests(unittest.TestCase):
    def explicit(self, text='romance', quote='sin romance', polarity='negative', modality='lyrics'):
        return constraint(modality, polarity, text, 'user_explicit', quote)

    def test_wrong_translation_cannot_be_promoted_to_explicit(self):
        for text in ('violence', 'romance and violence', 'romantic love and heartbreak'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                quoted_constraint(self.explicit(text, 'sin amor romántico'), 'Una letra sin amor romántico')

    def test_reviewed_translation_and_exact_phrase_are_accepted(self):
        for text, quote in [('romantic love', 'sin amor romántico'), ('violence', 'sin violencia'),
                            ('fear of darkness', 'without fear of darkness')]:
            self.assertEqual(quoted_constraint(self.explicit(text, quote), quote)['text'], text)

    def test_positive_mistranslation_and_polarity_reversal_fail(self):
        for text, quote in [('violence', 'amor romántico'), ('romance', 'sin romance'), ('not romance', 'romance')]:
            with self.subTest(text=text, quote=quote), self.assertRaises(ValueError):
                quoted_constraint(self.explicit(text, quote, 'positive'), quote)

    def test_accents_are_not_silently_removed_in_exact_evidence(self):
        with self.assertRaises(ValueError):
            quoted_constraint(self.explicit('si', 'sí', 'positive'), 'sí')

    def test_unreviewed_case_changes_cannot_change_proper_noun_meaning(self):
        with self.assertRaises(ValueError):
            quoted_constraint(self.explicit('us', 'US', 'positive'), 'US')

    def test_double_negation_and_unverified_expansion_fail(self):
        for text, quote in [('romance', 'sin no romance'), ('romance', 'no solo romance'),
                            ('romance', 'sin romance con violencia')]:
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                quoted_constraint(self.explicit(text, quote), quote)

    def test_compound_quote_must_preserve_all_negative_concepts(self):
        quote = 'sin romance y violencia'
        raw = dict(lyric_positive=['home'], lyric_negative=['romance'], clap_positive=[], clap_negative=[],
                   constraints=[self.explicit('romance', quote)])
        with self.assertRaises(ValueError): prepare_plan(raw, quote)
        raw['lyric_negative'].append('violence')
        raw['constraints'].append(self.explicit('violence', quote))
        self.assertEqual(prepare_plan(raw, quote)['lyric_negative'], ['romance', 'violence'])

    def test_previous_explicit_constraints_are_revalidated(self):
        raw = dict(lyric_positive=['home'], lyric_negative=['violence'], clap_positive=[], clap_negative=[], constraints=[])
        previous = dict(raw, constraints=[self.explicit('violence')])
        with self.assertRaises(ValueError): prepare_plan(raw, 'sin romance', previous)

    def test_short_quote_cannot_hide_a_continued_negative_list(self):
        for query in ('sin romance y violencia', 'sin romance, violencia', 'sin romance or violence'):
            with self.subTest(query=query), self.assertRaises(ValueError):
                quoted_constraint(self.explicit(), query)

    def test_quote_substrings_cannot_invent_a_negation_or_concept(self):
        for item, query in [(self.explicit(), 'basin romance'),
                            (self.explicit(), 'sin romances'),
                            (self.explicit('piano', 'piano', 'positive', 'audio'), 'pianola')]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                quoted_constraint(item, query)

    def test_ambiguous_not_only_is_never_explicit_negative(self):
        for quote in ('no solo romance', 'not only romance'):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                quoted_constraint(self.explicit(quote.split(' ', 1)[1], quote), quote)

    def test_opposing_verified_constraints_require_clarification(self):
        query = 'romance pero sin amor romántico'
        raw = dict(lyric_positive=['romance'], lyric_negative=['romantic love'], clap_positive=[], clap_negative=[],
            constraints=[self.explicit('romance', 'romance', 'positive'), self.explicit('romantic love', 'sin amor romántico')])
        with self.assertRaises(ValueError): prepare_plan(raw, query)

    def test_orchestrator_cannot_select_a_legacy_approval(self):
        graph = build_workflow(planner=lambda _: dict(lyric_positive=['home'], lyric_negative=[], clap_positive=[], clap_negative=[]),
            search=lambda **_: ([(1, .9)], [], [(1, .9)]),
            judge=lambda **_: dict(status='approved', format='legacy', reason='old approval'), refiner=Mock(),
            lookup=lambda _: dict(title='Control', lyrics='Returning home'), infer_weights=lambda *_: (1., 0.))
        state = graph.invoke(initial_state('about home', 'legacy'))
        self.assertIsNone(state['selected_song'])
        self.assertEqual(state['judge_log'][0]['status'], 'abstained')

    def test_wrong_translation_stops_workflow_before_search(self):
        raw = dict(lyric_positive=['home'], lyric_negative=['violence'], clap_positive=[], clap_negative=[],
                   constraints=[self.explicit('violence')])
        search = Mock()
        graph = build_workflow(planner=lambda _: raw, search=search, judge=Mock(), refiner=Mock(),
                               lookup=Mock(), infer_weights=lambda *_: (1., 0.))
        result = graph.invoke(initial_state('Una letra sobre volver a casa sin romance', 'grounding'))
        self.assertEqual(result['status'], 'clarification_needed')
        search.assert_not_called()

    def evaluate(self, answers, positives=None):
        llm = Mock(side_effect=None)
        llm.invoke.side_effect = [SimpleNamespace(content=json.dumps(row) if isinstance(row, dict) else row,
            tool_calls=[], invalid_tool_calls=[]) for row in answers]
        with patch.object(judge, '_get_judge_llm', return_value=llm):
            result = judge.evaluate_song('home', 'Returning home with my family',
                ['home'] if positives is None else positives, [], 1, 'Control')
        return result, llm

    def test_legacy_approval_retries_then_abstains(self):
        result, llm = self.evaluate(['APPROVED', 'APPROVED'])
        self.assertEqual(result['status'], 'abstained')
        self.assertFalse(result['approved'])
        self.assertEqual(llm.invoke.call_count, 2)

    def test_legacy_approval_can_recover_with_verified_json(self):
        result, llm = self.evaluate(['APPROVED', dict(status='approved', reason='home', evidence=['Returning home'])])
        self.assertEqual(result['status'], 'approved')
        self.assertEqual(result['format'], 'structured')
        self.assertEqual(result['model_calls'], 2)

    def test_negative_only_approval_also_requires_evidence(self):
        result, llm = self.evaluate([dict(status='approved', reason='no violence', evidence=[])]*2, positives=[])
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['model_calls'], 2)

    def test_repeated_invented_quotes_do_not_approve_or_reject(self):
        for status in ['approved', 'rejected']:
            result, llm = self.evaluate([dict(status=status, reason='claim', evidence=['invented'])]*2)
            self.assertEqual(result['status'], 'abstained')

    def test_legacy_tool_cannot_exclude_even_if_repeated(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content='', invalid_tool_calls=[],
            tool_calls=[dict(name='exclude_song', args=dict(song_id='1', reason='wrong'))])
        pool = judge.ExclusionPool(2)
        with patch.object(judge, '_get_judge_llm', return_value=llm):
            result = judge.evaluate_song('q', 'words', ['q'], [], 1, 'Control', pool)
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(pool.excluded_ids, set())

    def packet(self):
        snapshot = SimpleNamespace(version='control-version', manifest={'sha256': {'control': 'hash'}},
            song_id_map={'0': 1}, get_song_by_id=lambda sid: dict(song_id=sid, title='Control', lyrics='Home'))
        fixture = dict(queries=[dict(id='home', query='about home', lyric_positive=['home'], lyric_negative=[],
                                    clap_positive=[], clap_negative=[])])
        return snapshot, fixture, review_packet(snapshot, fixture)

    def test_annotation_export_does_not_invent_human_judgments(self):
        snapshot, fixture, packet = self.packet()
        self.assertEqual(packet['judgments']['home']['grades'], {})
        self.assertIsNone(packet['reviewed_by'])
        with self.assertRaises(ValueError): reviewed_qrels(packet, snapshot, fixture)

    def test_import_requires_grades_and_matching_catalog_queries(self):
        snapshot, fixture, packet = self.packet()
        packet.update(status='human_reviewed', reviewed_by='Control reviewer')
        with self.assertRaises(ValueError): reviewed_qrels(packet, snapshot, fixture)
        packet['judgments']['home']['grades']['1'] = 3
        result = reviewed_qrels(packet, snapshot, fixture)
        self.assertEqual(result['judgments']['home']['grades'], {'1': 3})
        for key in ('catalog', 'queries'):
            wrong = deepcopy(packet)
            wrong[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError): reviewed_qrels(wrong, snapshot, fixture)

    def test_import_rejects_invalid_and_unknown_song_grades(self):
        snapshot, fixture, packet = self.packet()
        packet.update(status='human_reviewed', reviewed_by='Control reviewer')
        for grades in ({'1': None}, {'1': True}, {'1': 4}, {'2': 3}):
            packet['judgments']['home']['grades'] = grades
            with self.subTest(grades=grades), self.assertRaises(ValueError): reviewed_qrels(packet, snapshot, fixture)


if __name__ == '__main__': unittest.main()
